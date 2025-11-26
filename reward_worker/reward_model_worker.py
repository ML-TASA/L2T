# Copyright 2025 ML-TASA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file is largely borrowed from FastChat (https://github.com/lm-sys/FastChat) and OpenR (https://github.com/openreasoner/openr)
"""

import argparse
import base64
import gc
import json
import os
from typing import List, Optional
import uuid
import functools
import time
import torch
import torch.nn.functional as F
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM, AutoModel
from safetensors import safe_open
import uvicorn

from fastchat.constants import ErrorCode, SERVER_ERROR_MSG
from fastchat.model.model_adapter import (
    load_model,
    add_model_args,
    get_generate_stream_function,
)
from fastchat.modules.awq import AWQConfig
from fastchat.modules.exllama import ExllamaConfig
from fastchat.modules.xfastertransformer import XftConfig
from fastchat.modules.gptq import GptqConfig
from fastchat.utils import (
    build_logger,
    get_context_length,
    str_to_torch_dtype,
)
from reward_worker.base_model_worker import BaseModelWorker, app
from reward_worker.rm_call import get_infer_fn, get_prm_special_tokens
# from reason.llm_service.workers.Process_Q_Model.value_model import AutoModelForCausalLMWithValueHead

worker_id = str(uuid.uuid4())[:8]
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))[2:]
logger = build_logger("reward_model_worker", f"rm_{time_str}.log")


class ModelWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        device: str,
        num_gpus: int,
        max_gpu_memory: str,
        revision: str = None,
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        gptq_config: Optional[GptqConfig] = None,
        awq_config: Optional[AWQConfig] = None,
        exllama_config: Optional[ExllamaConfig] = None,
        xft_config: Optional[XftConfig] = None,
        stream_interval: int = 2,
        conv_template: Optional[str] = None,
        embed_in_truncate: bool = False,
        seed: Optional[int] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template=conv_template,
        )

        logger.info(f"Loading the model {self.model_names} on worker {worker_id} ...")

        infer_fn = get_infer_fn(model_path, rm_serve_type='fastchat')
        if 'skywork' in model_path.lower():

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map=device).eval()
            prm_step_tag = '\n'
            step_tag_id = self.tokenizer.encode(prm_step_tag)[-1]

            self.infer_fn = functools.partial(infer_fn, model=self.model, tokenizer=self.tokenizer, device=device, step_tag_id=step_tag_id)
        elif 'qwen2.5-math' in model_path.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
            self.model.config.use_cache = False
            prm_step_tag = "<extra_0>"
            step_tag_id = self.tokenizer.encode(prm_step_tag)[0]

            self.infer_fn = functools.partial(infer_fn, model=self.model, tokenizer=self.tokenizer, device=device, special_tag_id=step_tag_id)
        elif 'pqm' in model_path.lower(): 
            from transformers import AutoModelForCausalLMWithValueHead

            prm_step_tag = '[PRM]'
            if '.safetensors' not in model_path:
                model_path = os.path.join(model_path, 'model.safetensors')
            model_dir = ''.join(model_path.split('/')[:-2])
            backbone_model_path = os.path.join(model_dir, 'models--deepseek-ai--deepseek-math-7b-base')
            self.tokenizer = AutoTokenizer.from_pretrained(backbone_model_path)
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(backbone_model_path, torch_dtype=torch.bfloat16)
            self.tokenizer.add_special_tokens({'additional_special_tokens': [prm_step_tag]})
            step_tag_id = self.tokenizer.encode(prm_step_tag, add_special_tokens=False)[-1]
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model = AutoModelForCausalLMWithValueHead(self.model)
            if '.safetensors' in model_path:
                state_dict = {}
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_path)

            self.model.load_state_dict(state_dict)
            self.model.to(device).eval()

            self.infer_fn = functools.partial(infer_fn, model=self.model, tokenizer=self.tokenizer, device=device, step_tag_id=step_tag_id)
        else:
            self.model, self.tokenizer = load_model(
                model_path,
                revision=revision,
                device=device,
                num_gpus=num_gpus,
                max_gpu_memory=max_gpu_memory,
                dtype=dtype,
                load_8bit=load_8bit,
                cpu_offloading=cpu_offloading,
                gptq_config=gptq_config,
                awq_config=awq_config,
                exllama_config=exllama_config,
                xft_config=xft_config,
                debug=debug,
            )

            step_tag_id, returned_token_ids = get_prm_special_tokens(model_path, self.tokenizer)
            if "math-shepherd" in model_path.lower():
                self.infer_fn = functools.partial(
                    infer_fn, model=self.model, tokenizer=self.tokenizer, device=device, returned_token_ids=returned_token_ids, step_tag_id=step_tag_id
                )
            elif "mistral-data" in model_path.lower():
                self.tokenizer.padding_side = "right"
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id
                self.infer_fn = functools.partial(infer_fn, model=self.model, tokenizer=self.tokenizer, device=device, returned_token_ids=returned_token_ids)
            elif "deepseek-data" in model_path.lower():
                self.tokenizer.padding_side = "right"
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id
                self.infer_fn = functools.partial(infer_fn, model=self.model, tokenizer=self.tokenizer, device=device, returned_token_ids=returned_token_ids)
            elif "llama3.1-math-prm" in model_path.lower():
                self.tokenizer.padding_side = "right"
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id
                self.infer_fn = functools.partial(infer_fn, model=self.model, tokenizer=self.tokenizer, device=device, returned_token_ids=returned_token_ids)
                self.infer_fn = functools.partial(
                    infer_fn, model=self.model, tokenizer=self.tokenizer, device=device, returned_token_ids=returned_token_ids, step_tag_id=step_tag_id
                )
            else:
                self.infer_fn = functools.partial(infer_fn, model=self.model, tokenizer=self.tokenizer, device=device)

        self.device = device
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_len = get_context_length(self.model.config)
        self.generate_stream_func = get_generate_stream_function(self.model, model_path)
        self.stream_interval = stream_interval
        self.embed_in_truncate = embed_in_truncate
        self.seed = seed

        if not no_register:
            self.init_heart_beat()

    @torch.inference_mode()
    def reward_inference_gate(self, params):
        input_str = params["input_str"]
        try:
            if isinstance(input_str, list):
                reward = [r if isinstance(r, list) else r.tolist() for r in self.infer_fn(input_str)]
            else:
                reward = self.infer_fn(input_str).tolist()
            # ret = {"input": input_str, "reward": reward}
            ret = {"reward": reward}
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            ret = {
                "text": f"{SERVER_ERROR_MSG}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }
        return ret


def create_model_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=10081)
    parser.add_argument("--worker-address", type=str, default="http://localhost:10081")
    parser.add_argument("--controller-address", type=str, default="http://localhost:10014")
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--conv-template", type=str, default=None, help="Conversation prompt template.")
    parser.add_argument("--embed-in-truncate", action="store_true")
    parser.add_argument(
        "--limit-worker-concurrency",
        type=int,
        default=5,
        help="Limit the model concurrency to prevent OOM.",
    )
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Overwrite the random seed for each generation.",
    )
    parser.add_argument("--debug", type=bool, default=False, help="Print debugging messages")
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )

    # parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--good_tag", type=str, default="+")
    parser.add_argument("--bad_tag", type=str, default="-")
    parser.add_argument("--prm_step_tag", type=str, default="ки\n")
    parser.add_argument("--prm_format_str", type=str, default="{question} {answer}")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )
    awq_config = AWQConfig(
        ckpt=args.awq_ckpt or args.model_path,
        wbits=args.awq_wbits,
        groupsize=args.awq_groupsize,
    )
    if args.enable_exllama:
        exllama_config = ExllamaConfig(
            max_seq_len=args.exllama_max_seq_len,
            gpu_split=args.exllama_gpu_split,
            cache_8bit=args.exllama_cache_8bit,
        )
    else:
        exllama_config = None
    if args.enable_xft:
        xft_config = XftConfig(
            max_seq_len=args.xft_max_seq_len,
            data_type=args.xft_dtype,
        )
        if args.device != "cpu":
            print("xFasterTransformer now is only support CPUs. Reset device to CPU")
            args.device = "cpu"
    else:
        xft_config = None

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        revision=args.revision,
        no_register=args.no_register,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        gptq_config=gptq_config,
        awq_config=awq_config,
        exllama_config=exllama_config,
        xft_config=xft_config,
        stream_interval=args.stream_interval,
        conv_template=args.conv_template,
        embed_in_truncate=args.embed_in_truncate,
        seed=args.seed,
        debug=args.debug,
        args=args,
    )
    return args, worker


if __name__ == "__main__":
    args, worker = create_model_worker()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
