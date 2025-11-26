# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from reward_worker.rm_call import RMRemoteCaller, RemoteRewardModelConfig, get_prm_special_tokens


# -*- coding: utf-8 -*-
from typing import List
from transformers import AutoTokenizer


PRM_STEP_TAG = " ки\n"
CONTROLLER_ADDR = "http://127.0.0.1:1234"
RM_SERVE_TYPE = "fastchat"
MULTI_GPU = True
LM = "qwen2.5"
RM = "/workspace/verl/Qwen2.5-Math-PRM-7B"

COT_PROMPT_DICT = {
    'llama_official': """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.""",
    'qwen': """Please reason step by step, and put your final answer within \\boxed{}.""",
    'default': """Please reason step by step, and put your final answer within \\boxed{}.""",
}

LLM_STEP_TAG_DICT = {"llama": "## Step ", "qwen": "\nStep ", "default": "\nStep "}
SEP_DICT = {"llama": ["## Step"], "qwen": ["\nStep"], "default": ["\nStep"]}
STOP_STR_DICT = {"llama": ["\\boxed"], "qwen": ["\\boxed"], "default": ["\\boxed"]}

LM_SYLE = {
    "cot_prompt": COT_PROMPT_DICT["qwen"],
    "llm_step_tag": LLM_STEP_TAG_DICT["qwen"],
    "sep": SEP_DICT["qwen"],
    "stop_str": STOP_STR_DICT["qwen"],
}

def parse_think_response(text: str):
    """
    Parse a reasoning response formatted as:
        <think>
            <episode_1>...</episode_1>
            <episode_2>...</episode_2>
            ...
        </think>
    """
    result = {"episodes": []}

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if not think_match:
        return result 

    think_content = think_match.group(1).strip()

    episodes = re.findall(r"<episode_(\d+)>(.*?)</episode_\1>", think_content, re.DOTALL)
    episodes = sorted(episodes, key=lambda x: int(x[0])) 

    result["episodes"] = [content.strip() for _, content in episodes]
    return result


def score_with_rm(
    rm_call: RMRemoteCaller,
    rm_model_name: str,
    query: str,
    response_segments: List[str],
    prm_step_tag: str,
    verbose: bool = False,
):

    joined_response = f"{prm_step_tag}".join(response_segments)
    input_list = [(query, joined_response)]
    value_list = rm_call(input_list, model_names=rm_model_name, verbose=verbose)
    return value_list

def find_episode_end_positions(token_ids: torch.Tensor, tokenizer, text) -> list:
    """
    Find all positions of '</episode_k>' tokens in a sequence.
    
    Args:
        token_ids (torch.Tensor): shape (L,), dtype=torch.int64
        tokenizer: a tokenizer (e.g., from transformers.AutoTokenizer)
    
    Returns:
        List[int]: positions (end indices in token_ids) where '</episode_k>' ends
    """
    matches = list(re.finditer(r"</episode_\d+>", text))

    positions = []
    for m in matches:
        substring = text[:m.end()] 
        sub_tokens = tokenizer.encode(substring, add_special_tokens=False)
        positions.append(len(sub_tokens) - 1) 

    return positions

def extract_user_content(text: str) -> str:
    match = re.search(r"user\n\s*(.*?)\s*\nassistant\n", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return ""

def fill_episode_deltas(zero_tensor,
                        episode_positions,
                        prm_scores,
                        i,
                        info_decay,
                        beta = 1e-3
                        ):
    """
    Fill zero_tensor[i, start:end] with PRM score differences between consecutive episodes.

    Args:
        zero_tensor (torch.Tensor): shape [N, FIX_length]
        episode_positions (list[int]): e.g. [22, 37, 54]
        prm_scores (list[float]): e.g. [0.953125, 0.9921875, 0.9921875]
        i (int): index along the batch dimension to update
    """
    if len(episode_positions) >= 2:
        for j in range(len(episode_positions) - 1):
            start = episode_positions[j]
            end = episode_positions[j + 1]
            delta = prm_scores[j + 1] - prm_scores[j]
            zero_tensor[i, start:end] = delta

@register("l2t")
class L2TRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the L2TRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

        # This is the code for calling a pre-deployed reward model
        self.rm_caller = RMRemoteCaller

        rm_tokenizer = AutoTokenizer.from_pretrained(RM, trust_remote_code=True)
        step_tag_id, returned_token_ids = get_prm_special_tokens(RM, rm_tokenizer)

        prm_format_str = "{question} {answer}"

        self.rm_config = RemoteRewardModelConfig(
            prm_step_tag=PRM_STEP_TAG,
            format_str=prm_format_str,
            model_name=RM,
            controller_addr=CONTROLLER_ADDR,
            step_tag_id=step_tag_id,
            returned_token_ids=returned_token_ids,
            rm_serve_type=RM_SERVE_TYPE,
            multi_gpu=MULTI_GPU,
        )
        self.rm_call = RMRemoteCaller(self.rm_config, tokenizer=rm_tokenizer)

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            episode_positions = find_episode_end_positions(valid_response_ids, self.tokenizer, response_str)

            # This is for PRM score
            user_prompt = extract_user_content(prompt_str)
            episodes = parse_think_response(response_str)["episodes"]
            
            value_list = score_with_rm(self.rm_call, RM, user_prompt, episodes, PRM_STEP_TAG)[0]

            # This is for correctness
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            # penalize invalid format
            if len(value_list) != len(episode_positions):
                reward -= 0.01
            elif len(value_list) == 0:
                reward -= 0.01
            else:
                # fill reward tensor
                info_decay_i = data_item.batch.get("info_decay", None)
                if info_decay_i is None:
                    info_decay_i = 0.0
                fill_episode_deltas(reward_tensor, episode_positions, value_list, i, info_decay_i)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
