#!/bin/bash


VALUE_MODEL_PATH='/workspace/verl/Qwen2.5-Math-PRM-7B'
HOST_ADDR='127.0.0.1'
CONTROLLER_PORT='1234'
WORKER_PORT='4321'

export CUDA_VISIBLE_DEVICES=7
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export PYTHONPATH=$(pwd)
PYTHON_EXECUTABLE=$(which python)
LOGDIR=${PYTHONPATH}/logs_fastchat
export LOGDIR=$LOGDIR

session_name=tts
if tmux has-session -t $session_name 2>/dev/null; then
    echo "Session $session_name already exists. Killing it."
    tmux kill-session -t $session_name
fi

tmux start-server

tmux new-session -s $session_name -n controller -d
tmux send-keys "source ~/.bashrc && conda activate tts && cd ${PYTHONPATH}" Enter
tmux send-keys "${PYTHON_EXECUTABLE} -m fastchat.serve.controller --port ${CONTROLLER_PORT} --host ${HOST_ADDR}" Enter
echo "Controller started at ${HOST_ADDR}:${CONTROLLER_PORT}"

sleep 5

tmux new-window -n rm_worker
tmux send-keys "source ~/.bashrc && conda activate tts && cd ${PYTHONPATH}" Enter
if [[ "$VALUE_MODEL_PATH" =~ "dummy" ]]; then
    command="pwd"
else
    command="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ${PYTHON_EXECUTABLE} -m reward_worker.reward_model_worker \
        --model-path $VALUE_MODEL_PATH \
        --controller-address http://$HOST_ADDR:$CONTROLLER_PORT \
        --host $HOST_ADDR \
        --port $WORKER_PORT \
        --worker-address http://$HOST_ADDR:$WORKER_PORT \
        --limit-worker-concurrency 20"
fi
tmux send-keys "$command" Enter

echo "Reward model worker started on GPU 0, port $WORKER_PORT"
