#!/usr/bin/env bash
set -euo pipefail

# Example single-node run (edit paths + hyperparams for your platform).
# This is intended to be passed as arguments to `entrypoint.sh`, or executed inside the container.

HF_CKPT="${HF_CKPT:-/models/gpt-oss-20b-bf16}"
PROMPT_DATA="${PROMPT_DATA:-/data/prompts.jsonl}"

python -m async_rl.miles_train \
  --train-backend fsdp \
  --bf16 \
  --qkv-format bshd \
  --attn-implementation "kernels-community/vllm-flash-attn3" \
  --hf-checkpoint "${HF_CKPT}" \
  --model-name "gpt-oss-20b" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --rollout-num-gpus-per-engine 8 \
  --sglang-tensor-parallel-size 8 \
  --sglang-dtype bfloat16 \
  --prompt-data "${PROMPT_DATA}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type deepscaler \
  --num-rollout 1000 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 4 \
  --rollout-max-response-len 2048 \
  --rollout-temperature 0.8 \
  --advantage-estimator grpo \
  --kl-loss-coef 0.0 \
  --kl-coef 0.0 \
  --entropy-coef 0.0 \
  --optimizer adam \
  --lr 1e-6 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98

