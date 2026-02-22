#!/usr/bin/env bash
set -euo pipefail

# Example launch for GPT-OSS-120B unified FP8 RL
# (Megatron backend + Megatron Bridge in Miles).

HF_CKPT_FP8="${HF_CKPT_FP8:-/models/gpt-oss-120b-fp8}"
HF_CKPT_BF16="${HF_CKPT_BF16:-/models/gpt-oss-120b}"
PROMPT_DATA="${PROMPT_DATA:-/data/dapo-math-17k.jsonl}"
EVAL_DATA="${EVAL_DATA:-/data/aime-2024.jsonl}"

python -m async_rl.miles_train \
  --train-backend megatron \
  --megatron-to-hf-mode bridge \
  --model-name gpt_oss \
  --transformer-impl transformer_engine \
  --bf16 \
  --fp8-format e4m3 \
  --fp8-recipe blockwise \
  --hf-checkpoint "${HF_CKPT_FP8}" \
  --ref-load "${HF_CKPT_BF16}" \
  --actor-num-nodes 2 \
  --actor-num-gpus-per-node 8 \
  --num-gpus-per-node 8 \
  --colocate \
  --tensor-model-parallel-size 1 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 \
  --prompt-data "${PROMPT_DATA}" \
  --input-key prompt \
  --label-key label \
  --apply-chat-template \
  --rollout-shuffle \
  --rm-type deepscaler \
  --num-rollout 3000 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 8 \
  --rollout-max-response-len 8192 \
  --rollout-temperature 1.0 \
  --global-batch-size 256 \
  --advantage-estimator grpo \
  --use-kl-loss \
  --kl-loss-coef 0.0 \
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --use-tis \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --rollout-num-gpus-per-engine 16 \
  --sglang-ep-size 8 \
  --sglang-mem-fraction-static 0.7 \
  --use-miles-router \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --eval-interval 20 \
  --eval-prompt-data aime "${EVAL_DATA}" \
  --n-samples-per-eval-prompt 16 \
  --eval-max-response-len 16384 \
  --eval-top-p 1
