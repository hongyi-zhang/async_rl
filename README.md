## Miles + GPT-OSS RL Training

This repo now documents two GPT-OSS training paths:

1. **FSDP BF16 kernelized training** (default in this repo):
   - FA3 sink attention patch (`kernels-community/vllm-flash-attn3` + sink backward fix)
   - MegaBlocks MoE training kernels (`kernels-community/megablocks`)
2. **Unified FP8 training/inference for GPT-OSS-120B** (Megatron + Bridge path):
   - shared FP8 quantization flow between rollout and training
   - implemented in the forked Miles branch pinned in this repo Dockerfile

## What this repo provides

- **`Dockerfile`**: image build on top of your CUDA/Python base image.
- **`entrypoint.sh`**: Ray start/join wrapper and head-node launcher.
- **`async_rl` patches**: FSDP GPT-OSS kernel patching via `sitecustomize.py`.
- **Example configs**:
  - `configs/run_gpt_oss_20b_grpo_fsdp.example.sh` (FSDP BF16)
  - `configs/run_gpt_oss_120b_grpo_megatron_fp8_bridge.example.sh` (Megatron unified FP8)

## Build the Docker image

For the FSDP BF16 path, a base with Debian 12 + CUDA 12.9 + Python 3.12 is sufficient.
For Megatron unified FP8 on a bare CUDA+Python base, set `INSTALL_MEGATRON_STACK=1` so the Dockerfile installs the required Megatron/TE/Apex/Bridge stack.

```bash
docker build -t miles-gpt-oss:dev \
  --build-arg BASE_IMAGE="<your-debian12-cuda12.9-py3.12-base>" \
  --build-arg TORCH_VERSION="2.9.1" \
  --build-arg TORCH_CUDA_TAG="cu129" \
  --build-arg INSTALL_MEGATRON_STACK=1 \
  --build-arg NCCL_VERSION="v2.27.7-1" \
  --build-arg MILES_REPO="https://github.com/hongyi-zhang/miles.git" \
  --build-arg MILES_REF="3cba08a9a8a4f53f1d128a85efb86364cf39589b" \
  .
```

Set `INSTALL_MEGATRON_STACK=0` (default) if your base image already provides a compatible Megatron stack.
When `INSTALL_MEGATRON_STACK=1`, the image now builds NCCL from source (default `NCCL_VERSION=v2.27.7-1`) so `transformer_engine` can compile on bare CUDA 12.9 bases.
`MILES_REPO` / `MILES_REF` are configurable; defaults in `Dockerfile` already point to the fork commit with GPT-OSS unified-FP8 bridge fixes.

## Run on a multi-node cluster

Run one container per node with the same entrypoint:

- Head node:
  - `NODE_RANK=0`
  - `RAY_HEAD_ADDR=<head-ip-or-hostname>`
- Worker nodes:
  - `NODE_RANK=<0..N-1>`
  - `RAY_HEAD_ADDR=<head-ip-or-hostname>`

Common:

- `NUM_GPUS_PER_NODE` (default: 8)
- `RAY_PORT` (default: 6379)
- `RAY_DASHBOARD_PORT` (default: 8265)

Example (head node, FSDP BF16):

```bash
docker run --gpus all --rm \
  -e NODE_RANK=0 \
  -e RAY_HEAD_ADDR="$(hostname -i | awk '{print $1}')" \
  -e NUM_GPUS_PER_NODE=8 \
  miles-gpt-oss:dev \
  --train-backend fsdp \
  --bf16 \
  --qkv-format bshd \
  --attn-implementation "kernels-community/vllm-flash-attn3" \
  --hf-checkpoint /models/gpt-oss-20b-bf16 \
  --rollout-batch-size 32 \
  --n-samples-per-prompt 4
```

Workers run the same container with `NODE_RANK!=0` and wait in the Ray worker loop.

## Convert GPT-OSS MXFP4 -> BF16 (optional)

For BF16 FSDP fine-tuning, it is common to dequantize MXFP4 MoE weights first:

```bash
python scripts/convert_gpt_oss_mxfp4_to_bf16.py \
  --model-id openai/gpt-oss-20b \
  --output-dir /models/gpt-oss-20b-bf16
```

## Unified FP8 (GPT-OSS-120B) quick start

Use the example file:

```bash
bash configs/run_gpt_oss_120b_grpo_megatron_fp8_bridge.example.sh
```

This path uses:

- `--train-backend megatron`
- `--megatron-to-hf-mode bridge`
- `--model-name gpt_oss`
- `--transformer-impl transformer_engine`
- `--fp8-format e4m3 --fp8-recipe blockwise`

## FSDP kernel patch toggles

These are only relevant to the FSDP kernelized GPT-OSS path:

- `ASYNC_RL_ENABLE_PATCHES` (default `1`)
- `ASYNC_RL_ENABLE_HF_KERNELS` (default `1`)
- `ASYNC_RL_PATCH_FA3_SINK_BWD` (default `1`)
- `ASYNC_RL_FORCE_GPT_OSS_ATTN_IMPL` (default `1`)
- `ASYNC_RL_GPT_OSS_ATTN_IMPL` (default `kernels-community/vllm-flash-attn3`)
- `ASYNC_RL_KERNEL_DEVICE` (default `cuda`)
- `ASYNC_RL_KERNEL_TORCH_COMPILE` (default `0`)

