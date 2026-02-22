## Miles + GPT-OSS RL Training (H100 / Hopper)

Standalone repo to run **Miles** RL training jobs for **GPT‑OSS** models on a multi-node H100 cluster, using the **correct GPT‑OSS kernels**:

- **Attention sinks (fwd/bwd)**: `kernels-community/vllm-flash-attn3` (FlashAttention‑3 + attention sinks).
- **MoE (fwd/bwd)**: `kernels-community/megablocks` (MegaBlocks MoE kernels), applied to GPT‑OSS `MegaBlocksMoeMLP` during training.

This repo provides:

- **`Dockerfile`**: builds a training image on top of a *clean* `Debian 12 + CUDA 12.9 + Python 3.12` base image.
- **`entrypoint.sh`**: starts/joins a Ray cluster and launches Miles training on the head node.

## Notes on the kernel integration

- **Attention sinks**: GPT‑OSS uses a learnable per-head sink (`s_aux`) in attention. The Hub kernel implements sink logic in forward; upstream historically did **not** return a gradient for the sink parameter. This repo applies a **Python-level backward patch** (no CUDA rebuild) so `s_aux` receives gradients during training.
- **MoE kernels**: GPT‑OSS MoE is kernelized by mapping `MegaBlocksMoeMLP` → `kernels-community/megablocks:MegaBlocksMoeMLP` in **TRAINING** mode (includes backward).
- **Patch propagation to Ray workers**: patches are applied via `sitecustomize.py`, enabled by `ASYNC_RL_ENABLE_PATCHES=1` (set by `entrypoint.sh`).

## Build the docker image

Your platform should provide a base image that already has **Debian 12 + CUDA 12.9 + Python 3.12**.

```bash
docker build -t miles-gpt-oss:dev \
  --build-arg BASE_IMAGE="<your-debian12-cuda12.9-py3.12-base>" \
  .
```

## Run on a multi-node cluster

Run **one container per node**, with the same `entrypoint.sh` and args. Minimal environment:

- **Head node**:
  - `NODE_RANK=0`
  - `RAY_HEAD_ADDR=<head-ip-or-hostname>`
- **Worker nodes**:
  - `NODE_RANK=<0..N-1>`
  - `RAY_HEAD_ADDR=<head-ip-or-hostname>`

Common:

- `NUM_GPUS_PER_NODE` (default: 8)
- `RAY_PORT` (default: 6379)
- `RAY_DASHBOARD_PORT` (default: 8265)

Example (head node):

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

Workers run the same container but with `NODE_RANK!=0` and will just join Ray and block.

## Convert GPT‑OSS MXFP4 → BF16 (optional)

For RL fine-tuning it’s common to **dequantize** the MXFP4 MoE weights to BF16 first (the sample in Miles does this).

```bash
python scripts/convert_gpt_oss_mxfp4_to_bf16.py \
  --model-id openai/gpt-oss-20b \
  --output-dir /models/gpt-oss-20b-bf16
```

## Example command

See `configs/run_gpt_oss_20b_grpo_fsdp.example.sh` for a fuller GRPO example.

## Environment toggles

- **`ASYNC_RL_ENABLE_PATCHES`**: enable repo patches in all Python processes (Ray workers too). Default set to `1` by `entrypoint.sh`.
- **`ASYNC_RL_ENABLE_HF_KERNELS`**: kernelize GPT‑OSS MoE with MegaBlocks during training. Default `1`.
- **`ASYNC_RL_PATCH_FA3_SINK_BWD`**: apply sink-parameter backward patch for FA3. Default `1`.
- **`ASYNC_RL_FORCE_GPT_OSS_ATTN_IMPL`**: force GPT‑OSS attention backend after model init (default `1`).
- **`ASYNC_RL_GPT_OSS_ATTN_IMPL`**: the attention backend string to force (default `kernels-community/vllm-flash-attn3`).
- **`ASYNC_RL_KERNEL_DEVICE`**: device string for kernel loading (default `cuda`).
- **`ASYNC_RL_KERNEL_TORCH_COMPILE`**: also request compile-friendly kernels (default `0`).

