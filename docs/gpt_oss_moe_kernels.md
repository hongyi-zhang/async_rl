## GPT‑OSS MoE: fused GPU kernels (forward + backward)

GPT‑OSS models are **token-choice MoEs** (e.g. `gpt-oss-20b` has 128 experts, top‑k=4). The MoE block must support:

- **Forward**: route tokens → run per-expert MLPs → combine outputs.
- **Backward**: gradients w.r.t. **hidden states**, **expert weights/biases**, and typically **router weights** (through the top‑k scores / routing weights).

This note summarizes what is available today in open source and what this repo relies on for **training**.

### 1) Baseline (Transformers eager MoE)

In `transformers.models.gpt_oss`, the reference MoE implementation is correct but not performant:

- Router: `Linear(hidden → num_experts)` + `topk` + `softmax`
- Experts: per-expert MLP + weighted sum

The default expert computation is not a single fused kernel and can fall back to very slow Python-level loops depending on the path. It supports backward because it is pure PyTorch, but it is not the “fused MoE kernel” you want for large-scale training.

### 2) Training-capable fused kernels via Hugging Face Kernel Hub (MegaBlocks)

**Best current option for fused MoE forward+backward on CUDA/Hopper** is the Kernel Hub implementation:

- **Kernel repo**: [`kernels-community/megablocks`](https://huggingface.co/kernels-community/megablocks)
- **Layer used by GPT‑OSS**: `MegaBlocksMoeMLP`

Transformers’ GPT‑OSS MLP is decorated with `@use_kernel_forward_from_hub("MegaBlocksMoeMLP")`, which makes it “kernel-aware”. When the model is kernelized (e.g. `use_kernels=True` or explicit `kernels.kernelize`), the MLP forward is replaced by `kernels-community/megablocks:MegaBlocksMoeMLP`.

#### What is actually fused

MegaBlocks is not one monolithic MoE CUDA kernel. Instead it combines several **specialized fused kernels**:

- **Token permutation (gather/scatter)** into expert-major layout:
  - `binned_gather`, `binned_scatter`, plus expert-weight (`topk_weights`) handling
  - Implemented as Triton kernels in `megablocks/backend/kernels.py`
- **Gradients for routing weights**:
  - `binned_scatter_wgrad` / `scatter_wgrad` compute `d(topk_weights)` efficiently
- **Expert MLP compute**:
  - uses batched GEMMs (`torch.bmm`, i.e. cuBLAS) and/or grouped GEMM (CUTLASS-backed) depending on the path (`mlp_impl="grouped"` is common)

This combination is the “fused MoE kernel” story in practice: the expensive data movement and the top‑k weighted combination are handled by custom kernels, while the dense math uses highly optimized GEMMs.

#### Backward support (what gets gradients)

The crucial point for training is that MegaBlocks supplies **autograd wrappers** for its custom kernels. Two important examples:

- `binned_gather` backward uses `binned_scatter` to return `d(x)`
- `binned_scatter` backward returns both:
  - `d(x)` via `binned_gather`
  - **`d(weights)`** via a dedicated kernel `binned_scatter_wgrad` when routing weights require grad

That means gradients can flow to:

- **hidden states** `x`
- **expert weights/biases** (through GEMMs)
- **router weights** (because `topk_weights` depends on router logits and receives gradients via the scatter wgrad path)

So, on CUDA, `kernels-community/megablocks` supports **both forward and backward** for GPT‑OSS MoE.

#### Practical limitations

- **Dtype / weight format**: MegaBlocks kernels are designed for FP16/BF16 training. GPT‑OSS checkpoints ship with **MXFP4 weight-only quantization on MoE weights**. In practice, training commonly:
  - **dequantizes to BF16** for fine-tuning (what Miles’ `run-gptoss-20b-fsdp.sh` does), or
  - uses QAT (e.g. NVIDIA ModelOpt) and converts back to MXFP4 for inference.

### 3) MXFP4 MoE kernels (mostly inference-focused)

OpenAI’s `gpt-oss` repo and Triton include **MXFP4 MoE kernels** (e.g. “fused experts” style kernels) that are excellent for **inference** and single-GPU reference implementations, but **training-grade backward support is not broadly available** in the same way as MegaBlocks BF16/FP16 kernels.

If your goal is RL fine-tuning in Miles today, the pragmatic path is:

- convert / load GPT‑OSS as **BF16** for training, and
- use MegaBlocks for fused MoE forward/backward.

### 4) How this repo enables MegaBlocks for GPT‑OSS training

This repo kernelizes the GPT‑OSS MoE MLP explicitly in training mode:

- `async_rl/patches/hf_kernels.py` registers the mapping:
  - `MegaBlocksMoeMLP` → `kernels-community/megablocks:MegaBlocksMoeMLP` (Mode.TRAINING)
- the patch is applied in Miles FSDP actor init (so it runs inside Ray workers as well).

