from __future__ import annotations

from typing import Any


def kernelize_gpt_oss_for_training(model: Any, *, device: str = "cuda", compile: bool = False):
    """
    Kernelize GPT-OSS MoE layers for training using Hugging Face Kernel Hub.

    See `docs/gpt_oss_moe_kernels.md` for details on what is fused, what gradients
    are supported, and practical limitations (e.g. MXFP4 vs BF16 training).

    We intentionally keep the mapping minimal and explicit:
    - **MoE**: `MegaBlocksMoeMLP` from `kernels-community/megablocks`

    Attention sinks are handled separately via `attn_implementation="kernels-community/vllm-flash-attn3"`.
    """
    from kernels import LayerRepository, Mode, kernelize, register_kernel_mapping

    mode = Mode.TRAINING
    if compile:
        mode = mode | Mode.TORCH_COMPILE

    mapping = {
        # GPT-OSS uses @use_kernel_forward_from_hub("MegaBlocksMoeMLP") on its MLP module.
        "MegaBlocksMoeMLP": {
            device: {
                mode: LayerRepository(
                    repo_id="kernels-community/megablocks",
                    layer_name="MegaBlocksMoeMLP",
                )
            }
        }
    }

    # Keep any mappings that might already exist (e.g., for RMSNorm).
    register_kernel_mapping(mapping, inherit_mapping=True)

    # `kernelize` returns the model (same instance, but typed as nn.Module).
    return kernelize(model, mode=mode, device=device)

