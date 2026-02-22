from __future__ import annotations

import os
import sys
from typing import Any, Callable


_PATCHED = False


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _is_gpt_oss_config(hf_config: Any) -> bool:
    try:
        return getattr(hf_config, "model_type", None) == "gpt_oss"
    except Exception:
        return False


def apply_miles_fsdp_gpt_oss_patches() -> None:
    """
    Patches Miles FSDP training actor to:
    - kernelize GPT-OSS MoE (MegaBlocks) for TRAINING mode
    - optionally patch FA3 sink backward for learnable sinks gradient

    This is intended to run in *every* process (driver + Ray workers), via `sitecustomize.py`.
    """
    global _PATCHED
    if _PATCHED:
        return

    if not _env_flag("ASYNC_RL_ENABLE_MILES_FSDP_PATCH", default="1"):
        _PATCHED = True
        return

    try:
        import miles.backends.fsdp_utils.actor as actor_mod
    except Exception:
        # Miles not installed / not used in this process.
        _PATCHED = True
        return

    orig_init_model: Callable[..., Any] = actor_mod.FSDPTrainRayActor.init_model

    def init_model_patched(self, args, with_ref: bool = True):  # type: ignore[no-untyped-def]
        start_rollout_id = orig_init_model(self, args, with_ref=with_ref)

        try:
            # Only do this for GPT-OSS. This avoids accidentally kernelizing other models
            # with incompatible MoE implementations.
            if not _is_gpt_oss_config(getattr(self, "hf_config", None)):
                return start_rollout_id

            if getattr(self, "_async_rl_kernelized", False):
                return start_rollout_id

            # Force a GPT-OSS compatible attention implementation.
            # GPT-OSS requires attention sinks; FA2/SDPA backends are typically incompatible.
            if _env_flag("ASYNC_RL_FORCE_GPT_OSS_ATTN_IMPL", default="1"):
                impl = os.environ.get("ASYNC_RL_GPT_OSS_ATTN_IMPL", "kernels-community/vllm-flash-attn3")
                cfg = getattr(self.model, "config", None)
                if cfg is not None:
                    try:
                        cfg._attn_implementation = impl
                    except Exception:
                        pass
                    try:
                        cfg.attn_implementation = impl
                    except Exception:
                        pass

            # Attention sinks backward for learnable `s_aux` is missing upstream.
            # We provide a safe Python-level gradient patch.
            if _env_flag("ASYNC_RL_PATCH_FA3_SINK_BWD", default="1"):
                from async_rl.patches.vllm_flash_attn3_sink_bwd import patch_vllm_flash_attn3_s_aux_backward

                patch_vllm_flash_attn3_s_aux_backward()

            # Kernelize MoE for TRAINING (needs backward).
            if _env_flag("ASYNC_RL_ENABLE_HF_KERNELS", default="1"):
                kernel_device = os.environ.get("ASYNC_RL_KERNEL_DEVICE", "cuda")
                compile_mode = _env_flag("ASYNC_RL_KERNEL_TORCH_COMPILE", default="0")

                from async_rl.patches.hf_kernels import kernelize_gpt_oss_for_training

                self.model = kernelize_gpt_oss_for_training(self.model, device=kernel_device, compile=compile_mode)

            setattr(self, "_async_rl_kernelized", True)
        except Exception as e:  # pragma: no cover
            print(f"[async_rl] WARNING: GPT-OSS kernelization/patch failed: {e}", file=sys.stderr)

        return start_rollout_id

    actor_mod.FSDPTrainRayActor.init_model = init_model_patched  # type: ignore[assignment]

    _PATCHED = True

