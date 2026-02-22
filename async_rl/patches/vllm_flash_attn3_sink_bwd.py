from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Optional


def _try_import(name: str) -> Optional[ModuleType]:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _locate_flash_attn_interface_module(kernel_mod: ModuleType) -> Optional[ModuleType]:
    """
    The kernels library loads hub kernels into uniquely-named Python packages.
    We try a few reasonable module name patterns to find the `flash_attn_interface` module.
    """
    candidates = []
    if getattr(kernel_mod, "__name__", None):
        candidates.append(f"{kernel_mod.__name__}.flash_attn_interface")
    if getattr(kernel_mod, "__package__", None):
        candidates.append(f"{kernel_mod.__package__}.flash_attn_interface")

    for name in candidates:
        m = _try_import(name)
        if m is not None:
            return m

    # Fallback: scan already-imported modules for the interface.
    for name, mod in list(sys.modules.items()):
        if not isinstance(mod, ModuleType):
            continue
        if name.endswith(".flash_attn_interface") and hasattr(mod, "FlashAttnFunc"):
            return mod
    return None


def patch_vllm_flash_attn3_s_aux_backward() -> bool:
    """
    Patch `kernels-community/vllm-flash-attn3` so the learnable sink parameter (`s_aux`)
    receives a gradient during backward.

    See `docs/flash_attn3_sink_backward_patch.md` for the derivation, integration details,
    and performance expectations.

    Upstream FA3 sink support adds `s_aux` to the forward path but does not return `ds_aux`
    from backward (see HF discussion #3 on the kernel repo). This patch computes:

      dL/ds = - sum_{b,t,d} (dout * out) * p_sink
      p_sink = exp(s - logZ)

    where `logZ` is `softmax_lse` returned by the kernel (log-sum-exp of logits incl. sink).
    """
    try:
        from kernels import get_kernel
    except Exception:
        return False

    try:
        kernel_mod = get_kernel("kernels-community/vllm-flash-attn3")
    except Exception:
        return False

    iface = _locate_flash_attn_interface_module(kernel_mod)
    if iface is None:
        return False

    FlashAttnFunc = getattr(iface, "FlashAttnFunc", None)
    if FlashAttnFunc is None:
        return False

    if getattr(FlashAttnFunc, "_async_rl_sink_bwd_patched", False):
        return True

    import torch

    orig_forward = FlashAttnFunc.forward
    orig_backward = FlashAttnFunc.backward

    @staticmethod
    def forward_patched(  # type: ignore[no-untyped-def]
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        s_aux=None,
    ):
        out, softmax_lse = orig_forward(
            ctx,
            q,
            k,
            v,
            softmax_scale,
            causal,
            qv,
            q_descale,
            k_descale,
            v_descale,
            window_size,
            softcap,
            num_splits,
            pack_gqa,
            deterministic,
            sm_margin,
            s_aux,
        )
        # Store for ds_aux computation in backward (do not add to saved_tensors list).
        ctx._async_rl_s_aux = s_aux
        return out, softmax_lse

    @staticmethod
    def backward_patched(ctx, dout, *args):  # type: ignore[no-untyped-def]
        grads = list(orig_backward(ctx, dout, *args))
        s_aux = getattr(ctx, "_async_rl_s_aux", None)
        if s_aux is None:
            return tuple(grads)

        # ctx.saved_tensors = (q, k, v, out, softmax_lse)
        try:
            _, _, _, out, softmax_lse = ctx.saved_tensors
        except Exception:
            return tuple(grads)

        # softmax_lse: (B, H, S)
        # out/dout:    (B, S, H, D)
        # s_aux:       (H,) or (B, H)
        softmax_lse_f = softmax_lse.float()
        out_f = out.float()
        dout_f = dout.float()

        # (B, S, H)
        dot = (dout_f * out_f).sum(dim=-1)
        # (B, H, S)
        dot_bhs = dot.permute(0, 2, 1).contiguous()

        if s_aux.dim() == 1:
            # (B, H, S)
            p_sink = torch.exp(s_aux.float().view(1, -1, 1) - softmax_lse_f)
            # (H,)
            ds = -(dot_bhs * p_sink).sum(dim=(0, 2))
        elif s_aux.dim() == 2:
            # (B, H, S)
            p_sink = torch.exp(s_aux.float().unsqueeze(-1) - softmax_lse_f)
            # (B, H)
            ds = -(dot_bhs * p_sink).sum(dim=2)
        else:
            return tuple(grads)

        # Replace last grad (for s_aux) with ds.
        grads[-1] = ds.to(dtype=s_aux.dtype)
        return tuple(grads)

    FlashAttnFunc.forward = forward_patched  # type: ignore[assignment]
    FlashAttnFunc.backward = backward_patched  # type: ignore[assignment]
    FlashAttnFunc._async_rl_sink_bwd_patched = True  # type: ignore[attr-defined]
    return True

