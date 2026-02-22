## FlashAttention-3 attention-sink backward patch

This repo uses `kernels-community/vllm-flash-attn3` (FlashAttention-3) to run GPT‑OSS attention **with attention sinks** on Hopper GPUs.

GPT‑OSS differs from most architectures because each attention layer has a **learnable per-head sink** parameter (`s_aux` in the kernel API, `module.sinks` in Transformers). The sink modifies the **softmax denominator** by adding an extra logit term per query row.

Historically, the public sink-enabled FA3 implementations shipped **sink-aware forward** but did **not** return a gradient for `s_aux` in backward. That forces training to fall back to eager attention if you want correct gradients.

This repo avoids that fallback by patching only the Python autograd wrapper used by the hub kernel:

- CUDA kernels still compute the expensive pieces (`dq`, `dk`, `dv`) using FA3.
- We compute only `ds_aux` in PyTorch using tensors the kernel already returns/saves.

The implementation lives in:

- `async_rl/patches/vllm_flash_attn3_sink_bwd.py`

### What exactly gets patched

The hub kernel exposes a Python wrapper `FlashAttnFunc(torch.autograd.Function)` that:

- calls the CUDA **forward** op and returns `(out, softmax_lse)`
- calls the CUDA **backward** op and returns `(dq, dk, dv, ...)`

The patch monkey-patches:

- `FlashAttnFunc.forward`: stores the `s_aux` tensor on the autograd context (`ctx`) so backward can see it.
- `FlashAttnFunc.backward`: calls the original CUDA backward first (so the main gradients are unchanged/fast), then computes and returns the missing `ds_aux` gradient.

No CUDA code is recompiled.

### Derivation of the sink gradient

Let the attention row logits (without sink) be \(a_j\) for keys \(j\), and let the sink logit be \(s\) (per head).
The normalization is:

\[
Z = \sum_j e^{a_j} + e^{s}
\]

The output is:

\[
O = \sum_j \frac{e^{a_j}}{Z} v_j
\]

The sink only affects the denominator, so:

\[
\frac{\partial O}{\partial s}
= \frac{\partial}{\partial s}\left(\frac{1}{Z}\right)\sum_j e^{a_j} v_j
= -\frac{e^{s}}{Z} O
= -p_{\text{sink}}\, O
\]

where:

\[
p_{\text{sink}} = \frac{e^{s}}{Z} = \exp(s - \log Z)
\]

Given `dout = dL/dO`, the gradient for the sink is:

\[
\frac{\partial L}{\partial s}
= \sum_{d} \frac{\partial L}{\partial O_d} \frac{\partial O_d}{\partial s}
= - (dout \cdot O)\, p_{\text{sink}}
\]

Summing over batch and sequence gives the per-head gradient used to update `module.sinks`.

### What tensors we use

The FA3 kernel already provides:

- `out` (attention output)
- `softmax_lse` (log-sum-exp per query row, i.e. \(\log Z\), including the sink term)

So we compute:

- `p_sink = exp(s_aux - softmax_lse)`
- `ds = -sum((dout * out).sum(-1) * p_sink)` (reduce over batch and sequence)

The implementation uses float32 for this calculation (for stability) and casts back to the original dtype.

### Performance expectations

Compared to a hypothetical fully fused sink-aware backward kernel that also returns `ds_aux`, this patch adds:

- a few extra pointwise/reduction ops (`mul`, `sum`, `exp`, reductions)
- additional reads of `out`, `dout`, and `softmax_lse`

Asymptotically it is \(O(B \cdot S \cdot H \cdot D)\) extra work, but it **does not materialize** an attention matrix and does not change the CUDA FA3 path for `dq/dk/dv`.

In practice:

- **Much faster** than falling back to eager attention (which is \(O(S^2)\) memory/compute).
- Typically **a modest overhead** vs a fully fused sink backward, since the dominant cost remains FA3 backward for `dq/dk/dv`.

### Limitations

- The patch currently targets the **non-varlen** `FlashAttnFunc` wrapper. If you switch training to use the varlen wrapper (`FlashAttnVarlenFunc`), you should extend the same idea there.
- This computes the mathematically correct `ds_aux`, but the exact numerical behavior may differ slightly from a future fused implementation (e.g., due to internal accumulation types).

