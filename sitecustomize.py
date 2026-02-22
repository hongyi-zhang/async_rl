"""
Auto-applied runtime patches for this repo.

Python automatically imports `sitecustomize` (if found on `sys.path`) during interpreter startup.
We use it so Ray worker processes inherit the same patches as the driver.
"""

from __future__ import annotations

import os
import sys


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


if _env_flag("ASYNC_RL_ENABLE_PATCHES", default="0"):
    try:
        from async_rl.patches.miles_fsdp_gpt_oss import apply_miles_fsdp_gpt_oss_patches

        apply_miles_fsdp_gpt_oss_patches()
    except Exception as e:  # pragma: no cover
        print(f"[async_rl] WARNING: failed to apply runtime patches: {e}", file=sys.stderr)

