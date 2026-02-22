#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GPT-OSS MXFP4 checkpoint to BF16 HF folder.")
    parser.add_argument("--model-id", type=str, default="openai/gpt-oss-20b", help="HF repo id or local path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for BF16 HF checkpoint")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        help="Persisted config hint only; runtime can override via from_pretrained(attn_implementation=...).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Transformers device_map for conversion (default: "auto").',
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        print(f"Found existing checkpoint at {args.output_dir}; skipping.")
        return

    quantization_config = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        attn_implementation=args.attn_implementation,
        use_cache=False,
        device_map=args.device_map,
        trust_remote_code=True,
    )

    # Make sure the saved config carries a sane default (training will override anyway).
    try:
        model.config.attn_implementation = args.attn_implementation
    except Exception:
        pass

    model.save_pretrained(args.output_dir)
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tok.save_pretrained(args.output_dir)

    # Small sync to surface potential device errors early.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"Saved BF16 checkpoint to: {args.output_dir}")


if __name__ == "__main__":
    main()

