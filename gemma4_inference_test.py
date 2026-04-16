#!/usr/bin/env python3
"""Run a real Gemma 4 generation pass with optional Kakeya KV cache."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, set_seed

from kakeya_kv_codec import build_gemma4_kakeya_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real Gemma 4 inference pass with Kakeya KV cache.",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-4-e4b-it",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain why KV cache compression matters for long-context inference.",
        help="Prompt text for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("kakeya", "dynamic"),
        default="kakeya",
        help="Cache implementation used during generation.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
        help="Model dtype.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Device map passed to transformers, e.g. "auto", "cpu", "cuda:0".',
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help='Attention implementation, e.g. "eager", "sdpa", "flash_attention_2".',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer.",
    )
    parser.add_argument(
        "--print-config-only",
        action="store_true",
        help="Print resolved runtime configuration and exit without loading the model.",
    )

    # Kakeya-specific cache settings.
    parser.add_argument("--variance-ratio", type=float, default=0.99)
    parser.add_argument("--k-segments", type=int, default=16)
    parser.add_argument("--d-res", type=int, default=8)
    parser.add_argument("--residual-length", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--min-rows-to-build", type=int, default=8)
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype | str:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def maybe_parse_device_map(device_map: str) -> Any:
    if device_map == "auto":
        return "auto"
    if device_map in {"cpu", "cuda", "mps"} or ":" in device_map:
        return device_map
    return device_map


def get_primary_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for _, device_name in model.hf_device_map.items():
            if device_name not in {"disk", "cpu"}:
                return torch.device(device_name)
    return next(model.parameters()).device


def prepare_inputs(tokenizer, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt")
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def build_cache(args: argparse.Namespace, model) -> DynamicCache:
    if args.cache_mode == "dynamic":
        return DynamicCache(config=model.config)
    return build_gemma4_kakeya_cache(
        model,
        variance_ratio=args.variance_ratio,
        K=args.k_segments,
        d_res=args.d_res,
        residual_length=args.residual_length,
        block_size=args.block_size,
        min_rows_to_build=args.min_rows_to_build,
    )


def describe_run(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model": args.model,
        "cache_mode": args.cache_mode,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": args.max_new_tokens,
        "variance_ratio": args.variance_ratio,
        "k_segments": args.k_segments,
        "d_res": args.d_res,
        "residual_length": args.residual_length,
        "block_size": args.block_size,
        "min_rows_to_build": args.min_rows_to_build,
    }


def main() -> None:
    args = parse_args()
    run_config = describe_run(args)

    if args.print_config_only:
        print(json.dumps(run_config, indent=2, ensure_ascii=False))
        return

    set_seed(args.seed)

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_dtype(args.dtype),
        "device_map": maybe_parse_device_map(args.device_map),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )
    model.eval()

    device = get_primary_device(model)
    inputs = prepare_inputs(tokenizer, args.prompt, device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    cache = build_cache(args, model)

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            past_key_values=cache,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    output_tokens = int(outputs.shape[-1])
    new_tokens = output_tokens - prompt_tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = {
        "run_config": run_config,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "new_tokens": new_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(new_tokens / elapsed, 4) if elapsed > 0 and new_tokens > 0 else None,
        "generated_text": generated_text,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
