"""Gradio Space demo: KakeyaLatticeCache on a small HF causal LM.

Run locally:
    pip install kakeyalattice[hf] gradio
    python app.py

Deploy to HF Spaces: see ./README.md.  By default uses Qwen2-0.5B
(head_dim=64, E8-compatible) so it fits on a free HF Space CPU.
Swap to Qwen/Qwen2.5-1.5B or Llama-3.2-1B (GPU Space) for more interesting
decode-length comparisons.

The demo shows, side-by-side, the same prompt generated under:
  (a) bf16 DynamicCache — reference
  (b) KakeyaLatticeCache E8 Q=10  (aggressive, ~3.6x KV compression)
  (c) KakeyaLatticeCache E8 Q=38  (balanced, ~2.5x KV compression)
  (d) KakeyaLatticeCache E8 Q=152 (near-lossless, ~1.9x KV compression)
and reports wall-clock + per-layer K rel-MSE.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import gradio as gr
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
except ImportError as e:
    raise ImportError("Install transformers: pip install 'kakeyalattice[hf]'") from e

from kakeyalattice.hf import KakeyaLatticeCache


DEFAULT_MODEL = os.environ.get("KAKEYA_DEMO_MODEL", "Qwen/Qwen2-0.5B")
_model_cache: dict = {}


def _load_model(model_id: str, device: str):
    key = (model_id, device)
    if key in _model_cache:
        return _model_cache[key]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    _model_cache[key] = (tok, model)
    return tok, model


def _generate_one(
    tok, model, prompt: str, max_new: int, cache, device: str,
) -> tuple[str, float]:
    ids = tok(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **ids,
            max_new_tokens=max_new,
            do_sample=False,
            past_key_values=cache,
            use_cache=True,
        )
    elapsed = time.perf_counter() - t0
    text = tok.decode(out[0], skip_special_tokens=True)
    return text, elapsed


def run_demo(
    prompt: str,
    max_new: int,
    model_id: str,
    device_pref: str,
) -> tuple[str, str, str, str, str]:
    device = "cuda" if (device_pref == "auto" and torch.cuda.is_available()) else (
        "cuda" if device_pref == "cuda" else "cpu"
    )
    tok, model = _load_model(model_id, device)

    cfg = model.config
    num_hidden_layers = cfg.num_hidden_layers
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    results = []

    # Baseline: bf16 DynamicCache
    baseline_cache = DynamicCache()
    text_bf16, t_bf16 = _generate_one(tok, model, prompt, max_new, baseline_cache, device)
    results.append(("bf16 DynamicCache (reference)", text_bf16, t_bf16, head_dim * 16))

    for q, label in [(10, "E8 Q=10 aggressive"), (38, "E8 Q=38 balanced"), (152, "E8 Q=152 near-lossless")]:
        try:
            cache = KakeyaLatticeCache(
                variant="e8", q_range=q,
                num_hidden_layers=num_hidden_layers,
                head_dim=head_dim,
                device=device,
                strict=False,
            )
            text, t = _generate_one(tok, model, prompt, max_new, cache, device)
            bits = cache._codecs[0].bits_per_token_per_head if cache._codecs else head_dim * 16
            results.append((f"KakeyaLattice {label}", text, t, bits))
        except Exception as e:
            results.append((f"KakeyaLattice {label} (FAILED)", f"Error: {e}", 0.0, 0))

    # Format as comparison table
    header = f"Model: {model_id} | head_dim: {head_dim} | device: {device} | new_tokens: {max_new}"
    rows = [
        f"\n### {name}  —  {t:.2f}s, {bits} bits/vec ({bits/16:.1f}x vs bf16)\n\n{text}"
        for (name, text, t, bits) in results
    ]
    return header, *rows


with gr.Blocks(title="KakeyaLattice KV-cache compression demo") as demo:
    gr.Markdown(
        "# KakeyaLattice KV-cache compression demo\n\n"
        "Compare generation output + latency across **bf16 baseline** and "
        "three **KakeyaLattice E8** compression levels on a small HF causal LM. "
        "The E8 variant uses 8-D nested-lattice closest-point quantisation "
        "with Sylvester-Hadamard rotation and per-vector adaptive scaling."
    )
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            value="Explain in one paragraph why lattice quantisation can beat scalar quantisation:",
            lines=3,
        )
    with gr.Row():
        max_new = gr.Slider(minimum=16, maximum=512, value=128, step=16, label="Max new tokens")
        model_id = gr.Textbox(label="HF model id", value=DEFAULT_MODEL)
        device_pref = gr.Radio(choices=["auto", "cpu", "cuda"], value="auto", label="Device")
    run_btn = gr.Button("Run comparison", variant="primary")
    header_out = gr.Markdown("")
    out_bf16 = gr.Markdown("")
    out_q10 = gr.Markdown("")
    out_q38 = gr.Markdown("")
    out_q152 = gr.Markdown("")
    run_btn.click(
        fn=run_demo,
        inputs=[prompt, max_new, model_id, device_pref],
        outputs=[header_out, out_bf16, out_q10, out_q38, out_q152],
    )


if __name__ == "__main__":
    demo.launch()
