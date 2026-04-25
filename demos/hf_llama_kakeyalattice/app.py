"""Gradio Space demo: KakeyaLatticeCache on a small HF causal LM.

Run locally:
    pip install kakeyalattice[hf] gradio
    python app.py

Deploy to HF Spaces: see ./SPACE_README.md and ./HF_SPACE_DEPLOY.md.
By default uses Qwen2-0.5B (head_dim=64, E8-compatible) so it fits on a
free HF Space CPU. Swap to Qwen/Qwen2.5-1.5B or Llama-3.2-1B (GPU Space)
for more interesting decode-length comparisons.

The demo shows, side-by-side, the same prompt generated under:
  (a) bf16 DynamicCache — reference
  (b) KakeyaLatticeCache E8 Q=10  (aggressive, highest KV compression)
  (c) KakeyaLatticeCache E8 Q=38  (balanced)
  (d) KakeyaLatticeCache E8 Q=152 (near-lossless)
and reports wall-clock + bits/vec vs bf16 baseline.
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
DEFAULT_PROMPT = "List five countries in Africa:"
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
    bf16_bits = head_dim * 16  # reference: bits per token per head in bf16

    results = []

    baseline_cache = DynamicCache()
    text_bf16, t_bf16 = _generate_one(tok, model, prompt, max_new, baseline_cache, device)
    results.append(("bf16 DynamicCache (reference)", text_bf16, t_bf16, bf16_bits))

    for q, label in [
        (10, "E8 Q=10 aggressive"),
        (38, "E8 Q=38 balanced"),
        (152, "E8 Q=152 near-lossless"),
    ]:
        try:
            cache = KakeyaLatticeCache(
                variant="e8", q_range=q,
                num_hidden_layers=num_hidden_layers,
                head_dim=head_dim,
                device=device,
                strict=False,
            )
            text, t = _generate_one(tok, model, prompt, max_new, cache, device)
            bits = cache._codecs[0].bits_per_token_per_head if cache._codecs else bf16_bits
            results.append((f"KakeyaLattice {label}", text, t, bits))
        except Exception as e:
            results.append((f"KakeyaLattice {label} (FAILED)", f"Error: {e}", 0.0, 0))

    header = (
        f"**Model:** `{model_id}` | **head_dim:** {head_dim} | "
        f"**device:** {device} | **new_tokens:** {max_new} | "
        f"**bf16 reference bits/vec:** {bf16_bits}"
    )

    rows = []
    for (name, text, t, bits) in results:
        if bits > 0:
            cr = bf16_bits / bits
            bit_saving = (1 - bits / bf16_bits) * 100
            cr_str = f"{cr:.2f}x"
            cr_detail = f"{bit_saving:+.0f}% bits vs bf16"
        else:
            cr_str = "n/a"
            cr_detail = "failed"
        rows.append(
            f"\n### {name}\n\n"
            f"- **latency:** {t:.2f}s\n"
            f"- **bits/vec:** {bits} (bf16 ref: {bf16_bits})\n"
            f"- **Compression:** {cr_str} ({cr_detail})\n\n"
            f"{text}"
        )
    return header, *rows


EXAMPLE_PROMPTS = [
    ["List five countries in Africa:"],
    ["Translate 'good morning' into French, Spanish, German, and Japanese:"],
    ["Write a two-sentence summary of what a transformer is in machine learning:"],
    ["What is 17 times 23? Show your work step by step."],
]


with gr.Blocks(title="KakeyaLattice KV-cache compression") as demo:
    gr.Markdown(
        "# KakeyaLattice KV-cache compression\n\n"
        "Compare generation output + latency across **bf16 baseline** and "
        "three **KakeyaLattice E8** compression levels on a small HF causal LM. "
        "The E8 variant uses 8-D nested-lattice closest-point quantisation "
        "with Sylvester-Hadamard rotation and per-vector adaptive scaling."
    )
    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            value=DEFAULT_PROMPT,
            lines=3,
        )
    with gr.Row():
        max_new = gr.Slider(minimum=16, maximum=512, value=128, step=16, label="Max new tokens")
        model_id = gr.Textbox(label="HF model id", value=DEFAULT_MODEL)
        device_pref = gr.Radio(choices=["auto", "cpu", "cuda"], value="auto", label="Device")
    run_btn = gr.Button("Run comparison", variant="primary")

    gr.Examples(
        examples=EXAMPLE_PROMPTS,
        inputs=[prompt],
        label="Example prompts (click to fill)",
    )

    gr.Markdown(
        "### About the default model\n\n"
        f"The default model is **{DEFAULT_MODEL}** (0.5B params). It runs on a "
        "free HF Space CPU but is *small*. Small models can fall into "
        "greedy-decode repetition loops on open-ended prompts — that is a "
        "property of the **model**, not the codec. If you see all four outputs "
        "repeating the same phrase, try a short, fact-shaped prompt (e.g. "
        "\"List five countries in Africa:\") or switch to a larger model "
        "(`KAKEYA_DEMO_MODEL=Qwen/Qwen2.5-1.5B`) on a GPU Space."
    )

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
    demo.launch(server_name="0.0.0.0", server_port=7860)
