"""M7 / Phase B.2d probe: can vllm.LLM() initialise with our backend?

Minimal smoke: register, set calibration, instantiate LLM, generate 20
tokens on 'The capital of France is'.  Success criterion matches M1's
TurboQuant k8v4 sanity — coherent text, no crash.
"""
import os
import sys
import traceback

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

# Tell our plugin where to find M2 calibration artifacts.  The plugin
# (installed via pyproject entry point) auto-loads them in every
# vllm process (parent + engine-core subprocess).
CAL = "/workspace/LLM-KV--Cache-compress/reports/v1_3_ppl/vllm_backend/calibration"
os.environ.setdefault("KAKEYA_SIGMA_Q_PATH", f"{CAL}/qwen3_4b_sigma_q.safetensors")
os.environ.setdefault("KAKEYA_K_CENTROIDS_PATH", f"{CAL}/qwen3_4b_lloyd_max_K_b3.f32")
os.environ.setdefault("KAKEYA_V_CENTROIDS_PATH", f"{CAL}/qwen3_4b_lloyd_max_V_b2.f32")
os.environ.setdefault("KAKEYA_SKIP_LAYERS", "0,1,34,35")


def main() -> int:
    print("[probe] plugin auto-registers via pyproject entry point", flush=True)

    print("[probe] importing vllm...", flush=True)
    from vllm import LLM, SamplingParams

    print("[probe] instantiating LLM (this is where things usually break)...",
          flush=True)
    try:
        llm = LLM(
            model="Qwen/Qwen3-4B",
            dtype="bfloat16",
            kv_cache_dtype="kakeya_v1_3_ppl",
            block_size=512,
            max_model_len=1024,
            gpu_memory_utilization=0.4,
            enforce_eager=True,
            attention_backend="CUSTOM",   # force selector to our CUSTOM override
        )
    except Exception as e:
        print("[probe] LLM() raised:", type(e).__name__, flush=True)
        traceback.print_exc()
        return 2

    print("[probe] LLM loaded.  Running generate...", flush=True)
    try:
        out = llm.generate(
            ["The capital of France is"],
            SamplingParams(max_tokens=20, temperature=0),
        )
        text = out[0].outputs[0].text
    except Exception as e:
        print("[probe] generate raised:", type(e).__name__, flush=True)
        traceback.print_exc()
        return 3

    print("=" * 60, flush=True)
    print("[probe] OUTPUT:", repr(text), flush=True)
    print("=" * 60, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
