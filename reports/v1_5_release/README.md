# v1.5 KakeyaLattice (E8) — Release Reports

This directory contains all evaluation data and reports produced for
the **v1.5 KakeyaLattice** release.  v1.5 introduces the E8 nested
lattice as the aggressive-point head-of-line codec, replacing v1.4's
D4 at Q ≤ 10 deployment points.

Release object: https://github.com/FluffyAIcode/LLM-KV--Cache-compress/releases/tag/v1.5

## What's here

| File / dir | Content |
|:-|:-|
| `V15_FULL_4MODEL_REPORT.md` | **Primary report** — PPL + MSE + CR + latency + NIAH across four models |
| `V15_VS_V14_VS_TQ_REPORT.md` | Qwen3-4B first-measurement detail (pre-dates the 4-model expansion) |
| `e8_latency_benchmark.json` + `.log` | E8 vs D4 vs TQ pure codec latency (H200, N=2048×8×128, 500 iters) |
| `qwen3_4b_v15_phase_beta_inforward.json` + `.log` | Qwen3-4B n=32 KV-mode sweep, aggressive + balanced |
| `{deepseek_1p5b,gemma4_e4b,glm4_9b}_nobdry_inforward.json` + `.log` | Other 3 models, no-boundary in-forward n=32 |
| `{model}_tqb2_bdry2_inforward.json` + `.log` | TQ b=2 guardrail run with boundary=2 (required to prevent catastrophic divergence) |
| `niah/` | NIAH long-context retrieval (ctx ∈ {4k, 8k, 16k} × depth ∈ {0.1, 0.5, 0.9} × n_trials=3) |

## Relationship to `reports/v1_4_release/`

- `v1_4_release/` is **frozen** — all the data that went into the v1.4
  release (tagged `v1.4`, commit `6b02711`).  Do not modify.
- `v1_5_release/` (this directory) contains **only v1.5-new data**.
  Nothing here supersedes v1.4; it's additive.

Earlier in development some v1.5 measurements were temporarily staged
at `reports/v1_4_release/rigorous_eval/v15_vs_v14_vs_tq/`.  That
sub-directory has been promoted to `reports/v1_5_release/` (this
directory) to match the two-release structure.

## Reproducibility

See the "Reproducibility" section at the bottom of
`V15_FULL_4MODEL_REPORT.md` for exact per-model commands.  Short
version:

```bash
pip install -e kakeyalattice
pip install -e vllm_backend
export VLLM_ENABLE_V1_MULTIPROCESSING=0 KAKEYA_SNAPSHOT_QWEN3=1

# Main 4-model PPL + MSE + CR sweep
python benchmarks/rigorous_eval.py \
    --model-path <HF-id> --model-name <short>_nobdry \
    --mode inforward --no-boundary \
    --q-values 4,10 --v15-q-values 4,10 --tq-b-values 3 \
    --kv-modes KV \
    --ctx-len 2048 --n-eval 64 --n-passages 32 \
    --out-dir reports/v1_5_release

# TQ b=2 guardrail
python benchmarks/rigorous_eval.py \
    --model-path <HF-id> --model-name <short>_tqb2_bdry2 \
    --mode inforward --boundary-size 2 \
    --q-values "" --v15-q-values "" --tq-b-values 2 \
    --kv-modes KV \
    --ctx-len 2048 --n-eval 64 --n-passages 32 \
    --out-dir reports/v1_5_release

# Latency (no model needed)
python benchmarks/e8_latency_benchmark.py --n-iters 500 \
    --out-dir reports/v1_5_release

# NIAH
python benchmarks/niah_eval.py \
    --model-path <HF-id> --model-name <short> \
    --mode inforward --boundary-size 2 --n-trials 3 \
    --ctx-lengths 4096,8192,16384 --depths 0.1,0.5,0.9 \
    --q-values 4,10 --v15-q-values 4,10 --tq-b-values 2,3 \
    --out-dir reports/v1_5_release/niah
```
