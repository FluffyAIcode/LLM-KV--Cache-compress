# RESUME — Option C (full v1.3 PPL vLLM backend)

**For the next cloud agent.**

## HEAD-OF-LINE BLOCKER (2026-04-22): git push is forbidden

**Symptom.** On the current cloud-agent profile, `git push` to
`FluffyAIcode/LLM-KV--Cache-compress.git` returns:

```
remote: Permission to FluffyAIcode/LLM-KV--Cache-compress.git denied to cursor[bot].
fatal: unable to access ...: The requested URL returned error: 403
```

The workspace ships a pre-baked token for `FluffyAIcode/AgentMemorySystem`
(the original workspace repo), but there is no token with `contents:write`
scope on `FluffyAIcode/LLM-KV--Cache-compress`. `gh auth token` returns
the same token and 403s identically. Nothing on the agent side can fix
this — it is a GitHub permissions / Cursor Dashboard secrets issue.

**User action required.** In the Cursor Dashboard → Cloud Agents →
Secrets, add a fine-grained PAT with `contents:write` on
`FluffyAIcode/LLM-KV--Cache-compress` (name it e.g. `GH_TOKEN_KVCACHE`)
and/or grant `cursor[bot]` write on that repo. Until that lands,
subsequent M2..M7 work will be *committed locally* but **not pushed**,
and every new cloud-agent VM starts from whatever is on origin — so
unpushed commits WILL be lost across sessions.

**What is on origin right now** (pre-M1):
  * `61a288f` — RESUME doc (the previous agent's version of this file)
  * `521c546` — Option C plan
  * `2fae6ff` — Plan: v1.3 PPL as a deployable vLLM attention backend

**What lives only on the current agent's local branch** (M1 artifacts;
please push ASAP once credentials are fixed):

```
a750351 M1: vLLM cu130 install + TurboQuant k8v4 baseline reproduced on H200
7eb0b36 M1: add GSM8K accuracy + TPOT benchmark harnesses (Qwen3-4B, H200)
```

If you wake up on a fresh Vast.ai VM and those SHAs are not on origin,
the corresponding work needs to be re-run. Everything needed is already
encoded in this file (install commands, wheel URL, sanity tests) plus
`benchmarks/m1_*.py` — but those files are unpushed too, so you'd have
to re-create the harnesses.

## Why you are here

You are continuing the v1.3 PPL vLLM backend work. A previous agent
got blocked at M1 because the old instance's Nvidia driver was
575.57.08 (supports CUDA 12.9), while vLLM nightly (the only build
with TurboQuant PR #38479) requires torch 2.11 + CUDA 13.0 which
needs driver ≥ 580. A new instance with driver 580.95.05 / CUDA 13.0
has been provisioned; this agent has completed M1 on it.

## Ground truth

- **Branch**: `AgentMemory/v1-3-ppl-vllm-backend-102e` in
  `FluffyAIcode/LLM-KV--Cache-compress`. Do NOT create a new branch;
  keep committing to this one.
- **Plan**: `reports/v1_3_ppl/vllm_backend/PLAN.md` — the Option C
  design. Non-negotiables are at the bottom of that file (no
  simplification, no fallback, no mock, no overfit).
- **Open PRs on this topic**:
  - PR #14 (scaffolding, merged-ish), PR #15 (in-forward harness),
    PR #17 (snapshot-mode harness)
  - No open PR yet for this backend work; you will open PR #18 when
    you have numbers.

## What the previous agent established

| Item | Status | Notes |
|:---|:---|:---|
| PLAN.md — full algorithm preserved in-kernel | Committed | Lists the 5 codec steps + 4 guardrails; ban-list at the bottom |
| Paged-cache slot layout design | Committed | Per-block skeleton + per-token residuals + fixed-budget outlier side-buffer; compression ~3.8×–4× at 8 % outlier budget |
| Runtime per-block PCA decision | Decided | Randomised HMT SVD (matches `kakeyaturbo/src/pca.rs::fit_weighted_pca_randomized`) — zero semantic change from v1.3 |
| Runtime K-means decision | Decided | Port `kakeyaturbo/src/kmeans.rs` Lloyd iteration exactly, no approximation |
| Outlier side-buffer | Decided | Fixed 8 % worst-case budget, zero-pad unused; this is a compression-ratio tax, not a correctness compromise |
| Partial-block trailing segment | Decided | Trailing bf16 staging buffer, sealed when full — same semantic as HF 2-pass harness |
| Block-size alignment | Decided | Require `--block-size 512` so vLLM cache block = codec block |
| vLLM nightly contains TurboQuant | Verified | wheel `vllm-0.19.2rc1.dev98+gcefa5281a-cp38-abi3-manylinux_2_31_x86_64.whl` at `https://wheels.vllm.ai/nightly` |

## Completed milestones

- [x] **M0** Plan document — on origin
- [x] **M1** Environment setup + TQ-k8v4 baseline reproduction — **LOCAL ONLY**, blocked on push (see top-of-file). M1_REPORT.md in this directory has full results: GSM8K 0.8673 (eager) / 0.8620 (compiled) on full 1319-question test split, TPOT 5.53 ms (TQ-k8v4 compiled) vs 3.84 ms (baseline compiled) on H200, single stream. Accuracy bar cleanly passed; TPOT ratio analyzed in §4 of the report.

## Incomplete milestones

- [ ] **M2** Offline calibration (Σ_q + Lloyd-Max centroids) on Qwen3-4B
- [ ] **M3** Rust reference codec refactored to in-process library
- [ ] **M4** Triton STORE kernel (encode path)
- [ ] **M5** Triton DECODE kernel
- [ ] **M6** Backend registration + integration smoke
- [ ] **M7** Head-to-head benchmark
- [ ] **M8** Open PR #18

## Immediate next steps on the new instance

1. **Verify the new driver actually supports CUDA 13**:
    ```
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    ```
    Required: driver ≥ 580 (reports `CUDA Version: 13.0` or higher in
    `nvidia-smi`). If not, SSH back to Vast.ai and get a real
    CUDA-13 instance before doing anything else.
2. **Install vLLM nightly that contains TurboQuant** — the default wheel is cu12 and does NOT work on driver 580 + CUDA 13. You must pull the **cu130 variant** from the nested cu130 index:
    ```bash
    /venv/main/bin/pip install --pre --no-cache-dir \
      --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
      "vllm==0.19.2rc1.dev100+gf946659ff.cu130" \
      "transformers>=4.56,<5" datasets safetensors
    ```
    If that exact commit is gone from the nightly index, go one level
    up (`https://wheels.vllm.ai/nightly/cu130`) and pick the newest
    `*.cu130` wheel. Verify with:
    ```bash
    python -c "import vllm._C; print('_C OK')"
    python -c "from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend; print('TQ OK')"
    ```
    Common failure: installing the plain `vllm==...` wheel (no
    `.cu130` suffix) loads cleanly but then `import vllm._C` fails
    with `libcudart.so.12: cannot open shared object file`, because
    the plain wheel links CUDA 12 runtime. This is how the previous
    agent got confused on the CUDA-13 instance; avoid it.
3. **Verify TurboQuant is importable**:
    ```python
    from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    ```
4. **Verify torch-CUDA works on the new driver**:
    ```python
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0).startswith("NVIDIA H")
    ```
5. **Download Qwen3-4B**:
    ```bash
    HF_HOME=/workspace/.hf_home \
      huggingface-cli download Qwen/Qwen3-4B \
      --exclude "*.bin" "*.pt" "*.msgpack" "*.gguf"
    ```
6. **Sanity test TurboQuant k8v4** (one-shot coherent text):
    ```python
    import os
    os.environ["HF_HOME"] = "/workspace/.hf_home"
    from vllm import LLM, SamplingParams
    llm = LLM(
        model="Qwen/Qwen3-4B",
        dtype="bfloat16",
        kv_cache_dtype="turboquant_k8v4",
        max_model_len=512,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
    )
    print(llm.generate(["The capital of France is"],
                       SamplingParams(max_tokens=20, temperature=0))[0]
          .outputs[0].text)
    ```

## Disk budget

The old instance had only 32 GB overlay. If the new one also has
limited root disk:

- `/workspace/.hf_home/hub`: don't re-download old models; only
  Qwen3-4B is needed (~8 GB).
- `/venv/main/`: vLLM nightly install adds ~5 GB (new torch 2.11
  + cutlass + cudnn + flashinfer).
- `/workspace/LLM-KV--Cache-compress/kakeyaturbo/target/`: Rust
  release build ~200 MB; rebuild on demand.

Target `< 20 GB` total after setup.

## File inventory on this branch

Files the previous agent created that you need to use:

- `reports/v1_3_ppl/vllm_backend/PLAN.md` — the Option C plan
- `reports/v1_3_ppl/vllm_backend/RESUME.md` — this file
- `benchmarks/q_calibration.py` — Σ_q Cholesky offline fitter
- `benchmarks/lloyd_max_calibration.py` — Lloyd-Max centroid offline fitter
- `kakeyaturbo/src/codec.rs` — the reference codec (Rust); STOP reading from stdin / disk — next agent may need to refactor it to pyo3/cffi in M3
- `kakeyaturbo/src/pca.rs` — `fit_weighted_pca_randomized` is the
  semantic spec for the Triton PCA step
- `kakeyaturbo/src/kmeans.rs` — Lloyd iteration spec

Directory that the next agent will create (per PLAN.md):

```
vllm_backend/kakeya_v1_3_ppl/
  __init__.py
  backend.py
  spec.py
  calibration.py
  kernels/
    store.py
    decode.py
    partial.py
    reference.py
  tests/
    test_store_roundtrip.py
    test_decode.py
    test_attention.py
```

## Timeline of prior agents' work (for audit)

### Pre-CUDA-13 agent

- Cleaned /workspace disk from 8 GB free → 17 GB free (DS-Distill
  and stale Qwen2.5 caches removed)
- Installed vLLM nightly wheel (~3 min)
- Downloaded Qwen3-4B (~1 min)
- Tried to run TurboQuant `k8v4` — got `RuntimeError: Engine core
  initialization failed` due to driver too old for CUDA 13
- Pivoted: wrote RESUME (this file), let user switch instance

### Post-CUDA-13 agent (this one, 2026-04-22)

- Verified H200 + driver 580.95.05 + CUDA 13.0 at IP 208.64.254.72:19253
- Installed `vllm==0.19.2rc1.dev100+gf946659ff.cu130` (the
  `gcefa5281a` pin from the original RESUME was no longer on the
  nightly index; picked the newest post-2026-04-15 build)
- Imported TurboQuant backend and config successfully
- Downloaded Qwen3-4B into `/workspace/.hf_home` (7.6 GB)
- First TQ-k8v4 forward pass (temperature 0, prompt "The capital of
  France is"): produced coherent factual continuation, confirming
  the backend is live
- Wrote `benchmarks/m1_gsm8k_eval.py` and `benchmarks/m1_tpot_bench.py`
  (identical harness across every kv-cache-dtype — no config-specific
  branches, no fallback paths, no prompt tuning per dtype)
- Ran full GSM8K (1319 questions) × 4 configs (baseline eager/compiled,
  TQ-k8v4 eager/compiled) + single-stream TPOT 4096→1024 × 4 configs.
  Logs in `reports/v1_3_ppl/vllm_backend/logs/`, JSON artifacts next
  to them. Headline: TQ-k8v4 compiled 0.8620 accuracy (vs 0.8726
  baseline); TQ-k8v4/baseline TPOT ratio +44 % on H200 vs PR's +28 %
  on 4× Blackwell (see M1_REPORT.md §4 for root-cause analysis)
- Blocked on `git push` (see top-of-file); wrote this RESUME update,
  stopped before M2 to avoid losing work across sessions.

## Key decisions already made — do NOT relitigate

- **Full v1.3 PPL algorithm in-kernel** (Option C; no simplification)
- **Block size 512**, requires `--block-size 512`
- **Per-block PCA: runtime randomised SVD** (not calibration-time)
- **K-means: runtime Lloyd iteration**
- **Outliers: fixed 8 % worst-case budget**
- **Partial-block staging buffer: bf16 until filled**
- **Correctness gating before benchmark**: ≥ 1000 random block
  round-trips bit-exact vs `kakeyaturbo/src/codec.rs` reference

## Do NOT do

- Do NOT introduce any `if … else: identity_codec()` fallback
- Do NOT mock outliers as "zero-length sparse buffer"
- Do NOT skip Σ_q whitening with `assume_identity=True`
- Do NOT use the same prompts for calibration and benchmark
- Do NOT report compression ratios that don't match actually-stored
  bytes per vector

## If you hit another blocker before reaching M7

Stop and write a new RESUME with the current blocker at the top,
rather than pressing forward with a compromised approach.

## After M7 (head-to-head), expected shape of PR #18 body

| Config | Δppl | GSM8K | NIAH | TPOT | TTFT | Tok/s | Peak GPU mem |
|:---|--:|--:|--:|--:|--:|--:|--:|
| baseline (bf16) | 0 | ? | 100 % | ? ms | ? ms | ? | ? GB |
| TurboQuant `k8v4` | ? | ? | ? | ? | ? | ? | ? |
| TurboQuant `4bit_nc` | ? | ? | ? | ? | ? | ? | ? |
| **kakeya_v1_3_ppl** | ? | ? | ? | ? | ? | ? | ? |

All numbers on **the same H200** (new instance), **the same
Qwen/Qwen3-4B**, the **same ctx lengths / passages**. No cross-
machine or cross-model comparisons in that table.

## Good luck

The heavy algorithmic work is already done — the previous agents
figured out calibration, Q-preconditioning, PCA + K-means + WHT +
Lloyd-Max + outliers. Your job is to port it faithfully into a
Triton kernel that lives inside a vLLM attention backend.
