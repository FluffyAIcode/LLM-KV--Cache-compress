# RESUME — Option C (full v1.3 PPL vLLM backend) on new H200 instance

**For the next cloud agent on the new Vast.ai instance.**

You are continuing the v1.3 PPL vLLM backend work. The previous
agent got blocked at M1 because the old instance's Nvidia driver
was 575.57.08 (supports CUDA 12.9), while vLLM nightly (the only
build with TurboQuant PR #38479) requires torch 2.11 + CUDA 13.0
which needs driver ≥ 580. A new instance with a newer driver has
been provisioned and you are on it.

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

- [x] **M0** Plan document
- [x] **M1 (partial)** Disk cleanup done; vLLM nightly install attempted; BLOCKED on driver

## Incomplete milestones

- [ ] **M1** Environment setup
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
2. **Install vLLM nightly that contains TurboQuant**:
    ```bash
    /venv/main/bin/pip install --pre --no-cache-dir \
      --extra-index-url https://wheels.vllm.ai/nightly \
      "vllm==0.19.2rc1.dev98+gcefa5281a" \
      "transformers>=4.56,<5" datasets safetensors
    ```
    (If that commit is no longer on the nightly index, pick the
    newest commit there — TurboQuant has been merged on main since
    2026-04-15 so any post-4-15 nightly has it.)
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

## Timeline of the previous agent's work (for audit)

- Cleaned /workspace disk from 8 GB free → 17 GB free (DS-Distill
  and stale Qwen2.5 caches removed)
- Installed vLLM nightly wheel (~3 min)
- Downloaded Qwen3-4B (~1 min)
- Tried to run TurboQuant `k8v4` — got `RuntimeError: Engine core
  initialization failed` due to driver too old for CUDA 13
- Pivoted: wrote RESUME (this file), let user switch instance

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
