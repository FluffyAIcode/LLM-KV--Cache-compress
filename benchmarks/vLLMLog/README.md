# `benchmarks/vLLMLog/` — vast.ai H200 log + data archive

This directory is an **archive of the raw vLLM-run artifacts** that lived only
on the vast.ai H200 dev instance (`root@208.64.254.72:19253`, `vast` SSH alias)
and not in any other place in this repository.

Snapshot date: `2026-04-25T00:20Z` (see top-level `MANIFEST.txt` for the exact
`ssh + rsync` audit trail).

## Why this directory exists

Most of this project's live results already live under `reports/` in the git
repo (e.g.\ `reports/v1_4_release/`, `reports/v1_5_release/`). But two subtrees
on the vast instance were **never committed**:

- `reports/v1_3_ppl/` (~242 MB on vast) — the v1.3 PPL sprint raw logs,
  calibration `.safetensors`, per-passage JSON, and bench outputs.
- `reports/v1_4_q_pca/` (~11 MB on vast) — the v1.4 Q-PCA flagship calibration
  snapshot used during the v1.4 codec bring-up.

If the vast instance were lost, we would lose these. This directory ships a
git-committed subset of the same data so the repo is self-contained for
rebuilds. Everything under `reports/v1_4_release/` and `reports/v1_5_release/`
is already in git and is **not duplicated** here.

## Layout

```
benchmarks/vLLMLog/
├── README.md                                    (this file)
├── v1_3_ppl/                                    (~7.9 MB, 145 files)
│   ├── snapshot_mode_qwen3/                     (4.1 MB)
│   │   ├── budget_sweep/, bdry_sweep/           (budget / boundary sweeps)
│   │   ├── rsvd_wht_ablation/                   (RSVD-WHT ablation probe)
│   │   ├── pcaExact/                            (exact-SVD vs RSVD comparison)
│   │   ├── direct_codebook/                     (flat-codebook PPL reference)
│   │   ├── non_gaussianity/                     (kurtosis / Wasserstein audit)
│   │   ├── single_layer_probe/                  (per-layer diagnostics)
│   │   ├── rvq/                                 (residual-VQ smoke)
│   │   ├── bridges_abc/                         (Dvir-to-Euclidean bridges A/B/C)
│   │   ├── multimodel/                          (4-model multi-head bundle)
│   │   └── headtohead/                          (TQ head-to-head JSONs)
│   └── vllm_backend/                            (3.8 MB, excl. 240 MB .safetensors)
│       ├── calibration/                         (Σ_q JSON, Lloyd-Max tables — BINARY .safetensors EXCLUDED, see below)
│       ├── logs/                                (11 stdout logs for M1/M2 baseline runs)
│       ├── m7/                                  (M7 milestone outputs)
│       ├── m7_boundary_skip/                    (M7 with boundary-skip variant)
│       ├── gsm8k_*.json                         (GSM-8k per-sample prefill + logprob dumps)
│       └── tpot_*.json                          (time-per-output-token micro-benches)
└── v1_4_q_pca/                                  (11 MB, 5 files)
    ├── calibrated_codebook/                     (calibrated flat codebook JSON)
    └── flagship/
        ├── deepseek_distill_q_calib.json        (metadata + metrics)
        └── deepseek_distill_q_calib.safetensors (10.5 MB — included)
```

## Excluded binary artifacts (intentional)

Six very large `.safetensors` files under `v1_3_ppl/vllm_backend/calibration/`
totalling **~234 MB** are **NOT committed**, because GitHub warns above 50 MB
per file and permanently bloats repo history. Their `.json` sidecars (metadata,
metrics, per-layer stats) ARE committed, and the binary tensors are fully
regenerable from the code in `kakeyalattice/` + `benchmarks/`:

| File | Size | Regeneration |
| --- | --- | --- |
| `qwen3_4b_sigma_q.safetensors`       | 54 MB | run the Σ_q calibration in `benchmarks/v14_streaming_diag.py` against Qwen/Qwen3-4B |
| `qwen3_4b_sigma_q_reg10.safetensors` | 36 MB | same, with `--sigma-reg 10` |
| `qwen3_4b_sigma_q_reg25.safetensors` | 36 MB | same, with `--sigma-reg 25` |
| `qwen3_4b_sigma_q_reg50.safetensors` | 36 MB | same, with `--sigma-reg 50` |
| `qwen3_4b_sigma_q_reg100.safetensors`| 36 MB | same, with `--sigma-reg 100` |
| `qwen3_4b_sigma_q_reg200.safetensors`| 36 MB | same, with `--sigma-reg 200` |

The original binary files are safely backed up at
`/workspace/vast_backup_2026-04-25/` on the local dev machine (total ~260 MB
full-fidelity mirror; see that directory's `README.md` for tarball + SHA-256
manifest).

## Relationship to other reports directories

| Directory | Status | Lineage |
| --- | --- | --- |
| `reports/v1_3_ppl/` (not present) | archived here (`benchmarks/vLLMLog/v1_3_ppl/`) | vast-only historical v1.3 sprint data |
| `reports/v1_4_q_pca/` (not present) | archived here (`benchmarks/vLLMLog/v1_4_q_pca/`) | vast-only historical v1.4 Q-PCA snapshot |
| `reports/v1_4_release/` | ✅ git-tracked | frozen v1.4 release — see that directory's `INVENTORY.md` |
| `reports/v1_5_release/` | ✅ git-tracked | v1.5 release + Stage 0.5 DSV4 probe |
| `reports/paper/` | ✅ git-tracked | the canonical paper LaTeX source + PDF |

## SHA-256 manifest of what's here

Generated at commit time with:

```bash
cd benchmarks/vLLMLog
find . -type f ! -name 'MANIFEST.sha256' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
```

See `MANIFEST.sha256` in this directory for the per-file hashes.

## Size summary

```
benchmarks/vLLMLog/             ~19 MB, 150 files committed
  ├── v1_3_ppl/                 7.9 MB, 145 files  (240 MB of .safetensors excluded)
  └── v1_4_q_pca/                11 MB,   5 files  (all included; largest 10.5 MB)
```

No code / no benchmark harness changes; this is pure data archival.
