# B2 Release — acceptance-rate benchmark outputs

空目录占位。真实 B2 benchmark 输出（需要 Apple Silicon + MLX + DFlash）
跑出来后，8 个 JSON 文件会落在这里:

```
reports/b2_release/
├── b2_dflash_kakeya_gsm8k_bf16.json
├── b2_dflash_kakeya_gsm8k_e8-q38.json
├── b2_dflash_kakeya_gsm8k_e8-q10.json
├── b2_dflash_kakeya_gsm8k_e8-q4.json
├── b2_dflash_kakeya_humaneval_bf16.json
├── b2_dflash_kakeya_humaneval_e8-q38.json
├── b2_dflash_kakeya_humaneval_e8-q10.json
├── b2_dflash_kakeya_humaneval_e8-q4.json
└── FINDINGS.md                              (narrative + aggregate tables)
```

跑法见 `benchmarks/b2_dflash_kakeya/README.md`。

schema 版本 + 每条 JSON 结构见 `benchmarks/b2_dflash_kakeya/schema.py`。
