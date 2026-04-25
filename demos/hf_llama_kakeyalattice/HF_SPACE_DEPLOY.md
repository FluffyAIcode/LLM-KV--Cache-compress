# Deploying to HuggingFace Space (Docker SDK)

Since you selected **Docker** (not Gradio) when creating the Space,
follow these steps. Docker is a perfectly valid choice — Gradio still
runs inside the container; the Docker SDK just gives you full control
over the Python + system-package environment.

## Files to push to the Space repo

The Space repo needs exactly these four files at its root:

| file in this repo | file in Space repo | purpose |
| --- | --- | --- |
| `demos/hf_llama_kakeyalattice/app.py` | `app.py` | Gradio app |
| `demos/hf_llama_kakeyalattice/Dockerfile` | `Dockerfile` | Container build spec |
| `demos/hf_llama_kakeyalattice/requirements.txt` | `requirements.txt` | Pip deps (installed by Dockerfile) |
| `demos/hf_llama_kakeyalattice/SPACE_README.md` | `README.md` | HF-rendered page + YAML frontmatter with `sdk: docker` |

**Important**: the Space repo's `README.md` is the file with YAML
frontmatter. Use `SPACE_README.md` from this repo (renamed to
`README.md`), NOT the other `README.md` in this directory (which is
for GitHub readers and has `sdk: gradio` — wrong for your chosen
Docker Space).

## One-shot deployment script

```bash
# 1. Clone the empty Space repo HF just created for you
git clone https://huggingface.co/spaces/<your-username>/kakeyalattice-demo
cd kakeyalattice-demo

# 2. Copy the four files from our repo (adjust path to your local clone)
KAKEYA_REPO=/path/to/LLM-KV--Cache-compress
cp $KAKEYA_REPO/demos/hf_llama_kakeyalattice/app.py .
cp $KAKEYA_REPO/demos/hf_llama_kakeyalattice/Dockerfile .
cp $KAKEYA_REPO/demos/hf_llama_kakeyalattice/requirements.txt .
cp $KAKEYA_REPO/demos/hf_llama_kakeyalattice/SPACE_README.md README.md

# 3. Commit + push
git add app.py Dockerfile requirements.txt README.md
git commit -m "Initial Space deploy: KakeyaLattice KV-cache compression demo"
git push

# 4. HF starts building the Docker image automatically. Check status at:
#    https://huggingface.co/spaces/<your-username>/kakeyalattice-demo
#    Build logs are visible in the Logs tab.
```

## Expected build behaviour

- **First build**: ~4–6 minutes (downloads Python 3.11-slim base,
  installs CPU torch + transformers + gradio + kakeyalattice). Image
  size ~2 GB (mostly torch-cpu).
- **Subsequent builds**: ~30 seconds if only `app.py` or `README.md`
  changed (Docker layer cache reuses the pip install layer).
- **Runtime**: ~5 seconds cold start. Each "Run comparison" button
  click takes ~10-30 seconds on free CPU tier (four generations ×
  model load already cached after first click).

## Free-tier specs (sanity check)

HF Space CPU-Basic (free):

- 2 CPU cores
- 16 GiB RAM (plenty for Qwen2-0.5B)
- Persistent storage: 50 GiB for /data (we don't need it; model cached
  in `/home/user/.cache`)
- No GPU

This is enough to run the default `Qwen/Qwen2-0.5B` (0.5 B params at
fp32 = 2 GB RAM used). For bigger models (Qwen3-4B, LLaMA-3.1-8B) you
need T4-Small ($0.60/hr) or better.

## Upgrading to GPU

If you later want to serve Qwen3-4B:

1. In the Space settings, switch hardware to "T4-small" or "A10G-small".
2. Edit `Dockerfile`: change `--extra-index-url https://download.pytorch.org/whl/cpu`
   to `https://download.pytorch.org/whl/cu124` (or cu128 for newer).
3. Set env var `KAKEYA_DEMO_MODEL=Qwen/Qwen3-4B` in the Space settings.
4. Re-push — Dockerfile will rebuild with CUDA torch.

## Troubleshooting

- **"No Dockerfile found"**: make sure the four files are at the Space
  repo ROOT, not inside a subdirectory.
- **"Port 7860 not binding"**: check `EXPOSE 7860` in Dockerfile and
  `app_port: 7860` in README YAML. Gradio defaults to 7860.
- **Slow first generation**: the model weights are downloaded on first
  run into `/home/user/.cache/huggingface`. Each subsequent call
  reuses cache.
- **OOM on free tier**: the default model `Qwen/Qwen2-0.5B` fits
  comfortably in 16 GB. If you switch to a larger model without
  upgrading hardware, you'll OOM. Stick to ≤1.5B params on free CPU
  tier.
- **Build failure on `kakeyalattice` install**: check
  `pip install kakeyalattice` works locally first. Our package is at
  https://pypi.org/project/kakeyalattice/1.5.0/.

## Validation after deploy

Open the Space URL, paste a prompt like:

> List five countries in Africa:

Click "Run comparison". You should see **four** generated paragraphs:

1. bf16 DynamicCache (reference)
2. KakeyaLattice E8 Q=10 aggressive
3. KakeyaLattice E8 Q=38 balanced
4. KakeyaLattice E8 Q=152 near-lossless

Text quality should degrade smoothly from identical to reference
(Q=152) to slightly different but coherent (Q=10). If all four are
byte-identical, the codec isn't firing — check the Logs tab.
