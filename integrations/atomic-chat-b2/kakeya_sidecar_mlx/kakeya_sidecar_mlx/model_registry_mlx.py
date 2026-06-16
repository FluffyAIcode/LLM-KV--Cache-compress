"""MLX-variant deployment profiles.

Structurally mirrors ``integrations/atomic-chat/kakeya_sidecar/
kakeya_sidecar/model_registry.py`` (B1) but:

1. Resolves to MLX-specific HF repos (e.g. ``mlx-community/Qwen3-4B-4bit``)
   where available; falls back to the upstream repo when mlx-lm can
   load safetensors directly.
2. Adds ``dflash_draft_repo`` per channel, referencing the z-lab
   pre-trained DFlash draft models that target each base model.
3. Drops the "Gemma-4 + DeepSeek-R1-Distill" entries that don't have
   DFlash drafts today — we keep them for future M4 PRs but mark
   them ``dflash_draft_repo=None`` and ``dflash_available=False``.

The v1.5 report numbers (est_delta_ppl_pct) are transferred unchanged:
they characterise the KakeyaLattice E8 codec, which is bit-identical
between B1 (PyTorch) and B2 (MLX) per ``kakeyalattice_mlx`` parity
tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MLXChannel:
    """Deployment channel for the MLX + DFlash + KakeyaLattice stack."""

    variant: str                    # "e8" only for B2
    q_range: int
    boundary: int = 0
    est_compression: float = 1.0
    est_delta_ppl_pct: float | None = None
    label: str = ""
    dflash_draft_repo: str | None = None
    dflash_available: bool = False

    def __post_init__(self) -> None:
        # ``dflash_available`` must track ``dflash_draft_repo`` to stop us
        # from shipping "feature enabled but no draft" in the UI.
        if self.dflash_available and self.dflash_draft_repo is None:
            object.__setattr__(self, "dflash_available", False)


@dataclass(frozen=True)
class MLXDeploymentProfile:
    short_id: str
    hf_repo_id: str                  # target model repo
    mlx_repo_id: str | None          # mlx-community quant repo if any
    head_dim: int | tuple[int, ...]
    num_hidden_layers: int
    channels: tuple[MLXChannel, ...]
    default_channel: MLXChannel
    notes: str = ""

    def channel_id(self, ch: MLXChannel) -> str:
        suffix = f"-b{ch.boundary}" if ch.boundary else ""
        return f"{self.short_id}@{ch.variant}-q{ch.q_range}{suffix}"

    def all_channel_ids(self) -> list[str]:
        return [self.channel_id(c) for c in self.channels]


# ---------------------------------------------------------------------------
# z-lab DFlash draft catalogue (2026-04-30)
# ---------------------------------------------------------------------------

DFLASH_DRAFTS = {
    "Qwen/Qwen3-4B":                  "z-lab/Qwen3-4B-DFlash-b16",
    "Qwen/Qwen3-8B":                  "z-lab/Qwen3-8B-DFlash-b16",
    "Qwen/Qwen3.5-4B":                "z-lab/Qwen3.5-4B-DFlash",
    "Qwen/Qwen3.5-9B":                "z-lab/Qwen3.5-9B-DFlash",
    "Qwen/Qwen3.5-27B":               "z-lab/Qwen3.5-27B-DFlash",
    "Qwen/Qwen3-Coder-30B-A3B":       "z-lab/Qwen3-Coder-30B-A3B-DFlash",
    "meta-llama/Llama-3.1-8B-Instruct": "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
    "openai/gpt-oss-20b":             "z-lab/gpt-oss-20b-DFlash",
    "openai/gpt-oss-120b":            "z-lab/gpt-oss-120b-DFlash",
}


def _mk_channels_128(hf_repo: str) -> tuple[MLXChannel, ...]:
    """E8 channel set for head_dim=128 models with DFlash support."""
    draft = DFLASH_DRAFTS.get(hf_repo)
    available = draft is not None
    return (
        MLXChannel("e8", 38, 0, 2.50, None, "near-lossless",
                   dflash_draft_repo=draft, dflash_available=available),
        MLXChannel("e8", 10, 0, 3.37, None, "balanced",
                   dflash_draft_repo=draft, dflash_available=available),
        MLXChannel("e8",  4, 0, 4.57, None, "aggressive",
                   dflash_draft_repo=draft, dflash_available=available),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY_MLX: dict[str, MLXDeploymentProfile] = {
    # Qwen3 family — primary DFlash target.
    "qwen3-4b": MLXDeploymentProfile(
        short_id="qwen3-4b",
        hf_repo_id="Qwen/Qwen3-4B",
        mlx_repo_id="mlx-community/Qwen3-4B-4bit",
        head_dim=128,
        num_hidden_layers=36,
        channels=(
            MLXChannel("e8", 38, 0, 2.50, None, "near-lossless",
                       "z-lab/Qwen3-4B-DFlash-b16", True),
            MLXChannel("e8", 10, 0, 3.37, 3.85, "balanced",
                       "z-lab/Qwen3-4B-DFlash-b16", True),
            MLXChannel("e8",  4, 0, 4.57, 17.00, "aggressive",
                       "z-lab/Qwen3-4B-DFlash-b16", True),
        ),
        default_channel=MLXChannel("e8", 10, 0, 3.37, 3.85, "balanced",
                                   "z-lab/Qwen3-4B-DFlash-b16", True),
        notes="v1.5 hero model; DFlash-b16 drafter available.",
    ),
    "qwen3-8b": MLXDeploymentProfile(
        short_id="qwen3-8b",
        hf_repo_id="Qwen/Qwen3-8B",
        mlx_repo_id="mlx-community/Qwen3-8B-4bit",
        head_dim=128,
        num_hidden_layers=32,
        channels=_mk_channels_128("Qwen/Qwen3-8B"),
        default_channel=MLXChannel("e8", 38, 0, 2.50, None, "near-lossless",
                                   "z-lab/Qwen3-8B-DFlash-b16", True),
        notes="B2 primary benchmark target (z-lab official DFlash result).",
    ),

    # Llama-3.1 — DFlash supported.
    "llama-3.1-8b-instruct": MLXDeploymentProfile(
        short_id="llama-3.1-8b-instruct",
        hf_repo_id="meta-llama/Llama-3.1-8B-Instruct",
        mlx_repo_id="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        head_dim=128,
        num_hidden_layers=32,
        channels=_mk_channels_128("meta-llama/Llama-3.1-8B-Instruct"),
        default_channel=MLXChannel("e8", 10, 0, 3.37, None, "balanced",
                                   "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
                                   True),
        notes="HF gated; requires token. DFlash UltraChat variant.",
    ),

    # Qwen3.5 family — DFlash supported.
    "qwen3.5-4b": MLXDeploymentProfile(
        short_id="qwen3.5-4b",
        hf_repo_id="Qwen/Qwen3.5-4B",
        mlx_repo_id="mlx-community/Qwen3.5-4B-4bit",
        head_dim=128,
        num_hidden_layers=36,
        channels=_mk_channels_128("Qwen/Qwen3.5-4B"),
        default_channel=MLXChannel("e8", 10, 0, 3.37, None, "balanced",
                                   "z-lab/Qwen3.5-4B-DFlash", True),
        notes="Qwen3.5 generation, DFlash draft available.",
    ),

    # Mistral — no DFlash draft yet; MLX-only path.
    "mistral-7b-instruct-v0.3": MLXDeploymentProfile(
        short_id="mistral-7b-instruct-v0.3",
        hf_repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        mlx_repo_id="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        head_dim=128,
        num_hidden_layers=32,
        channels=(
            MLXChannel("e8", 38, 0, 2.50, None, "near-lossless"),
            MLXChannel("e8", 10, 0, 3.37, None, "balanced"),
        ),
        default_channel=MLXChannel("e8", 38, 0, 2.50, None, "near-lossless"),
        notes="No DFlash draft; single-track MLX decode only.",
    ),
}


def iter_mlx_channel_ids() -> Iterable[str]:
    for prof in MODEL_REGISTRY_MLX.values():
        yield from prof.all_channel_ids()


def resolve_mlx_model(channel_id: str) -> tuple[MLXDeploymentProfile, MLXChannel]:
    """Parse ``<short>@<variant>-q<Q>[-b<B>]`` into ``(profile, channel)``.

    A bare short id resolves to the profile's ``default_channel``.
    """
    if "@" not in channel_id:
        short = channel_id
        if short not in MODEL_REGISTRY_MLX:
            raise KeyError(f"unknown MLX model id {short!r}")
        prof = MODEL_REGISTRY_MLX[short]
        return prof, prof.default_channel

    short, channel_suffix = channel_id.split("@", 1)
    if short not in MODEL_REGISTRY_MLX:
        raise KeyError(f"unknown MLX model id {short!r}")
    prof = MODEL_REGISTRY_MLX[short]

    parts = channel_suffix.split("-")
    variant = parts[0].lower()
    q_range: int | None = None
    boundary = 0
    for part in parts[1:]:
        if part.startswith("q") and part[1:].isdigit():
            q_range = int(part[1:])
        elif part.startswith("b") and part[1:].isdigit():
            boundary = int(part[1:])
    if q_range is None:
        raise ValueError(
            f"channel suffix {channel_suffix!r} missing q<N> segment"
        )

    for ch in prof.channels:
        if ch.variant == variant and ch.q_range == q_range and ch.boundary == boundary:
            return prof, ch

    raise KeyError(
        f"MLX model {short!r} has no channel matching variant={variant} "
        f"q_range={q_range} boundary={boundary}. Available: "
        + ", ".join(prof.all_channel_ids())
    )
