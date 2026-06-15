"""Per-model deployment profiles.

The registry maps a user-facing model id (e.g. ``qwen3-4b``) to:

1. The HuggingFace repo id we pull weights from.
2. The canonical KakeyaLattice channels we support for that model
   (variant ∈ {"d4", "e8"}, q_range ∈ {...}, boundary ∈ {...}).
3. Human-readable metadata (head_dim, compression-ratio estimate,
   measured |Δppl|) so the UI / ``/v1/models`` response can show it.

The numbers in ``channels[].est_*`` come directly from
``reports/v1_5_release/V15_FULL_4MODEL_REPORT.md`` where we have
measurements, and from the compression-ratio formula otherwise.

Nothing in this module imports torch or transformers — it is pure
metadata, safe to import in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Channel:
    """One (variant, Q, boundary) deployment channel for a model."""

    variant: str                 # "d4" | "e8"
    q_range: int                 # 4, 10, 38, 152 …
    boundary: int = 0            # number of front/back layers to skip
    est_compression: float = 1.0  # KV cache compression vs bf16
    est_delta_ppl_pct: float | None = None  # measured if available
    label: str = ""              # e.g. "balanced", "aggressive", "near-lossless"


@dataclass(frozen=True)
class DeploymentProfile:
    """Full deployment profile for one HF model."""

    short_id: str                # "qwen3-4b"
    hf_repo_id: str              # "Qwen/Qwen3-4B"
    head_dim: int | tuple[int, ...]  # tuple for heterogeneous (Gemma-4)
    num_hidden_layers: int
    channels: tuple[Channel, ...]
    default_channel: Channel
    notes: str = ""

    def channel_id(self, ch: Channel) -> str:
        """UI/API id = ``<short>@<variant>-q<Q>[-b<boundary>]``."""
        suffix = f"-b{ch.boundary}" if ch.boundary else ""
        return f"{self.short_id}@{ch.variant}-q{ch.q_range}{suffix}"

    def all_channel_ids(self) -> list[str]:
        return [self.channel_id(c) for c in self.channels]


def _std_channels_128() -> tuple[Channel, ...]:
    """Canonical channels for head_dim=128 (Qwen/Llama/Mistral/GLM/DeepSeek).

    Compression ratios from v1.5 report Table §3.
    """
    return (
        Channel("e8", q_range=38, boundary=0,
                est_compression=2.50, est_delta_ppl_pct=None,
                label="near-lossless"),
        Channel("e8", q_range=10, boundary=0,
                est_compression=3.37, est_delta_ppl_pct=None,
                label="balanced"),
        Channel("e8", q_range=4, boundary=0,
                est_compression=4.57, est_delta_ppl_pct=None,
                label="aggressive"),
    )


# NOTE: est_delta_ppl_pct is populated per-model where v1.5 measured it;
# empty entries stay ``None`` and the UI shows "estimated only".

MODEL_REGISTRY: dict[str, DeploymentProfile] = {
    # --- Qwen family ----------------------------------------------------
    "qwen3-4b": DeploymentProfile(
        short_id="qwen3-4b",
        hf_repo_id="Qwen/Qwen3-4B",
        head_dim=128,
        num_hidden_layers=36,
        channels=(
            Channel("e8", 38, 0, 2.50, None, "near-lossless"),
            Channel("e8", 10, 0, 3.37, 3.85, "balanced"),
            Channel("e8",  4, 0, 4.57, 17.00, "aggressive"),
        ),
        default_channel=Channel("e8", 10, 0, 3.37, 3.85, "balanced"),
        notes="v1.5 E8 first-measurement model; numbers from V15_FULL_4MODEL_REPORT.md.",
    ),
    "qwen2-1.5b": DeploymentProfile(
        short_id="qwen2-1.5b",
        hf_repo_id="Qwen/Qwen2-1.5B",
        head_dim=128,
        num_hidden_layers=28,
        channels=_std_channels_128(),
        default_channel=Channel("e8", 38, 0, 2.50, None, "near-lossless"),
        notes="Small Qwen2; conservative default (Q=38) on Mac 8-16 GB.",
    ),

    # --- Llama family ---------------------------------------------------
    "llama-3.2-3b-instruct": DeploymentProfile(
        short_id="llama-3.2-3b-instruct",
        hf_repo_id="meta-llama/Llama-3.2-3B-Instruct",
        head_dim=128,
        num_hidden_layers=28,
        channels=_std_channels_128(),
        default_channel=Channel("e8", 10, 0, 3.37, None, "balanced"),
        notes="Main Mac-16GB target; requires HF token (gated).",
    ),
    "llama-3.1-8b-instruct": DeploymentProfile(
        short_id="llama-3.1-8b-instruct",
        hf_repo_id="meta-llama/Llama-3.1-8B-Instruct",
        head_dim=128,
        num_hidden_layers=32,
        channels=_std_channels_128(),
        default_channel=Channel("e8", 10, 0, 3.37, None, "balanced"),
        notes="Best quality/size trade-off for Mac-32GB.",
    ),

    # --- Mistral --------------------------------------------------------
    "mistral-7b-instruct-v0.3": DeploymentProfile(
        short_id="mistral-7b-instruct-v0.3",
        hf_repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        head_dim=128,
        num_hidden_layers=32,
        channels=_std_channels_128(),
        default_channel=Channel("e8", 10, 0, 3.37, None, "balanced"),
    ),

    # --- Gemma ----------------------------------------------------------
    # Heterogeneous head_dim (20×256 + 4×512 MatFormer); KakeyaLatticeCache
    # handles per-layer head_dim internally.
    "gemma-4-e4b": DeploymentProfile(
        short_id="gemma-4-e4b",
        hf_repo_id="google/gemma-4-E4B",
        head_dim=(256, 512),
        num_hidden_layers=24,
        channels=(
            Channel("e8", 38, 0, 2.30, None, "near-lossless"),
            Channel("e8", 10, 0, 3.47, 1.56, "balanced"),
            Channel("e8",  4, 0, 4.77, 5.79, "aggressive"),
        ),
        default_channel=Channel("e8", 10, 0, 3.47, 1.56, "balanced"),
        notes="Heterogeneous head_dim; measured in v1.5 report.",
    ),

    # --- DeepSeek (R1-Distill series) ----------------------------------
    # Small DeepSeek models are structurally fragile under no-boundary.
    # Force boundary=2 on every channel.
    "deepseek-r1-distill-qwen-1.5b": DeploymentProfile(
        short_id="deepseek-r1-distill-qwen-1.5b",
        hf_repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        head_dim=128,
        num_hidden_layers=28,
        channels=(
            Channel("e8", 38, 2, 2.40, None, "near-lossless"),
            Channel("e8", 10, 2, 3.20, None, "balanced"),
        ),
        default_channel=Channel("e8", 38, 2, 2.40, None, "near-lossless"),
        notes=(
            "boundary=2 is REQUIRED; no-boundary in-forward explodes "
            "to >50 000% |Δppl| per V15_FULL_4MODEL_REPORT.md §1."
        ),
    ),
    "deepseek-r1-distill-qwen-7b": DeploymentProfile(
        short_id="deepseek-r1-distill-qwen-7b",
        hf_repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        head_dim=128,
        num_hidden_layers=28,
        channels=(
            Channel("e8", 38, 2, 2.40, None, "near-lossless"),
            Channel("e8", 10, 2, 3.20, None, "balanced"),
        ),
        default_channel=Channel("e8", 10, 2, 3.20, None, "balanced"),
    ),

    # --- GLM ------------------------------------------------------------
    "glm-4-9b-chat": DeploymentProfile(
        short_id="glm-4-9b-chat",
        hf_repo_id="zai-org/GLM-4-9B-Chat",
        head_dim=128,
        num_hidden_layers=40,
        channels=(
            Channel("e8", 38, 0, 2.30, None, "near-lossless"),
            Channel("e8", 10, 0, 3.37, 6.96, "balanced"),
            Channel("e8",  4, 0, 4.57, 32.36, "aggressive"),
        ),
        default_channel=Channel("e8", 10, 0, 3.37, 6.96, "balanced"),
        notes="Requires trust_remote_code=True.",
    ),
}


def iter_model_channel_ids() -> Iterable[str]:
    for prof in MODEL_REGISTRY.values():
        yield from prof.all_channel_ids()


def resolve_model(channel_id: str) -> tuple[DeploymentProfile, Channel]:
    """Parse ``<short>@<variant>-q<Q>[-b<B>]`` into a (profile, channel) pair.

    Accepts the short id alone (no channel suffix) and returns the
    profile's ``default_channel``.
    """
    if "@" not in channel_id:
        short = channel_id
        if short not in MODEL_REGISTRY:
            raise KeyError(f"unknown model id {short!r}")
        prof = MODEL_REGISTRY[short]
        return prof, prof.default_channel

    short, channel_suffix = channel_id.split("@", 1)
    if short not in MODEL_REGISTRY:
        raise KeyError(f"unknown model id {short!r}")
    prof = MODEL_REGISTRY[short]

    parts = channel_suffix.split("-")
    variant = parts[0].lower()
    q_range = None
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
        f"model {short!r} has no channel matching variant={variant} "
        f"q_range={q_range} boundary={boundary}. Available: "
        + ", ".join(prof.all_channel_ids())
    )
