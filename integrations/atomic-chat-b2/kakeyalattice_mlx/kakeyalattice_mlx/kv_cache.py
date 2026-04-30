"""MLX-lm compatible KV cache with KakeyaLattice E8 compression.

Design mirrors ``kakeyalattice.hf.KakeyaLatticeCache`` but adapted to
mlx-lm's per-layer ``KVCache`` model. Key differences vs HF:

1. mlx-lm creates ONE cache object per layer (``model.make_cache()``
   returns a list of per-layer caches). There's no single global
   object with ``layer_idx`` routing.
2. ``update_and_fetch(keys, values)`` is the hot path; it concatenates
   new K/V to the existing cache and returns the full K/V tensor for
   attention. We intercept just before the concat.
3. mlx-lm has several cache variants (StandardKVCache, RotatingKVCache
   for SWA layers, ConcatenateKVCache). We wrap any of them via
   composition rather than inheritance — the wrapper forwards
   everything except ``update_and_fetch`` to the inner cache.

Usage:

    from kakeyalattice_mlx import KakeyaLatticeMLXCache, E8LatticeCodebookMLX

    caches = []
    for layer_idx in range(num_layers):
        inner = StandardKVCache()            # whatever mlx-lm would build
        if layer_idx < boundary or layer_idx >= num_layers - boundary:
            caches.append(inner)             # boundary skip
        else:
            codec = E8LatticeCodebookMLX(D=head_dim, q_range=q_range)
            caches.append(KakeyaLatticeMLXCache(inner=inner, codec=codec))
    # pass `caches` to `model(..., cache=caches)` in mlx-lm.

The helper ``make_kakeya_caches(model, variant, q_range, boundary)``
does the layer-by-layer wrapping in one call; it reads the model's
num_hidden_layers + head_dim from ``model.args`` (standard mlx-lm
convention).
"""
from __future__ import annotations

from typing import Any


class KakeyaLatticeMLXCache:
    """Wrap any mlx-lm KV cache with E8 codec roundtrip on writes.

    The inner cache is consulted through delegation, so this class
    works transparently with ``StandardKVCache``, ``RotatingKVCache``,
    ``ConcatenateKVCache``, and any future mlx-lm variant whose
    interface includes ``update_and_fetch(keys, values)`` + ``offset``.
    """

    def __init__(self, inner: Any, codec: Any) -> None:
        self._inner = inner
        self._codec = codec
        self._fired = 0
        self._skipped = 0

    # ----- core hot path -----

    def update_and_fetch(self, keys, values):
        """Roundtrip new K/V through the codec, then delegate concat.

        ``keys`` / ``values`` have mlx-lm's standard shape
        ``[B, H_kv, S, D]`` where ``D`` is the head dim. The codec's
        ``.roundtrip(x)`` only touches the last dim, so this shape is
        accepted unchanged.
        """
        try:
            k_rt = self._codec.roundtrip(keys)
            v_rt = self._codec.roundtrip(values)
            self._fired += 1
        except Exception:  # pragma: no cover — defensive
            self._skipped += 1
            return self._inner.update_and_fetch(keys, values)
        return self._inner.update_and_fetch(k_rt, v_rt)

    # ----- introspection / diagnostics -----

    @property
    def offset(self):
        return self._inner.offset

    @property
    def state(self):
        return self._inner.state

    @state.setter
    def state(self, v):
        self._inner.state = v

    @property
    def fire_count(self) -> int:
        return self._fired

    @property
    def skip_count(self) -> int:
        return self._skipped

    def __repr__(self) -> str:
        return (
            f"KakeyaLatticeMLXCache(inner={type(self._inner).__name__}, "
            f"codec={getattr(self._codec, 'name', type(self._codec).__name__)}, "
            f"fired={self._fired}, skipped={self._skipped})"
        )

    # Forward every attribute not explicitly overridden to the inner
    # cache. This keeps us compatible with attributes we don't know
    # about yet (mlx-lm adds new ones across releases).
    def __getattr__(self, name: str):
        # Only reached if the attribute was not found on `self`.
        inner = object.__getattribute__(self, "_inner")
        return getattr(inner, name)


def make_kakeya_caches(
    model: Any,
    *,
    variant: str = "e8",
    q_range: int = 38,
    boundary: int = 0,
    inner_factory=None,
    dtype=None,
):
    """Build a list of per-layer caches wrapping mlx-lm's defaults.

    Args:
        model: an mlx-lm model (must expose ``.args`` with
            ``num_hidden_layers`` and either ``head_dim`` or
            ``hidden_size``+``num_attention_heads``).
        variant: must be ``"e8"`` (we ship E8 only for B2).
        q_range: E8 Q parameter. Pass 38 for near-lossless, 10 for
            balanced, 4 for aggressive.
        boundary: number of front + back layers to keep uncompressed.
        inner_factory: a zero-arg callable returning a fresh mlx-lm
            cache per layer. Defaults to ``mlx_lm.models.cache.
            KVCache`` (aka StandardKVCache).

    Returns:
        list[KakeyaLatticeMLXCache | <inner>] of length
        ``model.args.num_hidden_layers``.
    """
    if variant.lower() != "e8":
        raise ValueError(f"B2 ships E8 only, got {variant!r}")

    from .codec import E8LatticeCodebookMLX

    args = model.args if hasattr(model, "args") else model.config
    num_layers = getattr(args, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("model.args must expose num_hidden_layers")

    head_dim = getattr(args, "head_dim", None)
    if head_dim is None:
        hidden = getattr(args, "hidden_size", None)
        n_heads = getattr(args, "num_attention_heads", None)
        if hidden is None or n_heads is None:
            raise ValueError(
                "cannot infer head_dim; pass a model with args.head_dim "
                "or args.hidden_size + args.num_attention_heads"
            )
        head_dim = hidden // n_heads

    if inner_factory is None:
        from mlx_lm.models.cache import KVCache as _InnerCls  # type: ignore
        inner_factory = _InnerCls

    caches: list[Any] = []
    for layer_idx in range(num_layers):
        inner = inner_factory()
        is_boundary = (
            boundary > 0
            and (layer_idx < boundary or layer_idx >= num_layers - boundary)
        )
        if is_boundary:
            caches.append(inner)
        else:
            codec = E8LatticeCodebookMLX(D=head_dim, q_range=q_range, dtype=dtype)
            caches.append(KakeyaLatticeMLXCache(inner=inner, codec=codec))
    return caches


__all__ = ["KakeyaLatticeMLXCache", "make_kakeya_caches"]
