#!/usr/bin/env python3
"""
KakeyaCodec-based KV cache compression for Gemma 4 / Hugging Face transformers.

This module keeps the main KakeyaCodec logic intact:
  1. PCA projection
  2. temporal direction separation
  3. spherical K-means on perpendicular directions
  4. sparse residual storage
  5. approximate decode on demand

It does not modify model parameters. It only swaps the runtime KV cache object.

Design notes:
  - Full-attention layers use Kakeya compression for old KV history.
  - Sliding-window layers stay uncompressed because their cache is already bounded.
  - Gemma 4 shared-KV layers remain compatible because transformers excludes them
    from the cache layer list and stores shared tensors on `cache.shared_layers`.
  - The implementation is optimized for correctness and integration simplicity.
    It is not a fused kernel and may trade speed for memory savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicLayer, DynamicSlidingWindowLayer


@dataclass
class CompressedVec:
    """Compressed representation of a batch of vectors."""

    seg_id: torch.Tensor
    alpha: torch.Tensor
    t: torch.Tensor
    residual_vals: torch.Tensor
    residual_idx: torch.Tensor


@dataclass
class KakeyaSkeleton:
    """The Kakeya-like skeleton: basis + temporal direction + segment centers."""

    basis: torch.Tensor
    mean: torch.Tensor
    t_dir: torch.Tensor
    centers: torch.Tensor
    d_eff: int
    K: int
    d_res: int


@dataclass
class KakeyaCompressedBlock:
    """Compressed block for a KV tensor slice."""

    skeleton: KakeyaSkeleton
    encoded: CompressedVec
    shape: Tuple[int, int, int, int]
    dtype: torch.dtype


class KakeyaCodec:
    """
    Kakeya-like set compression codec generalized to vectors of size `d_model`.

    This intentionally preserves the main implementation logic from the original
    semantic embedding codec, but makes it reusable for KV cache vectors where
    `d_model == head_dim`.
    """

    def __init__(
        self,
        d_model: int,
        variance_ratio: float = 0.99,
        K: int = 16,
        d_res: int = 8,
        min_rows_to_build: int = 8,
    ):
        self.d_model = d_model
        self.variance_ratio = variance_ratio
        self.K = K
        self.d_res = d_res
        self.min_rows_to_build = min_rows_to_build

    def _compute_pca(self, vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        mu = vecs.mean(0)
        centered = vecs - mu.unsqueeze(0)
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        denom = S.pow(2).sum().clamp_min(1e-8)
        cumvar = S.pow(2).cumsum(0) / denom
        d_eff_arr = (cumvar >= self.variance_ratio).nonzero(as_tuple=True)[0]
        d_eff = (d_eff_arr[0].item() + 1) if len(d_eff_arr) > 0 else len(S)
        d_eff = max(min(d_eff, self.d_model), 2)
        basis = Vh[:d_eff]
        return basis, mu, d_eff

    def _spherical_kmeans(
        self,
        dirs: torch.Tensor,
        K: int,
        max_iter: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, _ = dirs.shape
        K = min(K, N)
        if K <= 1:
            return dirs[:1].clone(), torch.zeros(N, dtype=torch.long, device=dirs.device)

        centers = [dirs[0].clone()]
        for _ in range(K - 1):
            sims = torch.stack([dirs @ c for c in centers], dim=1)
            max_sim = sims.max(dim=1)[0]
            farthest = max_sim.argmin()
            centers.append(dirs[farthest].clone())

        centers = torch.stack(centers)
        assignments = torch.zeros(N, dtype=torch.long, device=dirs.device)

        for _ in range(max_iter):
            sims = dirs @ centers.T
            new_assign = sims.argmax(dim=1)
            if torch.equal(new_assign, assignments):
                break
            assignments = new_assign
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    centers[k] = F.normalize(dirs[mask].mean(0), dim=0, eps=1e-8)
                else:
                    far = (dirs @ centers.T).max(1)[0].argmin()
                    centers[k] = dirs[far].clone()
                    assignments[far] = k
        return centers, assignments

    def fit(self, vecs: torch.Tensor) -> KakeyaSkeleton:
        if vecs.ndim != 2 or vecs.shape[-1] != self.d_model:
            raise ValueError(f"Expected [N, {self.d_model}] input, got {tuple(vecs.shape)}")
        if vecs.shape[0] < self.min_rows_to_build:
            raise ValueError(
                f"Need at least {self.min_rows_to_build} rows to build a Kakeya skeleton, "
                f"got {vecs.shape[0]}"
            )

        basis, mu, d_eff = self._compute_pca(vecs)
        coeffs = (vecs - mu.unsqueeze(0)) @ basis.T
        mu_coeff = coeffs.mean(0)
        mu_norm = mu_coeff.norm()
        if mu_norm > 1e-8:
            t_dir = mu_coeff / mu_norm
        else:
            t_dir = torch.zeros(d_eff, device=vecs.device, dtype=vecs.dtype)
            t_dir[0] = 1.0

        alpha = coeffs @ t_dir
        perp = coeffs - alpha.unsqueeze(-1) * t_dir.unsqueeze(0)
        perp_norms = perp.norm(dim=-1)
        valid_mask = perp_norms > 1e-8

        if valid_mask.sum() >= 2:
            perp_dirs = F.normalize(perp[valid_mask], dim=-1)
            K_actual = min(self.K, perp_dirs.shape[0])
            centers, _ = self._spherical_kmeans(perp_dirs, K_actual)
        else:
            centers = F.normalize(torch.randn(1, d_eff, device=vecs.device, dtype=vecs.dtype), dim=-1)
            K_actual = 1

        return KakeyaSkeleton(
            basis=basis.detach().cpu(),
            mean=mu.detach().cpu(),
            t_dir=t_dir.detach().cpu(),
            centers=centers.detach().cpu(),
            d_eff=d_eff,
            K=K_actual,
            d_res=self.d_res,
        )

    def encode(self, vecs: torch.Tensor, skeleton: KakeyaSkeleton) -> CompressedVec:
        coeff = (vecs - skeleton.mean.to(vecs.device, vecs.dtype)) @ skeleton.basis.to(vecs.device, vecs.dtype).T
        t_dir = skeleton.t_dir.to(vecs.device, vecs.dtype)
        centers = skeleton.centers.to(vecs.device, vecs.dtype)

        alpha = coeff @ t_dir
        perp = coeff - alpha.unsqueeze(-1) * t_dir.unsqueeze(0)
        perp_norm = perp.norm(dim=-1, keepdim=True)

        nonzero = perp_norm.squeeze(-1) > 1e-8
        perp_dir = torch.zeros_like(perp)
        perp_dir[nonzero] = perp[nonzero] / perp_norm[nonzero]

        sims = perp_dir @ centers.T
        seg_id = sims.argmax(dim=1)
        chosen_centers = centers[seg_id]
        t = (perp * chosen_centers).sum(dim=-1)
        residual = perp - t.unsqueeze(-1) * chosen_centers

        d_res = min(skeleton.d_res, skeleton.d_eff)
        if d_res < skeleton.d_eff:
            _, top_idx = residual.abs().topk(d_res, dim=-1)
            r_vals = residual.gather(-1, top_idx)
        else:
            top_idx = (
                torch.arange(skeleton.d_eff, device=vecs.device)
                .unsqueeze(0)
                .expand(vecs.shape[0], -1)
                .contiguous()
            )
            r_vals = residual

        return CompressedVec(
            seg_id=seg_id.detach().cpu(),
            alpha=alpha.detach().cpu(),
            t=t.detach().cpu(),
            residual_vals=r_vals.detach().cpu(),
            residual_idx=top_idx.detach().cpu(),
        )

    def decode(
        self,
        comp: CompressedVec,
        skeleton: KakeyaSkeleton,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        n = comp.seg_id.shape[0]
        d_eff = skeleton.d_eff

        residual = torch.zeros(n, d_eff, device=device, dtype=dtype)
        idx = comp.residual_idx.to(device)
        vals = comp.residual_vals.to(device, dtype)
        residual.scatter_(1, idx, vals)

        centers = skeleton.centers.to(device, dtype)
        t_dir = skeleton.t_dir.to(device, dtype)
        basis = skeleton.basis.to(device, dtype)
        mean = skeleton.mean.to(device, dtype)

        perp_approx = comp.t.to(device, dtype).unsqueeze(-1) * centers[comp.seg_id.to(device)] + residual
        coeff_approx = comp.alpha.to(device, dtype).unsqueeze(-1) * t_dir + perp_approx
        return coeff_approx @ basis + mean


class KakeyaCompressedLayer(DynamicLayer):
    """
    KV cache layer that stores:
      - recent tokens in exact precision
      - older tokens as Kakeya-compressed blocks

    Returned tensors are always materialized in standard KV shape, so the model
    remains unaware of the compression.
    """

    is_sliding = False
    is_compileable = False

    def __init__(
        self,
        variance_ratio: float = 0.99,
        K: int = 16,
        d_res: int = 8,
        residual_length: int = 2048,
        block_size: int = 512,
        min_rows_to_build: int = 8,
    ):
        super().__init__()
        self.variance_ratio = variance_ratio
        self.K = K
        self.d_res = d_res
        self.residual_length = residual_length
        self.block_size = block_size
        self.min_rows_to_build = min_rows_to_build

        self.dtype: Optional[torch.dtype] = None
        self.device: Optional[torch.device] = None
        self.cumulative_length = 0

        self._compressed_key_blocks: List[KakeyaCompressedBlock] = []
        self._compressed_value_blocks: List[KakeyaCompressedBlock] = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(compressed_blocks={len(self._compressed_key_blocks)})"

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def _make_codec(self, head_dim: int) -> KakeyaCodec:
        return KakeyaCodec(
            d_model=head_dim,
            variance_ratio=self.variance_ratio,
            K=self.K,
            d_res=self.d_res,
            min_rows_to_build=self.min_rows_to_build,
        )

    def _compress_block(self, tensor: torch.Tensor) -> KakeyaCompressedBlock:
        bsz, n_heads, n_tokens, head_dim = tensor.shape
        flat = tensor.permute(0, 1, 2, 3).reshape(-1, head_dim).float()
        codec = self._make_codec(head_dim)
        skeleton = codec.fit(flat)
        encoded = codec.encode(flat, skeleton)
        return KakeyaCompressedBlock(
            skeleton=skeleton,
            encoded=encoded,
            shape=(bsz, n_heads, n_tokens, head_dim),
            dtype=tensor.dtype,
        )

    def _decode_block(self, block: KakeyaCompressedBlock, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        codec = self._make_codec(block.shape[-1])
        flat = codec.decode(block.encoded, block.skeleton, device=device, dtype=dtype)
        bsz, n_heads, n_tokens, head_dim = block.shape
        return flat.reshape(bsz, n_heads, n_tokens, head_dim)

    def _materialize_blocks(
        self,
        blocks: Sequence[KakeyaCompressedBlock],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not blocks:
            return torch.tensor([], device=device, dtype=dtype)
        decoded = [self._decode_block(block, device, dtype) for block in blocks]
        return torch.cat(decoded, dim=-2)

    def _full_keys(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        prefix = self._materialize_blocks(self._compressed_key_blocks, device, dtype)
        if self.keys is None or self.keys.numel() == 0:
            return prefix
        if prefix.numel() == 0:
            return self.keys.to(device=device, dtype=dtype)
        return torch.cat([prefix, self.keys.to(device=device, dtype=dtype)], dim=-2)

    def _full_values(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        prefix = self._materialize_blocks(self._compressed_value_blocks, device, dtype)
        if self.values is None or self.values.numel() == 0:
            return prefix
        if prefix.numel() == 0:
            return self.values.to(device=device, dtype=dtype)
        return torch.cat([prefix, self.values.to(device=device, dtype=dtype)], dim=-2)

    def _append_exact(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if self.keys is None or self.values is None:
            raise RuntimeError("Layer must be initialized before appending KV states.")
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)

    def _compress_oldest_exact_block(self) -> bool:
        if self.keys is None or self.values is None:
            return False
        exact_len = self.keys.shape[-2]
        overflow = exact_len - self.residual_length
        if overflow < self.block_size:
            return False

        block_len = self.block_size
        key_block = self.keys[:, :, :block_len, :].contiguous()
        value_block = self.values[:, :, :block_len, :].contiguous()

        self._compressed_key_blocks.append(self._compress_block(key_block))
        self._compressed_value_blocks.append(self._compress_block(value_block))
        self.keys = self.keys[:, :, block_len:, :].contiguous()
        self.values = self.values[:, :, block_len:, :].contiguous()
        return True

    def _rebalance(self) -> None:
        while self._compress_oldest_exact_block():
            pass

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.cumulative_length += key_states.shape[-2]

        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self._append_exact(key_states, value_states)
        self._rebalance()

        full_k = self._full_keys(key_states.device, key_states.dtype)
        full_v = self._full_values(value_states.device, value_states.dtype)
        return full_k, full_v

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        kv_offset = 0
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1

    def _reset_storage(self) -> None:
        self._compressed_key_blocks.clear()
        self._compressed_value_blocks.clear()
        if self.keys is not None:
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        if self.values is not None:
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def _rebuild_from_full(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.shape != values.shape:
            raise ValueError("Key/value tensors must have the same shape when rebuilding cache.")
        if keys.ndim != 4:
            raise ValueError(f"Expected [batch, heads, seq, dim] cache tensor, got {tuple(keys.shape)}")

        if not self.is_initialized:
            self.lazy_initialization(keys, values)

        self._reset_storage()
        total_len = keys.shape[-2]
        split = max(total_len - self.residual_length, 0)
        compressed_prefix_len = (split // self.block_size) * self.block_size

        for start in range(0, compressed_prefix_len, self.block_size):
            end = start + self.block_size
            self._compressed_key_blocks.append(self._compress_block(keys[:, :, start:end, :].contiguous()))
            self._compressed_value_blocks.append(self._compress_block(values[:, :, start:end, :].contiguous()))

        self.keys = keys[:, :, compressed_prefix_len:, :].to(device=self.device, dtype=self.dtype).contiguous()
        self.values = values[:, :, compressed_prefix_len:, :].to(device=self.device, dtype=self.dtype).contiguous()
        self.cumulative_length = total_len

    def crop(self, max_length: int) -> None:
        if self.get_seq_length() == 0:
            return
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return

        full_k = self._full_keys(self.device, self.dtype)
        full_v = self._full_values(self.device, self.dtype)
        self._rebuild_from_full(full_k[..., :max_length, :], full_v[..., :max_length, :])

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        full_k = self._full_keys(self.device, self.dtype).repeat_interleave(repeats, dim=0)
        full_v = self._full_values(self.device, self.dtype).repeat_interleave(repeats, dim=0)
        self._rebuild_from_full(full_k, full_v)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        full_k = self._full_keys(self.device, self.dtype)[indices, ...]
        full_v = self._full_values(self.device, self.dtype)[indices, ...]
        self._rebuild_from_full(full_k, full_v)

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx.to(self.device))

    def offload(self) -> None:
        if self.is_initialized and self.keys is not None and self.values is not None:
            self.keys = self.keys.to("cpu", non_blocking=True)
            self.values = self.values.to("cpu", non_blocking=True)

    def prefetch(self) -> None:
        if self.is_initialized and self.keys is not None and self.values is not None and self.device is not None:
            if self.keys.device != self.device:
                self.keys = self.keys.to(self.device, non_blocking=True)
                self.values = self.values.to(self.device, non_blocking=True)

    def reset(self) -> None:
        self._compressed_key_blocks.clear()
        self._compressed_value_blocks.clear()
        self.cumulative_length = 0
        if self.is_initialized:
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)


class KakeyaKVCache(Cache):
    """
    Drop-in transformers cache for Gemma 4 style decoding.

    Non-sliding attention layers use Kakeya compression on old KV history.
    Sliding-window layers are left as standard `DynamicSlidingWindowLayer`.
    """

    def __init__(
        self,
        config,
        variance_ratio: float = 0.99,
        K: int = 16,
        d_res: int = 8,
        residual_length: int = 2048,
        block_size: int = 512,
        min_rows_to_build: int = 8,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        decoder_config = config.get_text_config(decoder=True)
        sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
            decoder_config, "attention_chunk_size", None
        )
        layer_types = getattr(decoder_config, "layer_types", None)
        if layer_types is None:
            layer_types = []
            for _ in range(decoder_config.num_hidden_layers):
                if sliding_window is not None:
                    layer_types.append("sliding_attention")
                else:
                    layer_types.append("full_attention")

        num_kv_shared_layers = getattr(decoder_config, "num_kv_shared_layers", 0) or 0
        if num_kv_shared_layers > 0:
            layer_types = layer_types[:-num_kv_shared_layers]

        layers = []
        for layer_type in layer_types:
            if layer_type in ("sliding_attention", "chunked_attention"):
                layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
            elif layer_type in ("full_attention", "hybrid"):
                layers.append(
                    KakeyaCompressedLayer(
                        variance_ratio=variance_ratio,
                        K=K,
                        d_res=d_res,
                        residual_length=residual_length,
                        block_size=block_size,
                        min_rows_to_build=min_rows_to_build,
                    )
                )
            else:
                layers.append(
                    KakeyaCompressedLayer(
                        variance_ratio=variance_ratio,
                        K=K,
                        d_res=d_res,
                        residual_length=residual_length,
                        block_size=block_size,
                        min_rows_to_build=min_rows_to_build,
                    )
                )

        super().__init__(
            layers=layers,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

        # Kept for compatibility with helpers that inspect cache-level metadata.
        # Gemma 4's shared-KV layers are fed through a per-forward `shared_kv_states`
        # dict built inside the model, not through this attribute, so it stays empty.
        self.shared_layers: dict = {}


def build_gemma4_kakeya_cache(
    model,
    variance_ratio: float = 0.99,
    K: int = 16,
    d_res: int = 8,
    residual_length: int = 2048,
    block_size: int = 512,
    min_rows_to_build: int = 8,
    offloading: bool = False,
) -> KakeyaKVCache:
    """
    Convenience factory for Gemma 4 generation.

    Example:
        cache = build_gemma4_kakeya_cache(model)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            past_key_values=cache,
        )
    """

    return KakeyaKVCache(
        config=model.config,
        variance_ratio=variance_ratio,
        K=K,
        d_res=d_res,
        residual_length=residual_length,
        block_size=block_size,
        min_rows_to_build=min_rows_to_build,
        offloading=offloading,
    )


__all__ = [
    "CompressedVec",
    "KakeyaSkeleton",
    "KakeyaCompressedBlock",
    "KakeyaCodec",
    "KakeyaCompressedLayer",
    "KakeyaKVCache",
    "build_gemma4_kakeya_cache",
]
