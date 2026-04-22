"""`AttentionImpl` + metadata builder for the v1.3 PPL codec.

This is the core of M6: the layer-level glue between vLLM's paged-cache
API (`do_kv_cache_update`, `forward`) and the M4/M5 Triton kernels.

Structure (PLAN.md §"The key design decision"):

  * Per-layer STATE
      - `partial_block_staging` : torch.bfloat16[block_size_codec, n_kv_heads, head_size]
                                  — tokens that arrived after the last sealed block
      - `partial_block_count`   : int — how many tokens are in the staging buffer

  * STORE path (called per forward on new KV):
      1. Append incoming tokens to the partial-block staging buffer.
      2. While `partial_block_count >= block_size_codec`:
           a. Peel the first `block_size_codec` tokens out of staging.
           b. Call `encode_block_triton_stage2` on K, then on V.
           c. Serialise the returned dict into the raw-byte slot.
           d. Advance the block table / slot mapping.

  * READ path (called in forward):
      1. For each sealed block in the block table: unpack the slot
         into a parts dict and call `decode_block_triton_from_parts`.
      2. For the trailing partial block: read the bf16 staging buffer
         directly.
      3. Concat sealed + partial, feed the reconstructed K/V into
         FlashAttention-varlen.

Phase A (this commit) lays down the data structures + routing + unit
tests for slot serialisation.  The two pieces that require vLLM's
live engine context — `do_kv_cache_update` dispatch and `forward`
FlashAttention integration — are stubbed with `NotImplementedError`
and finished in Phase B on the H200.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import torch

try:
    from vllm.v1.attention.backend import (
        AttentionImpl,
        AttentionLayer,
        AttentionMetadata,
        AttentionMetadataBuilder,
        AttentionType,
        CommonAttentionMetadata,
    )
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
    class AttentionImpl:  # type: ignore[no-redef]
        pass
    class AttentionMetadata:  # type: ignore[no-redef]
        pass
    class AttentionMetadataBuilder:  # type: ignore[no-redef]
        pass
    class AttentionLayer:  # type: ignore[no-redef]
        pass
    class AttentionType:  # type: ignore[no-redef]
        DECODER = "decoder"
    class CommonAttentionMetadata:  # type: ignore[no-redef]
        pass


from .config import KakeyaV13PPLConfig
from .spec import KakeyaV13PPLAttentionSpec


@dataclass
class KakeyaV13PPLMetadata:
    """Phase-A metadata carrier.  When M6 Phase B wires the backend
    into vLLM, this dataclass becomes a subclass of `AttentionMetadata`.

    All fields are plain torch tensors — nothing here is vLLM-specific,
    so unit tests can construct them without a live engine.
    """

    seq_lens: torch.Tensor              # [num_reqs]
    slot_mapping: torch.Tensor          # [num_tokens]
    block_table: torch.Tensor           # [num_reqs, max_num_blocks]
    query_start_loc: torch.Tensor       # [num_reqs + 1]
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False


class KakeyaV13PPLMetadataBuilder(AttentionMetadataBuilder):
    """Minimal metadata builder that mirrors what TurboQuant does.

    Phase A shim — extends to full vLLM compatibility in Phase B."""

    _cudagraph_support: ClassVar[Any] = None

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        if _HAS_VLLM:
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self._kv_cache_spec = kv_cache_spec
        self._layer_names = layer_names
        self._device = device

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> KakeyaV13PPLMetadata:
        # CUDA-graph capture produces a dummy batch; fill seq_lens=1.
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> KakeyaV13PPLMetadata:
        cam = common_attn_metadata
        return KakeyaV13PPLMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
        )


# ---------------------------------------------------------------------------
# Per-layer staging buffer for the partial-block (< block_size_codec)
# trailing tokens.  PLAN.md §Consequence:
#   "We require `block_size_codec` to be a multiple of `block_size_vllm`.
#    Easiest: set vLLM `block_size = 512` so one vLLM cache block =
#    one codec block."
# With block_size == block_size_codec, the staging buffer is only
# non-empty during prefill / decode.  Decode increments by 1 token
# per call, so the staging buffer fills by 1 per step until a full
# codec block is ready.
# ---------------------------------------------------------------------------


@dataclass
class _PerLayerState:
    """Held on `layer._kakeya_state` once the impl has initialised
    its buffers for that layer."""

    staging_k_bf16: torch.Tensor  # [block_size_codec, n_kv_heads, head_size] bf16
    staging_v_bf16: torch.Tensor  # same
    staging_count: int = 0        # how many valid rows in the staging buffer

    # Calibrated codec constants (loaded from M2 artefacts).
    sigma_q_chol: torch.Tensor | None = None      # [n_kv, D, D] fp32
    sigma_q_inv_chol: torch.Tensor | None = None  # [n_kv, D, D] fp32
    k_centroids: torch.Tensor | None = None       # [2^b_K] fp32
    v_centroids: torch.Tensor | None = None       # [2^b_V] fp32


class KakeyaV13PPLAttentionImpl(AttentionImpl):
    """v1.3 PPL codec attention impl.

    Phase A delivers:
      - Config loading
      - Per-layer state construction
      - Slot serialisation helpers (`_pack_parts_into_slot`,
        `_unpack_slot_into_parts`) with unit tests
      - `do_kv_cache_update` + `forward` skeletons; the full bodies
        land in Phase B.

    Phase B wires these into vLLM's engine loop on H200.
    """

    supports_quant_query_input: ClassVar[bool] = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        # Default codec config — Phase B will plumb per-layer overrides
        # from a vllm plugin-args surface.
        self.k_config = KakeyaV13PPLConfig.default_for_head_dim(head_size)
        self.v_config = KakeyaV13PPLConfig(
            head_dim=head_size,
            d_eff=self.k_config.d_eff,
            block_size_codec=self.k_config.block_size_codec,
            variance_ratio=self.k_config.variance_ratio,
            k_centers=self.k_config.k_centers,
            bit_width=2,
            outlier_budget_frac=0.0,
        )
        self.block_size_codec = self.k_config.block_size_codec

    # ---- Phase A: slot (de)serialisation ------------------------------

    def _pack_parts_into_slot(
        self,
        parts: dict,
        config: KakeyaV13PPLConfig,
    ) -> np.ndarray:
        """Serialise a kakeya_py.encode_block_codes dict into a flat
        uint8 buffer of length `config.slot_size_bytes`.

        Layout: see `KakeyaV13PPLConfig` property accessors.
        """
        out = np.zeros(config.slot_size_bytes, dtype=np.uint8)
        # --- HEADER (48 B) ---
        # 0..4   : magic "KK13"
        # 4..8   : d_eff (u32)
        # 8..12  : k_centers (u32)
        # 12..16 : bit_width (u32)
        # 16..20 : outlier_count_total (u32) — sum across all rows
        # 20..24 : reserved
        # 24..48 : reserved
        header = out[:config.header_bytes]
        header[0:4] = np.frombuffer(b"KK13", dtype=np.uint8)
        header[4:8] = np.frombuffer(
            np.uint32(parts["d_eff"]).tobytes(), dtype=np.uint8,
        )
        header[8:12] = np.frombuffer(
            np.uint32(parts["k"]).tobytes(), dtype=np.uint8,
        )
        header[12:16] = np.frombuffer(
            np.uint32(parts["bit_width"]).tobytes(), dtype=np.uint8,
        )
        oc_total = int(np.asarray(parts["outlier_count"]).sum())
        header[16:20] = np.frombuffer(
            np.uint32(oc_total).tobytes(), dtype=np.uint8,
        )
        # --- PCA basis (fp16) ---
        basis_fp16 = np.asarray(parts["basis"]).astype(np.float16).tobytes(order="C")
        assert len(basis_fp16) == config.pca_basis_bytes, (
            f"basis serialise: got {len(basis_fp16)} B, expected "
            f"{config.pca_basis_bytes} B"
        )
        out[config.offset_pca_basis:
            config.offset_pca_basis + config.pca_basis_bytes
        ] = np.frombuffer(basis_fp16, dtype=np.uint8)
        # --- PCA mean (fp16) ---
        mean_fp16 = np.asarray(parts["mean"]).astype(np.float16).tobytes(order="C")
        assert len(mean_fp16) == config.pca_mean_bytes
        out[config.offset_pca_mean:
            config.offset_pca_mean + config.pca_mean_bytes
        ] = np.frombuffer(mean_fp16, dtype=np.uint8)
        # --- K-means centroids (fp16) ---
        c_fp16 = np.asarray(parts["centers"]).astype(np.float16).tobytes(order="C")
        assert len(c_fp16) == config.kmeans_centroids_bytes
        out[config.offset_kmeans_centroids:
            config.offset_kmeans_centroids + config.kmeans_centroids_bytes
        ] = np.frombuffer(c_fp16, dtype=np.uint8)

        # --- Per-block parallel-array code layout (PLAN.md §cache layout) ---
        # Four parallel arrays rather than interleaved per-vec records:
        #
        #   seg_id_block   : bit-packed `seg_id_bits_per_vec` per vec
        #   t_block        : fp16 per vec
        #   norm_block     : fp16 per vec
        #   residual_block : concatenation of per-vec residual_packed bytes
        #
        # This matches what a coalesced Triton STORE / LOAD kernel
        # wants: each (vec → field) access is a strided read against
        # one of the four arrays.
        seg = np.asarray(parts["seg_id"]).astype(np.uint32)
        t_fp16 = np.asarray(parts["t"]).astype(np.float16)
        nrm_fp16 = np.asarray(parts["norm"]).astype(np.float16)
        pk = np.asarray(parts["residual_packed"])       # [n, per_vec_residual_bytes]
        n = seg.shape[0]
        assert n == config.block_size_codec, (
            f"per_vec count {n} != block_size_codec "
            f"{config.block_size_codec}"
        )
        # (a) seg_id block: bit-pack seg_id_bits_per_vec per vec.
        seg_bits = config.seg_id_bits_per_vec
        seg_bytes = config.seg_id_bytes_per_block
        seg_block = np.zeros(seg_bytes, dtype=np.uint8)
        seg_mask = (1 << seg_bits) - 1
        for i in range(n):
            val = int(seg[i]) & seg_mask
            bit_off = i * seg_bits
            byte_i = bit_off // 8
            shift = bit_off % 8
            seg_block[byte_i] |= (val << shift) & 0xFF
            # Spill to next byte if the value straddles a byte boundary.
            bits_in_first_byte = 8 - shift
            if seg_bits > bits_in_first_byte:
                seg_block[byte_i + 1] |= (val >> bits_in_first_byte) & 0xFF
                # Spill to 3rd byte if seg_bits + shift > 16 (seg_bits ≤ 8
                # and shift ≤ 7 means this is impossible, but guard anyway)
                if (seg_bits + shift) > 16:
                    seg_block[byte_i + 2] |= (val >> (16 - shift)) & 0xFF
        out[config.offset_seg_id_block:
            config.offset_seg_id_block + seg_bytes] = seg_block
        # (b) t block
        out[config.offset_t_block:
            config.offset_t_block + config.t_bytes_per_block] = np.frombuffer(
                t_fp16.tobytes(), dtype=np.uint8,
        )
        # (c) norm block
        out[config.offset_norm_block:
            config.offset_norm_block + config.norm_bytes_per_block] = np.frombuffer(
                nrm_fp16.tobytes(), dtype=np.uint8,
        )
        # (d) residual block: concatenation of per-vec packed bytes.
        # Each per-vec packed length = ceil(wht_len * bit_width / 8).
        per_vec_residual_bytes = pk.shape[1]
        expected_per_vec_rb = (config.wht_len * config.bit_width + 7) // 8
        assert per_vec_residual_bytes == expected_per_vec_rb, (
            f"Rust-packed per-vec residual {per_vec_residual_bytes} B != "
            f"expected {expected_per_vec_rb} B for wht_len={config.wht_len}, "
            f"bit_width={config.bit_width}"
        )
        total_residual_bytes = n * per_vec_residual_bytes
        assert total_residual_bytes == config.residual_bytes_per_block
        out[config.offset_residual_block:
            config.offset_residual_block + total_residual_bytes] = pk.reshape(-1)

        # --- Outlier side-buffer ---
        # Layout (PLAN.md §cache layout):
        #   [ outlier_row_counts  | n × u16  ]   per-row outlier count
        #   [ flat_entries        | k × (u16 idx, f16 val) ]
        # Row boundaries are recovered by prefix-summing the counts at
        # decode time.
        max_outliers = int(parts.get("max_outliers", 0))
        if max_outliers > 0 and config.outlier_budget_bytes > 0:
            oi = np.asarray(parts["outlier_idx"]).astype(np.uint16)
            ov = np.asarray(parts["outlier_val"]).astype(np.float16)
            oc = np.asarray(parts["outlier_count"]).astype(np.uint32)
            off_counts = config.offset_outlier_side_buffer
            off_entries = off_counts + config.outlier_row_count_bytes
            # Write per-row counts.
            out[off_counts:off_counts + config.outlier_row_count_bytes] = np.frombuffer(
                oc.astype("<u2").tobytes(), dtype=np.uint8,
            )
            # Write flat entries.  Each entry = u16 idx + f16 val = 4 B.
            entry_bytes = 4
            max_entries = config.outlier_entry_bytes_budget // entry_bytes
            used = 0
            for i in range(n):
                cnt = int(oc[i])
                for j in range(cnt):
                    if used >= max_entries:
                        raise RuntimeError(
                            f"outlier budget exceeded: {used + 1} entries "
                            f"but only {max_entries} fit (row {i}, j {j}); "
                            "increase outlier_budget_frac."
                        )
                    base = off_entries + used * entry_bytes
                    out[base:base + 2] = np.frombuffer(
                        np.uint16(oi[i, j]).tobytes(), dtype=np.uint8,
                    )
                    out[base + 2:base + 4] = np.frombuffer(
                        ov[i, j].tobytes(), dtype=np.uint8,
                    )
                    used += 1

        return out

    def _unpack_slot_into_parts(
        self,
        slot: np.ndarray,
        config: KakeyaV13PPLConfig,
        head_size: int,
    ) -> dict:
        """Inverse of `_pack_parts_into_slot`.  Reads `slot` (uint8
        buffer of `slot_size_bytes`) and rebuilds the dict that
        `decode_block_triton_from_parts` takes.

        Uses byte-aligned per-vec layout (matches the Phase-A packer).
        """
        if slot.shape != (config.slot_size_bytes,):
            raise ValueError(
                f"slot shape {slot.shape} != ({config.slot_size_bytes},)"
            )
        header = slot[:config.header_bytes]
        magic = bytes(header[0:4])
        if magic != b"KK13":
            raise ValueError(f"bad slot magic {magic!r}")
        d_eff = int(np.frombuffer(bytes(header[4:8]), dtype="<u4")[0])
        k_centers = int(np.frombuffer(bytes(header[8:12]), dtype="<u4")[0])
        bit_width = int(np.frombuffer(bytes(header[12:16]), dtype="<u4")[0])
        if d_eff != config.d_eff:
            raise ValueError(f"slot d_eff={d_eff}, config d_eff={config.d_eff}")

        basis = np.frombuffer(
            bytes(slot[config.offset_pca_basis:
                       config.offset_pca_basis + config.pca_basis_bytes]),
            dtype=np.float16,
        ).astype(np.float32).reshape(d_eff, head_size).copy()
        mean = np.frombuffer(
            bytes(slot[config.offset_pca_mean:
                       config.offset_pca_mean + config.pca_mean_bytes]),
            dtype=np.float16,
        ).astype(np.float32).copy()
        centers = np.frombuffer(
            bytes(slot[config.offset_kmeans_centroids:
                       config.offset_kmeans_centroids + config.kmeans_centroids_bytes]),
            dtype=np.float16,
        ).astype(np.float32).reshape(k_centers, d_eff).copy()

        n = config.block_size_codec
        pbytes = (config.wht_len * bit_width + 7) // 8
        seg_id = np.zeros(n, dtype=np.uint32)
        t = np.empty(n, dtype=np.float32)
        norm = np.empty(n, dtype=np.float32)
        residual_packed = np.empty((n, pbytes), dtype=np.uint8)

        # (a) seg_id — bit-unpack the parallel-array layout.
        seg_bits = config.seg_id_bits_per_vec
        seg_mask = (1 << seg_bits) - 1
        seg_block = slot[
            config.offset_seg_id_block:
            config.offset_seg_id_block + config.seg_id_bytes_per_block
        ]
        for i in range(n):
            bit_off = i * seg_bits
            byte_i = bit_off // 8
            shift = bit_off % 8
            v = int(seg_block[byte_i]) >> shift
            bits_in_first_byte = 8 - shift
            if seg_bits > bits_in_first_byte:
                v |= int(seg_block[byte_i + 1]) << bits_in_first_byte
            seg_id[i] = v & seg_mask

        # (b) t
        t[:] = np.frombuffer(
            bytes(slot[
                config.offset_t_block:
                config.offset_t_block + config.t_bytes_per_block
            ]),
            dtype=np.float16,
        ).astype(np.float32)

        # (c) norm
        norm[:] = np.frombuffer(
            bytes(slot[
                config.offset_norm_block:
                config.offset_norm_block + config.norm_bytes_per_block
            ]),
            dtype=np.float16,
        ).astype(np.float32)

        # (d) residual — concatenated per-vec packed bytes.
        residual_flat = slot[
            config.offset_residual_block:
            config.offset_residual_block + config.residual_bytes_per_block
        ]
        residual_packed[:] = np.asarray(residual_flat).reshape(n, pbytes)

        # Unpack outliers (if any).
        oc_total = int(np.frombuffer(bytes(header[16:20]), dtype="<u4")[0])
        outlier_count = np.zeros(n, dtype=np.uint32)
        outlier_idx = np.zeros((n, max(1, oc_total or 1)), dtype=np.uint16)
        outlier_val = np.zeros_like(outlier_idx, dtype=np.float32)
        max_outliers = 0
        if oc_total > 0 and config.outlier_budget_bytes > 0:
            off_counts = config.offset_outlier_side_buffer
            off_entries = off_counts + config.outlier_row_count_bytes
            # Read per-row counts.
            outlier_count = np.frombuffer(
                bytes(slot[off_counts:off_counts + config.outlier_row_count_bytes]),
                dtype="<u2",
            ).astype(np.uint32).copy()
            assert int(outlier_count.sum()) == oc_total, (
                f"outlier header total {oc_total} != row-count sum "
                f"{int(outlier_count.sum())}"
            )
            max_outliers = int(outlier_count.max()) if oc_total > 0 else 0
            outlier_idx = np.zeros((n, max_outliers), dtype=np.uint16)
            outlier_val = np.zeros((n, max_outliers), dtype=np.float32)
            entry_bytes = 4
            used = 0
            for i in range(n):
                cnt = int(outlier_count[i])
                for j in range(cnt):
                    base = off_entries + used * entry_bytes
                    outlier_idx[i, j] = int(np.frombuffer(
                        bytes(slot[base:base + 2]), dtype="<u2",
                    )[0])
                    outlier_val[i, j] = float(np.frombuffer(
                        bytes(slot[base + 2:base + 4]),
                        dtype=np.float16,
                    )[0])
                    used += 1

        return {
            "mean": mean,
            "basis": basis,
            "centers": centers,
            "d": head_size,
            "d_eff": d_eff,
            "k": k_centers,
            "rotation_seed": 3405691582,     # Phase A: hardcoded to match M2 cal
            "wht_len": config.wht_len,
            "bit_width": bit_width,
            "metric": "mse",                  # Phase B: stored in header
            "seg_id": seg_id,
            "t": t,
            "norm": norm,
            "residual_packed": residual_packed,
            "outlier_idx": outlier_idx,
            "outlier_val": outlier_val,
            "outlier_count": outlier_count,
            "max_outliers": max_outliers,
        }

    # ---- Phase B: live vLLM integration (stubs) ------------------------

    def _ensure_layer_state(
        self,
        layer: AttentionLayer,
        device: torch.device,
    ) -> _PerLayerState:
        """Initialise the per-layer staging buffer + calibrated
        constants.  Phase B plumbs the M2 safetensors load here."""
        state = getattr(layer, "_kakeya_state", None)
        if state is not None:
            return state
        D = self.head_size
        state = _PerLayerState(
            staging_k_bf16=torch.zeros(
                (self.block_size_codec, self.num_kv_heads, D),
                dtype=torch.bfloat16, device=device,
            ),
            staging_v_bf16=torch.zeros(
                (self.block_size_codec, self.num_kv_heads, D),
                dtype=torch.bfloat16, device=device,
            ),
            staging_count=0,
        )
        layer._kakeya_state = state   # type: ignore[attr-defined]
        return state

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Stage-1: append → seal full blocks → write.

        Phase B implements; Phase A leaves this as a documented stub so
        downstream test code can verify the rest of the pipeline without
        needing a live vLLM engine.  The stub raises NotImplementedError
        to enforce PLAN.md's 'no silent fallback' rule.
        """
        raise NotImplementedError(
            "M6 Phase B: do_kv_cache_update lives here.  Phase A "
            "delivers the slot (de)serialisation helpers "
            "(_pack_parts_into_slot / _unpack_slot_into_parts) that "
            "this method will call.  See vllm_backend/kakeya_v1_3_ppl/"
            "impl.py docstring for the pipeline sketch."
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: KakeyaV13PPLMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage-2: decode sealed blocks + read partial block + flash-attn.

        Phase B implements.
        """
        raise NotImplementedError(
            "M6 Phase B: forward lives here.  Wiring is:\n"
            "  * For each sealed block in attn_metadata.block_table: "
            "unpack the slot via _unpack_slot_into_parts and call "
            "kakeyaturbo_py.decode_block_triton_from_parts.\n"
            "  * For the trailing partial block: read from "
            "layer._kakeya_state.staging_[kv]_bf16 via "
            "kakeyaturbo_py.decode_partial_block_bf16.\n"
            "  * Concat and feed into flash_attn_varlen_func."
        )
