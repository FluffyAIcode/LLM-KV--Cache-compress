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

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import torch

try:
    from vllm.v1.attention.backend import (
        AttentionCGSupport,
        AttentionImpl,
        AttentionLayer,
        AttentionMetadata,
        AttentionMetadataBuilder,
        AttentionType,
        CommonAttentionMetadata,
    )
    _HAS_VLLM = True
    _CG_SUPPORT_NEVER = AttentionCGSupport.NEVER
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
    _CG_SUPPORT_NEVER = None


from .calibration import CalibrationBundle, load_calibration_bundle
from .config import KakeyaV13PPLConfig
from .spec import KakeyaV13PPLAttentionSpec


# Process-global calibration bundle.  vLLM instantiates AttentionImpl
# per (layer, worker) and we don't own the construction site, so the
# clean way to plumb calibration data is a module-level register that
# the plugin entry point (Phase B.2d) populates before model init.
_GLOBAL_CALIBRATION: CalibrationBundle | None = None


def set_global_calibration(bundle: CalibrationBundle | None) -> None:
    """Install a calibration bundle that every new
    `KakeyaV13PPLAttentionImpl` instance will pick up.

    Call this once at plugin init (Phase B.2d) with the M2
    safetensors + Lloyd-Max tables loaded via
    `calibration.load_calibration_bundle(...)`.

    Setting to None reverts to Gaussian-default centroids and no
    Σ_q whitening — the Phase B.1 behaviour.
    """
    global _GLOBAL_CALIBRATION
    _GLOBAL_CALIBRATION = bundle


def _parse_layer_idx_from_name(layer_name: str | None) -> int | None:
    """vLLM exposes `layer.layer_name = "model.layers.{L}.self_attn.attn"`.
    Parse L; return None on unknown format so calibration can stay
    identity instead of crashing."""
    if not layer_name:
        return None
    parts = layer_name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


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
    """Minimal metadata builder.

    Phase B.1 sets `_cudagraph_support = NEVER` because our impl
    keeps per-block staging in a Python dict + uses CPU-side
    `.tolist()` in `do_kv_cache_update`, which breaks CUDAGraph
    capture.  Phase B.2b will move staging to a dense tensor and
    switch to `UNIFORM_BATCH`, matching TurboQuant's support level.
    """

    # When vllm is installed: _CG_SUPPORT_NEVER is AttentionCGSupport.NEVER
    # When vllm missing (CPU-only dev): None (tests don't exercise this).
    _cudagraph_support: ClassVar[Any] = _CG_SUPPORT_NEVER

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
class _BlockStaging:
    """Staging buffer for a single in-progress codec block.

    One of these exists per (layer, partial-block-cache-index) combo
    until the block seals.  After sealing, the slot bytes are written
    to the paged cache and the staging entry is dropped.

    PLAN.md §Consequence:
        The paged cache has two slot types per layer:
        (1) sealed codec blocks (compressed, full pipeline applied),
        (2) trailing partial block (bf16, < block_size_codec tokens).

    This is the carrier for (2).  The staging buffer is NOT in the
    paged cache allocation — it would blow the 74 KB slot budget by
    ~23×.  It lives on the impl, keyed by the paged-cache block_idx
    the partial block will eventually write to.
    """

    k_bf16: torch.Tensor      # [block_size_codec, n_kv_heads, head_size] bf16
    v_bf16: torch.Tensor      # same
    count: int = 0            # how many rows have been filled


@dataclass
class _PerLayerState:
    """Held on `layer._kakeya_state` once the impl has initialised
    per-layer buffers.

    Multiple requests share a layer but not their partial blocks,
    because two requests rarely write the same cache block.  We
    therefore key the staging dict by *paged-cache block_idx* (which
    vLLM's block_manager guarantees is per-request uniquely allocated)
    rather than by request_id.
    """

    # block_idx → _BlockStaging.  Dropped on seal.
    staging_per_block: dict[int, _BlockStaging] = field(default_factory=dict)

    # Debug shadow: {block_idx: (k_bf16, v_bf16)}.  Populated on
    # seal when KAKEYA_DEBUG_BF16_SHADOW=1; consumed in
    # _decode_sealed to bypass the codec path.  Used to isolate
    # "codec quality" vs "assembly plumbing" PPL regressions.
    bf16_shadow: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    # Layer index inside the model (parsed from `layer.layer_name`).
    # None iff we couldn't parse — in which case calibration is disabled.
    layer_idx: int | None = None

    # Σ_q Cholesky factors for this specific layer's kv-heads, if the
    # calibration bundle covers this layer.  None = no whitening
    # (identity) for this layer.
    sigma_q_chol: torch.Tensor | None = None      # [n_kv, D, D] fp32
    sigma_q_inv_chol: torch.Tensor | None = None  # [n_kv, D, D] fp32

    # Per-stream Lloyd-Max tables (None = Gaussian default).
    k_centroids: np.ndarray | None = None         # [2^b_K] fp32
    v_centroids: np.ndarray | None = None         # [2^b_V] fp32


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

        # Phase B.2a: calibration bundle (shared across all layers).
        # Set by `set_calibration_bundle` or fetched lazily from a
        # process-global default.  Tests can construct an impl with
        # `bundle=None` to exercise the Gaussian-default path.
        self._calibration = _GLOBAL_CALIBRATION

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
        # 20..24 : metric id (u32, 0=mse, 1=inner_product, 2=linf)
        # 24..32 : rotation_seed (u64)
        # 32..48 : reserved
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
        metric_id = {"mse": 0, "inner_product": 1, "linf": 2}[str(parts["metric"])]
        header[20:24] = np.frombuffer(
            np.uint32(metric_id).tobytes(), dtype=np.uint8,
        )
        rotation_seed = int(parts.get("rotation_seed", 3405691582))
        header[24:32] = np.frombuffer(
            np.uint64(rotation_seed).tobytes(), dtype=np.uint8,
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

        metric_id = int(np.frombuffer(bytes(header[20:24]), dtype="<u4")[0])
        metric = {0: "mse", 1: "inner_product", 2: "linf"}.get(metric_id, "mse")
        rotation_seed = int(np.frombuffer(bytes(header[24:32]), dtype="<u8")[0])

        return {
            "mean": mean,
            "basis": basis,
            "centers": centers,
            "d": head_size,
            "d_eff": d_eff,
            "k": k_centers,
            "rotation_seed": rotation_seed,
            "wht_len": config.wht_len,
            "bit_width": bit_width,
            "metric": metric,
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
        """Initialise the per-layer staging dict + calibration.

        Phase B.2a: if a `CalibrationBundle` is registered via
        `set_global_calibration`, pull this layer's Σ_q factors
        (indexed by `layer_idx` parsed from `layer.layer_name`) and
        the per-stream Lloyd-Max centroid tables.  Missing layers
        (e.g. boundary layers skipped during calibration) fall
        through to identity whitening + Gaussian-default Lloyd-Max.
        """
        state = getattr(layer, "_kakeya_state", None)
        if state is not None:
            return state

        layer_idx = _parse_layer_idx_from_name(
            getattr(layer, "layer_name", None)
        )
        state = _PerLayerState(
            staging_per_block={}, layer_idx=layer_idx,
        )

        bundle = self._calibration
        if bundle is not None:
            # Attach per-stream Lloyd-Max tables (None if the bundle
            # didn't include them).
            state.k_centroids = bundle.lloyd_max_k
            state.v_centroids = bundle.lloyd_max_v
            # Σ_q factors are per-layer; look up by parsed idx.
            if layer_idx is not None and layer_idx in bundle.sigma_q_chol:
                L = bundle.sigma_q_chol[layer_idx]
                Linv = bundle.sigma_q_inv_chol[layer_idx]
                if L.shape != (self.num_kv_heads, self.head_size, self.head_size):
                    raise ValueError(
                        f"layer {layer_idx} Σ_q chol shape {L.shape} != "
                        f"({self.num_kv_heads}, {self.head_size}, "
                        f"{self.head_size})"
                    )
                state.sigma_q_chol = torch.from_numpy(L).to(
                    device, torch.float32,
                ).contiguous()
                state.sigma_q_inv_chol = torch.from_numpy(Linv).to(
                    device, torch.float32,
                ).contiguous()

        layer._kakeya_state = state   # type: ignore[attr-defined]
        return state

    def _get_or_create_staging(
        self,
        state: _PerLayerState,
        block_idx: int,
        device: torch.device,
    ) -> _BlockStaging:
        st = state.staging_per_block.get(block_idx)
        if st is not None:
            return st
        D = self.head_size
        st = _BlockStaging(
            k_bf16=torch.zeros(
                (self.block_size_codec, self.num_kv_heads, D),
                dtype=torch.bfloat16, device=device,
            ),
            v_bf16=torch.zeros(
                (self.block_size_codec, self.num_kv_heads, D),
                dtype=torch.bfloat16, device=device,
            ),
            count=0,
        )
        state.staging_per_block[block_idx] = st
        return st

    # ---- GPU-encode helpers (M4 Phase C Path D) -----------------------

    def _get_wht_sign(
        self,
        rotation_seed: int,
        wht_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Cached WHT ±1 sign pattern for this (seed, wht_len, device).

        Rust's `kakeyaturbo::wht::sign_pattern` is deterministic given
        the rotation_seed; we fetch it once via the pyo3 `_core`
        helper and keep the resulting CUDA tensor on the impl so
        repeated seals don't re-derive it.
        """
        cache = getattr(self, "_wht_sign_cache", None)
        if cache is None:
            cache = {}
            self._wht_sign_cache = cache
        key = (int(rotation_seed), int(wht_len), str(device))
        if key not in cache:
            from kakeyaturbo_py import _core
            arr = np.asarray(
                _core.wht_sign_pattern(int(rotation_seed), int(wht_len))
            ).reshape(-1).astype(np.float32)
            cache[key] = torch.from_numpy(arr).to(device)
        return cache[key]

    def _boundary_skip_set(self) -> frozenset[int]:
        """Parsed `KAKEYA_BOUNDARY_SKIP_LAYERS` as an int set, cached
        on the impl so we don't re-parse on every seal.

        Format: comma-separated ints, e.g. `"0,1,34,35"`.  Empty or
        unset → no skip (codec applies to every layer, unchanged
        behaviour).
        """
        cached = getattr(self, "_boundary_skip_cache", None)
        if cached is not None:
            return cached
        raw = os.environ.get("KAKEYA_BOUNDARY_SKIP_LAYERS", "")
        ids: list[int] = []
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                ids.append(int(tok))
            except ValueError:
                # Surface the typo loudly — silently dropping a boundary
                # skip would be a subtle PPL bug at runtime.
                raise RuntimeError(
                    f"KAKEYA_BOUNDARY_SKIP_LAYERS: cannot parse "
                    f"{tok!r} as int (full value: {raw!r})"
                )
        cached = frozenset(ids)
        self._boundary_skip_cache = cached
        return cached

    def _is_boundary_layer(self, layer_idx: int | None) -> bool:
        return layer_idx is not None and layer_idx in self._boundary_skip_set()

    def _seal_block_as_bf16_boundary(
        self,
        st: _BlockStaging,
        block_idx: int,
        kv_cache: torch.Tensor,
        state: _PerLayerState,
    ) -> None:
        """Skip-layer path: the block's K/V are kept as bf16 in the
        per-layer shadow dict `state.bf16_shadow`, keyed by block_idx.
        `_decode_sealed` consults the shadow first and returns the
        uncompressed tensor directly, short-circuiting the codec.

        Two constraints make this the right place to stash the bf16
        (rather than trying to write into `kv_cache`):

        1.  The `kv_cache` byte slot was sized from
            `KakeyaV13PPLConfig.slot_size_bytes` (~37 KB per K-slot
            at d_eff=64/k=16/b=3), which is ~3x smaller than a raw
            bf16 block for the same (block_size_codec, num_kv_heads,
            head_size) geometry (~128 KB).  The paged-cache byte
            tensor has no room.

        2.  vLLM's paged allocator doesn't know about our
            out-of-cache shadow, so the shadow consumes GPU memory
            beyond `gpu_memory_utilization`.  For boundary layers
            (typically 4–6 out of 32–36) at snapshot-sized workloads
            this is a few MB per block × a small blocks-per-request
            count — negligible.  For long production contexts this
            could be tracked in a follow-up if it ever matters.

        The K/V staging tensors are .clone()'d so the shadow owns
        its own storage; without the clone, releasing `st` (the
        caller drops it after seal) would free the shadow's memory.
        """
        state.bf16_shadow[block_idx] = (
            st.k_bf16.clone(), st.v_bf16.clone(),
        )
        # Write zeros into the paged-cache slot so that a downstream
        # consumer that accidentally reads uncalibrated bytes gets a
        # deterministic value rather than undefined memory.  The
        # forward path always checks the shadow first for skipped
        # layers, so these zeros are never actually decoded.
        v_end = self.k_config.slot_size_bytes + self.v_config.slot_size_bytes
        kv_cache[block_idx, :, :v_end].zero_()

    def _config_offsets(self, cfg: "KakeyaV13PPLConfig") -> dict:
        """Extract the byte-layout properties that
        `gpu_encode.encode_and_pack_batched` needs into a plain dict
        (so that hot path code doesn't hit property getters inside
        the seal critical section)."""
        return {
            "header_bytes": cfg.header_bytes,
            "offset_pca_basis": cfg.offset_pca_basis,
            "pca_basis_bytes": cfg.pca_basis_bytes,
            "offset_pca_mean": cfg.offset_pca_mean,
            "pca_mean_bytes": cfg.pca_mean_bytes,
            "offset_kmeans_centroids": cfg.offset_kmeans_centroids,
            "kmeans_centroids_bytes": cfg.kmeans_centroids_bytes,
            "offset_seg_id_block": cfg.offset_seg_id_block,
            "seg_id_bits_per_vec": cfg.seg_id_bits_per_vec,
            "seg_id_bytes_per_block": cfg.seg_id_bytes_per_block,
            "offset_t_block": cfg.offset_t_block,
            "t_bytes_per_block": cfg.t_bytes_per_block,
            "offset_norm_block": cfg.offset_norm_block,
            "norm_bytes_per_block": cfg.norm_bytes_per_block,
            "offset_residual_block": cfg.offset_residual_block,
            "residual_bytes_per_block": cfg.residual_bytes_per_block,
            "offset_outlier_side_buffer": cfg.offset_outlier_side_buffer,
            "outlier_budget_bytes": cfg.outlier_budget_bytes,
            "outlier_row_count_bytes": cfg.outlier_row_count_bytes,
            "outlier_entry_bytes_budget": cfg.outlier_entry_bytes_budget,
            "block_size_codec": cfg.block_size_codec,
            "wht_len": cfg.wht_len,
        }

    # ---- Phase B.1: encode-seal + write + read paths ------------------

    def _seal_and_write_block(
        self,
        st: _BlockStaging,
        block_idx: int,
        kv_cache: torch.Tensor,
        layer: AttentionLayer,
    ) -> None:
        """Call the M4 Triton encoder on a full `_BlockStaging` (K and
        V separately) and write the two slots into the per-layer
        paged cache.

        `kv_cache` has shape `[num_blocks, num_kv_heads, slot_budget_bytes]`
        per `KakeyaV13PPLAttentionSpec.get_kv_cache_shape`; the K-slot
        starts at byte 0 and the V-slot at `k_config.slot_size_bytes`
        within each `kv_cache[block_idx, head, :]`.

        Phase B.2a: if `_PerLayerState.sigma_q_chol` is set, whiten
        K per-kv-head before encode (`K_tilde = K @ L[h]`).  Decode
        will later unwhiten.  Σ_q whitening is K-stream only — V
        doesn't have a Q-preconditioning analogue.
        """
        # M4 Phase C (Path D): fully GPU-resident encode pipeline.
        # Replaces the old per-head serial CPU loop
        #   for h: encode_block_codes(Xh.cpu()) + triton_stage2 + numpy pack
        # with a single batched RSVD + Triton-match argmin + GPU bit-pack
        # + scatter into a [H, slot_bytes] uint8 tensor — O(1000x)
        # faster on H200 and aligned with PLAN.md §"Key design decision"
        # (randomised PCA, matching kakeyaturbo::pca::fit_weighted_pca_randomized).
        from kakeyaturbo_py.gpu_skeleton import fit_skeleton_batched
        from kakeyaturbo_py.gpu_encode import encode_and_pack_batched
        from kakeyaturbo_py import _core as _kt_core
        n = self.block_size_codec
        assert st.count == n, f"seal called on partial block ({st.count}/{n})"

        state: _PerLayerState = layer._kakeya_state  # type: ignore[attr-defined]

        # ---- Guardrail #3: boundary-layer skip (PR #17 recipe) ----
        # PR #17's production cell for DeepSeek-R1-Distill-Qwen-1.5B used
        # a 6-layer boundary skip: [0, 1, 7, 14, 26, 27] passed through
        # as bf16 (no codec) because those layers' attention patterns
        # are disproportionately sensitive to codec reconstruction error.
        # FINDINGS.md §"V-only + outlier" measured this configuration
        # as the closest-to-MARGINAL verdict (+7.04% Δppl, 75.39% top-1).
        #
        # Environment-var driven so one backend binary can serve
        # different models / recipes:
        #
        #   KAKEYA_BOUNDARY_SKIP_LAYERS="0,1,7,14,26,27"    # PR #17 on DS-1.5B
        #   KAKEYA_BOUNDARY_SKIP_LAYERS="0,1,34,35"         # Qwen3-4B boundary
        #   KAKEYA_BOUNDARY_SKIP_LAYERS=""                  # no skip (default)
        #
        # NOTE: distinct from `KAKEYA_SKIP_LAYERS` (plugin.py) which
        # controls which layers the Σ_q calibration bundle covers —
        # those two concepts happen to coincide in the DeepSeek recipe
        # but are semantically independent.
        if self._is_boundary_layer(state.layer_idx):
            self._seal_block_as_bf16_boundary(
                st, block_idx, kv_cache, state,
            )
            return
        k_cfg = self.k_config
        v_cfg = self.v_config
        K_fp32 = st.k_bf16.to(torch.float32)              # [n, n_kv, D]
        V_fp32 = st.v_bf16.to(torch.float32)

        # Optional Σ_q whitening on K (per-kv-head).
        #
        # NOTE: The Σ_q bundle in M2 was calibrated on *pre-RoPE*
        # queries, whereas vLLM's attention backend sees *post-RoPE*
        # K activations (the Q/K rotation happens before `Attention`
        # calls into the backend).  Applying pre-RoPE-fitted Σ_q to
        # post-RoPE K is a semantic mismatch: the reconstruction
        # error ε in the whitened space gets amplified to ε · L^{-1}
        # in the decode path, and for ill-conditioned heads the
        # condition number of L^{-1} can be >5×, blowing PPL.
        #
        # Env toggle: set KAKEYA_DISABLE_SIGMA_Q=1 to bypass.
        # DEFAULT: ON for disable (= OFF for whitening) — the M2 Σ_q
        # bundle was calibrated on pre-RoPE Q but vLLM's attention
        # backend sees post-RoPE K; applying pre-RoPE-fitted Σ_q to
        # post-RoPE K introduces a semantic mismatch that gets
        # amplified by L⁻¹ in the decode path.  CPU reference
        # (benchmarks/e2e_ppl_validation_vllm.py) does NOT use Σ_q
        # whitening at all — it's a separate PLAN.md Option.
        disable_sigma_q = os.environ.get("KAKEYA_DISABLE_SIGMA_Q", "1") == "1"
        if state.sigma_q_chol is not None and not disable_sigma_q:
            K_fp32 = torch.einsum(
                "thj,hjk->thk",
                K_fp32,
                state.sigma_q_chol,
            ).contiguous()

        # Per-stream Lloyd-Max tables — CPU reference uses Gaussian
        # default (`_core.centroids_gaussian(b)`); M2 empirical tables
        # were fit on pre-RoPE data, same mismatch as Σ_q.  Env
        # toggle: set KAKEYA_USE_M2_CENTROIDS=1 to opt in.
        use_m2_centroids = os.environ.get("KAKEYA_USE_M2_CENTROIDS", "0") == "1"
        k_centroids = state.k_centroids if use_m2_centroids else None
        v_centroids = state.v_centroids if use_m2_centroids else None

        # K-stream outlier threshold: CPU reference bench CLI does not
        # set --outlier-threshold, so the side-buffer path is inactive
        # in v1.3 PPL-ref.  Mirror that.  Env toggle:
        # KAKEYA_OUTLIER_THRESHOLD=<float> to enable.
        _olt_env = os.environ.get("KAKEYA_OUTLIER_THRESHOLD", "")
        k_outlier_threshold = float(_olt_env) if _olt_env else None

        def _pad_to_d_eff(parts: dict, target_d_eff: int, head_size: int) -> dict:
            """Pad PCA basis / per-vec coeffs / K-means centers / seg_id /
            norm to a fixed `target_d_eff`.

            Rationale: the Rust codec returns a data-dependent `d_eff`
            (the intrinsic PCA rank of the block after variance-ratio +
            exact_rank_cap).  For vLLM slot sizing, we need a
            *configuration-constant* `d_eff` so the slot layout is
            deterministic.  On rank-deficient blocks (e.g. repetitive
            text), d_eff can come back smaller than the configured
            value; we pad the basis with zero-eigenvectors and the
            K-means centroids with zero-rows so that the round-trip is
            still exact (projecting onto a zero basis row adds 0 to
            the reconstruction).
            """
            cur = int(parts["d_eff"])
            if cur >= target_d_eff:
                return parts
            padded = dict(parts)
            pad = target_d_eff - cur
            # basis: (d_eff, head_size)
            b = np.asarray(parts["basis"]).reshape(cur, head_size)
            padded["basis"] = np.concatenate(
                [b, np.zeros((pad, head_size), dtype=b.dtype)], axis=0,
            )
            # K-means centers: (k, d_eff)
            c = np.asarray(parts["centers"]).reshape(-1, cur)
            padded["centers"] = np.concatenate(
                [c, np.zeros((c.shape[0], pad), dtype=c.dtype)], axis=1,
            )
            padded["d_eff"] = target_d_eff
            return padded

        # Stage 1 (batched, GPU): fit PCA + K-means skeleton for K
        # and V of all kv-heads in a single torch.linalg.eigh call
        # (batch = 2·num_kv_heads) — both streams share d_eff and
        # k_centers per the KakeyaV13PPLConfig invariant, so they
        # compose trivially along the batch axis.  The stream split
        # happens when we slice `KV_batched` back apart.
        KV_batched = torch.cat(
            [K_fp32.permute(1, 0, 2), V_fp32.permute(1, 0, 2)],
            dim=0,
        ).contiguous()                                           # [2·H, n, D]
        assert k_cfg.d_eff == v_cfg.d_eff
        assert k_cfg.k_centers == v_cfg.k_centers

        kv_skel_batch = fit_skeleton_batched(
            KV_batched,
            d_eff=k_cfg.d_eff,
            k=k_cfg.k_centers,
            seed=3405691582,
        )

        # Split back into K and V skeletons.  kv_skel_batch holds the
        # per-batch tensors with K rows first (indices 0..H) and V
        # rows second (H..2H).
        def _slice_skel(batch_dict, idx_lo, idx_hi):
            return {
                **{k_: v_[idx_lo:idx_hi] if hasattr(v_, "shape") else v_
                   for k_, v_ in batch_dict.items()},
            }
        H = self.num_kv_heads
        k_skel_batch = _slice_skel(kv_skel_batch, 0, H)
        v_skel_batch = _slice_skel(kv_skel_batch, H, 2 * H)

        # Build WHT sign patterns once per (seed, wht_len) — cached
        # on the impl since it's the same across all layers / blocks.
        #
        # The K and V streams share the same wht_len (both have
        # d_eff = k_cfg.d_eff = v_cfg.d_eff), so one sign tensor
        # suffices.
        k_sign = self._get_wht_sign(
            int(k_skel_batch["rotation_seed"]),
            int(k_skel_batch["wht_len"]),
            st.k_bf16.device,
        )
        v_sign = self._get_wht_sign(
            int(v_skel_batch["rotation_seed"]),
            int(v_skel_batch["wht_len"]),
            st.v_bf16.device,
        )

        # Config-offset dicts — extract the byte-layout properties
        # once; gpu_encode.encode_and_pack_batched is pure (doesn't
        # hold a ref to the config object).
        k_offsets = self._config_offsets(k_cfg)
        v_offsets = self._config_offsets(v_cfg)

        # Move custom Lloyd-Max centroid tables to CUDA once.
        k_centroids_gpu = (
            torch.as_tensor(k_centroids, device=st.k_bf16.device,
                            dtype=torch.float32)
            if k_centroids is not None else None
        )
        v_centroids_gpu = (
            torch.as_tensor(v_centroids, device=st.v_bf16.device,
                            dtype=torch.float32)
            if v_centroids is not None else None
        )

        # Batched encode + pack for K (all kv_heads at once).
        # K_fp32 is [n, H, D] → permute to [H, n, D] for the batch axis.
        K_batch = K_fp32.permute(1, 0, 2).contiguous()   # [H, n, D]
        V_batch = V_fp32.permute(1, 0, 2).contiguous()

        k_slot_tensor = encode_and_pack_batched(      # [H, k_slot_bytes] uint8
            K_batch, k_skel_batch,
            bit_width=k_cfg.bit_width,
            metric="inner_product",
            slot_size_bytes=k_cfg.slot_size_bytes,
            config_offsets=k_offsets,
            custom_centroids=k_centroids_gpu,
            outlier_threshold=k_outlier_threshold,
            wht_sign=k_sign,
        )
        v_slot_tensor = encode_and_pack_batched(      # [H, v_slot_bytes] uint8
            V_batch, v_skel_batch,
            bit_width=v_cfg.bit_width,
            metric="mse",
            slot_size_bytes=v_cfg.slot_size_bytes,
            config_offsets=v_offsets,
            custom_centroids=v_centroids_gpu,
            outlier_threshold=None,
            wht_sign=v_sign,
        )

        # Concatenate K and V along the byte axis → single [H, total]
        # memcpy into kv_cache[block_idx, :, :].
        combined = torch.cat([k_slot_tensor, v_slot_tensor], dim=1)
        k_end = k_cfg.slot_size_bytes
        v_end = k_end + v_cfg.slot_size_bytes
        assert combined.shape[1] == v_end, (
            f"combined slot {combined.shape[1]} != K+V bytes {v_end}"
        )
        kv_cache[block_idx, :, :v_end].copy_(combined)

        # Debug: stash a bf16 shadow so _decode_sealed can bypass
        # the codec path and verify the rest of the assembly pipe.
        if os.environ.get("KAKEYA_DEBUG_BF16_SHADOW", "0") == "1":
            state.bf16_shadow[block_idx] = (
                st.k_bf16.clone(), st.v_bf16.clone()
            )

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Stage-1: append to staging → seal full blocks → write.

        PLAN.md §Consequence: the paged cache has sealed codec blocks
        (compressed) and a trailing partial block (bf16 staging).
        `slot_mapping[i]` is the absolute slot index in the paged
        cache where token `i` lands; we compute
        `block_idx = slot // block_size_codec` and
        `pos_in_block = slot % block_size_codec`.

        For each token (in order):
          1. Look up `(block_idx, pos_in_block)`.
          2. Write `key[i]` / `value[i]` into `staging[block_idx]
             .{k,v}_bf16[pos_in_block]` and bump that block's count.
          3. If the count hits `block_size_codec`, seal: encode via
             Triton, pack, write to `kv_cache[block_idx]`, drop the
             staging entry.
        """
        N = int(slot_mapping.shape[0])
        if N <= 0:
            return
        device = key.device
        state = self._ensure_layer_state(layer, device)

        # Reshape to (N, n_kv, D) even if the caller passed a 2-D view.
        k_view = key[:N].view(N, self.num_kv_heads, self.head_size)
        v_view = value[:N].view(N, self.num_kv_heads, self.head_size)

        # CPU-side slot → (block_idx, pos) conversion, then group tokens
        # by block so we do at most one encode per block per step.
        # Calling .tolist() moves once per update; at prefill this is
        # O(num_prefill_tokens).
        slot_cpu = slot_mapping[:N].detach().cpu().tolist()

        # Gather writes per block_idx, then process in sorted order so
        # any block that seals is sealed in a deterministic order.
        per_block: dict[int, list[tuple[int, int]]] = {}
        for i, slot in enumerate(slot_cpu):
            if slot < 0:
                continue
            blk = slot // self.block_size_codec
            pos = slot % self.block_size_codec
            per_block.setdefault(blk, []).append((i, pos))

        for blk in sorted(per_block.keys()):
            entries = per_block[blk]
            st = self._get_or_create_staging(state, blk, device)
            for i, pos in entries:
                st.k_bf16[pos] = k_view[i]
                st.v_bf16[pos] = v_view[i]
                # `count` tracks the high-water mark of filled rows so
                # we know when to seal; rewriting an existing pos is
                # idempotent since vLLM guarantees per-slot write-once.
                if pos + 1 > st.count:
                    st.count = pos + 1
            # Seal iff the block is full.  vLLM's block manager does
            # not emit more writes for a sealed block (the block table
            # advances), so seal-once-and-drop is safe.
            if st.count == self.block_size_codec:
                self._seal_and_write_block(st, blk, kv_cache, layer)
                state.staging_per_block.pop(blk, None)

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
        """Stage-2: assemble K/V from sealed + partial blocks, then
        attend.

        For each request:
          1. For every sealed block in the block_table (up to but not
             including the trailing partial block), unpack the slot
             via `_unpack_slot_into_parts` and decode K and V via
             `kakeyaturbo_py.decode_block_triton_from_parts`.
          2. For the trailing partial block (if any tokens staged for
             this block and not yet sealed), read directly from
             `state.staging_per_block[block_idx].{k,v}_bf16[:count]`
             via `decode_partial_block_bf16`.
          3. Concatenate sealed + partial K/V along seq-dim.
          4. Feed Q (from the forward args) and assembled K/V into
             `flash_attn_varlen_func`.
        """
        from kakeyaturbo_py import (
            decode_block_triton_from_parts,
            decode_partial_block_bf16,
        )
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        if output is None:
            output = torch.zeros(
                query.shape[0], self.num_heads * self.head_size,
                dtype=query.dtype, device=query.device,
            )
        if attn_metadata is None:
            return output.fill_(0)
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        device = query.device
        state = self._ensure_layer_state(layer, device)

        num_reqs = attn_metadata.seq_lens.shape[0]
        seq_lens_cpu = attn_metadata.seq_lens[:num_reqs].detach().cpu().tolist()
        block_table_cpu = attn_metadata.block_table.detach().cpu().tolist()
        query_start_loc_cpu = attn_metadata.query_start_loc.detach().cpu().tolist()

        # Per-request assembled K/V, then concat + varlen flash-attn.
        assembled_k: list[torch.Tensor] = []
        assembled_v: list[torch.Tensor] = []
        cu_seqlens_k = [0]

        for req in range(num_reqs):
            seq_len = int(seq_lens_cpu[req])
            if seq_len <= 0:
                cu_seqlens_k.append(cu_seqlens_k[-1])
                continue
            blocks_needed = (seq_len + self.block_size_codec - 1) // self.block_size_codec
            block_ids = [int(block_table_cpu[req][b]) for b in range(blocks_needed)]
            full_blocks = seq_len // self.block_size_codec
            tail = seq_len - full_blocks * self.block_size_codec

            req_k_chunks: list[torch.Tensor] = []
            req_v_chunks: list[torch.Tensor] = []

            for bi, block_idx in enumerate(block_ids):
                if bi < full_blocks:
                    # Sealed block.
                    k_block = self._decode_sealed(
                        kv_cache, block_idx, stream="K", device=device,
                        layer=layer,
                    )
                    v_block = self._decode_sealed(
                        kv_cache, block_idx, stream="V", device=device,
                        layer=layer,
                    )
                    req_k_chunks.append(k_block)
                    req_v_chunks.append(v_block)
                else:
                    # Trailing partial block.
                    st = state.staging_per_block.get(block_idx)
                    if st is None or tail == 0:
                        continue
                    k_partial = decode_partial_block_bf16(st.k_bf16[:tail])
                    v_partial = decode_partial_block_bf16(st.v_bf16[:tail])
                    # [tail, n_kv, D]  → [tail, n_kv, D]
                    req_k_chunks.append(k_partial)
                    req_v_chunks.append(v_partial)

            if not req_k_chunks:
                cu_seqlens_k.append(cu_seqlens_k[-1])
                continue
            k_req = torch.cat(req_k_chunks, dim=0)   # [seq_len, n_kv, D]
            v_req = torch.cat(req_v_chunks, dim=0)
            assembled_k.append(k_req)
            assembled_v.append(v_req)
            cu_seqlens_k.append(cu_seqlens_k[-1] + int(k_req.shape[0]))

        if not assembled_k:
            return output.fill_(0)

        K_total = torch.cat(assembled_k, dim=0)      # [sum_seq, n_kv, D] fp32
        V_total = torch.cat(assembled_v, dim=0)
        # Match Q's dtype for flash_attn.
        K_total = K_total.to(query.dtype)
        V_total = V_total.to(query.dtype)

        cu_seqlens_k_t = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, device=device,
        )

        q = query[:N].view(N, self.num_heads, self.head_size)
        out_bhd = flash_attn_varlen_func(
            q=q,
            k=K_total,
            v=V_total,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=cu_seqlens_k_t,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
        )
        # out_bhd: [N, n_heads, D]
        if output.ndim == 3:
            output[:N] = out_bhd.to(output.dtype)
        else:
            output[:N] = out_bhd.reshape(N, -1).to(output.dtype)
        return output

    def _decode_sealed(
        self,
        kv_cache: torch.Tensor,
        block_idx: int,
        stream: str,
        device: torch.device,
        *,
        layer: AttentionLayer | None = None,
    ) -> torch.Tensor:
        """Decode one sealed (K or V) codec block to `[block_size_codec,
        n_kv_heads, head_size]` fp32.

        Iterates over kv_heads since each head has its own
        skeleton + codes in our slot layout.

        Phase B.2a: if `layer._kakeya_state` carries calibrated
        centroids, pass them as `custom_centroids` to the Triton
        decoder.  If it carries a `sigma_q_inv_chol` tensor and this
        is the K stream, un-whiten the decoded K per-head
        (`K_hat = K_hat_tilde @ L^{-1}`).
        """
        from kakeyaturbo_py import decode_block_triton_from_parts
        cfg = self.k_config if stream == "K" else self.v_config
        slot_off = 0 if stream == "K" else self.k_config.slot_size_bytes

        # ---- Guardrail #3 read side: boundary-layer bf16 shadow ----
        # If _seal_and_write_block detected this layer was on the
        # `KAKEYA_BOUNDARY_SKIP_LAYERS` list, the codec was bypassed
        # at seal time and the bf16 K/V are in state.bf16_shadow.
        # Return them directly — the paged-cache slot for this block
        # is zeros (kept deterministic but never decoded).
        #
        # Also used by the older `KAKEYA_DEBUG_BF16_SHADOW=1` probe
        # that forced the shadow on every block regardless of layer
        # id, so the check is merged.
        state: _PerLayerState | None = (
            getattr(layer, "_kakeya_state", None) if layer is not None else None
        )
        boundary = (
            state is not None and self._is_boundary_layer(state.layer_idx)
        )
        debug_shadow = (
            os.environ.get("KAKEYA_DEBUG_BF16_SHADOW", "0") == "1"
        )
        if (boundary or debug_shadow) and state is not None:
            entry = state.bf16_shadow.get(block_idx)
            if entry is not None:
                k_shadow, v_shadow = entry
                sh = k_shadow if stream == "K" else v_shadow
                return sh.to(device=device, dtype=torch.float32)
            elif boundary:
                # This shouldn't happen — if we marked the layer as
                # boundary at seal time the shadow must be populated.
                # Fail loud to surface the mismatch at test time.
                raise RuntimeError(
                    f"boundary layer {state.layer_idx} block {block_idx} "
                    f"{stream}-stream missing from bf16_shadow — seal "
                    f"and decode saw inconsistent skip sets?"
                )

        # Pull calibration from layer state (if available).  We must
        # use the SAME centroid table that _seal_and_write_block used
        # when encoding this block, or the Lloyd-Max codebook
        # mismatches and every quantised coordinate decodes wrong —
        # catastrophic PPL blowup.  The encode-side opt-in flag is
        # KAKEYA_USE_M2_CENTROIDS (default off matches CPU reference
        # which uses the Gaussian default).
        # `state` already bound above for the boundary-skip check.
        use_m2_centroids = os.environ.get("KAKEYA_USE_M2_CENTROIDS", "0") == "1"
        custom_centroids = None
        if state is not None and use_m2_centroids:
            custom_centroids = (
                state.k_centroids if stream == "K" else state.v_centroids
            )

        rows = []
        for h in range(self.num_kv_heads):
            slot_bytes = kv_cache[block_idx, h, slot_off:slot_off + cfg.slot_size_bytes]
            slot_np = slot_bytes.detach().cpu().numpy()
            parts = self._unpack_slot_into_parts(slot_np, cfg, head_size=self.head_size)
            decoded = decode_block_triton_from_parts(
                parts, custom_centroids=custom_centroids, device=str(device),
            )
            rows.append(torch.from_numpy(decoded).to(device))
        # Stack to [block_size_codec, n_kv_heads, head_size].
        out = torch.stack(rows, dim=1)

        # Σ_q un-whitening (K only; V-stream has no whitening).
        # Must match the toggle in `_seal_and_write_block`.
        # Default: disabled (matches CPU reference; see the paired
        # comment in _seal_and_write_block).
        disable_sigma_q = os.environ.get("KAKEYA_DISABLE_SIGMA_Q", "1") == "1"
        if (stream == "K"
            and state is not None
            and state.sigma_q_inv_chol is not None
            and not disable_sigma_q):
            out = torch.einsum(
                "thj,hjk->thk",
                out,
                state.sigma_q_inv_chol,
            ).contiguous()

        return out

