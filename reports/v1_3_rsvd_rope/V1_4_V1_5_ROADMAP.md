# v1.4 / v1.5 Sprint Roadmap — Attention-aware KV + Session Memory

**Status**: planning document, no code yet. This is the agreed-upon
scope for the next sprint chain after v1.3 shipped in PR #11.

**Hard scope boundary** (do not re-open without new evidence):

| Memory layer | In scope | Rationale |
|---|---|---|
| **L0** KV cache | ✅ yes — attention-weighted + sink preservation | codec problem domain; v1.3 `weights` interface already reserved |
| **L1** Session memory | ✅ yes — recall-aware v1.3 + per-block retention | agent workloads have 10× capacity ask; same codec family |
| L2 Agent long-term memory | ❌ no | use off-the-shelf PQ / ScaNN; not a codec problem |
| L3 RAG document index | ❌ no | Faiss / Milvus / ScaNN own this Pareto frontier |
| L4 Tool output cache | ❌ no | heterogeneous blobs, different codec per object type |

## Three hard invariants (every phase must respect)

1. **Ship discipline aligned with v1.3**:
   MSE inflation ≤ 10% → ACCEPT; real-data ablation on the same
   7-model corpus at ctx=4096 before ship; no mock / no fallback.
2. **Shadow mode before rollout**: every new dynamic attention signal
   must be runnable side-by-side with its static equivalent to produce
   identical outputs when disabled.
3. **L0 prefill overhead ≤ 5%**: any attention-signal extraction must
   stay on-device (GPU-native) and be bounded to that ceiling.

## Delivery order (ROI × risk)

| Phase | Content | Risk | Exit criteria |
|---|---|---|---|
| **v1.4.1** | L0 attention sink preservation (first N tokens + per-layer top-k exact) | low | LongBench-E / RULER no regression; byte overhead ≤ 5% |
| **v1.4.0** | L0 attention-weighted `weights: &[f32]` wired to prefill scores | medium | 7-model ablation: 6 ACCEPT + PPL +≤1% + prefill +≤5% latency |
| **v1.5.0** | L1 session memory codec + recall-by-block-id | high (new subsystem) | 20-turn session test: final-token-match ≥ 95%; RAM saving ≥ 4× |
| **v1.5.1** | L1 semantic recall (per-block embeddings + top-k lookup) | high (retrieval quality) | head-to-head vs RAG baseline: recall@10 ≥ 90%, storage < RAG × 0.7 |
| **v1.6?** | entropy-adaptive bit / cross-head codec sharing | speculative | ablation first; only proceed if gain ≥ 5% on 5+ models |

**Sequencing rules**:
- v1.4.1 ships before v1.4.0 (lower risk, faster feedback).
- v1.4.0 and v1.4.1 may be developed in parallel PRs but ship independently.
- v1.5.0 strictly blocks on v1.4.0 (L1 priority needs the L0 attention signal).
- v1.5.1 strictly blocks on v1.5.0.

---

## Phase v1.4.1 — L0 attention sink preservation

**Hypothesis**: front-N tokens (and, optionally, per-layer static
heavy-hitters) carry "attention sink" load — compressing them
degrades long-generation stability. Preserving them exact costs <1%
bytes but stabilises streaming behaviour.

### Interface addition
```rust
pub struct CodecParams {
    // ... existing fields ...
    /// Number of leading tokens per layer kept at bf16 (0 = off).
    /// Streaming-LLM default: 4. Higher for models with more diffuse
    /// attention-sink behaviour.
    pub sink_tokens: usize,
}
```

### Test matrix
- LongBench-E at ctx ∈ {8k, 32k} on all 7 open-source models.
- RULER multi-needle recall at ctx=32k.
- Byte-overhead measurement: (sink_tokens × L × n_kv × head_dim × 2) / total_compressed.

### Exit criteria
- 6/7 models: LongBench-E delta ≤ +0.5 PPL vs v1.3 baseline.
- RULER recall ≥ 95% (bf16 baseline: 97%).
- Byte overhead ≤ 5% on the smallest-ctx config (4k).

### Failure modes and rollback
- **If N=4 is insufficient**: try N=8; if still insufficient,
  implement per-layer heavy-hitter detection (defer to v1.4.2).
- **If byte overhead > 5%**: models with small head_dim (64) may need
  fewer sinks; ship with `sink_tokens = min(4, ctx/256)`.

---

## Phase v1.4.0 — L0 attention-weighted codec

**Hypothesis**: prefill attention scores, used as per-token weights in
the weighted PCA, let the codec allocate basis capacity to
heavy-hitter tokens. H2O / Scissorhands literature shows 10–50% of
tokens carry >90% attention mass.

### Signal source (choice to commit)

| Option | Formula | Pros | Cons | Default |
|---|---|---|---|---|
| **A** cumulative full-prefill | `w[i] = Σ_j Σ_head softmax(q_j·k_i)[i]` | H2O-faithful; kernel-friendly | prefill≠decode distribution | v1.4.0 default |
| B last-N-query | same sum but `j ∈ last N queries` | closer to decode proxy | new hparam N | per-request override |
| C temporal-decay | `Σ_j decay(j) · softmax(q_j·k_i)` | convex combo of A+B | two hparams; defer | v1.4.2 candidate |

### Kernel strategy

| Approach | HBM | Latency | Implementation | Verdict |
|---|---|---|---|---|
| hook to store full attn_weights | O(L·H·N²) | +15–30% | trivial | **do not** |
| custom kernel column-sum | O(L·H·N) | +3–5% | medium | **v1.4.0 default** |
| patch into flash-attention numerator | O(L·H·N) | +2% | high (fork flash-attn) | **v1.4.2 upgrade** |

The attention tensor must stay on device; pipe into the Rust codec's
GPU-side shim, not back to CPU. CPU round-trip is an instant
disqualification.

### Test matrix (must run before ship)

Against v1.3 tier-1 at ctx=4096 on all 7 models:

| Metric | Baseline | v1.4.0 | Acceptance |
|---|---|---|---|
| byte ratio | v1.3 tier-1 measured | new | 6/7 ≥ v1.3 × 1.05 |
| K MSE inflation | v1.3 K MSE | new | 7/7 ≤ v1.3 × 1.05 |
| V MSE inflation | v1.3 V MSE | new | 6/7 ≤ v1.3 × 1.10 |
| PPL (WikiText-103) | bf16 baseline | new | 7/7 ≤ baseline × 1.01 |
| needle-in-haystack | bf16 baseline | new | 6/7 ≥ 95% of baseline accuracy |
| prefill latency | v1.3 tier-1 | new | ≤ +5% median on H100 |

Any single criterion missed → do not ship; fall back to a
`per_request_attention_weighted: bool` opt-in flag with default off.

### Failure-mode playbook

| Failure | Cause | Rollback |
|---|---|---|
| PPL regression > 1% on 3+ models | Option A's distribution drift | Switch to Option C (temporal decay); if still fails, disable by default |
| Spiky weights (>90% mass on <5% tokens) | attention sink un-handled | Bump v1.4.1 to P0 |
| Prefill latency > +5% | kernel not optimised | Degrade to "sample 1-in-128 tokens" (sampled H2O) |
| Needle-in-haystack accuracy drop > 3% | compressed needle tokens | Reserve 2–5% "never-compressed" per-block budget |

---

## Phase v1.5.0 — L1 session memory codec (MVP)

**Hypothesis**: agent workloads accumulate 3–30 GB of session state
per user per day, most of which is dormant. A local-only,
session-scoped codec with more aggressive parameters than L0 yields
10–20× compression at ≥ 95% recall quality when blocks are needed back.

### Scope lockdown (write these as MUST NOT in the crate doc)

- ✅ single-node, single-user, single-session, local disk
- ✅ block-level retention and eviction
- ✅ breaking format changes allowed (session TTL ≤ 7 days)
- ❌ no cross-session search
- ❌ no multi-user memory sharing
- ❌ no object-storage tier
- ❌ no long-term persistence
- ❌ no backward compatibility guarantee across versions

### Minimal interface

```rust
pub trait SessionMemoryStore {
    /// Archive a fully-encoded block; returns opaque id.
    fn archive_block(
        &mut self,
        session_id: Uuid,
        kv_vectors: &[f32],
        dim: usize,
        priority: f32,
    ) -> BlockId;

    /// Recall a block by id, returning its decoded bf16 form.
    fn retrieve_block(&self, id: BlockId) -> Option<Vec<f32>>;

    /// Evict all but the top-N priority blocks in a session.
    fn evict_by_priority(&mut self, session_id: Uuid, keep_top_n: usize);

    /// Total on-disk bytes used by a session.
    fn session_size_bytes(&self, session_id: Uuid) -> usize;
}
```

### Codec parameter delta vs L0

| Parameter | L0 (v1.4) | **L1 (v1.5.0)** | Reason |
|---|---|---|---|
| `bit_width` | 2 | **2** | keep fidelity for recall |
| `pca_method` / `target_rank` | Randomized, r=D/2 | **Randomized, r=D/3** | more aggressive; L1 allows more loss |
| `variance_ratio` | 0.95 | **0.90** | same |
| `block_size` | 512 | **1024** | not bound to attention block, amortise skeleton |
| `share_basis` (V) | true | **true** | preserve v1.2 B' |

**Predicted L1 ratio**: 1.5–2.0× over L0 (33% from target_rank cut +
15% from block_size doubling). On v1.3's 6× L0 this gives L1 codec
ratio of **9–12×** vs bf16.

### Retention priority

```
priority[block] = α · max_attention_ever_received(block)
                + β · recency(block)
                + γ · user_explicit_flag(block)
```

v1.5.0 defaults: α = 1.0, β = 0.3, γ = +∞ (user-flagged blocks are
never evicted). Rationale: recency is partly subsumed by semantic
retrieval (v1.5.1); explicit user flag overrides everything.

### Data boundary between L0 and L1

At session end, **do not** copy L0 codes directly to L1. Instead:
1. Decode L0 codes to bf16 KV tokens (already part of DynamicCache).
2. L1 encoder re-encodes those tokens with **L1 codec params**.
3. L1 stores with `session_id` tag.

Justification: L0 codec's PCA basis is block_size=512-tuned; L1
block_size=1024 requires a fresh fit. Re-encoding is a one-off CPU
cost that runs async after the session ends.

### Session-end consolidation job

```
on_session_end(session_id):
    for block in L0.blocks(session_id):
        bf16 = L0.decode(block)
        priority = aggregate_priority_from_logs(block)
        L1.archive_block(session_id, bf16, dim, priority)
    L0.evict(session_id)
    # optional: schedule L1 eviction if session_size_bytes > user_budget
```

Latency budget: ≤ 30 seconds per 10k-token session, async
(non-blocking to the next user turn).

### Recall-time re-injection

When a session's L1 block is recalled and spliced into current prefill:
- **position encoding** must be re-applied: the block's original RoPE
  rotation is stripped (if applicable) during recall, and the new
  position in the current prompt re-applies RoPE.
- **relevance gate**: only recall blocks whose retention priority >
  threshold, to avoid cluttering the prefill with dormant context.

### Test matrix

| Scenario | Baseline | v1.5.0 | Acceptance |
|---|---|---|---|
| 20-turn multi-turn with back-reference | full bf16 session | L1 compressed | final-token exact-match ≥ 95% |
| 100-turn agent trace, recall from 20 turns ago | full bf16 session | L1 compressed | recall@10 of reference block ≥ 90% |
| RAM saving | full bf16 | L1 | ≥ 4× total session-memory RAM reduction |
| archive latency | — | — | ≤ 30 s / 10k tokens |
| recall latency | — | — | ≤ 50 ms / block on local SSD |

### Risk: recalled-K distribution after re-injection

When L1 blocks are re-injected mid-prefill, the current attention
computes `q_new · k_recalled`. If recall quality is too low, the
attention distribution for this forward pass skews and downstream
generation drifts.

Mitigation:
1. **Recall on demand** via a small embedding-similarity classifier
   (`sim(query_emb, block_emb) > τ`); avoids always-on recall.
2. **Per-layer relevance check** at attention time; skip layers where
   recalled block would not be the top-k K attended to.
3. Start with τ tuned on dev set; expose as a runtime knob.

---

## Phase v1.5.1 — L1 semantic recall

**Hypothesis**: per-block 256-dim embeddings (stored alongside the
compressed block, ~512 B overhead per block) enable top-k semantic
retrieval without decompression. This extends L1 from
recall-by-block-id to recall-by-semantic-query.

### Interface addition

```rust
pub trait SemanticRecall {
    /// Attach a semantic embedding to a block at archive time.
    fn archive_with_embedding(
        &mut self,
        session_id: Uuid,
        kv_vectors: &[f32],
        dim: usize,
        embedding: &[f32],
        priority: f32,
    ) -> BlockId;

    /// Retrieve top-k blocks by cosine similarity to `query_emb`.
    fn retrieve_semantic(
        &self,
        session_id: Uuid,
        query_emb: &[f32],
        top_k: usize,
    ) -> Vec<(BlockId, f32)>;
}
```

### Embedding source

Two candidate paths (benchmark both before committing):

| Option | Embedding source | Pros | Cons |
|---|---|---|---|
| A | Pool K vectors of each block (mean / attention-weighted mean) | free (already have K); no extra model | may not capture semantic content; depends on layer depth |
| B | Small sentence encoder run on decoded text tokens | semantic-grade embedding | extra compute; model dependency |

### Test matrix

Head-to-head against RAG baseline (Qdrant / Chroma with default
all-MiniLM-L6 embeddings):

| Metric | Baseline (RAG) | v1.5.1 | Acceptance |
|---|---|---|---|
| recall@10 on MTEB-session subset | — | — | ≥ 90% of RAG baseline |
| storage per session | RAG docs + index | L1 compressed + embeddings | ≤ 0.7 × RAG baseline |
| retrieval latency | — | — | ≤ 100 ms local |

**If v1.5.1 cannot match RAG baseline on recall quality, do not ship.**
L1 stays as recall-by-id-only. This is the honest exit door.

---

## SKU structure after v1.4 / v1.5 chain completes

| SKU | Active features | Target workload | Positioning |
|---|---|---|---|
| **Base** | v1.3 tier-1 | one-shot inference | "beats turbo3 at ACCEPT quality" |
| **Pro** | v1.3 + v1.4.0 + v1.4.1 | high-quality inference, streaming | "+ lower PPL, streaming-safe" |
| **Agent** | Pro + v1.5.0 (+ v1.5.1) | multi-turn / agentic | "+ 10× session capacity, fast recall" |

SKUs strictly dominate their predecessors. An operator picks one
based on the workload; no combinatorial config search required.

## Decisions explicitly deferred

Not in v1.4 / v1.5 scope:

- **entropy-adaptive bit budget**: benchmark requires per-layer bit
  control infrastructure we don't have; defer to v1.6 with an
  ablation prerequisite.
- **cross-head codec basis sharing**: speculative; no clear win path
  on measured 7-model corpus.
- **pre-RoPE K on L1**: open research question; v1.3 tier-1.5 already
  handles this at L0, pushing it to L1 would require re-applying
  inverse-RoPE at recall time. Defer.
- **L0 on-device Rust codec**: all current bench runs happen on CPU.
  GPU deployment of the kakeyaturbo Rust encoder is a separate
  orthogonal track (v1.4.0 can initially run the codec on CPU and
  just pipe attention weights from GPU).

## Open items to decide before v1.4 kickoff

1. **Attention scores kernel owner**: flash-attn fork? Triton kernel?
   vLLM custom attention? Needs an engineering owner before v1.4.0
   code starts.
2. **Reference PPL harness**: WikiText-103 minimum; LongBench-E for
   regression; RULER for needle-in-haystack. Acceptance gates require
   a stable harness; set one up before v1.4.1.
3. **Session store backend**: sqlite + local files for v1.5.0? Or
   rocksdb? Decision affects the interface implementation but not the
   contract.
4. **LongBench subset for acceptance**: full LongBench is expensive;
   pick a ≤ 30-minute-per-model subset that covers long-range deps.

## Reproducibility plan (mirrors v1.3)

Every phase delivers:
- Rust code on a new branch `cursor/v1-<phase>-…-12f5` per v1.4.1, v1.4.0, v1.5.0, v1.5.1.
- Unit tests + integration tests (`cargo test --release` must stay green).
- Real-data ablation under `reports/v1_4_attention_aware/` (v1.4.1 /
  v1.4.0) and `reports/v1_5_session_memory/` (v1.5.0 / v1.5.1).
- DECISION.md per phase with the exact verdict against exit criteria.
- Shadow-mode A/B snippet in the PR body.

## Final recommendation

**Kick off with v1.4.1** (lowest risk, quick feedback, no signal-extraction
infrastructure required). **Use v1.4.1 outcomes to inform the kernel
owner decision** for v1.4.0. Do not start v1.5.0 until v1.4.0 has
passed acceptance gates.

This is a **bounded, phased, failure-tolerant** roadmap — every phase
has a rollback path, every acceptance gate is a hard number, and
scope creep to L2+ is fenced off by the hard invariants at the top.
