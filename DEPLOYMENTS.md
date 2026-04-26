# Deployments

Named deployments of KakeyaLattice in production, research, and demo
contexts. This is the canonical public list; if you want to be on it,
see [How to add your deployment](#how-to-add-your-deployment) below.

The list exists for three purposes:

1. **Credit you.** If your team is shipping KakeyaLattice we want to
   say so here, with a link to whatever you want to point users at.
2. **Give other teams prior art.** A procurement engineer evaluating
   whether to adopt KV-cache compression will ask, "who else is
   already running this?" — and will trust a named public list more
   than anything we can say ourselves.
3. **Document operating-point distribution.** The
   [`q_range`](docs/faq.md#choosing-q_range) value real deployments
   converge on is much more informative than the 2.4×–2.8× CR range
   we quote in the README.

## Current deployments

*(Empty at release 1.5.0, 2026-04-24. Be first.)*

| date | organisation / individual | model served | variant · q_range | context | notes | contact / link |
|:-----|:--------------------------|:-------------|:------------------|:--------|:------|:---------------|
| _YYYY-MM-DD_ | _your org_ | _e.g. Qwen3-4B_ | _e.g. e8 · 38_ | _e.g. 32 k_ | _one-line deployment context_ | _GitHub @handle, URL, or email_ |

### Reference demo

| date | operator | model served | variant · q_range | context | notes | contact |
|:-----|:---------|:-------------|:------------------|:--------|:------|:--------|
| 2026-04-25 | FluffyAIcode | Qwen/Qwen3-0.6B | e8 · {10, 38, 152} | 2 k | Live side-by-side demo vs bf16 baseline on free CPU tier | [🤗 Space](https://huggingface.co/spaces/FluffyAIcode/LLM-KA-Cache-Compress) |

## Case studies

Longer write-ups of specific deployments live here as
subdirectories under `deployments/`. If you want a case-study slot,
open an issue with the title `Case study: <your name>` — we'll coordinate
wording and, if you're willing, a short quote we can pull into the
README.

*(No case studies yet at release 1.5.0.)*

## How to add your deployment

Two paths, both fine:

### Path A — PR (preferred if you can)

1. Fork the repo.
2. Add one row to the table under **Current deployments** above. Fill
   in as many columns as you are comfortable disclosing; the `contact
   / link` column can be a GitHub @handle, a URL, an email, or left
   blank.
3. Open a PR titled `deployment: <your organisation>`.

We merge same-day on business days. We do not QA the claims you make
— you own them. We reserve the right to remove entries that are
clearly false or spam.

### Path B — Issue (if you can't PR)

Open an issue at
<https://github.com/FluffyAIcode/LLM-KV--Cache-compress/issues> with
the title `deployment: <your organisation>` and the row you'd like
added. We will PR it for you.

### What we'd like but don't require

Any of the following is useful context, but none is mandatory:

- The `q_range` value you shipped (the operating-point distribution of
  real deployments is the single most useful piece of data for the
  next wave of adopters).
- The KV compression ratio you measured on your model × traffic pattern.
- Perplexity / task-metric delta if you measured it (`lm-eval-harness`,
  internal eval harness, LiveBench, etc.).
- Whether you stacked KakeyaLattice with HQQ / AWQ / GPTQ weight
  quantisation, or with SnapKV / H2O eviction. Real stacks are
  informative.
- A one-line quote we can pull into the README / blog.

### What we will not ask for

We will not ask for traffic numbers, revenue numbers, commercial
details, or any internal deployment details you have not volunteered.

## Research-community deployments

Citations in academic papers that use KakeyaLattice in experiments
are tracked separately in
[`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md#early-users-and-contributors)
to keep this table focused on production / pre-production runs. If
your paper uses KakeyaLattice, please also open an issue titled
`Acknowledgment: paper <short title>` so we can add the citation.

## Anti-shill policy

We will not:

- Pay for entries on this list.
- Accept entries from organisations that have not actually run the
  code. If in doubt, we may ask to see a `git rev-parse HEAD` or a
  snippet of the compressed-cache output.
- Remove entries on request from anyone other than the lister, unless
  the entry violates the above.

The credibility of this list to *future* readers depends on it being
accurate; we would rather be empty and honest than full and padded.
