# Search Arena: Citations, Cost Tiers, and Conversational Friction

Analysis of the **Search Arena 24k** human-preference dataset, asking what actually
predicts whether users prefer one search-augmented LLM response over another.

Three strands:

1. **Citations** — does showing more visible citations predict winning, and does that
   survive controls for model identity and response length (verbosity)?
2. **Cost tiers** — do pricier models win more, and is "cost tier" even a meaningful
   grouping given how much individual models differ within a tier?
3. **Conversational friction** — a direct signal of user frustration classified from
   real conversation text, rather than a proxy like citation count.

Results are reported two ways for every core finding: the original OLS-residualization
approach, and a Bradley-Terry model with conversation-level bootstrap, which is the
standard for pairwise preference data. Where the two disagree, both are shown.

A separate strand compares these preference-based rankings against **objective
task-success** data from `tau2-bench`. The two datasets share **zero** overlapping
models, so that comparison is a generalization test, not a validation.

## Requirements

- Python **3.12+**
- `numpy`, `pandas`, `scipy`

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy
```

`frustration_classify.py` additionally needs a **Gemini API key** (standard library
only otherwise — no SDK required):

```bash
export GEMINI_API_KEY="your-key-here"
```

Never commit the key. `.gitignore` excludes `.env` files and any filename containing
`key` or `secret`.

## Getting `train.jsonl`

**Not included in this repo** — it is ~1.2 GB, well over GitHub's 100 MB per-file
limit, and the dataset is not ours to redistribute.

It is the **Search Arena 24k** dataset (LMArena; Miroyan et al.), `train` split,
exported as newline-delimited JSON to `train.jsonl` in the repo root. It is published
on HuggingFace — confirm the current dataset id on the hub rather than trusting a
hardcoded path here.

```python
from datasets import load_dataset
ds = load_dataset("lmarena-ai/search-arena-24k", split="train")
ds.to_json("train.jsonl", orient="records", lines=True)
```

Verify the export before running anything — each line should be one conversation with
these keys:

```
judge, languages, messages_a, messages_b, model_a, model_b,
primary_intent, secondary_intent, system_a_metadata, system_b_metadata,
timestamp, turn, winner
```

Expected: **24,069** conversations, of which **8,613** are decisive (non-tie, voted).

> **Two data quirks worth knowing before you trust any output.** First, in multi-turn
> battles the user's turns are **byte-identical on both sides** — the user addresses
> both models at once — so any user-derived signal is a property of the conversation,
> not of one side. Second, the `turn` field is the arena's round count, which can
> exceed the number of rounds actually stored: 724 of the 1,714 decisive "multi-turn"
> conversations contain only **one** stored user turn. Filter on
> `len(get_user_messages(...)) >= 2`, not on `turn > 1`.

## Run order

Each step depends on outputs from the previous one.

| # | Script | Needs | Notes |
|---|---|---|---|
| 1 | `inspect_search_arena.py` | `train.jsonl` | Exploration only. Browse records, inspect citations and search traces. Nothing downstream requires running it — it is the shared helper module the others import. |
| 2 | `intent_complexity_analysis_v2.py` | `train.jsonl` | Establishes and verifies the **fixed per-turn citation aggregation** (dedupe markers *within* each assistant turn, then sum — the earlier version deduped across the whole concatenated conversation and undercounted). |
| 3 | `final_analysis.py` | `train.jsonl` | The OLS-based core analysis. Builds `final_analysis_derived.csv`, the cached per-conversation table every later script reads. Full scan of the 1.2 GB file — slowest step. |
| 4 | `bradley_terry_analysis.py` | `final_analysis_derived.csv` | Bradley-Terry rebuild with 1,000-replicate bootstrap. Reads the cached table, so it does **not** re-scan `train.jsonl`. |
| 5 | `frustration_classify.py` | `train.jsonl`, `GEMINI_API_KEY` | Classifies conversational friction via the Gemini API. Takes `--input`/`--out-*` arguments; supports `--resume` to continue after a daily quota wall. |

```bash
python scripts/intent_complexity_analysis_v2.py
python scripts/final_analysis.py
python scripts/bradley_terry_analysis.py
GEMINI_API_KEY=... python scripts/frustration_classify.py \
    --input <sample>.pkl --out-txt manual_review/review.txt --out-jsonl labels.jsonl
```

> **Free-tier quota.** Gemini free-tier request limits are **per model per day** and can
> be as low as 20/day on some models. `frustration_classify.py` detects a hard daily
> wall, aborts immediately rather than burning the remaining allowance on retries, and
> writes results incrementally so `--resume` can pick up where it stopped.

## Outputs

### Core analysis

| File | Contents |
|---|---|
| `final_analysis_derived.csv` | **The cached per-conversation table** (24,069 rows): intent, turn count, per-side citation counts, per-side response length, model identities, winner, tie/decisive flags. Every later script reads this instead of re-scanning `train.jsonl`. |
| `final_analysis_by_intent.csv` | Citation-winning correlation per intent category, reported raw, model-controlled, and model+length-controlled, with Fisher-z significance. |
| `final_analysis_cost_tier_comparison.csv` | Head-to-head win rates between cost tiers, per intent, with z-tests. |
| `final_analysis_stratified_9x2.csv` | The 9 intents × single/multi-turn split. **Read with the `turn`-field caveat above** — the multi-turn cell is 42% single-round. |
| `final_analysis_significance_test.csv` | Formal test of the single- vs multi-turn citation-correlation gap. |
| `final_analysis_citation_length_correlation.csv` | Correlation between citation count and response length (verbosity confound check). |
| `final_analysis_vote_presence.csv` | Whether citation count predicts that a vote happened at all. |
| `final_analysis_summary.md` | Narrative write-up with explicit "confident" vs "uncertain" verdicts. |

### Bradley-Terry rebuild

| File | Contents |
|---|---|
| `bradley_terry_by_intent.csv` | Citation and length coefficients per intent, with bootstrap 95% CIs and p-values. |
| `bradley_terry_cost_tier.csv` | Tier coefficients relative to cheap, plus the expensive-minus-mid contrast with its own bootstrap interval. |
| `bradley_terry_model_strengths.csv` | Per-model latent strengths from one pooled fit — shows within-tier spread is far larger than between-tier spread. |
| `bradley_terry_tier_boundary_check.csv` | Each model's distance from the $1.00 and $5.00 tier boundaries, flagging models that sit on a boundary or have no published price. |
| `bradley_terry_summary.md` | Methodology, results, the OLS-vs-Bradley-Terry comparison, and the tau2-bench connection. |

### tau2-bench comparison

| File | Contents |
|---|---|
| `tau2_precomputed_summary.csv` | Pass@1, tool calls, turns, tool error rate, termination reasons and measured cost per task for each model/domain in tau2-bench's precomputed trajectories. Flags ablation conditions, which are **not** comparable to the standard condition. |
| `tau2_analysis_summary.md` | What the precomputed data shows, the generalization test, and the zero-model-overlap limitation. |

### Friction classification

| File | Contents |
|---|---|
| `frustration_prompt_design.txt` | The classifier prompt, with its labels, decision rules, and the reasoning behind the conversation-level unit of analysis. |
| `frustration_pilot.jsonl` / `frustration_pilot_review.txt` | 20-conversation pilot. The `.txt` carries full verbatim turns, label, and one-sentence justification for every item so each call can be checked by hand. |
| `frustration_sample300.jsonl` / `frustration_sample300_review.txt` | 300-conversation stratified sample (intent shares matched to the population within 0.1pp; 42 distinct language groups). |

### Manual review files

Human-readable samples used to validate the automated measures rather than trust them:

| File | Contents |
|---|---|
| `zero_citation_manual_review.txt` | Responses the citation parser scored as zero, for checking false negatives. |
| `intent_manual_validation.txt` | Sampled conversations per intent label, for checking the dataset's own intent classifications. |
| `recommendation_citation_examples.txt` | Recommendation-intent examples, the category with the strongest citation effect. |

## Caveats

- **Citation counts are a visible-marker proxy.** Models that cite without inline
  markers are undercounted.
- **Cost tiers are fragile.** Two models sit exactly on a tier boundary and one has no
  published price at all; tier membership for those is a convention, not a measurement.
  Within-tier strength spread exceeds between-tier spread, so tier is a weak proxy for
  quality — route on individual models, not price bands.
- **Correlational throughout.** No causal claim is supported; models that cite more
  differ in other ways too.
- **Multiple comparisons.** Per-intent tests are reported without family-wise
  correction; roughly one in twenty "significant" results is expected by chance.
