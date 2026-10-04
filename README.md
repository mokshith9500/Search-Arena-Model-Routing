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

## Folder structure

```
search-arena-24k/
├── README.md                  # this file — stays in the repo root
├── .env                        # GEMINI_API_KEY — gitignored, never commit
├── train.jsonl                 # raw dataset — gitignored (1.2 GB, see below)
├── frustration_*.jsonl         # friction-classification label files (see Friction classification below)
├── scripts/                    # every .py file — all run from the repo root (see Run order)
├── outputs/                    # every result CSV and markdown summary — see Outputs
├── charts/                     # the 8 report chart PNGs, embedded in outputs/bradley_terry_summary.md
└── manual_review/              # human-readable .txt samples for validating the automated measures
```

All scripts are invoked from the **repo root**, e.g. `python scripts/bradley_terry_analysis.py`
— every relative path inside a script (`outputs/...`, `charts/...`, `manual_review/...`)
is written relative to that root, not to `scripts/`. If you move `outputs/bradley_terry_summary.md`
elsewhere, note its chart embeds use `../charts/...`, relative to its own location in `outputs/`.

## Run order

Each step depends on outputs from the previous one. All scripts write into `outputs/`
(data files like `frustration_*.jsonl` and `train.jsonl` stay in the repo root).

| # | Script | Needs | Notes |
|---|---|---|---|
| 1 | `inspect_search_arena.py` | `train.jsonl` | Exploration only. Browse records, inspect citations and search traces. Nothing downstream requires running it — it is the shared helper module the others import. |
| 2 | `intent_complexity_analysis_v2.py` | `train.jsonl` | Establishes and verifies the **fixed per-turn citation aggregation** (dedupe markers *within* each assistant turn, then sum — the earlier version deduped across the whole concatenated conversation and undercounted). Only two helper functions are imported downstream; its own outputs aren't currently regenerated as part of the main run order. |
| 3 | `final_analysis.py` | `train.jsonl` | The OLS-based core analysis. Builds `outputs/final_analysis_derived.csv`, the cached per-conversation table every later script reads. Full scan of the 1.2 GB file — slowest step. |
| 4 | `bradley_terry_analysis.py` | `outputs/final_analysis_derived.csv` | Bradley-Terry rebuild with 1,000-replicate bootstrap. Reads the cached table, so it does **not** re-scan `train.jsonl`. **Overwrites `outputs/bradley_terry_summary.md` from scratch** (core + addendum sections only) — run this *first* in the chain, before the additive scripts below, or you'll need to re-run them afterward to restore their sections. |
| 5 | `frustration_classify.py` | `train.jsonl`, `GEMINI_API_KEY` | Classifies conversational friction via the Gemini API. Takes `--input`/`--out-*` arguments; supports `--resume` to continue after a daily quota wall. |
| 6 | `bradley_terry_frustration.py` | `frustration_full990.jsonl`, `outputs/final_analysis_derived.csv` | Frustration-as-moderator analysis. Appends/replaces its own section in `outputs/bradley_terry_summary.md` — safe to re-run without disturbing other sections. |
| 7 | `bradley_terry_family.py` | `outputs/final_analysis_derived.csv` | Tests whether the cost-tier effect is actually a model-family effect. Appends/replaces its own section. |
| 8 | `bradley_terry_english_only.py` | `outputs/final_analysis_derived.csv`, `train.jsonl` (for `languages`) | English-only replication of the by-intent, cost-tier, and family-vs-tier results. Appends/replaces its own section. |
| 9 | `bh_correction.py` | `outputs/bradley_terry_by_intent.csv`, `outputs/bradley_terry_cost_tier.csv` | Benjamini-Hochberg correction over every p-value in those two files. Appends/replaces its own section; **re-run this last** if you re-ran script 4, since that overwrites the two CSVs it corrects and drops its added columns. |
| 10 | `build_charts.py` | the CSVs above + `outputs/bradley_terry_summary.md` | Builds all 8 report charts into `charts/`. Chart embeds in the markdown are inserted by hand, not regenerated by this script — if you re-run script 4, the embeds and the English-only/family/BH sections will need restoring (see note on script 4). |

```bash
python scripts/intent_complexity_analysis_v2.py
python scripts/final_analysis.py
python scripts/bradley_terry_analysis.py
GEMINI_API_KEY=... python scripts/frustration_classify.py \
    --input <sample>.pkl --out-txt manual_review/review.txt --out-jsonl labels.jsonl
python scripts/bradley_terry_frustration.py --input frustration_full990.jsonl
python scripts/bradley_terry_family.py
python scripts/bradley_terry_english_only.py
python scripts/bh_correction.py
python scripts/build_charts.py
```

> **Free-tier quota.** Gemini free-tier request limits are **per model per day** and can
> be as low as 20/day on some models. `frustration_classify.py` detects a hard daily
> wall, aborts immediately rather than burning the remaining allowance on retries, and
> writes results incrementally so `--resume` can pick up where it stopped.

## Outputs

**All result CSVs and markdown summaries live in `outputs/`.** Charts live in `charts/`
(see Folder structure above); `outputs/bradley_terry_summary.md`'s chart embeds are
`../charts/...`, relative to its own location.

### Core analysis (`outputs/`)

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

### Bradley-Terry rebuild and its extensions (`outputs/`)

| File | Contents |
|---|---|
| `bradley_terry_by_intent.csv` | Citation and length coefficients per intent, with bootstrap 95% CIs and p-values, plus Benjamini-Hochberg-corrected p-values/significance (`*_bh` columns). |
| `bradley_terry_cost_tier.csv` | Tier coefficients relative to cheap, the expensive-minus-mid contrast, and the same BH-corrected columns. |
| `bradley_terry_model_strengths.csv` | Per-model latent strengths from one pooled fit — shows within-tier spread is far larger than between-tier spread. |
| `bradley_terry_tier_boundary_check.csv` | Each model's distance from the $1.00 and $5.00 tier boundaries, flagging models that sit on a boundary or have no published price. |
| `bradley_terry_family.csv` / `bradley_terry_family_joint.csv` | Model-family-as-player fit, and the joint tier+family fit that tests whether the tier effect is actually a family effect in disguise (verdict: inseparable — both remain significant together). |
| `bradley_terry_english_only_*.csv` | The by-intent, cost-tier, family-joint, and pooled-vs-English-only comparison results restricted to English-only conversations (60.5% of the decisive sample) — a replication check. |
| `bradley_terry_frustration_moderator.csv` | Whether conversation-level frustration moderates the citation effect, fit on the complete 989-conversation frustration-labeled sample. |
| `bradley_terry_summary.md` | **The main report** — methodology, all results above, the OLS-vs-Bradley-Terry comparison, the tau2-bench connection, and all 8 charts embedded with captions. |

### tau2-bench comparison (`outputs/`)

| File | Contents |
|---|---|
| `tau2_precomputed_summary.csv` | Pass@1, tool calls, turns, tool error rate, termination reasons and measured cost per task for each model/domain in tau2-bench's precomputed trajectories. Flags ablation conditions, which are **not** comparable to the standard condition. |
| `tau2_analysis_summary.md` | What the precomputed data shows, the generalization test, and the zero-model-overlap limitation. |

### Report charts (`charts/`)

8 PNGs built by `build_charts.py`, every number read directly from an `outputs/` CSV (or
parsed from `outputs/bradley_terry_summary.md` for the frustration-evidence chart) — see
that script's per-chart source comments for the exact file/columns behind each one.

### Friction classification (repo root)

These stay in the repo root, not `outputs/` — they're label data, not analysis results.

| File | Contents |
|---|---|
| `frustration_prompt_design.txt` *(in `manual_review/`)* | The classifier prompt, with its labels, decision rules, and the reasoning behind the conversation-level unit of analysis. |
| `frustration_pilot.jsonl` / `frustration_pilot_review.txt` | 20-conversation pilot. The `.txt` carries full verbatim turns, label, and one-sentence justification for every item so each call can be checked by hand. |
| `frustration_sample300.jsonl` / `frustration_sample300_review.txt` | 300-conversation stratified sample (intent shares matched to the population within 0.1pp; 42 distinct language groups). First labeling pass. |
| `frustration_full990.jsonl` | **The complete, current label set** (989 of 990 eligible multi-turn conversations; one hit a permanent Gemini content-safety block, not a quota issue). Combines `frustration_sample300.jsonl` with the remaining 689 conversations classified in later runs. This is what `bradley_terry_frustration.py` should be pointed at. |
| `frustration_partial687.jsonl`, `frustration_remaining690.jsonl`, `frustration_remaining690_review.txt` | **Superseded checkpoints** from before classification completed, fully absorbed into `frustration_full990.jsonl`. Zero references anywhere in the code. Kept for now pending a decision on deletion. |

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
- **Cost tiers are fragile at two boundaries.** `sonar` and `sonar-reasoning-pro` each
  sit exactly on a tier boundary — tier membership for those is a convention, not a
  measurement. (`sonar-reasoning`'s price was previously unpublished and assumed; it is
  now verified at $3.00/Mtok, comfortably inside the mid tier — see
  `outputs/bradley_terry_summary.md` → "Tier boundary sensitivity".) Within-tier
  strength spread exceeds between-tier spread, so tier is a weak proxy for quality —
  route on individual models, not price bands. A follow-up joint fit found the tier
  effect and a model-family effect are **both** independently significant and
  inseparable with this dataset — tier is not simply a family effect in disguise, but
  family isn't fully controlled for by tier either.
- **Correlational throughout.** No causal claim is supported; models that cite more
  differ in other ways too.
- **Multiple comparisons — now corrected.** Per-intent and cost-tier tests were
  originally reported without family-wise correction. Benjamini-Hochberg correction has
  since been applied to every p-value in `outputs/bradley_terry_by_intent.csv` and
  `outputs/bradley_terry_cost_tier.csv` (see `*_boot_p_bh` / `*_significant_bh`
  columns and the "Multiple-Comparisons Correction" section of the summary) — 4 of 50
  cost-tier tests and 0 of 18 by-intent tests lost significance under correction.
- **English-only replication.** The by-intent, cost-tier, and family-vs-tier findings
  were re-checked on the English-only subset (60.5% of the decisive sample) — 15 of 16
  checked coefficients held their significance status; see "English-Only Replication"
  in the summary.

## A note on coordinating with collaborators before pushing

GitHub currently has an upload from a teammate (**itsAvester**) that has not been
pulled into this local project, including a file called `ZeroCitationInvestigation.ipynb`
that does not exist in this local checkout. **Coordinate with that teammate before
pushing** so their work isn't silently overwritten — pull first, or keep this
reorganized version as a separate branch/folder rather than force-pushing over it.
