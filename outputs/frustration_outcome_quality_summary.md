# Frustration vs Outcome Quality: Correcting a Structurally Flawed Test

## What this replaces, and why

`embedding_loop_detector.py`'s Step 6 tested whether the loop-similarity metric predicts which **arbitrary side** (`model_a` vs `model_b`) wins a battle. Frustration is a property of the user's own turns, which are **identical on both sides of every battle** -- so that test was **structurally incapable of detecting anything**, regardless of whether frustration actually matters. A null result there was guaranteed by construction, not evidence about frustration.

This script tests the right question instead: do frustrated conversations have **worse outcomes overall** (ties, both-bad results) than engaged ones? "Worse outcome" here is a property of the *conversation*, not of an arbitrary side -- so it does not inherit the earlier flaw.

## Methodology

**Outcome classification, reused rather than reinvented (Step 1).** `both_bad_rate.py`'s `load_and_classify()` already classifies every row of `outputs/final_analysis_derived.csv` by its raw `winner` string into `decisive` / `tie_good` (tie, both acceptable) / `both_bad` (tie, both judged bad) / `no_vote`. That exact logic is replicated here against the raw `winner` field (not reinvented). `loss_anatomy.py` was checked too and found **not** to have this classification -- it starts from `load_decisive()`, which already drops every tie before that script ever sees the data, making it the wrong starting point for this task.

**Population (Step 2):** all 986 analyzable frustration-labeled conversations from `frustration_full990.jsonl` (989 total, minus 3 privacy-redacted, same exclusion as `embedding_loop_detector.py`). Every one received a valid outcome category -- 0 `UNRECOGNIZED`.

**Test (Step 3):** a bootstrap comparison of proportions, resampling each group independently with replacement (1,000 replications, seed `20260930`), percentile 95% CI, two-sided bootstrap p -- the same style of interval used by `bootstrap_bt` throughout this project, adapted to a two-group proportion difference rather than a Bradley-Terry coefficient. Run twice: once on the keyword-classifier `frustrated`/`engaged` labels, once checking the embedding metric (`mean_cos_consecutive`) directly against a both-bad indicator across all 986 conversations.

## KEY FINDING: the test as specified cannot be run on this population

**0 of 986** analyzable conversations have a tie or both-bad outcome. **Every single one is a decisive win/loss.**

This is not a bug and not a null result -- it is a **data availability limit**, confirmed against two independent sources (the `winner` field already carried in `frustration_full990.jsonl`/`embedding_loop_metrics.csv`, and a fresh lookup against `outputs/final_analysis_derived.csv`). `bradley_terry_frustration.py` documents exactly why: the frustration-labeling population was defined, from the start, as **decisive multi-turn battles only** ("Total eligible population this label set is drawn from: decisive multi-turn battles ... with >= 2 actually-stored user turns"). Ties and both-bad outcomes were excluded from the labeling population before this task -- or `embedding_loop_detector.py` -- ever existed.

A consequence worth stating plainly: the *only* outcome variation that exists within this labeled population is decisive win vs. loss -- and testing frustration against THAT would simply reintroduce the exact same arbitrary-side-label flaw this script set out to fix (`model_a` vs `model_b` is still an arbitrary label with respect to the user's turns). The tie/both-bad axis is the only outcome dimension that is NOT arbitrary with respect to frustration -- and it has zero variance in this specific labeled population. There is currently no way to test the intended hypothesis without labeling frustration on a population that includes ties, which this task's scope does not include.

## Results (reported for completeness, not as a confident null)

| signal | outcome | group A rate | group B rate | diff | 95% CI | p | note |
|---|---|---|---|---|---|---|---|
| keyword_classifier | both_bad | 0.0000 (frustrated) | 0.0000 (engaged) | +0.0000 | [+0.0000, +0.0000] | 1.0000 | DEGENERATE: zero variance in both groups |
| keyword_classifier | tie_good | 0.0000 (frustrated) | 0.0000 (engaged) | +0.0000 | [+0.0000, +0.0000] | 1.0000 | DEGENERATE: zero variance in both groups |
| embedding_metric | both_bad_correlation | 0.0000 (all 986) | -- | -- | -- | -- | correlation undefined (zero variance in both-bad indicator) |

Every proportion-comparison row above is **0.0000 vs 0.0000** with a degenerate (zero-width) interval and p=1 -- the mathematically correct output of bootstrapping a variable that never varies, not evidence that frustration has no effect on both-bad/tie rates. The embedding-metric correlation is undefined for the same reason: there is no both-bad variation for `mean_cos_consecutive` to correlate against.

## Honest bottom line

**This is genuinely untestable with the current data, not a second null result to add alongside the first.** The earlier Step 6 test (`embedding_loop_detector.py`) returned a null because it was structurally incapable of finding anything -- testing against an arbitrary side label. This test is structurally incapable of finding anything for a *different* reason -- the population it was asked to use contains no outcome variation on the dimension being tested. Both are honestly reported as **not usable evidence**, for two distinct and now well-documented reasons, rather than pushed toward looking like a confirmed null or redesigned a third time on the same population. **To actually test whether frustration relates to both-bad/tie outcomes, frustration would need to be labeled on a population that includes tied battles** -- a new labeling pass, out of scope for this script.

## Caveats

1. **Population mismatch is the whole finding here.** Nothing about citation count, response length, or model identity was examined -- the blocker is purely which conversations got a frustration label in the first place.
2. **The bootstrap and correlation code are reported for transparency, not as findings.** Both were deliberately left running (rather than short-circuited) so the degenerate output is visible and auditable, not hidden behind an early return.
3. **This does not retroactively validate or invalidate the keyword vs. embedding frustration labels themselves** -- it only shows neither can be related to tie/both-bad outcomes using the conversations currently labeled.

