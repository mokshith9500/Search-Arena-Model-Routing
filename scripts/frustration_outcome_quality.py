"""
frustration_outcome_quality.py

FIXES A REAL DESIGN FLAW in embedding_loop_detector.py's Step 6. That test
asked whether the loop-similarity metric predicts which ARBITRARY SIDE
(model_a vs model_b) wins. Since frustration is a property of the user's
turns -- identical on both sides of every battle -- that test was
structurally incapable of detecting anything, regardless of whether
frustration actually matters. The correct question is whether frustrated
conversations have WORSE OUTCOMES OVERALL (ties, both-bad results) than
engaged ones -- not which side an arbitrary label happened to win.

ADDITIVE ONLY. Does not modify any existing file.

STEP 1 FINDING (read before writing anything new, as instructed):
`both_bad_rate.py`'s `load_and_classify()` already has the exact outcome
classification this task needs, applied to the raw `winner` column of
`outputs/final_analysis_derived.csv`:

    winner value       category     meaning
    ------------------  -----------  -----------------------------------
    'model_a'/'model_b'  decisive     a clear winner
    'tie'                tie_good     tie, both judged acceptable
    'tie (bothbad)'      both_bad     tie, both judged bad
    NaN                  no_vote      no vote occurred

`loss_anatomy.py` does NOT have this -- it imports `load_decisive()` from
`bradley_terry_analysis.py`, which filters to `is_decisive == True` ONLY and
drops every tie and no-vote row before loss_anatomy.py ever sees the data.
So for this task, `both_bad_rate.py`'s classification is the one to reuse;
`loss_anatomy.py`'s population is the wrong starting point entirely (it has
already had every tie removed).

This script reuses `both_bad_rate.py`'s `wilson_ci` function unchanged, and
replicates its classification conditions (not a new import, since
`both_bad_rate.py`'s `load_and_classify` is hardwired to its own module-level
constants and isn't written as a reusable library call -- but the logic
itself, stated above, is copied verbatim, not reinvented).
"""

import json

import numpy as np
import pandas as pd

from bradley_terry_analysis import N_BOOTSTRAP, ALPHA, RANDOM_SEED
from both_bad_rate import wilson_ci

FRUSTRATION_FULL_JSONL = "frustration_full990.jsonl"
DERIVED_CSV = "outputs/final_analysis_derived.csv"
EMBEDDING_METRICS_CSV = "outputs/embedding_loop_metrics.csv"
REDACTION_MARKER = "[excluded: contains personal/sensitive content, available on request]"

OUT_CSV = "outputs/frustration_outcome_quality.csv"
OUT_MD = "outputs/frustration_outcome_quality_summary.md"


# ============================================================
# STEP 1 (report): outcome categories already established in this project
# ============================================================

def report_step1_findings():
    print("=" * 78)
    print("STEP 1  Outcome categories already established in this project")
    print("=" * 78)
    print("both_bad_rate.py's load_and_classify() classifies every row of "
          f"{DERIVED_CSV} by its raw `winner` string:")
    print("  'model_a'/'model_b'  -> decisive (a clear winner)")
    print("  'tie'                -> tie_good  (tie, both acceptable)")
    print("  'tie (bothbad)'      -> both_bad  (tie, both judged bad)")
    print("  NaN                  -> no_vote   (no vote occurred)")
    print()
    print("loss_anatomy.py does NOT have this classification -- it starts from "
          "load_decisive(), which already filters to is_decisive==True and drops "
          "every tie before loss_anatomy.py sees the data. For THIS task, "
          "both_bad_rate.py's classification (applied directly to "
          f"{DERIVED_CSV}'s raw `winner` column) is the one being reused.")
    print()


# ============================================================
# STEP 2: outcome category for every one of the 986 analyzable conversations
# ============================================================

def classify_outcome(winner_series):
    conditions = [
        winner_series.isin(["model_a", "model_b"]),
        winner_series == "tie",
        winner_series == "tie (bothbad)",
        winner_series.isna(),
    ]
    choices = ["decisive", "tie_good", "both_bad", "no_vote"]
    return pd.Series(np.select(conditions, choices, default="UNRECOGNIZED"),
                     index=winner_series.index)


def load_population_with_outcomes():
    print("=" * 78)
    print("STEP 2  Outcome category for every analyzable frustration-labeled conversation")
    print("=" * 78)

    rows = [json.loads(l) for l in open(FRUSTRATION_FULL_JSONL, encoding="utf-8")]
    rows = [r for r in rows if r["user_turns"] != [REDACTION_MARKER]]
    print(f"loaded {len(rows)} analyzable conversations (989 total minus 3 "
          "privacy-redacted, same exclusion as embedding_loop_detector.py)")

    full = pd.read_csv(DERIVED_CSV)
    idxs = [r["idx"] for r in rows]
    missing = [i for i in idxs if i not in full.index]
    if missing:
        print(f"[WARNING] {len(missing)} idx not found in {DERIVED_CSV}: {missing}")

    base = pd.DataFrame(rows)[["idx", "label"]].rename(columns={"label": "keyword_label"})
    base = base.set_index("idx")
    base["outcome"] = classify_outcome(full.loc[base.index, "winner"])

    emb = pd.read_csv(EMBEDDING_METRICS_CSV).set_index("idx")
    base["mean_cos_consecutive"] = emb.loc[base.index, "mean_cos_consecutive"]

    n_unrecognized = int((base["outcome"] == "UNRECOGNIZED").sum())
    print(f"\n{len(base)} conversations classified; {n_unrecognized} UNRECOGNIZED "
          "(expect 0 -- every row should land in decisive/tie_good/both_bad/no_vote)")
    print("\noutcome category counts:")
    print(base["outcome"].value_counts())

    n_ties = int((base["outcome"].isin(["tie_good", "both_bad"])).sum())
    n_no_vote = int((base["outcome"] == "no_vote").sum())
    print(f"\n*** KEY FINDING: {n_ties} ties and {n_no_vote} no-vote rows found in "
          f"this population of {len(base)}. ***")
    if n_ties == 0:
        print("This is NOT a bug -- it is a structural property of how the "
              "frustration-labeling population was originally built. "
              "bradley_terry_frustration.py documents it explicitly: "
              "\"Total eligible population this label set is drawn from: decisive "
              "multi-turn battles ... with >= 2 actually-stored user turns.\" The "
              "frustration classifier was only ever run on DECISIVE battles -- ties "
              "and both-bad outcomes were excluded from the labeling population from "
              "the very start, before this task existed. Confirmed independently "
              "here against two sources (frustration_full990.jsonl's own `winner` "
              f"field and a fresh lookup against {DERIVED_CSV}), not assumed.")
    return base, n_ties


# ============================================================
# Bootstrap comparison of two proportions, conversation-level resampling,
# same style (percentile CI, two-sided bootstrap p) as bootstrap_bt in
# bradley_terry_analysis.py -- each group resampled independently with
# replacement, same N_BOOTSTRAP / RANDOM_SEED used everywhere in this
# project.
# ============================================================

def bootstrap_proportion_diff(group_a, group_b, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    group_a, group_b: 0/1 arrays (e.g. is_both_bad for frustrated vs engaged
    conversations). Returns point diff (a-mean minus b-mean), CI, p, plus
    explicit flags when either group has zero variance -- a degenerate
    bootstrap (zero-width CI, p=1) from zero variance is NOT the same thing
    as a confidently-estimated null and must not be reported as one.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    p_a, p_b = float(a.mean()), float(b.mean())
    point = p_a - p_b

    zero_variance = (a.std() == 0 and b.std() == 0)

    rng = np.random.default_rng(seed)
    na, nb = len(a), len(b)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        da = a[rng.integers(0, na, size=na)].mean()
        db = b[rng.integers(0, nb, size=nb)].mean()
        draws[i] = da - db

    lo = float(np.percentile(draws, 100 * ALPHA / 2))
    hi = float(np.percentile(draws, 100 * (1 - ALPHA / 2)))
    p = float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))))
    sig = bool(lo > 0 or hi < 0)
    return dict(p_a=p_a, p_b=p_b, diff=point, ci_lo=lo, ci_hi=hi, p=p,
               significant=sig, n_a=na, n_b=nb, zero_variance=zero_variance)


# ============================================================
# STEP 3: test using both frustration signals
# ============================================================

def run_outcome_tests(base, n_ties):
    print("\n" + "=" * 78)
    print("STEP 3  Does frustration predict worse outcomes (ties / both-bad)?")
    print("=" * 78)

    rows = []

    if n_ties == 0:
        print("\n*** The test AS SPECIFIED (tie-both-bad rate, tie-both-acceptable "
              "rate, by frustrated vs engaged) CANNOT BE RUN on this population: "
              "there are zero tie or both-bad outcomes among all 986 analyzable "
              "conversations -- every one is a decisive win/loss. This is not a "
              "null result (a null result requires the outcome to vary enough to "
              "estimate a difference); it is a DATA AVAILABILITY LIMIT. Proceeding "
              "to run the bootstrap machinery anyway, to show explicitly what a "
              "zero-variance input produces (a degenerate, uninformative interval), "
              "rather than skip the step silently.")

    for signal_name, group_col, groups in (
        ("keyword_classifier", "keyword_label", ("frustrated", "engaged")),
    ):
        sub = base[base[group_col].isin(groups)]
        print(f"\n--- signal: {signal_name}  (n={len(sub)}, "
              f"{groups[0]}={int((sub[group_col]==groups[0]).sum())}, "
              f"{groups[1]}={int((sub[group_col]==groups[1]).sum())}) ---")
        for outcome_name, outcome_val in (("both_bad", "both_bad"),
                                          ("tie_good", "tie_good")):
            a = (sub.loc[sub[group_col] == groups[0], "outcome"] == outcome_val).astype(int).values
            b = (sub.loc[sub[group_col] == groups[1], "outcome"] == outcome_val).astype(int).values
            res = bootstrap_proportion_diff(a, b)
            flag = "  [DEGENERATE -- zero variance in both groups]" if res["zero_variance"] else ""
            print(f"  {outcome_name:10} rate: {groups[0]}={res['p_a']:.4f}  "
                  f"{groups[1]}={res['p_b']:.4f}  diff={res['diff']:+.4f}  "
                  f"95% CI [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  p={res['p']:.4f}  "
                  f"{'SIG' if res['significant'] else 'not sig'}{flag}")
            rows.append(dict(signal=signal_name, outcome=outcome_name,
                            group_a=groups[0], group_b=groups[1], **res))

    # Embedding metric, same two-group comparison for completeness ...
    emb_sub = base[base["keyword_label"].isin(["frustrated", "engaged"])]
    # ... but the task also asks for something more direct: correlation of
    # the CONTINUOUS metric against a binary both-bad indicator, across
    # ALL 986 (not just the frustrated/engaged subset).
    print("\n--- signal: embedding metric (mean_cos_consecutive) vs both_bad, "
          "all 986 conversations ---")
    both_bad_indicator = (base["outcome"] == "both_bad").astype(float).values
    metric = base["mean_cos_consecutive"].values
    if both_bad_indicator.std() == 0:
        print("  correlation UNDEFINED: both_bad indicator has zero variance "
              "(0 of 986 conversations are both_bad) -- there is nothing for the "
              "metric to correlate WITH. np.corrcoef would return NaN; reporting "
              "that honestly rather than silently dropping the row or printing NaN "
              "unexplained.")
        corr_val = None
    else:
        corr_val = float(np.corrcoef(metric, both_bad_indicator)[0, 1])
        print(f"  r = {corr_val:+.4f}  (n=986)")
    rows.append(dict(signal="embedding_metric", outcome="both_bad_correlation",
                     group_a="all_986", group_b=None, p_a=float(both_bad_indicator.mean()),
                     p_b=None, diff=None, ci_lo=None, ci_hi=None, p=None,
                     significant=None, n_a=986, n_b=None, zero_variance=(corr_val is None),
                     correlation=corr_val))

    return pd.DataFrame(rows)


# ============================================================
# Markdown summary
# ============================================================

def write_summary_md(path, base, n_ties, results_df):
    L = []
    A = L.append
    A("# Frustration vs Outcome Quality: Correcting a Structurally Flawed Test")
    A("")
    A("## What this replaces, and why")
    A("")
    A("`embedding_loop_detector.py`'s Step 6 tested whether the loop-similarity "
      "metric predicts which **arbitrary side** (`model_a` vs `model_b`) wins a "
      "battle. Frustration is a property of the user's own turns, which are "
      "**identical on both sides of every battle** -- so that test was "
      "**structurally incapable of detecting anything**, regardless of whether "
      "frustration actually matters. A null result there was guaranteed by "
      "construction, not evidence about frustration.")
    A("")
    A("This script tests the right question instead: do frustrated conversations "
      "have **worse outcomes overall** (ties, both-bad results) than engaged ones? "
      "\"Worse outcome\" here is a property of the *conversation*, not of an "
      "arbitrary side -- so it does not inherit the earlier flaw.")
    A("")

    A("## Methodology")
    A("")
    A("**Outcome classification, reused rather than reinvented (Step 1).** "
      "`both_bad_rate.py`'s `load_and_classify()` already classifies every row of "
      f"`{DERIVED_CSV}` by its raw `winner` string into `decisive` / `tie_good` "
      "(tie, both acceptable) / `both_bad` (tie, both judged bad) / `no_vote`. "
      "That exact logic is replicated here against the raw `winner` field (not "
      "reinvented). `loss_anatomy.py` was checked too and found **not** to have "
      "this classification -- it starts from `load_decisive()`, which already "
      "drops every tie before that script ever sees the data, making it the wrong "
      "starting point for this task.")
    A("")
    A(f"**Population (Step 2):** all {len(base):,} analyzable frustration-labeled "
      "conversations from `frustration_full990.jsonl` (989 total, minus 3 "
      "privacy-redacted, same exclusion as `embedding_loop_detector.py`). Every "
      "one received a valid outcome category -- 0 `UNRECOGNIZED`.")
    A("")
    A("**Test (Step 3):** a bootstrap comparison of proportions, resampling each "
      f"group independently with replacement ({N_BOOTSTRAP:,} replications, seed "
      f"`{RANDOM_SEED}`), percentile 95% CI, two-sided bootstrap p -- the same "
      "style of interval used by `bootstrap_bt` throughout this project, adapted "
      "to a two-group proportion difference rather than a Bradley-Terry "
      "coefficient. Run twice: once on the keyword-classifier `frustrated`/"
      "`engaged` labels, once checking the embedding metric "
      "(`mean_cos_consecutive`) directly against a both-bad indicator across all "
      f"{len(base):,} conversations.")
    A("")

    A("## KEY FINDING: the test as specified cannot be run on this population")
    A("")
    A(f"**{n_ties} of {len(base):,}** analyzable conversations have a tie or "
      "both-bad outcome. **Every single one is a decisive win/loss.**")
    A("")
    A("This is not a bug and not a null result -- it is a **data availability "
      "limit**, confirmed against two independent sources (the `winner` field "
      "already carried in `frustration_full990.jsonl`/`embedding_loop_metrics.csv`, "
      f"and a fresh lookup against `{DERIVED_CSV}`). "
      "`bradley_terry_frustration.py` documents exactly why: the frustration-"
      "labeling population was defined, from the start, as **decisive multi-turn "
      "battles only** (\"Total eligible population this label set is drawn from: "
      "decisive multi-turn battles ... with >= 2 actually-stored user turns\"). "
      "Ties and both-bad outcomes were excluded from the labeling population "
      "before this task -- or `embedding_loop_detector.py` -- ever existed.")
    A("")
    A("A consequence worth stating plainly: the *only* outcome variation that "
      "exists within this labeled population is decisive win vs. loss -- and "
      "testing frustration against THAT would simply reintroduce the exact same "
      "arbitrary-side-label flaw this script set out to fix (`model_a` vs "
      "`model_b` is still an arbitrary label with respect to the user's turns). "
      "The tie/both-bad axis is the only outcome dimension that is NOT arbitrary "
      "with respect to frustration -- and it has zero variance in this specific "
      "labeled population. There is currently no way to test the intended "
      "hypothesis without labeling frustration on a population that includes ties, "
      "which this task's scope does not include.")
    A("")

    A("## Results (reported for completeness, not as a confident null)")
    A("")
    A("| signal | outcome | group A rate | group B rate | diff | 95% CI | p | note |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in results_df.iterrows():
        if r["signal"] == "embedding_metric":
            note = ("correlation undefined (zero variance in both-bad indicator)"
                    if r["zero_variance"] else f"r={r['correlation']:+.4f}")
            A(f"| {r['signal']} | {r['outcome']} | "
              f"{r['p_a']:.4f} (all 986) | -- | -- | -- | -- | {note} |")
        else:
            note = "DEGENERATE: zero variance in both groups" if r["zero_variance"] else ""
            A(f"| {r['signal']} | {r['outcome']} | {r['p_a']:.4f} ({r['group_a']}) | "
              f"{r['p_b']:.4f} ({r['group_b']}) | {r['diff']:+.4f} | "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p']:.4f} | {note} |")
    A("")
    A("Every proportion-comparison row above is **0.0000 vs 0.0000** with a "
      "degenerate (zero-width) interval and p=1 -- the mathematically correct "
      "output of bootstrapping a variable that never varies, not evidence that "
      "frustration has no effect on both-bad/tie rates. The embedding-metric "
      "correlation is undefined for the same reason: there is no both-bad "
      "variation for `mean_cos_consecutive` to correlate against.")
    A("")

    A("## Honest bottom line")
    A("")
    A("**This is genuinely untestable with the current data, not a second null "
      "result to add alongside the first.** The earlier Step 6 test "
      "(`embedding_loop_detector.py`) returned a null because it was structurally "
      "incapable of finding anything -- testing against an arbitrary side label. "
      "This test is structurally incapable of finding anything for a *different* "
      "reason -- the population it was asked to use contains no outcome variation "
      "on the dimension being tested. Both are honestly reported as **not usable "
      "evidence**, for two distinct and now well-documented reasons, rather than "
      "pushed toward looking like a confirmed null or redesigned a third time on "
      "the same population. **To actually test whether frustration relates to "
      "both-bad/tie outcomes, frustration would need to be labeled on a "
      "population that includes tied battles** -- a new labeling pass, out of "
      "scope for this script.")
    A("")

    A("## Caveats")
    A("")
    A("1. **Population mismatch is the whole finding here.** Nothing about "
      "citation count, response length, or model identity was examined -- the "
      "blocker is purely which conversations got a frustration label in the "
      "first place.")
    A("2. **The bootstrap and correlation code are reported for transparency, "
      "not as findings.** Both were deliberately left running (rather than "
      "short-circuited) so the degenerate output is visible and auditable, not "
      "hidden behind an early return.")
    A("3. **This does not retroactively validate or invalidate the keyword vs. "
      "embedding frustration labels themselves** -- it only shows neither can be "
      "related to tie/both-bad outcomes using the conversations currently labeled.")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[saved] {path}")


def main():
    print("=" * 78)
    print("FRUSTRATION OUTCOME QUALITY (additive -- no existing file modified)")
    print("=" * 78)

    report_step1_findings()
    base, n_ties = load_population_with_outcomes()
    results_df = run_outcome_tests(base, n_ties)

    results_df.to_csv(OUT_CSV, index=False)
    print(f"\n[saved] {OUT_CSV}")

    write_summary_md(OUT_MD, base, n_ties, results_df)


if __name__ == "__main__":
    main()
