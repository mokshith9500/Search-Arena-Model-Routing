"""
both_bad_rate.py

A3: rate at which BOTH responses in a battle are judged bad -- a distinct
failure signal from the decisive win/loss analysis (loss_anatomy.py), closer
to a churn/catastrophic-failure metric than a preference comparison.

STEP 0 FINDING (confirmed before writing any aggregation logic, not assumed):
the raw `winner` field in this dataset has a genuine, separately-labeled
"both bad" category, distinct from an ordinary "both good" tie:

    winner value       count    meaning
    ------------------  -------  -----------------------------------
    'model_a'             4255  model_a wins (decisive)
    'model_b'             4358  model_b wins (decisive)
    'tie'                 2502  tie -- both judged acceptable
    'tie (bothbad)'       1537  tie -- both judged bad (THE SIGNAL)
    NaN (no vote)        11417  no vote occurred
    ------------------  -------
    total                24069

This is case (a) from the task: a real label exists, so the both-bad rate is
computed directly from it, not from a proxy. This confirms something the
existing pipeline (final_analysis.py / intent_complexity_analysis_v2.py) was
already quietly aware of -- TIE_WINNERS = ("tie", "tie (bothbad)") collapses
both into one `is_tie` boolean for the citation/length analysis, which is
the right call for *that* analysis (both are "not decisive") but throws away
exactly the distinction this task needs. This script does not modify that
collapsing logic anywhere -- it reads the raw `winner` string column
(preserved as-is in outputs/final_analysis_derived.csv) and makes its own,
separate distinction.

Runs entirely locally, additive only, modifies no existing file.
"""

import math
import os

import numpy as np
import pandas as pd

DERIVED_CSV = "outputs/final_analysis_derived.csv"
INTENT_ORDER = [
    "Factual Lookup", "Info Synthesis", "Analysis", "Recommendation",
    "Explanation", "Guidance", "Creative Generation", "Text Processing",
    "Other",
]

ALPHA = 0.05
OUT_CSV = "outputs/both_bad_rate_by_intent.csv"
OUT_MD = "outputs/both_bad_rate_summary.md"
OUT_CHART = "charts/chart_both_bad_rate_by_intent.png"
OUT_CITATION_CSV = "outputs/both_bad_rate_by_citation_presence.csv"

# Minimum voted battles in an (intent, citation_group) cell before its rate
# is reported rather than flagged as too thin -- same cutoff this project
# already uses elsewhere (MIN_N_FOR_FIT in bradley_terry_analysis.py) for
# "enough data to say anything about a subgroup at all."
MIN_N_FOR_CITATION_SPLIT = 30


# ============================================================
# Wilson score interval for a binomial proportion
# ------------------------------------------------------------
# No proportion-CI method is already established elsewhere in this project
# (grepped: the rest of this project only ever bootstraps Bradley-Terry
# coefficients, never a raw proportion). Wilson is used here because it is
# the standard, well-calibrated interval for a binomial proportion -- unlike
# the normal (Wald) approximation, it stays sensible at small n and at rates
# far from 0.5, both of which apply to several intent/conversation_type
# cells here. This is a new addition to the project's statistical toolkit,
# stated explicitly rather than silently introduced.
# ============================================================

def wilson_ci(n_success, n_total, alpha=ALPHA):
    if n_total == 0:
        return float("nan"), float("nan"), float("nan")
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z_for_alpha(alpha)
    p = n_success / n_total
    denom = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denom
    half = (z * math.sqrt((p * (1 - p) / n_total) + (z ** 2 / (4 * n_total ** 2)))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _z_for_alpha(alpha):
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / 2))


# ============================================================
# STEP 1: classify every row (voted or not) by raw winner value
# ============================================================

def load_and_classify():
    df = pd.read_csv(DERIVED_CSV)
    w = df["winner"]
    conditions = [
        w.isin(["model_a", "model_b"]),
        w == "tie",
        w == "tie (bothbad)",
        w.isna(),
    ]
    choices = ["decisive", "tie_good", "both_bad", "no_vote"]
    df["category"] = np.select(conditions, choices, default="UNRECOGNIZED")
    n_unrecognized = int((df["category"] == "UNRECOGNIZED").sum())
    if n_unrecognized:
        print(f"[WARNING] {n_unrecognized} rows have an unrecognized winner value: "
              f"{df.loc[df['category'] == 'UNRECOGNIZED', 'winner'].unique()}")
    return df


def rate_table(df, group_col):
    print(f"\n{'='*78}\nBoth-bad rate by {group_col}\n{'='*78}")
    rows = []
    if group_col == "primary_intent":
        # df[col].unique() includes NaN if present -- include it explicitly
        # as its own row rather than silently dropping those rows by
        # iterating a fixed category list that doesn't cover them.
        present = set(df[group_col].dropna().unique())
        groups = [g for g in INTENT_ORDER if g in present]
        extra = present - set(INTENT_ORDER)
        groups += sorted(extra)
        n_missing_intent = int(df[group_col].isna().sum())
        if n_missing_intent:
            groups.append("(missing primary_intent)")
    else:
        groups = sorted(df[group_col].dropna().unique())

    for g in groups:
        if g == "(missing primary_intent)":
            sub = df[df[group_col].isna()]
        else:
            sub = df[df[group_col] == g]
        n_total = len(sub)
        n_decisive = int((sub["category"] == "decisive").sum())
        n_tie_good = int((sub["category"] == "tie_good").sum())
        n_both_bad = int((sub["category"] == "both_bad").sum())
        n_no_vote = int((sub["category"] == "no_vote").sum())
        n_voted = n_decisive + n_tie_good + n_both_bad

        # reconciliation -- must hold exactly, every row accounted for
        assert n_decisive + n_tie_good + n_both_bad + n_no_vote == n_total

        p_voted, lo_voted, hi_voted = wilson_ci(n_both_bad, n_voted)
        p_all, lo_all, hi_all = wilson_ci(n_both_bad, n_total)

        rows.append(dict(
            group=g, n_total=n_total, n_voted=n_voted, n_no_vote=n_no_vote,
            n_decisive=n_decisive, n_tie_good=n_tie_good, n_both_bad=n_both_bad,
            pct_both_bad_of_voted=100 * p_voted,
            ci_lo_of_voted=100 * lo_voted, ci_hi_of_voted=100 * hi_voted,
            pct_both_bad_of_all=100 * p_all,
            ci_lo_of_all=100 * lo_all, ci_hi_of_all=100 * hi_all,
        ))
        print(f"{g:22} n_total={n_total:6}  n_voted={n_voted:6}  "
              f"both_bad={n_both_bad:5}  "
              f"pct(of voted)={100*p_voted:5.1f}%  "
              f"95% CI [{100*lo_voted:.1f}, {100*hi_voted:.1f}]")
    return pd.DataFrame(rows)


# ============================================================
# PART A (added later): citation-presence crosstab
# ------------------------------------------------------------
# Motivated by an ANECDOTAL observation (2 of 3 eyeballed "both bad" battles
# in the Step-3 sanity check had zero citations on at least one side) --
# tested here as a real hypothesis against the full voted set, not assumed
# to hold just because 3 examples looked that way. Uses citations_a/b
# exactly as already present in outputs/final_analysis_derived.csv (the
# fixed per-turn-deduped-then-summed measure used everywhere else in this
# project) -- nothing recomputed. Same Wilson CI method as the rest of this
# script, no new methodology.
# ============================================================

def classify_citation_presence(df):
    cit_a, cit_b = df["citations_a"], df["citations_b"]
    n_nan = int(cit_a.isna().sum() + cit_b.isna().sum())
    if n_nan:
        print(f"[WARNING] {n_nan} NaN citation values among voted rows -- "
              "these rows cannot be classified and are excluded from the crosstab.")
    at_least_one_zero = (cit_a == 0) | (cit_b == 0)
    both_have = (cit_a > 0) & (cit_b > 0)
    group = pd.Series(np.where(at_least_one_zero, "at_least_one_zero_citation",
                               np.where(both_have, "both_have_citations", "UNCLASSIFIED")),
                      index=df.index)
    return group


def citation_presence_table(df):
    """
    Both-bad rate split by whether either side had zero visible citations,
    among VOTED battles only (both-bad is only defined where a vote
    occurred). Overall, then by primary_intent if the per-intent x
    citation-group cell meets MIN_N_FOR_CITATION_SPLIT -- flagged rather
    than silently reported if it doesn't.
    """
    print(f"\n{'='*78}\nPART A  Both-bad rate by citation presence\n{'='*78}")
    voted = df[df["category"] != "no_vote"].copy()
    voted["citation_group"] = classify_citation_presence(voted)
    n_unclassified = int((voted["citation_group"] == "UNCLASSIFIED").sum())
    if n_unclassified:
        print(f"[WARNING] {n_unclassified} voted rows unclassified by citation "
              "presence (NaN citations) -- excluded from this crosstab, included "
              "everywhere else.")
    voted = voted[voted["citation_group"] != "UNCLASSIFIED"]

    rows = []

    def _one(sub, intent_label, group_label):
        n_voted_cell = len(sub)
        n_both_bad = int((sub["category"] == "both_bad").sum())
        too_small = n_voted_cell < MIN_N_FOR_CITATION_SPLIT
        if too_small:
            p = lo = hi = float("nan")
        else:
            p, lo, hi = wilson_ci(n_both_bad, n_voted_cell)
        rows.append(dict(
            primary_intent=intent_label, citation_group=group_label,
            n_voted=n_voted_cell, n_both_bad=n_both_bad,
            pct_both_bad=np.nan if too_small else 100 * p,
            ci_lo=np.nan if too_small else 100 * lo,
            ci_hi=np.nan if too_small else 100 * hi,
            too_small_to_report=too_small,
        ))

    # overall (both citation groups pooled across all intents)
    for g in ("at_least_one_zero_citation", "both_have_citations"):
        _one(voted[voted["citation_group"] == g], "ALL", g)

    # per primary_intent
    for intent in sorted(voted["primary_intent"].dropna().unique()):
        for g in ("at_least_one_zero_citation", "both_have_citations"):
            sub = voted[(voted["primary_intent"] == intent) & (voted["citation_group"] == g)]
            _one(sub, intent, g)

    out = pd.DataFrame(rows)
    n_flagged = int(out["too_small_to_report"].sum())
    print(out.to_string(index=False))
    print(f"\n{n_flagged} of {len(out)} (intent, citation_group) cells below "
          f"MIN_N_FOR_CITATION_SPLIT={MIN_N_FOR_CITATION_SPLIT} and flagged "
          "rather than reported as a rate.")

    overall = out[out["primary_intent"] == "ALL"]
    r0, r1 = overall.iloc[0], overall.iloc[1]
    overlap = not (r0["ci_lo"] > r1["ci_hi"] or r1["ci_lo"] > r0["ci_hi"])
    print(f"\nOverall: at_least_one_zero_citation={r0['pct_both_bad']:.2f}% "
          f"[{r0['ci_lo']:.2f}, {r0['ci_hi']:.2f}]  vs  "
          f"both_have_citations={r1['pct_both_bad']:.2f}% "
          f"[{r1['ci_lo']:.2f}, {r1['ci_hi']:.2f}]  "
          f"-- CIs {'OVERLAP' if overlap else 'DO NOT OVERLAP'}")
    return out


# ============================================================
# STEP 3: sanity checks
# ============================================================

def sanity_checks(df, by_intent):
    print(f"\n{'='*78}\nSTEP 3  Sanity checks\n{'='*78}")
    total_rows = len(df)
    total_from_table = int(by_intent["n_total"].sum())
    print(f"total rows in derived table : {total_rows}")
    print(f"sum of n_total across intents (incl. missing-intent row): {total_from_table}")
    assert total_rows == total_from_table, "rows dropped across intent groups!"
    print("reconciliation OK -- no rows dropped across primary_intent groups "
          "(the single row with a missing primary_intent is its own explicit row, "
          "not silently excluded).")

    cat_counts = df["category"].value_counts()
    print(f"\noverall category counts:\n{cat_counts}")

    both_bad = df[df["category"] == "both_bad"]
    print(f"\n--- 3 example 'both bad' battles ---")
    sample = both_bad.sample(min(3, len(both_bad)), random_state=20260930)
    for _, r in sample.iterrows():
        print(f"  intent={r['primary_intent']:20} conv_type={r['conversation_type']:12} "
              f"winner_raw={r['winner']!r}  "
              f"model_a={r['base_model_a']}  model_b={r['base_model_b']}  "
              f"citations_a={r['citations_a']}  citations_b={r['citations_b']}  "
              f"len_a={r['response_length_a']}  len_b={r['response_length_b']}")


# ============================================================
# Chart
# ============================================================

def build_chart(by_intent, out_path=OUT_CHART):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # drop zero-voted cells (e.g. the single row with a missing
    # primary_intent, n_voted=0) -- nothing to plot, included in the CSV
    # and sanity-check reconciliation but not in this chart.
    sub = by_intent[by_intent["n_voted"] > 0].sort_values(
        "pct_both_bad_of_voted", ascending=False).copy()
    x = np.arange(len(sub))
    vals = sub["pct_both_bad_of_voted"].values
    err_lo = vals - sub["ci_lo_of_voted"].values
    err_hi = sub["ci_hi_of_voted"].values - vals

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x, vals, color="#d03b3b", width=0.55, edgecolor="white", linewidth=0.6)
    ax.errorbar(x, vals, yerr=[err_lo, err_hi], fmt="none", ecolor="#555555",
                elinewidth=1.3, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["group"], rotation=30, ha="right")
    ax.set_ylabel('% of voted battles judged "both bad"')
    ax.set_title('Both-bad rate by primary intent (95% Wilson CI)')
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i, r["ci_hi_of_voted"] + 0.3, f"n={int(r['n_voted'])}",
                ha="center", va="bottom", fontsize=8, color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ============================================================
# Markdown summary
# ============================================================

def write_summary_md(path, df, by_intent, by_conv_type, citation_presence):
    L = []
    A = L.append
    A("# Both-Bad Rate: A Churn/Catastrophic-Failure Signal")
    A("")
    A("## Methodology")
    A("")
    A("This is a distinct failure signal from the decisive win/loss analysis in "
      "`loss_anatomy_summary.md`: instead of asking \"which response won,\" it asks "
      "\"how often did the judge decide *neither* response was acceptable.\"")
    A("")
    A("**Step 0 finding, confirmed before building anything:** the raw `winner` field "
      "in this dataset has a genuine, separately-labeled **\"both bad\"** category, "
      "distinct from an ordinary \"both good\" tie:")
    A("")
    A("| raw `winner` value | count | meaning |")
    A("|---|---|---|")
    A("| `model_a` | 4,255 | model_a wins (decisive) |")
    A("| `model_b` | 4,358 | model_b wins (decisive) |")
    A("| `tie` | 2,502 | tie -- both judged acceptable |")
    A("| `tie (bothbad)` | 1,537 | tie -- both judged bad (**the signal this report measures**) |")
    A("| *(missing / NaN)* | 11,417 | no vote occurred |")
    A("| **total** | **24,069** | |")
    A("")
    A("This is case (a) from the brief: a real label exists, so the both-bad rate is "
      "computed directly from it -- **no proxy was built or needed.** Note that the "
      "existing pipeline (`intent_complexity_analysis_v2.py`'s "
      "`TIE_WINNERS = (\"tie\", \"tie (bothbad)\")`) collapses both tie types into one "
      "`is_tie` boolean for the citation/length analysis -- the right call for that "
      "analysis, since both are \"not decisive,\" but it throws away exactly the "
      "distinction this report needs. This script does not touch that collapsing "
      "logic; it reads the raw `winner` string column (preserved as-is in "
      "`outputs/final_analysis_derived.csv`) and draws its own, separate line.")
    A("")
    A("**Denominator, stated explicitly.** The headline rate "
      "`pct_both_bad_of_voted` is both-bad count divided by *voted* battles only "
      "(`decisive + tie + tie (bothbad)`), since \"both bad\" is a judgment that can "
      "only be made when a vote occurred -- the 11,417 no-vote conversations carry no "
      "information either way. A secondary column, `pct_both_bad_of_all`, is also "
      "reported against the full row count (including no-vote rows) for readers who "
      "want that denominator instead; the two are not interchangeable and the chart "
      "below uses `pct_both_bad_of_voted` only.")
    A("")
    A("**Interval method, stated explicitly.** No proportion confidence interval is "
      "already established elsewhere in this project -- the rest of the project only "
      "ever bootstraps Bradley-Terry coefficients, never a raw proportion. The **Wilson "
      "score interval** is used here instead of a normal (Wald) approximation because "
      "it stays well-calibrated at small n and at rates far from 50%, both of which "
      "apply to several cells below (e.g. Text Processing has only "
      f"{int(by_intent[by_intent['group']=='Text Processing']['n_voted'].iloc[0]) if 'Text Processing' in by_intent['group'].values else 0} "
      "voted battles). This is a new addition to the project's statistical toolkit, "
      "not a silent substitution for an existing method.")
    A("")
    A("**Scope note.** Unlike `loss_anatomy.py`, this analysis covers every voted "
      "battle, not just decisive ones -- ties are the subject here, not an excluded "
      "category.")
    A("")

    A("## Findings: both-bad rate by primary intent")
    A("")
    A("| primary_intent | n_voted | n_both_bad | % both bad (of voted) | 95% Wilson CI | % both bad (of all, incl. no-vote) |")
    A("|---|---|---|---|---|---|")
    sub = by_intent[by_intent["n_voted"] > 0].sort_values(
        "pct_both_bad_of_voted", ascending=False)
    for _, r in sub.iterrows():
        A(f"| {r['group']} | {int(r['n_voted']):,} | {int(r['n_both_bad']):,} | "
          f"**{r['pct_both_bad_of_voted']:.2f}%** | "
          f"[{r['ci_lo_of_voted']:.2f}%, {r['ci_hi_of_voted']:.2f}%] | "
          f"{r['pct_both_bad_of_all']:.2f}% |")
    A("")
    A(f"![Both-bad rate by intent](../{OUT_CHART})")
    A("")
    A('*Bars show the % of voted battles in each intent judged "both bad," with 95% '
      "Wilson CIs; `n` above each bar is the number of voted battles for that intent.*")
    A("")

    A("## Findings: both-bad rate by conversation type")
    A("")
    A("| conversation_type | n_voted | n_both_bad | % both bad (of voted) | 95% Wilson CI |")
    A("|---|---|---|---|---|")
    for _, r in by_conv_type.sort_values("pct_both_bad_of_voted", ascending=False).iterrows():
        A(f"| {r['group']} | {int(r['n_voted']):,} | {int(r['n_both_bad']):,} | "
          f"**{r['pct_both_bad_of_voted']:.2f}%** | "
          f"[{r['ci_lo_of_voted']:.2f}%, {r['ci_hi_of_voted']:.2f}%] |")
    A("")

    A("## Findings: both-bad rate by citation presence "
      "(testing the zero-citation hypothesis)")
    A("")
    A("**Motivation.** The Step-3 sanity check above eyeballed 3 example \"both bad\" "
      "battles; 2 of the 3 had zero citations on at least one side. This section tests "
      "whether that holds up across the full voted set, or whether it was an artifact "
      "of 3 small examples. Battles are split into two groups using `citations_a`/"
      "`citations_b` exactly as already computed in "
      "`outputs/final_analysis_derived.csv` (the fixed per-turn-deduped measure used "
      "everywhere else in this project -- nothing recomputed):")
    A("")
    A("- `at_least_one_zero_citation`: `citations_a == 0 OR citations_b == 0`")
    A("- `both_have_citations`: `citations_a > 0 AND citations_b > 0`")
    A("")
    cp_overall = citation_presence[citation_presence["primary_intent"] == "ALL"]
    r_zero = cp_overall[cp_overall["citation_group"] == "at_least_one_zero_citation"].iloc[0]
    r_both = cp_overall[cp_overall["citation_group"] == "both_have_citations"].iloc[0]
    cp_overlap = not (r_zero["ci_lo"] > r_both["ci_hi"] or r_both["ci_lo"] > r_zero["ci_hi"])
    A("**Overall (pooled across all intents):**")
    A("")
    A("| citation_group | n_voted | n_both_bad | % both bad | 95% Wilson CI |")
    A("|---|---|---|---|---|")
    A(f"| at_least_one_zero_citation | {int(r_zero['n_voted']):,} | "
      f"{int(r_zero['n_both_bad']):,} | **{r_zero['pct_both_bad']:.2f}%** | "
      f"[{r_zero['ci_lo']:.2f}%, {r_zero['ci_hi']:.2f}%] |")
    A(f"| both_have_citations | {int(r_both['n_voted']):,} | "
      f"{int(r_both['n_both_bad']):,} | **{r_both['pct_both_bad']:.2f}%** | "
      f"[{r_both['ci_lo']:.2f}%, {r_both['ci_hi']:.2f}%] |")
    A("")
    if cp_overlap:
        A(f"**The hypothesis does NOT clearly hold up at the overall level.** The two "
          f"95% CIs overlap "
          f"([{r_zero['ci_lo']:.2f}%, {r_zero['ci_hi']:.2f}%] vs "
          f"[{r_both['ci_lo']:.2f}%, {r_both['ci_hi']:.2f}%]), so the point-estimate "
          f"gap ({r_zero['pct_both_bad']:.2f}% vs {r_both['pct_both_bad']:.2f}%) is not "
          "clearly established as a real difference rather than noise. The 3-example "
          "anecdote does not generalize cleanly to the full voted set.")
    else:
        A(f"**The hypothesis holds up at the overall level.** The two 95% CIs do not "
          f"overlap: battles with at least one zero-citation side have a "
          f"**{'higher' if r_zero['pct_both_bad'] > r_both['pct_both_bad'] else 'lower'}** "
          "both-bad rate than battles where both sides cited something.")
    A("")
    A(f"**Per-intent breakdown.** Cells with fewer than "
      f"{MIN_N_FOR_CITATION_SPLIT} voted battles are flagged rather than reported as a "
      "rate (per-intent sample sizes split two ways get thin quickly):")
    A("")
    A("| primary_intent | citation_group | n_voted | % both bad | 95% Wilson CI |")
    A("|---|---|---|---|---|")
    per_intent_cp = citation_presence[citation_presence["primary_intent"] != "ALL"].sort_values(
        ["primary_intent", "citation_group"])
    for _, r in per_intent_cp.iterrows():
        if r["too_small_to_report"]:
            A(f"| {r['primary_intent']} | {r['citation_group']} | {int(r['n_voted'])} | "
              "*too few battles to report* | -- |")
        else:
            A(f"| {r['primary_intent']} | {r['citation_group']} | {int(r['n_voted']):,} | "
              f"{r['pct_both_bad']:.2f}% | [{r['ci_lo']:.2f}%, {r['ci_hi']:.2f}%] |")
    A("")
    n_flagged_cp = int(per_intent_cp["too_small_to_report"].sum())
    A(f"{n_flagged_cp} of {len(per_intent_cp)} (intent, citation_group) cells were too "
      f"thin (< {MIN_N_FOR_CITATION_SPLIT} voted battles) to report a rate for.")
    A("")

    top = sub.iloc[0]
    bottom = sub.iloc[-1]
    mt = by_conv_type[by_conv_type["group"] == "multi_turn"].iloc[0]
    st = by_conv_type[by_conv_type["group"] == "single_turn"].iloc[0]
    A("## Plain-language summary")
    A("")
    A(f"**Highest both-bad rate:** `{top['group']}` at "
      f"**{top['pct_both_bad_of_voted']:.2f}%** of voted battles "
      f"(95% CI [{top['ci_lo_of_voted']:.2f}%, {top['ci_hi_of_voted']:.2f}%], "
      f"n={int(top['n_voted']):,}) -- this is the intent category where judges most "
      "often found neither response acceptable.")
    A(f"**Lowest both-bad rate:** `{bottom['group']}` at "
      f"**{bottom['pct_both_bad_of_voted']:.2f}%** "
      f"(95% CI [{bottom['ci_lo_of_voted']:.2f}%, {bottom['ci_hi_of_voted']:.2f}%], "
      f"n={int(bottom['n_voted']):,}).")
    A("")
    mt_vs_st = ("higher" if mt['pct_both_bad_of_voted'] > st['pct_both_bad_of_voted']
               else "lower")
    overlap = not (mt['ci_lo_of_voted'] > st['ci_hi_of_voted'] or
                  st['ci_lo_of_voted'] > mt['ci_hi_of_voted'])
    A(f"**Multi-turn vs single-turn:** multi-turn conversations show a "
      f"**{mt_vs_st}** both-bad rate ({mt['pct_both_bad_of_voted']:.2f}% vs "
      f"{st['pct_both_bad_of_voted']:.2f}% for single-turn). The 95% CIs "
      f"{'overlap, so this difference is not clearly established' if overlap else 'do not overlap, a real difference'} "
      "-- read the CI columns above rather than the point estimates alone.")
    A("")
    if cp_overlap:
        A("**Zero-citation hypothesis:** did not hold up -- the 95% CIs overlap, so "
          "the 3-example anecdote does not generalize to the full voted set.")
    else:
        zero_direction = ("higher" if r_zero["pct_both_bad"] > r_both["pct_both_bad"]
                          else "lower")
        A(f"**Zero-citation hypothesis:** held up -- battles with at least one "
          f"zero-citation side have a real, CI-confirmed **{zero_direction}** "
          "both-bad rate than battles where both sides cited something.")
    A("")

    A("## Caveats")
    A("")
    A("1. **\"Both bad\" is the judge's call, not an independently verified content "
      "assessment.** This report takes the `tie (bothbad)` label at face value, "
      "consistent with how `is_decisive`/`is_tie` are already treated elsewhere in "
      "this project -- it is not re-derived from response content.")
    A("2. **No-vote conversations (11,417 of 24,069) are excluded from the headline "
      "rate.** They cannot be either confirmed or ruled out as \"both bad\" since no "
      "judgment was recorded; `pct_both_bad_of_all` is reported for transparency but "
      "treats every no-vote row as a non-both-bad, which is an assumption, not a fact.")
    A("3. **Small-n cells.** Several (intent, conversation_type) cells have limited "
      "voted battles; the Wilson CI widens accordingly and is the number to read, not "
      "the point estimate alone.")
    A("4. **No multiple-comparisons correction** is applied across the 9 intent "
      "categories or the 2 conversation-type categories reported here, consistent with "
      "how this was flagged (and later BH-corrected) for the citation/tier analyses "
      "elsewhere in this project -- it has not been corrected here.")
    A("5. **This is a rate, not a cause.** A high both-bad rate for an intent says "
      "nothing about *why* -- it could reflect harder queries, worse model coverage "
      "for that intent, or judge strictness; this report does not attempt to "
      "disambiguate those.")
    A("6. **Citation presence and both-bad are both observational.** Even where the "
      "zero-citation split shows a real CI-confirmed gap, this does not establish that "
      "missing citations *causes* a both-bad judgment -- queries where no good source "
      "exists may independently drive both zero citations and a bad judgment on both "
      "sides. No causal claim is made here.")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[saved] {path}")


def main():
    print("=" * 78)
    print("BOTH-BAD RATE ANALYSIS (A3, additive only)")
    print("=" * 78)

    df = load_and_classify()
    print(f"total rows: {len(df)}")
    print(df["category"].value_counts())

    by_intent = rate_table(df, "primary_intent")
    by_conv_type = rate_table(df, "conversation_type")

    combined = pd.concat([
        by_intent.assign(group_type="primary_intent"),
        by_conv_type.assign(group_type="conversation_type"),
    ], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"\n[saved] {OUT_CSV}")

    sanity_checks(df, by_intent)

    citation_presence = citation_presence_table(df)
    citation_presence.to_csv(OUT_CITATION_CSV, index=False)
    print(f"[saved] {OUT_CITATION_CSV}")

    build_chart(by_intent)
    write_summary_md(OUT_MD, df, by_intent, by_conv_type, citation_presence)


if __name__ == "__main__":
    main()
