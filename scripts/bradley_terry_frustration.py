"""
bradley_terry_frustration.py

Frustration as a moderator of the citation effect on Search Arena outcomes.

Pure data analysis on files already in the repo. `frustration_sample300.jsonl`
holds 300 conversations labeled frustrated / engaged / neutral at the
CONVERSATION level (user turns are byte-identical on both sides of a battle,
so frustration cannot be a per-side property -- see frustration_classify.py).
These are merged with the decisive-battle dataset already built and tested in
bradley_terry_analysis.py.

This is an ADDITIVE script: it imports and reuses fit_bt / bootstrap_bt /
build_design_model_identity / _scale_covariates / load_decisive from
bradley_terry_analysis.py UNCHANGED. No core machinery is rewritten here.

No API calls. No new data collection. Runs entirely against already-processed
local files.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bradley_terry_analysis import (   # noqa: E402  (sys.path must be set first)
    N_BOOTSTRAP,
    ALPHA,
    RANDOM_SEED,
    MIN_N_FOR_FIT,
    bootstrap_bt,
    build_design_model_identity,
    _scale_covariates,
    load_decisive,
)

FRUSTRATION_JSONL = "frustration_sample300.jsonl"
OUT_CSV = "outputs/bradley_terry_frustration_moderator.csv"
SUMMARY_MD = "outputs/bradley_terry_summary.md"

# Marker used to find and REPLACE a previously-appended section of the same
# kind on a re-run, rather than accumulating duplicate sections every time
# this script is pointed at a bigger input file.
SECTION_MARKER = "# Frustration as a Moderator of the Citation Effect"

# Total eligible population this label set is drawn from: decisive multi-turn
# battles (1,714, by the `turn` metadata field) with >= 2 actually-stored user
# turns. Verified by direct reconstruction from train.jsonl (724 of the 1,714
# have only 1 stored turn and were excluded) -- see the Step 2 population
# check run alongside this script, not re-derived here to avoid a second
# full pass over the 1.2GB raw file on every analysis run.
TOTAL_ELIGIBLE = 990

# Step 2: honesty threshold -- a group below this is reported as too small to
# support a reliable conclusion, not silently fit anyway.
MIN_GROUP_N = 50


# ============================================================
# STEP 1: load + merge
# ============================================================

def load_frustration_labels(path=FRUSTRATION_JSONL):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append(dict(idx=r["idx"], label=r.get("label")))
    df = pd.DataFrame(rows)
    return df


def merge_with_decisive(dec, frust):
    """
    `idx` in frustration_sample300.jsonl is the row index into the derived
    decisive-eligible table (verified by hand: base_model_a/b, winner,
    citations_a/b and response_length_a/b at that index match the jsonl
    record exactly for a sample of rows). `dec` preserves that same index
    (load_decisive only boolean-filters and drops NA rows -- it never calls
    reset_index), so a plain index join is correct and needs no fuzzy
    matching.
    """
    frust = frust.set_index("idx")
    matched_idx = frust.index.intersection(dec.index)
    unmatched_idx = frust.index.difference(dec.index)
    merged = dec.loc[matched_idx].copy()
    merged["frustration_label"] = frust.loc[matched_idx, "label"]
    return merged, unmatched_idx


def run_load_and_merge(input_path=FRUSTRATION_JSONL):
    print("=" * 78)
    print("STEP 1  Load frustration labels and merge with the decisive dataset")
    print("=" * 78)

    frust = load_frustration_labels(input_path)
    n_frust = len(frust)
    print(f"loaded {n_frust} records from {input_path}")

    full, dec, n_all, src = load_decisive()
    print(f"decisive dataset source: {src}")
    print(f"decisive battles available to merge against: {len(dec)}")

    merged, unmatched_idx = merge_with_decisive(dec, frust)
    n_matched = len(merged)
    print(f"\nmatched {n_matched} of {n_frust} frustration-labeled conversations "
          "to a decisive record with citation/length data")

    if len(unmatched_idx):
        print(f"[warn] {len(unmatched_idx)} unmatched -- investigating why:")
        for i in list(unmatched_idx)[:20]:
            if i in full.index:
                r = full.loc[i]
                reason = ("is_decisive=False" if not bool(r["is_decisive"])
                          else "present in full table, decisive, but dropped by "
                               "the dropna in load_decisive (missing citation, "
                               "length, model, or winner field)")
                print(f"    idx={i}: {reason}")
            else:
                print(f"    idx={i}: index not present in the derived table at all "
                      "(out of range or table rebuilt differently since labeling)")
    else:
        print(f"All {n_frust} frustration-labeled conversations matched cleanly; "
              "no unmatched records.")

    print(f"\nlabel distribution among the {n_matched} matched conversations:")
    print(merged["frustration_label"].value_counts().to_string())
    return merged, n_frust, n_matched, unmatched_idx


# ============================================================
# STEP 2: collapse to frustrated vs engaged
# ============================================================

def run_two_group_split(merged):
    print("\n" + "=" * 78)
    print("STEP 2  Collapse to frustrated vs engaged (drop neutral)")
    print("=" * 78)

    two = merged[merged["frustration_label"].isin(["frustrated", "engaged"])].copy()
    n_neutral = int((merged["frustration_label"] == "neutral").sum())
    n_frustrated = int((two["frustration_label"] == "frustrated").sum())
    n_engaged = int((two["frustration_label"] == "engaged").sum())

    print(f"dropped {n_neutral} 'neutral' conversations (no clean frustrated-vs-"
          "engaged contrast)")
    print(f"frustrated : n = {n_frustrated}")
    print(f"engaged    : n = {n_engaged}")
    print(f"total used for Steps 3-4 : {len(two)}")

    small = []
    for label, n in (("frustrated", n_frustrated), ("engaged", n_engaged)):
        if n < MIN_GROUP_N:
            small.append(label)
    if small:
        print(f"\n[FLAG] group(s) below the n={MIN_GROUP_N} reliability threshold: "
              f"{', '.join(small)}. Results below should be read as exploratory, "
              "not as a confident conclusion.")
    else:
        print(f"\nBoth groups meet the n={MIN_GROUP_N} threshold.")

    return two, n_frustrated, n_engaged, n_neutral


# ============================================================
# STEP 3: moderator analysis -- citation effect within each group
# ============================================================

def run_moderator_analysis(two, sd_cit, sd_len):
    print("\n" + "=" * 78)
    print("STEP 3  Citation effect, fit separately within frustrated and engaged")
    print("        (per-model strengths + citation diff + length diff, "
          f"{N_BOOTSTRAP:,}-rep bootstrap, same scaling as the main analysis)")
    print("=" * 78)

    rows = []
    group_results = {}
    for label in ("frustrated", "engaged"):
        sub = two[two["frustration_label"] == label].copy()
        n = len(sub)
        if n < MIN_N_FOR_FIT:
            print(f"\n{label:10} n={n} -- below MIN_N_FOR_FIT={MIN_N_FOR_FIT}, "
                  "not fit.")
            rows.append(dict(group=label, n=n, note="insufficient n for fit"))
            continue
        models = sorted(set(sub["base_model_a"]) | set(sub["base_model_b"]))
        res, cols, ref, nfail, ok, draws = bootstrap_bt(
            sub, build_design_model_identity,
            models=models, sd_cit=sd_cit, sd_len=sd_len)
        c = res["beta_citation"]
        l = res["beta_length"]
        group_results[label] = dict(res=res, cols=cols, draws=draws, n=n)
        rows.append(dict(
            group=label, n=n, n_models=len(models), reference_model=ref,
            beta_citation=c["coef"], cit_ci_lo=c["lo"], cit_ci_hi=c["hi"],
            cit_boot_p=c["p"], cit_significant=c["sig"],
            beta_length=l["coef"], len_ci_lo=l["lo"], len_ci_hi=l["hi"],
            len_boot_p=l["p"], len_significant=l["sig"],
            low_n=n < MIN_GROUP_N, converged=ok, n_boot_failed=nfail,
        ))
        print(f"\n{label:10} n={n}  models={len(models)}  reference={ref}  "
              f"converged={ok}  boot_failed={nfail}")
        print(f"  beta_citation = {c['coef']:+.4f}  95% CI [{c['lo']:+.3f}, "
              f"{c['hi']:+.3f}]  p={c['p']:.4f}  "
              f"{'SIG' if c['sig'] else 'not significant'}")
        print(f"  beta_length   = {l['coef']:+.4f}  95% CI [{l['lo']:+.3f}, "
              f"{l['hi']:+.3f}]  p={l['p']:.4f}  "
              f"{'SIG' if l['sig'] else 'not significant'}")

    # Explicit frustrated-vs-engaged contrast on the citation coefficient.
    # The two fits are on DISJOINT conversations (independent samples), so the
    # sampling distribution of the difference is validly formed by taking the
    # elementwise difference of the two independent bootstrap draw arrays --
    # unlike the correlated within-fit tier contrast elsewhere in this
    # project, no shared-draw alignment trick is needed or available here.
    contrast = None
    if "frustrated" in group_results and "engaged" in group_results:
        gf, ge = group_results["frustrated"], group_results["engaged"]
        jf = gf["cols"].index("beta_citation")
        je = ge["cols"].index("beta_citation")
        df_ = gf["draws"][:, jf]
        de_ = ge["draws"][:, je]
        df_ = df_[np.isfinite(df_)]
        de_ = de_[np.isfinite(de_)]
        m = min(len(df_), len(de_))
        if m >= 50:
            diff_draws = df_[:m] - de_[:m]
            point = float(gf["res"]["beta_citation"]["coef"]
                          - ge["res"]["beta_citation"]["coef"])
            lo = float(np.percentile(diff_draws, 100 * ALPHA / 2))
            hi = float(np.percentile(diff_draws, 100 * (1 - ALPHA / 2)))
            p = float(min(1.0, 2.0 * min(np.mean(diff_draws <= 0),
                                         np.mean(diff_draws >= 0))))
            sig = bool(lo > 0 or hi < 0)
            contrast = dict(point=point, lo=lo, hi=hi, p=p, sig=sig, n_draws=m)
            verdict = ("STRONGER in frustrated" if sig and point > 0 else
                       ("WEAKER in frustrated" if sig and point < 0 else
                        "statistically indistinguishable"))
            print(f"\nCONTRAST  beta_citation[frustrated] - beta_citation[engaged] "
                  f"= {point:+.4f}")
            print(f"          95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  "
                  f"({m} usable bootstrap pairs)")
            print(f"          => the citation effect is {verdict} between "
                  "frustrated and engaged conversations.")
            rows.append(dict(group="frustrated_minus_engaged_contrast",
                             n=min(group_results["frustrated"]["n"],
                                   group_results["engaged"]["n"]),
                             beta_citation=point, cit_ci_lo=lo, cit_ci_hi=hi,
                             cit_boot_p=p, cit_significant=sig))
        else:
            print(f"\n[warn] too few finite bootstrap pairs ({m}) to form the "
                  "frustrated-vs-engaged contrast reliably.")

    return pd.DataFrame(rows), group_results, contrast


# ============================================================
# STEP 4: frustration as a direct effect on the actual outcome
# ============================================================

def build_design_frustration_direct(df, sd_cit, sd_len):
    """
    Does being in a frustrated conversation (vs engaged) relate to the
    ACTUAL outcome (y = 1 if model_a is the decisive winner), controlling
    for citation and length differences?

    Frustration is a CONVERSATION-level label, not a per-side feature, so it
    cannot enter as a theta-style per-model strength or as a differenced
    covariate the way citations/length do. It enters as a single 0/1
    indicator (frustrated=1, engaged=0 baseline) plus an intercept, alongside
    the same citation/length difference covariates used everywhere else in
    this project. This is deliberately NOT a per-model fit -- there is no
    model-identity theta in this design -- because frustration is not being
    claimed as a property of any one model.
    """
    n = len(df)
    is_frust = (df["frustration_label"].values == "frustrated").astype(float)
    dcit = ((df["citations_a"] - df["citations_b"]).astype(float) / sd_cit).values
    dlen = ((df["response_length_a"] - df["response_length_b"]).astype(float) / sd_len).values
    X = np.column_stack([np.ones(n), is_frust, dcit, dlen])
    cols = ["intercept", "frustrated_vs_engaged", "beta_citation", "beta_length"]
    y = df["_y"].values.astype(float)
    return X, y, cols


def run_direct_effect(two, sd_cit, sd_len):
    print("\n" + "=" * 78)
    print("STEP 4  Frustration as a direct effect on the actual outcome")
    print("        (conversation-level covariate, NOT a per-model effect --")
    print("         controlling for which side had more citations / was longer)")
    print("=" * 78)

    n = len(two)
    if n < MIN_N_FOR_FIT:
        print(f"n={n} is below MIN_N_FOR_FIT={MIN_N_FOR_FIT}; not fit.")
        return None

    res, cols, _rest, nfail, ok, _draws = bootstrap_bt(
        two, build_design_frustration_direct, sd_cit=sd_cit, sd_len=sd_len)
    f = res["frustrated_vs_engaged"]
    c = res["beta_citation"]
    l = res["beta_length"]
    print(f"n={n}  converged={ok}  bootstrap replicates failed={nfail}")
    print(f"\n  frustrated_vs_engaged = {f['coef']:+.4f}  95% CI [{f['lo']:+.3f}, "
          f"{f['hi']:+.3f}]  p={f['p']:.4f}  "
          f"{'SIG' if f['sig'] else 'not significant'}")
    print(f"  beta_citation         = {c['coef']:+.4f}  95% CI [{c['lo']:+.3f}, "
          f"{c['hi']:+.3f}]  p={c['p']:.4f}")
    print(f"  beta_length           = {l['coef']:+.4f}  95% CI [{l['lo']:+.3f}, "
          f"{l['hi']:+.3f}]  p={l['p']:.4f}")
    print("\nINTERPRETATION: this coefficient asks whether the conversation-level "
          "frustration label relates to which side actually won, net of citation "
          "and length differences. It is NOT a claim that any particular model "
          "is more likely to win frustrated conversations -- frustration has no "
          "per-model identity in this dataset.")
    row = dict(group="direct_effect_on_outcome", n=n,
              frustration_coef=f["coef"], frustration_ci_lo=f["lo"],
              frustration_ci_hi=f["hi"], frustration_boot_p=f["p"],
              frustration_significant=f["sig"],
              beta_citation=c["coef"], cit_ci_lo=c["lo"], cit_ci_hi=c["hi"],
              cit_boot_p=c["p"], cit_significant=c["sig"],
              beta_length=l["coef"], len_ci_lo=l["lo"], len_ci_hi=l["hi"],
              len_boot_p=l["p"], len_significant=l["sig"],
              converged=ok, n_boot_failed=nfail)
    return row


# ============================================================
# STEP 6: write outputs
# ============================================================

def save_csv(mod_df, direct_row, out_csv=OUT_CSV):
    rows = mod_df.to_dict("records")
    if direct_row is not None:
        rows.append(direct_row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")


# Reference point from the original n=300 run (83 frustrated / 165 engaged),
# recorded here so every later, larger run can report an honest before/after
# rather than a reader having to dig up the prior CSV. Not recomputed -- this
# is a fixed historical comparison point.
ORIGINAL_300_CONTRAST = dict(point=0.0612, lo=-0.6326, hi=0.9866, p=0.92,
                             n_frustrated=83, n_engaged=165)
ORIGINAL_300_DIRECT = dict(coef=-0.0563, lo=-0.634, hi=0.506, p=0.8669)


def _truncate_existing_section(path):
    """
    Idempotency: if this script already appended a "Frustration as a
    Moderator" section on a previous (smaller-sample) run, remove it before
    appending the new one, rather than accumulating duplicate sections every
    time this is re-run against a bigger input file.

    Bounded to just THIS section's span (from its own "---" separator to the
    start of the next top-level section, or EOF if it was last) -- not from
    the marker to end-of-file. The EOF version was a real bug: if another
    script's section was appended after this one, re-running this script
    would silently delete that later content too.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    marker_idx = content.find(SECTION_MARKER)
    if marker_idx == -1:
        return
    sep = "---\n\n"
    sep_idx = content.rfind(sep, 0, marker_idx)
    start = sep_idx if sep_idx != -1 else marker_idx
    next_sep_idx = content.find("\n" + sep + "#", marker_idx + len(SECTION_MARKER))
    end = len(content) if next_sep_idx == -1 else next_sep_idx + 1
    content = content[:start].rstrip("\n") + "\n" + content[end:]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


def append_summary_section(input_path, n_frust_total, n_matched, unmatched_idx,
                           n_frustrated, n_engaged, n_neutral,
                           mod_df, contrast, direct_row, sd_cit, sd_len,
                           summary_md=SUMMARY_MD):
    too_small = n_frustrated < MIN_GROUP_N or n_engaged < MIN_GROUP_N
    # A group can clear the raw n=50 floor and still yield an uninformative
    # CI once 9 per-model strengths are estimated alongside it from a few
    # hundred binary outcomes. Call that out honestly rather than letting
    # "n >= 50" stand in for "the interval is actually narrow enough to mean
    # something". The threshold (0.5 log-odds) is set against the smallest
    # SIGNIFICANT per-intent/cost-tier beta_citation already established
    # elsewhere in this project (roughly 0.3-0.5 log-odds): a contrast CI
    # wider than that cannot rule out a moderator effect of the size this
    # project otherwise treats as real, even if it narrowed from a prior run.
    WIDE_CI_THRESHOLD = 0.5
    contrast_width = (contrast["hi"] - contrast["lo"]) if contrast is not None else float("nan")
    wide_contrast = contrast is not None and contrast_width > WIDE_CI_THRESHOLD
    n_remaining = max(0, TOTAL_ELIGIBLE - n_frust_total)
    is_partial = n_frust_total < TOTAL_ELIGIBLE

    L = []
    A = L.append
    A("")
    A("---")
    A("")
    A(SECTION_MARKER)
    A("")
    if is_partial:
        A(f"> **This is a partial sample, not the full eligible population.** "
          f"{n_frust_total} of {TOTAL_ELIGIBLE} eligible multi-turn decisive "
          f"conversations have been classified so far. The remaining "
          f"**{n_remaining}** are planned but not yet completed, blocked on a "
          "Gemini API billing/quota issue on the classification account "
          "(free-tier daily/per-minute request caps were hit; billing linkage "
          "has not yet taken effect for this project). **Results below should "
          "be read as preliminary** until the full sample is available. "
          "`frustration_classify.py --resume` will pick up exactly where this "
          "batch left off once the quota issue clears -- no completed work is "
          "redone.")
        A("")
    if too_small:
        A(f"> **Sample-size caveat, stated up front.** After collapsing to a clean "
          f"frustrated-vs-engaged contrast, the groups are **n={n_frustrated} "
          f"frustrated** and **n={n_engaged} engaged** conversations. "
          f"{'Frustrated' if n_frustrated < MIN_GROUP_N else 'Engaged'} falls below "
          f"the {MIN_GROUP_N}-conversation threshold this project uses as a floor "
          "for a reliable fit. The per-group citation coefficients below carry "
          "correspondingly wide confidence intervals. **Treat this section as "
          "exploratory, not as a confirmed finding** -- that is an honest property "
          "of a 300-conversation label budget, not a flaw in the method.")
    elif wide_contrast:
        A(f"> **Sample-size caveat, stated up front.** Both groups technically clear "
          f"this project's n={MIN_GROUP_N} floor (frustrated n={n_frustrated}, "
          f"engaged n={n_engaged}), but each fit also estimates 9 per-model "
          "strengths from that same handful of conversations, and the "
          f"frustrated-vs-engaged contrast on beta_citation comes back with a "
          f"95% CI **{contrast_width:.2f} log-odds wide** -- wider than the "
          "smallest significant citation effect found anywhere else in this "
          "project (roughly 0.3-0.5 log-odds). **The data still cannot rule out "
          "a moderator effect of the size this project otherwise treats as "
          "real.** Read everything below as exploratory, not confirmed.")
    else:
        A(f"Both groups meet this project's n={MIN_GROUP_N} reliability floor "
          f"(frustrated n={n_frustrated}, engaged n={n_engaged}), and the resulting "
          "intervals are narrow enough to support the conclusions below.")
    A("")
    A("## What this section does")
    A("")
    A(f"`{input_path}` holds {n_frust_total} conversations classified "
      "frustrated / engaged / neutral at the **conversation** level (the user's "
      "turns are identical on both sides of a battle, so frustration cannot be a "
      "per-side property -- see `frustration_classify.py`). "
      f"All {n_matched} of {n_frust_total} matched to a decisive record with "
      "citation and length data already computed"
      + (f"; {len(unmatched_idx)} did not match." if len(unmatched_idx) else
         ", with no unmatched records.") + " "
      f"Of those, {n_neutral} are labeled 'neutral' and are dropped: 'neutral' "
      "gives no clean frustrated-vs-engaged contrast. The remaining "
      f"{n_frustrated + n_engaged} conversations ({n_frustrated} frustrated, "
      f"{n_engaged} engaged) are used below.")
    A("")
    A("Both fits below reuse the exact Bradley-Terry-with-covariates "
      "specification, covariate scaling (1 SD of citation diff = "
      f"{sd_cit:.3f} citations; 1 SD of length diff = {sd_len:,.0f} characters, "
      "the SAME global scaling constants used throughout this project so "
      f"coefficients stay comparable), and {N_BOOTSTRAP:,}-replicate "
      "conversation-level bootstrap as the by-intent fits above -- only the "
      "subset of conversations changes.")
    A("")

    A("## Step 3: citation effect within each group")
    A("")
    A("| group | n | models | beta_citation | 95% CI | boot p | significant | "
      "beta_length | length sig |")
    A("|---|---|---|---|---|---|---|---|---|")
    for _, r in mod_df.iterrows():
        if r["group"] not in ("frustrated", "engaged"):
            continue
        if "note" in r and isinstance(r.get("note"), str) and pd.notna(r.get("note")):
            A(f"| {r['group']} | {int(r['n'])} | — | — | — | — | "
              f"{r['note']} | — | — |")
            continue
        A(f"| {r['group']} | {int(r['n'])} | {int(r['n_models'])} | "
          f"{r['beta_citation']:+.4f} | [{r['cit_ci_lo']:+.3f}, {r['cit_ci_hi']:+.3f}] | "
          f"{r['cit_boot_p']:.4f} | {'**yes**' if r['cit_significant'] else 'no'} | "
          f"{r['beta_length']:+.4f} | {'yes' if r['len_significant'] else 'no'} |")
    A("")

    if contrast is not None:
        verdict = ("**stronger in frustrated conversations**" if contrast["sig"] and contrast["point"] > 0 else
                   ("**weaker in frustrated conversations**" if contrast["sig"] and contrast["point"] < 0 else
                    "**statistically indistinguishable between frustrated and engaged conversations**"))
        A(f"**Frustrated − engaged contrast on beta_citation = {contrast['point']:+.4f}**, "
          f"95% CI [{contrast['lo']:+.3f}, {contrast['hi']:+.3f}], p = {contrast['p']:.4f}. "
          f"The citation effect is {verdict}.")
        A("")
        A("This contrast is formed from the elementwise difference of the two "
          "groups' independent bootstrap draw arrays (the two fits are on "
          "disjoint conversations, so this is a valid way to build the sampling "
          "distribution of the difference -- unlike the within-fit tier contrast "
          "elsewhere in this project, there is no shared-draw correlation to "
          "preserve here because there is none to begin with).")
        A("")

        if n_frust_total != 300:
            old = ORIGINAL_300_CONTRAST
            old_width = old["hi"] - old["lo"]
            narrowed_pct = 100.0 * (1.0 - contrast_width / old_width)
            A("### Compared to the original n=300 sample")
            A("")
            A("| sample | frustrated n | engaged n | contrast | 95% CI | CI width | boot p |")
            A("|---|---|---|---|---|---|---|")
            A(f"| original (n=300) | {old['n_frustrated']} | {old['n_engaged']} | "
              f"{old['point']:+.4f} | [{old['lo']:+.3f}, {old['hi']:+.3f}] | "
              f"{old_width:.3f} | {old['p']:.4f} |")
            A(f"| this run (n={n_frust_total}) | {n_frustrated} | {n_engaged} | "
              f"{contrast['point']:+.4f} | [{contrast['lo']:+.3f}, {contrast['hi']:+.3f}] | "
              f"{contrast_width:.3f} | {contrast['p']:.4f} |")
            A("")
            if contrast_width < old_width:
                A(f"The extra conversations **narrowed the CI by {narrowed_pct:.0f}%** "
                  f"({old_width:.3f} -> {contrast_width:.3f} log-odds).")
            else:
                A(f"The CI did **not narrow** -- it is {abs(narrowed_pct):.0f}% "
                  f"{'wider' if contrast_width > old_width else 'about the same'} "
                  f"than the original ({old_width:.3f} -> {contrast_width:.3f} log-odds).")
            A("")
            if wide_contrast:
                A(f"**Even after narrowing, the interval is still "
                  f"{contrast_width:.2f} log-odds wide -- still wider than the "
                  "smallest significant citation effect found elsewhere in this "
                  "project (~0.3-0.5 log-odds), so it still cannot distinguish a "
                  "null moderator effect from a real one of that size. This "
                  "partial sample moved the estimate but has not yet resolved "
                  "the question.**")
            else:
                A("**The added data brought the interval narrow enough to be "
                  "informative** -- see the verdict above.")
            A("")
    else:
        A("*(Contrast not computed -- one or both groups could not be fit; see "
          "console output / CSV.)*")
        A("")

    A("## Step 4: frustration as a direct effect on the actual outcome")
    A("")
    if direct_row is not None:
        f_sig = "**yes**" if direct_row["frustration_significant"] else "no"
        A("This is a separate, simpler question from the moderator analysis above: "
          "does the conversation being labeled 'frustrated' (vs 'engaged') relate "
          "to who actually won, controlling for which side had more citations and "
          "a longer response? Frustration has no per-side identity, so it enters "
          "as a single conversation-level 0/1 indicator (frustrated=1, "
          "engaged=0 baseline) rather than as a per-model strength -- **this is not "
          "a claim about any specific model**, only about whether the conversation's "
          "overall tenor relates to its outcome.")
        A("")
        A("| term | coefficient | 95% CI | boot p | significant |")
        A("|---|---|---|---|---|")
        A(f"| frustrated vs engaged (direct) | {direct_row['frustration_coef']:+.4f} | "
          f"[{direct_row['frustration_ci_lo']:+.3f}, {direct_row['frustration_ci_hi']:+.3f}] | "
          f"{direct_row['frustration_boot_p']:.4f} | {f_sig} |")
        A(f"| beta_citation (per 1 SD) | {direct_row['beta_citation']:+.4f} | "
          f"[{direct_row['cit_ci_lo']:+.3f}, {direct_row['cit_ci_hi']:+.3f}] | "
          f"{direct_row['cit_boot_p']:.4f} | "
          f"{'yes' if direct_row['cit_significant'] else 'no'} |")
        A(f"| beta_length (per 1 SD) | {direct_row['beta_length']:+.4f} | "
          f"[{direct_row['len_ci_lo']:+.3f}, {direct_row['len_ci_hi']:+.3f}] | "
          f"{direct_row['len_boot_p']:.4f} | "
          f"{'yes' if direct_row['len_significant'] else 'no'} |")
        A("")
        if direct_row["frustration_significant"]:
            A(f"The direct frustration term is significant "
              f"(p = {direct_row['frustration_boot_p']:.4f}): net of citation and "
              "length differences, there is a detectable association between the "
              "conversation's frustration label and which side won.")
        else:
            A(f"The direct frustration term is **not significant** "
              f"(p = {direct_row['frustration_boot_p']:.4f}, CI spans zero): once "
              "citation and length differences are controlled, there is no "
              "detectable net association between a conversation's frustration "
              "label and which side won.")
        A("")
        if n_frust_total != 300:
            od = ORIGINAL_300_DIRECT
            od_width = od["hi"] - od["lo"]
            new_width = direct_row["frustration_ci_hi"] - direct_row["frustration_ci_lo"]
            A(f"Compared to the original n=300 run: the direct-effect CI was "
              f"[{od['lo']:+.3f}, {od['hi']:+.3f}] (width {od_width:.3f}) and is now "
              f"[{direct_row['frustration_ci_lo']:+.3f}, "
              f"{direct_row['frustration_ci_hi']:+.3f}] (width {new_width:.3f}) -- "
              f"{'narrower' if new_width < od_width else 'not narrower'}, and still "
              f"{'not ' if not direct_row['frustration_significant'] else ''}"
              "significant.")
            A("")
    else:
        A("*(Not fit -- insufficient n; see console output.)*")
        A("")

    A("## Honest bottom line")
    A("")
    if is_partial:
        A(f"This is **{n_frust_total} of {TOTAL_ELIGIBLE} eligible conversations "
          f"({n_remaining} still unclassified)**, so every number above is a "
          "preliminary read, not a final one -- it should be revisited once the "
          "remaining conversations are classified and this script is re-run "
          f"against the complete {TOTAL_ELIGIBLE}-conversation set.")
        A("")
    if too_small or wide_contrast:
        A(f"With only {n_frustrated + n_engaged} usable conversations split "
          f"{n_frustrated}/{n_engaged} between frustrated and engaged -- and 9 "
          "model identities competing for signal within each subset -- this is a "
          "small-sample exploratory check, not a confirmed result. The confidence "
          "intervals above should be read at face value: where they are wide, that "
          "width is the honest answer, not a reason to round it down to a cleaner-"
          "sounding claim. **The data in hand supports only one conclusion with "
          "any confidence: it does not detect a difference in the citation effect "
          "between frustrated and engaged conversations** -- which is different "
          "from having shown there is no difference. A larger frustration-labeled "
          "sample would be needed before treating either the moderator contrast or "
          "the direct effect as established in either direction.")
    else:
        A("Both groups cleared this project's reliability floor with intervals "
          "narrow enough to be informative, so the estimates above can be read "
          "with the same confidence as the other Bradley-Terry fits in this "
          "project.")
    A("")

    _truncate_existing_section(summary_md)
    with open(summary_md, "a", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[appended] Frustration as a Moderator of the Citation Effect -> {summary_md}")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=FRUSTRATION_JSONL,
                   help="frustration-labeled jsonl to analyze "
                        f"(default: {FRUSTRATION_JSONL})")
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--summary-md", default=SUMMARY_MD)
    args = ap.parse_args()

    print("=" * 78)
    print("FRUSTRATION AS A MODERATOR OF THE CITATION EFFECT")
    print(f"input = {args.input}")
    print(f"bootstrap replications = {N_BOOTSTRAP}  seed = {RANDOM_SEED}")
    print("No API calls. No new data collection.")
    print("=" * 78)

    merged, n_frust_total, n_matched, unmatched_idx = run_load_and_merge(args.input)
    two, n_frustrated, n_engaged, n_neutral = run_two_group_split(merged)

    # Reuse the SAME global covariate scaling as the main analysis (computed
    # on the full decisive set), so coefficients here stay on identical units
    # and are directly comparable to every other BT fit in this project.
    _full, dec_all, _n_all, _src = load_decisive()
    sd_cit, sd_len = _scale_covariates(dec_all)
    print(f"\ncovariate scaling (reused from the main analysis): "
          f"1 SD citation diff = {sd_cit:.3f}; 1 SD length diff = {sd_len:.1f} chars")

    mod_df, group_results, contrast = run_moderator_analysis(two, sd_cit, sd_len)
    direct_row = run_direct_effect(two, sd_cit, sd_len)

    save_csv(mod_df, direct_row, args.out_csv)
    append_summary_section(args.input, n_frust_total, n_matched, unmatched_idx,
                           n_frustrated, n_engaged, n_neutral,
                           mod_df, contrast, direct_row, sd_cit, sd_len,
                           args.summary_md)


if __name__ == "__main__":
    main()
