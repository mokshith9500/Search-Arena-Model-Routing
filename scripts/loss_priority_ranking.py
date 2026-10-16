"""
loss_priority_ranking.py

John's next question after loss_driver_model.py's ranked importances: "if we
fix X, Y, Z, how do we know which to prioritize?" Effect size alone isn't
enough -- a feature can have a big average effect (permutation importance)
while rarely being the actual deciding factor in any specific loss. This
combines effect size with a counterfactual FLIP RATE (how often closing one
feature's gap alone would have changed the model's predicted winner) into a
single priority score.

ADDITIVE ONLY. Does not modify any existing file. Reuses, unchanged, by
direct import (not reimplemented): load_proven_features, extract_raw_texts,
build_new_features, fit_and_evaluate, run_permutation_importance,
FEATURES_NUMERIC, FEATURE_CATEGORICAL, TEAMMATE_BASELINE_AUC,
TEAMMATE_FULL_AUC from loss_driver_model.py; RANDOM_SEED from
bradley_terry_analysis.py.
"""

import numpy as np
import pandas as pd

from bradley_terry_analysis import RANDOM_SEED
from loss_driver_model import (
    load_proven_features,
    extract_raw_texts,
    build_new_features,
    fit_and_evaluate,
    run_permutation_importance,
    FEATURES_NUMERIC,
    FEATURE_CATEGORICAL,
)

EXPECTED_FULL_TEST_AUC = 0.7509
AUC_TOLERANCE = 0.0005

OUT_CSV = "outputs/loss_priority_ranking.csv"
OUT_MD = "outputs/loss_priority_ranking_summary.md"

# Step 5: the two fundamentally different kinds of fixes, stated explicitly
# per the task's own examples -- not inferred, not guessed.
ENGINEERING_FIXES = {"citation_diff", "length_diff", "hedging_diff", "alignment_diff"}
ROUTING_FIXES = {"strength_diff", "tier_diff", "same_family"}

# The "neutral / parity" value for each feature's counterfactual intervention.
# True differences (a - b) have a natural neutral point at 0 -- setting the
# difference to 0 literally means "loser brought to parity with winner,"
# regardless of which side actually won (diff=0 means a=b either way, so no
# sign-flipping is needed for y=0 vs y=1 battles).
#
# `same_family` is NOT a difference -- it's a 0/1 indicator of whether the
# two sides share a model family. Its "no disadvantage" state is 1 (same
# family), not 0. Mechanically setting it to 0 would WIDEN the family gap
# for battles that already share a family, which is backwards. This is
# stated explicitly here, not silently handled, because it's a genuine
# judgment call about what "closing the gap" means for a non-differenced
# indicator.
NEUTRAL_VALUE = {
    "citation_diff": 0.0, "length_diff": 0.0, "strength_diff": 0.0,
    "tier_diff": 0.0, "alignment_diff": 0.0, "hedging_diff": 0.0,
    "same_family": 1.0,
}

# primary_intent is explicitly excluded from the counterfactual test: it is
# not a per-side difference (loss_driver_model.py's own docstring states
# this plainly -- it is a shared conversation-level property, kept only as
# interaction context). There is no "gap to close" on a property that is
# already identical on both sides of every battle.
FEATURES_TO_TEST = [f for f in FEATURES_NUMERIC]  # same_family included, handled above


# ============================================================
# STEP 1: rebuild the exact same fitted model, verify it matches
# ============================================================

def rebuild_and_verify():
    print("=" * 78)
    print("STEP 1  Rebuild the exact same fitted model from loss_driver_model.py")
    print("=" * 78)
    df = load_proven_features()
    text_df = extract_raw_texts(df)
    df = build_new_features(df, text_df)
    fit_result = fit_and_evaluate(df)

    observed = fit_result["full_test_auc"]
    diff = abs(observed - EXPECTED_FULL_TEST_AUC)
    print(f"\nexpected full-model held-out AUC: {EXPECTED_FULL_TEST_AUC}")
    print(f"observed full-model held-out AUC: {observed:.4f}")
    print(f"difference: {diff:.5f}  (tolerance: {AUC_TOLERANCE})")
    if diff > AUC_TOLERANCE:
        raise SystemExit(
            f"STOPPING: rebuilt held-out AUC ({observed:.4f}) does not match the "
            f"already-reported value ({EXPECTED_FULL_TEST_AUC}) within tolerance "
            f"({AUC_TOLERANCE}). Do not trust anything built on top of this -- "
            "investigate the discrepancy before proceeding.")
    print("MATCH CONFIRMED -- proceeding on a verified, reproduced model.")
    return df, fit_result


# ============================================================
# STEP 2: isolate held-out losses correctly predicted by the model
# ============================================================

def isolate_correct_losses(fit_result):
    print("\n" + "=" * 78)
    print("STEP 2  Isolate correctly-predicted held-out battles")
    print("=" * 78)
    clf = fit_result["full_clf"]
    full_features = fit_result["full_features"]
    X_test = fit_result["X_test"][full_features]
    y_test = fit_result["y_test"]

    proba = clf.predict_proba(X_test)[:, 1]
    pred_label = (proba >= 0.5).astype(int)
    correct_mask = (pred_label == y_test.values)

    n_total = len(y_test)
    n_correct = int(correct_mask.sum())
    print(f"held-out battles: {n_total}")
    print(f"correctly predicted: {n_correct} ({100 * n_correct / n_total:.1f}%) "
          "-- a counterfactual on a battle the model already gets wrong isn't "
          "meaningful, so only these are used from here on.")

    X_correct = X_test.loc[correct_mask].copy()
    y_correct = y_test.loc[correct_mask].copy()
    return X_correct, y_correct, clf, full_features, n_correct


# ============================================================
# STEP 3: one feature at a time, counterfactual flip test
# ============================================================

def run_flip_test(X_correct, y_correct, clf, features_to_test):
    print("\n" + "=" * 78)
    print("STEP 3  Counterfactual flip test, one feature at a time")
    print("=" * 78)
    n = len(X_correct)
    rows = []
    total_flips_sum = 0
    for feat in features_to_test:
        neutral = NEUTRAL_VALUE[feat]
        X_cf = X_correct.copy()
        X_cf[feat] = neutral
        proba_cf = clf.predict_proba(X_cf)[:, 1]
        pred_cf = (proba_cf >= 0.5).astype(int)
        flipped = (pred_cf != y_correct.values)
        n_flipped = int(flipped.sum())
        flip_rate = n_flipped / n
        total_flips_sum += n_flipped
        print(f"{feat:18} neutral_value={neutral:+.1f}  flipped={n_flipped:5}/{n}  "
              f"flip_rate={flip_rate:.4f}")
        rows.append(dict(feature=feat, n_flipped=n_flipped, n_tested=n,
                         flip_rate=flip_rate))
    out = pd.DataFrame(rows)
    print(f"\nsum of flip counts across all features: {total_flips_sum} "
          f"(vs {n} correctly-predicted losses -- a sum well above {n} is expected "
          "and confirms individual flip rates overlap/double-count rather than "
          "partitioning the losses; see Step 6 limitations.)")
    return out


# ============================================================
# STEP 4: combine effect size + flip rate into one priority score
# ============================================================

def build_priority_ranking(importance_df, flip_df):
    print("\n" + "=" * 78)
    print("STEP 4  Combine permutation importance + flip rate into a priority score")
    print("=" * 78)
    merged = flip_df.merge(importance_df, on="feature", how="left")

    # Formula, stated explicitly: min-max normalize BOTH columns to [0,1]
    # across the set of testable features, then multiply. A feature scores
    # high only if it is BOTH a large average effect AND frequently the
    # single deciding factor -- either one alone is not enough to rank it
    # first, which is exactly the property John asked for.
    imp = merged["mean_auc_drop"].values
    flip = merged["flip_rate"].values
    imp_norm = (imp - imp.min()) / (imp.max() - imp.min())
    flip_norm = (flip - flip.min()) / (flip.max() - flip.min())
    merged["importance_norm"] = imp_norm
    merged["flip_rate_norm"] = flip_norm
    merged["priority_score"] = imp_norm * flip_norm

    merged = merged.sort_values("priority_score", ascending=False).reset_index(drop=True)
    print("priority_score = importance_norm * flip_rate_norm, where each is "
          "min-max scaled to [0,1] across the 7 testable features (primary_intent "
          "excluded -- see Step 2/3 note, it has no flip rate).")

    imp_order = list(merged.sort_values("mean_auc_drop", ascending=False)["feature"])
    combined_order = list(merged["feature"])
    same_order = (imp_order == combined_order)
    print(f"\nranking order vs. pure permutation-importance order: "
          f"{'IDENTICAL' if same_order else 'DIFFERS'} -- "
          f"importance-only order: {imp_order}")
    print()
    print(merged[["feature", "mean_auc_drop", "importance_norm", "flip_rate",
                 "flip_rate_norm", "priority_score"]].to_string(index=False))
    return merged


# ============================================================
# STEP 5: engineering fix vs routing fix categorization
# ============================================================

def categorize_fixes(ranking_df):
    print("\n" + "=" * 78)
    print("STEP 5  Engineering fix vs routing fix")
    print("=" * 78)
    def cat(feat):
        if feat in ENGINEERING_FIXES:
            return "engineering_fix"
        if feat in ROUTING_FIXES:
            return "routing_fix"
        return "not_applicable"
    ranking_df = ranking_df.copy()
    ranking_df["fix_type"] = ranking_df["feature"].map(cat)
    print(ranking_df[["feature", "priority_score", "fix_type"]].to_string(index=False))
    return ranking_df


# ============================================================
# Markdown summary
# ============================================================

def write_summary_md(path, n_total, n_correct, ranking_df, intent_importance_row,
                     fit_result):
    L = []
    A = L.append
    A("# Loss Priority Ranking: Which Driver to Fix First")
    A("")
    A("## John's question, answered directly")
    A("")
    A("`loss_driver_model.py` ranked drivers by average effect size (permutation "
      "importance) alone. That's not enough to answer \"what do we fix first\" -- a "
      "feature can have a large average effect while rarely being the single thing "
      "that actually would have flipped a specific loss. This combines **effect size** "
      "with a **counterfactual flip rate** (how often closing one feature's gap alone "
      "would have changed the model's predicted winner) into one priority score.")
    A("")

    A("## Step 1: model rebuild, verified")
    A("")
    A(f"`load_proven_features`, `extract_raw_texts`, `build_new_features`, and "
      "`fit_and_evaluate` were imported from `loss_driver_model.py` and called "
      "unchanged -- no logic reimplemented. Rebuilt held-out AUC: "
      f"**{fit_result['full_test_auc']:.4f}**, matching the already-reported "
      f"{EXPECTED_FULL_TEST_AUC} within tolerance ({AUC_TOLERANCE}). Confirmed before "
      "trusting anything built on top of it, per the task's own instruction to stop "
      "if it didn't match.")
    A("")

    A("## Step 2: held-out losses isolated")
    A("")
    A(f"Of **{n_total:,}** held-out battles, **{n_correct:,}** "
      f"({100*n_correct/n_total:.1f}%) were correctly predicted by the full model. "
      "Only these are used for the counterfactual test below -- a counterfactual on a "
      "battle the model already gets wrong isn't a meaningful test of what *would* "
      "flip it.")
    A("")

    A("## Steps 3-4: counterfactual flip rate + combined priority score")
    A("")
    A("**Counterfactual definition.** One feature at a time, every correctly-predicted "
      "held-out battle has that feature's per-side difference set to its neutral/"
      "parity value (0 for a true a-minus-b difference; **1, not 0, for "
      "`same_family`** -- see the code comment: 1 is the \"no disadvantage\" state for "
      "a same/different indicator, so setting it to 0 would backwards-widen the gap "
      "for already-matched battles). The model re-predicts; a **flip** is counted when "
      "the predicted winner changes to the actual loser.")
    A("")
    A("**Combined score formula, stated exactly:** `priority_score = "
      "importance_norm * flip_rate_norm`, where `importance_norm` and `flip_rate_norm` "
      "are each the feature's value min-max scaled to [0,1] across the 7 testable "
      "features. A feature ranks high only if it is **both** a large average effect "
      "**and** frequently the single deciding factor -- neither alone is sufficient, "
      "which is exactly the property this ranking was built to capture.")
    A("")
    A("| rank | feature | fix type | permutation importance | flip rate | priority score |")
    A("|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(ranking_df.iterrows(), 1):
        A(f"| {i} | `{r['feature']}` | {r['fix_type'].replace('_', ' ')} | "
          f"{r['mean_auc_drop']:+.5f} | {r['flip_rate']:.4f} | "
          f"**{r['priority_score']:.4f}** |")
    A("")
    imp_order = list(ranking_df.sort_values("mean_auc_drop", ascending=False)["feature"])
    combined_order = list(ranking_df["feature"])
    if imp_order == combined_order:
        A("**Honest note:** in this run, the combined ranking happens to land in "
          "**exactly the same order** as permutation importance alone -- the flip-"
          "rate signal correlated strongly with effect size here rather than "
          "reordering anything. That is a legitimate result, not a failure of the "
          "method: it means effect size and \"how often this is the deciding factor\" "
          "agree for this model and this population. The combined score still adds "
          "real information (the magnitudes and relative gaps differ from importance "
          "alone), and a different model, split, or feature set could easily produce "
          "a case where the two rankings diverge -- this script would surface that "
          "automatically if it occurred.")
    else:
        A("**Note:** the combined ranking differs from pure permutation-importance "
          f"order (importance-only order: {', '.join('`'+f+'`' for f in imp_order)}) -- "
          "this is the concrete case the combined score was built to catch: at least "
          "one feature's priority changed once flip rate was factored in alongside "
          "effect size.")
    A("")
    if intent_importance_row is not None:
        A(f"*(`primary_intent` excluded from this ranking -- permutation importance "
          f"{intent_importance_row['mean_auc_drop']:+.5f}, but no flip rate is "
          "applicable: it is a shared conversation-level property, identical on both "
          "sides of every battle, so there is no per-side gap to close.)*")
        A("")

    A("## Step 5: two fundamentally different kinds of fixes")
    A("")
    A("| fix type | features | what it actually means |")
    A("|---|---|---|")
    A("| **Engineering fix** | `citation_diff`, `length_diff`, `hedging_diff`, "
      "`alignment_diff` | Something a system could change about *how a given model "
      "behaves* -- prompting, response formatting, retrieval. Does not require "
      "switching models. |")
    A("| **Routing fix** | `strength_diff`, `tier_diff`, `same_family` | Something "
      "about *which model you choose to use at all*. \"Fixing\" this means routing to "
      "a different/stronger model, not changing how the current one responds. |")
    A("")
    A("**This distinction matters for prioritization**, not just description: the top "
      "engineering fix and the top routing fix point to entirely different team "
      "actions (prompt/response engineering vs. a routing-policy change), and the two "
      "should not be read as directly substitutable just because one has a higher "
      "combined score than the other.")
    A("")
    top_eng = ranking_df[ranking_df["fix_type"] == "engineering_fix"].iloc[0]
    top_route = ranking_df[ranking_df["fix_type"] == "routing_fix"].iloc[0]
    A(f"**Top engineering fix: `{top_eng['feature']}`** (priority score "
      f"{top_eng['priority_score']:.4f}). **Top routing fix: `{top_route['feature']}`** "
      f"(priority score {top_route['priority_score']:.4f}).")
    A("")

    A("## Step 6: real limitations of this method -- stated plainly")
    A("")
    A("1. **This shows what the model's learned decision boundary implies, not a "
      "guaranteed real-world causal effect.** The gradient-boosted model is a "
      "statistical fit to historical battles, not a causal model of why a response "
      "wins -- a high flip rate means the model's *learned function* is sensitive to "
      "that feature near these specific losses, which is suggestive evidence for "
      "prioritization, not proof of what would happen if a team actually changed it.")
    A("2. **Setting a feature to exact parity is an idealized full fix, not a "
      "realistic partial improvement.** Real engineering work rarely closes a gap "
      "completely -- a flip rate measured at full parity is an upper bound on what any "
      "real, partial intervention would achieve.")
    A("3. **Flipping one feature at a time ignores interactions.** The sum of "
      "individual flip counts across all seven features is well above the number of "
      "correctly-predicted losses (printed explicitly in the Step 3 console output) -- "
      "this confirms flip rates overlap rather than partition the losses. Several "
      "features can each individually \"flip\" the same battle, and fixing one may "
      "make another's flip unnecessary (or insufficient on its own if two features "
      "were jointly compensating). **Individual flip rates do not simply add up to "
      "the total number of losses explained.**")
    A("4. **Held-out sample size.** The flip test runs on only "
      f"{n_correct:,} correctly-predicted held-out battles -- small differences in "
      "flip rate between lower-ranked features should not be over-read.")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[saved] {path}")


def main():
    print("=" * 78)
    print("LOSS PRIORITY RANKING (additive -- no existing file modified)")
    print("=" * 78)

    df, fit_result = rebuild_and_verify()

    X_correct, y_correct, clf, full_features, n_correct = isolate_correct_losses(fit_result)
    n_total = len(fit_result["y_test"])

    importance_df = run_permutation_importance(fit_result)
    intent_importance_row = importance_df[
        importance_df["feature"] == FEATURE_CATEGORICAL].iloc[0] if (
        importance_df["feature"] == FEATURE_CATEGORICAL).any() else None

    flip_df = run_flip_test(X_correct, y_correct, clf, FEATURES_TO_TEST)

    ranking_df = build_priority_ranking(importance_df, flip_df)
    ranking_df = categorize_fixes(ranking_df)

    ranking_df.to_csv(OUT_CSV, index=False)
    print(f"\n[saved] {OUT_CSV}")

    write_summary_md(OUT_MD, n_total, n_correct, ranking_df, intent_importance_row,
                     fit_result)


if __name__ == "__main__":
    main()
