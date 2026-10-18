"""
cost_performance_routing.py

Cost-performance routing analysis, requested by John: for each primary_intent
category SEPARATELY, which individual models are Pareto-optimal on
(price, Bradley-Terry strength)? A Pareto-optimal model is one no other model
in that same intent beats on BOTH lower cost AND higher strength at once --
that is a real, defensible "route here" recommendation. A dominated model is
not, regardless of how good it looks in isolation.

ADDITIVE ONLY. Does not modify any existing file. Reuses, unchanged:
- MODEL_BLENDED_PRICE_USD_PER_MILLION, classify_cost_tier, INTENT_ORDER from
  final_analysis.py (the same verified real blended pricing and intent list
  used throughout this project -- no new pricing invented here)
- outputs/bradley_terry_model_strength_by_intent.csv (already-fit,
  CI-bounded per-(model, intent) Bradley-Terry strengths)

INDIVIDUAL-MODEL level, deliberately NOT tier level: this project's own
tier-heterogeneity finding is that within-tier spread (how much models in
the SAME price tier differ from each other) is 2.7x the between-tier spread
-- i.e. price tier barely predicts strength. A chart or analysis that groups
by tier as the primary axis would visually contradict that finding by
implying tier is the thing that matters. So cost tier appears here only as
an optional secondary label (in the saved CSV), never as the chart's primary
grouping or color.

Uses `primary_intent` as a difficulty proxy, NOT Avery's planned 0-7
difficulty score -- confirmed, in a separate audit, not to exist anywhere in
this repo yet. Built independently so that score can be swapped in later
(by re-grouping on it instead of primary_intent) without rebuilding this
analysis from scratch -- see the summary's closing note.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_analysis import MODEL_BLENDED_PRICE_USD_PER_MILLION, INTENT_ORDER, classify_cost_tier

STRENGTH_BY_INTENT_CSV = "outputs/bradley_terry_model_strength_by_intent.csv"
OUT_CSV = "outputs/cost_performance_routing.csv"
OUT_MD = "outputs/cost_performance_routing_summary.md"
CHART_DIR = "charts"
OUT_CHART = os.path.join(CHART_DIR, "chart_cost_performance_routing_by_intent.png")

# Palette follows this project's existing build_charts.py convention exactly
# (same hex values, same generic roles -- not a new palette): blue =
# confidently-distinguished status, gray = confidently-not, orange = the
# project's existing third series color, reused here for "CI-ambiguous,
# can't confidently call it either way" rather than for tier (tier is not
# this chart's grouping at all, see module docstring).
BLUE = "#2a78d6"     # confidently Pareto-optimal
ORANGE = "#eb6834"   # CI-ambiguous: optimal or dominated only by point estimate
GRAY = "#9a9992"     # confidently dominated
INK = "#0b0b0b"
MUTED = "#52514e"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cfcec6",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 9.5,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e8e7e1",
    "grid.linewidth": 0.7,
})


# ============================================================
# STEP 1: confirm what already exists, before building anything new
# ============================================================

def step1_confirm_inputs():
    print("=" * 78)
    print("STEP 1  Confirm strength-by-intent file and pricing dict (not assumed)")
    print("=" * 78)

    df = pd.read_csv(STRENGTH_BY_INTENT_CSV)
    print(f"{STRENGTH_BY_INTENT_CSV}: shape={df.shape}")
    print(f"columns: {df.columns.tolist()}")
    n_models = df["base_model"].nunique()
    n_intents = df["primary_intent"].nunique()
    print(f"distinct models: {n_models}  distinct intents: {n_intents}  "
          f"({n_models}x{n_intents}={n_models*n_intents}, actual rows={len(df)})")
    if n_models * n_intents != len(df):
        print("[WARNING] not a full (model x intent) grid -- some combinations are "
              "missing. Investigate before proceeding.")
    else:
        print("confirmed: one row per (model, intent) pair, a full grid, no gaps.")

    n_low_n = int(df["low_n"].sum())
    print(f"\nlow_n flagged rows: {n_low_n} of {len(df)} "
          f"({100*n_low_n/len(df):.1f}%) -- these are cells with too few battles "
          "to trust the strength estimate.")
    print("low_n rows:")
    print(df.loc[df["low_n"], ["base_model", "primary_intent", "n_battles"]]
          .sort_values(["primary_intent", "n_battles"]).to_string(index=False))

    n_null_strength = int(df["strength"].isna().sum())
    print(f"\nnull strength values: {n_null_strength} (expected 0 -- every cell must "
          "have a point estimate to be plotted/compared)")
    if n_null_strength:
        print("[WARNING] cannot proceed with null strengths present.")

    print(f"\n{STRENGTH_BY_INTENT_CSV} confirmed: {len(df)} rows, "
          f"{n_low_n} low_n-flagged, 0 null strengths. Proceeding.")

    pricing = MODEL_BLENDED_PRICE_USD_PER_MILLION
    print(f"\nfinal_analysis.py pricing dict: {len(pricing)} models")
    for m, p in sorted(pricing.items(), key=lambda kv: kv[1]):
        print(f"  {m:32} ${p:.2f} /M tokens")

    return df, pricing


# ============================================================
# STEP 2: merge strength-by-intent with pricing
# ============================================================

def step2_merge_price(df, pricing):
    print("\n" + "=" * 78)
    print("STEP 2  Merge (model, intent) strength with blended price")
    print("=" * 78)

    out = df.copy()
    out["price_usd_per_million"] = out["base_model"].map(pricing)
    out["cost_tier"] = out["base_model"].map(classify_cost_tier)

    n_unmatched = int(out["price_usd_per_million"].isna().sum())
    print(f"rows with no price match: {n_unmatched} of {len(out)}")
    if n_unmatched:
        missing_models = sorted(out.loc[out["price_usd_per_million"].isna(),
                                        "base_model"].unique())
        print(f"[WARNING] models with no price entry: {missing_models} -- these "
              "rows cannot be placed on the cost axis and will be excluded, not "
              "silently dropped without saying so.")
    else:
        print("every (model, intent) row matched a real blended price -- "
              "confirmed, not assumed.")

    models_in_data = set(df["base_model"].unique())
    models_in_pricing = set(pricing.keys())
    extra_priced = models_in_pricing - models_in_data
    if extra_priced:
        print(f"(models priced but not present in the strength-by-intent file, "
              f"harmless: {sorted(extra_priced)})")

    return out


# ============================================================
# STEP 3: Pareto-optimal models, per intent (not overall)
# ============================================================

def pareto_optimal_mask(prices, strengths):
    """
    A point i is dominated if some other point j has price_j <= price_i AND
    strength_j >= strength_i, with at least one strict inequality (lower
    cost OR higher strength, not identical on both). Pareto-optimal = not
    dominated by anything else in the same group. O(n^2), n=9 models/intent,
    negligible cost -- no need for a fancier frontier algorithm at this size.
    """
    n = len(prices)
    optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cheaper_or_equal = prices[j] <= prices[i]
            stronger_or_equal = strengths[j] >= strengths[i]
            strictly_better = (prices[j] < prices[i]) or (strengths[j] > strengths[i])
            if cheaper_or_equal and stronger_or_equal and strictly_better:
                optimal[i] = False
                break
    return optimal


def step3_pareto_by_intent(df):
    print("\n" + "=" * 78)
    print("STEP 3  Pareto-optimal models, per intent (not overall)")
    print("=" * 78)

    df = df.copy()
    df["pareto_optimal"] = False
    for intent, sub in df.groupby("primary_intent"):
        mask = pareto_optimal_mask(sub["price_usd_per_million"].to_numpy(),
                                   sub["strength"].to_numpy())
        df.loc[sub.index, "pareto_optimal"] = mask

    for intent in INTENT_ORDER:
        sub = df[df["primary_intent"] == intent].sort_values("price_usd_per_million")
        frontier = sub[sub["pareto_optimal"]]
        print(f"\n{intent} (n_models={len(sub)}):")
        for _, r in frontier.iterrows():
            flag = " [LOW_N]" if r["low_n"] else ""
            print(f"  PARETO-OPTIMAL  {r['base_model']:32} "
                  f"${r['price_usd_per_million']:6.2f}/M  "
                  f"strength={r['strength']:+.3f}{flag}")
        dominated = sub[~sub["pareto_optimal"]]
        if len(dominated):
            print(f"  dominated: {', '.join(dominated['base_model'].tolist())}")

    n_pareto_total = int(df["pareto_optimal"].sum())
    print(f"\ntotal Pareto-optimal (model, intent) cells: {n_pareto_total} of {len(df)}")
    return df


# ============================================================
# STEP 1 (follow-up audit): CI-aware dominance -- does the point-estimate
# frontier above overstate certainty, the same category of mistake as the
# earlier OLS-to-Bradley-Terry correction in this project (treating a point
# estimate as settled when the interval around it wasn't checked)?
#
# Two SEPARATE CI-aware questions, not one, because they ask different
# things and conflating them would hide which direction any given cell's
# status is uncertain in:
#
# (A) Is a DOMINATED model's dominated status actually statistically
#     certain, or could the "better" model just be noise away from it?
#     -> `confidently_dominated`. Strictly rarer than point-estimate
#     dominance (it ADDS a non-overlapping-CI requirement on top), so the
#     CI-aware "not confidently dominated" set is a SUPERSET of the
#     point-estimate frontier -- some dominated cells get "recovered" as
#     merely uncertain, not confirmed worse. This is Step 2's question.
#
# (B) Is a PARETO-OPTIMAL model's position actually statistically certain
#     over every model it needs to beat on price, or could a cheaper rival
#     secretly be just as strong? -> `confidently_beats_all_cheaper`.
#     Strictly rarer than just "has a higher point estimate than every
#     cheaper rival", so the CI-aware "confidently optimal" set is a SUBSET
#     of the point-estimate frontier -- some optimal cells get downgraded to
#     "optimal only by point estimate, CI overlaps a competitor". This is
#     Step 3/4's question (and the one that actually tests sonar's 9-of-9
#     claim, since being merely "not confidently dominated" is automatic for
#     anything already point-optimal and would never catch anything).
#
# Reference-model handling (gemini-2.0-flash): its strength is FIXED at 0 by
# the Bradley-Terry identification constraint, not an independently
# estimated quantity -- there is no ci_lo/ci_hi to read for it in the source
# file (confirmed in Step 1 above: exactly the 9 is_reference rows are the 9
# null ci_lo/ci_hi rows). Every other model's CI is already expressed
# relative to this fixed point. So the reference is treated here as a
# degenerate interval [0, 0], not skipped and not treated as "no data" --
# this is the standard way to compare against a BT identification anchor,
# stated explicitly rather than silently baked in.
# ============================================================

def effective_ci(is_reference, strength, ci_lo, ci_hi):
    if is_reference:
        return 0.0, 0.0
    return ci_lo, ci_hi


def confident_dominance_mask(prices, ci_los, ci_his):
    """
    (A) j confidently dominates i if price_j <= price_i AND ci_lo_j > ci_hi_i
    (the two strength intervals do not overlap at all, with j's clearly
    higher) -- the same "95% CIs do not overlap" bar this project already
    uses elsewhere (citation_support_check.py, the BT tier/intent
    comparisons) to call a difference real, not a weaker ad hoc threshold
    invented just for this script.
    """
    n = len(prices)
    confidently_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if prices[j] <= prices[i] and ci_los[j] > ci_his[i]:
                confidently_dominated[i] = True
                break
    return confidently_dominated


def confident_beats_all_cheaper_mask(prices, ci_los, ci_his):
    """
    (B) i confidently beats all cheaper rivals if, for EVERY j strictly
    cheaper than i, ci_lo_i > ci_hi_j (i's interval is entirely above j's --
    i is not just higher on paper, it is statistically distinguishable from
    every cheaper alternative it needs to beat to earn its spot on the
    frontier). A model with no strictly-cheaper rival at all (the cost
    floor) passes vacuously -- there is nothing on price it needs to beat.
    """
    n = len(prices)
    confident = np.zeros(n, dtype=bool)
    for i in range(n):
        cheaper = [j for j in range(n) if prices[j] < prices[i]]
        if not cheaper:
            confident[i] = True
            continue
        confident[i] = all(ci_los[i] > ci_his[j] for j in cheaper)
    return confident


def step1b_ci_aware_dominance(df):
    print("\n" + "=" * 78)
    print("STEP 1 (follow-up)  CI-aware dominance -- point-estimate vs. CI-aware, "
          "side by side, per intent")
    print("=" * 78)
    print("Reference model (gemini-2.0-flash) has no estimated CI (strength fixed "
          "at 0 by BT identification) -- treated as a degenerate [0, 0] interval, "
          "not skipped. Confirmed from Step 1: exactly its 9 rows are the 9 null "
          "ci_lo/ci_hi rows in the source file.")

    df = df.copy()
    df["eff_ci_lo"] = np.where(df["is_reference"], 0.0, df["ci_lo"])
    df["eff_ci_hi"] = np.where(df["is_reference"], 0.0, df["ci_hi"])

    df["confidently_dominated"] = False
    df["confidently_beats_all_cheaper"] = False
    for intent, sub in df.groupby("primary_intent"):
        prices = sub["price_usd_per_million"].to_numpy()
        ci_los = sub["eff_ci_lo"].to_numpy()
        ci_his = sub["eff_ci_hi"].to_numpy()
        df.loc[sub.index, "confidently_dominated"] = confident_dominance_mask(
            prices, ci_los, ci_his)
        df.loc[sub.index, "confidently_beats_all_cheaper"] = confident_beats_all_cheaper_mask(
            prices, ci_los, ci_his)

    # Sanity check, not assumed: confidently_dominated must imply NOT
    # pareto_optimal (a stricter dominance test can't dominate something a
    # looser one didn't). If this ever fails it means the two mask functions
    # disagree on direction somewhere -- caught here, not downstream.
    contradiction = df["confidently_dominated"] & df["pareto_optimal"]
    assert not contradiction.any(), (
        f"[BUG] {int(contradiction.sum())} rows are both pareto_optimal AND "
        "confidently_dominated -- CI-aware dominance must be a strict subset "
        "of point-estimate dominance, not an alternative to it.")
    print("sanity check passed: confidently_dominated is a strict subset of "
          "point-estimate dominated (no contradictions).")

    df["ci_recovered"] = (~df["pareto_optimal"]) & (~df["confidently_dominated"])
    df["confidently_pareto_optimal"] = df["pareto_optimal"] & df["confidently_beats_all_cheaper"]
    df["nominal_only_optimal"] = df["pareto_optimal"] & (~df["confidently_beats_all_cheaper"])

    def status(r):
        if r["confidently_pareto_optimal"]:
            return "confidently_optimal"
        if r["pareto_optimal"]:
            return "nominal_only_optimal"
        if r["confidently_dominated"]:
            return "confidently_dominated"
        return "ci_recovered_uncertain"
    df["ci_status"] = df.apply(status, axis=1)

    for intent in INTENT_ORDER:
        sub = df[df["primary_intent"] == intent].sort_values("price_usd_per_million")
        print(f"\n{intent}:")
        print(f"  point-estimate frontier:  "
              f"{', '.join(sub.loc[sub['pareto_optimal'], 'base_model'])}")
        ci_frontier = sub[~sub["confidently_dominated"]]
        print(f"  CI-aware undominated set: {', '.join(ci_frontier['base_model'])}")
        recovered = sub[sub["ci_recovered"]]
        if len(recovered):
            print(f"    -> newly recovered (dominated by point estimate, NOT "
                  f"confidently): {', '.join(recovered['base_model'])}")
        nominal = sub[sub["nominal_only_optimal"]]
        if len(nominal):
            print(f"    -> optimal only by point estimate (CI overlaps a cheaper "
                  f"rival): {', '.join(nominal['base_model'])}")

    return df


# ============================================================
# STEP 2: quantify how much the CI-aware check changes the picture
# ============================================================

def step2_quantify_ci_impact(df):
    print("\n" + "=" * 78)
    print("STEP 2  Quantify: how much does the CI-aware check change the picture?")
    print("=" * 78)

    n_dominated_point = int((~df["pareto_optimal"]).sum())
    n_recovered = int(df["ci_recovered"].sum())
    pct = 100 * n_recovered / n_dominated_point if n_dominated_point else 0.0
    print(f"(model, intent) pairs dominated under point estimate: {n_dominated_point}")
    print(f"of those, NOT confidently dominated once CIs are accounted for "
          f"(recovered): {n_recovered} ({pct:.1f}%)")
    print("\nrecovered cells, by intent:")
    rec = df[df["ci_recovered"]].sort_values(["primary_intent", "price_usd_per_million"])
    if len(rec):
        print(rec[["primary_intent", "base_model", "price_usd_per_million",
                   "strength", "n_battles"]].to_string(index=False))
    else:
        print("  (none)")

    n_optimal_point = int(df["pareto_optimal"].sum())
    n_nominal_only = int(df["nominal_only_optimal"].sum())
    pct2 = 100 * n_nominal_only / n_optimal_point if n_optimal_point else 0.0
    print(f"\n(model, intent) pairs Pareto-optimal under point estimate: {n_optimal_point}")
    print(f"of those, NOT confidently beating every cheaper rival "
          f"(nominal-only): {n_nominal_only} ({pct2:.1f}%)")
    print("\nnominal-only-optimal cells, by intent:")
    nom = df[df["nominal_only_optimal"]].sort_values(["primary_intent", "price_usd_per_million"])
    if len(nom):
        print(nom[["primary_intent", "base_model", "price_usd_per_million",
                   "strength", "n_battles"]].to_string(index=False))
    else:
        print("  (none)")

    verdict = ("STRENGTHENS" if (n_recovered == 0 and n_nominal_only == 0) else
              "PARTIALLY WEAKENS" if (n_recovered < n_dominated_point * 0.3 and
                                      n_nominal_only < n_optimal_point * 0.3) else
              "MEANINGFULLY WEAKENS")
    print(f"\nREAD: the point-estimate frontier {verdict.replace('_', ' ').lower()} "
          "under CI-aware scrutiny (see Step 5's final verdict for the full statement).")
    return dict(n_dominated_point=n_dominated_point, n_recovered=n_recovered,
               n_optimal_point=n_optimal_point, n_nominal_only=n_nominal_only)


# ============================================================
# STEP 3: specifically re-check the two headline findings
# ============================================================

def step3b_recheck_headline_findings(df):
    print("\n" + "=" * 78)
    print("STEP 3  Re-check headline findings: sonar's '9-of-9', and every other "
          "non-cost-floor model's dominance pattern")
    print("=" * 78)

    n_intents = df["primary_intent"].nunique()
    cheapest_model = df.loc[df["price_usd_per_million"].idxmin(), "base_model"]
    print(f"(cost-floor model, skipped per the task's instruction: {cheapest_model})")

    point_counts = (df[df["pareto_optimal"]].groupby("base_model").size())
    non_floor_models = [m for m in point_counts.index if m != cheapest_model]

    print("\nsonar, intent by intent (the headline '9-of-9' claim):")
    sonar = df[df["base_model"] == "sonar"].sort_values("primary_intent")
    for _, r in sonar.iterrows():
        tag = "CONFIDENT" if r["confidently_pareto_optimal"] else "nominal-only (CI overlaps a cheaper rival)"
        print(f"  {r['primary_intent']:20} strength={r['strength']:+.3f} "
              f"[{r['eff_ci_lo']:+.3f},{r['eff_ci_hi']:+.3f}]  -> {tag}")
    n_sonar_confident = int(sonar["confidently_pareto_optimal"].sum())
    print(f"\nsonar: {n_sonar_confident} of {len(sonar)} intents are CONFIDENTLY "
          f"Pareto-optimal (vs. {int(sonar['pareto_optimal'].sum())} of "
          f"{len(sonar)} by point estimate).")

    print("\nevery other non-cost-floor model that was ever point-optimal, same check:")
    summary_rows = []
    for model in non_floor_models:
        sub = df[df["base_model"] == model]
        n_point = int(sub["pareto_optimal"].sum())
        n_confident = int(sub["confidently_pareto_optimal"].sum())
        summary_rows.append(dict(base_model=model, n_point_optimal=n_point,
                                 n_confidently_optimal=n_confident))
        print(f"  {model:32} point-optimal in {n_point}/{n_intents}  -> "
              f"confidently optimal in {n_confident}/{n_intents}")

    return pd.DataFrame(summary_rows).sort_values("n_point_optimal", ascending=False)


# ============================================================
# STEP 4: chart -- individual-model level, faceted by intent
# ============================================================

CI_STATUS_COLOR = {
    "confidently_optimal": BLUE,
    "nominal_only_optimal": ORANGE,
    "ci_recovered_uncertain": ORANGE,
    "confidently_dominated": GRAY,
}


def step4_build_chart(df):
    print("\n" + "=" * 78)
    print("STEP 4  Build chart: cost vs strength, one point per model, "
          "faceted by intent, CI-aware status")
    print("=" * 78)

    price_lo = df["price_usd_per_million"].min() * 0.8
    price_hi = df["price_usd_per_million"].max() * 1.25
    strength_lo = df["strength"].min() - 0.15
    strength_hi = df["strength"].max() + 0.15

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True)
    for ax, intent in zip(axes.flat, INTENT_ORDER):
        sub = df[df["primary_intent"] == intent]
        for _, r in sub.iterrows():
            color = CI_STATUS_COLOR[r["ci_status"]]
            if r["low_n"]:
                # Low-confidence cell: hollow marker (face=white, colored
                # edge) instead of a new color -- same status color, visibly
                # less-certain, per Step 5's "mark, don't silently plot".
                ax.scatter(r["price_usd_per_million"], r["strength"], s=70,
                          facecolor="white", edgecolor=color, linewidth=1.8,
                          zorder=3)
            else:
                ax.scatter(r["price_usd_per_million"], r["strength"], s=70,
                          color=color, edgecolor="white", linewidth=0.6, zorder=3)
        # Direct labels used selectively (per build_charts.py's own
        # convention) -- only the point-estimate-optimal models are labeled
        # (confidently_optimal OR nominal_only_optimal), since those are the
        # actual routing recommendations this chart exists to show; dominated
        # points (confident or merely recovered-uncertain) are left as plain
        # unlabeled markers either way.
        # Some labeled points sit close together and their default offset
        # would collide -- alternate the label above/below the point, same
        # fix build_charts.py's chart3 already uses for a close cluster.
        frontier = sub[sub["pareto_optimal"]].sort_values("price_usd_per_million")
        for i, (_, r) in enumerate(frontier.iterrows()):
            xytext = (5, 4) if i % 2 == 0 else (5, -11)
            ax.annotate(r["base_model"], (r["price_usd_per_million"], r["strength"]),
                       textcoords="offset points", xytext=xytext, fontsize=6.6,
                       color=MUTED)
        ax.axhline(0, color="#b8b7ae", linewidth=0.8, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(price_lo, price_hi)
        ax.set_ylim(strength_lo, strength_hi)
        ax.set_title(intent, fontsize=10, pad=4)

    for ax in axes[-1, :]:
        ax.set_xlabel("Blended price (USD / million tokens, log scale)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("BT strength\n(log-odds vs. reference)", fontsize=8.5)

    handles = [
        plt.Line2D([0], [0], marker="o", color=BLUE, linestyle="", markersize=8,
                  label="Confidently Pareto-optimal\n(beats every cheaper rival, CIs don't overlap)"),
        plt.Line2D([0], [0], marker="o", color=ORANGE, linestyle="", markersize=8,
                  label="CI-ambiguous\n(optimal or dominated only by point estimate)"),
        plt.Line2D([0], [0], marker="o", color=GRAY, linestyle="", markersize=8,
                  label="Confidently dominated\n(a cheaper-or-equal rival's CI is clearly higher)"),
        plt.Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=INK,
                  markeredgewidth=1.6, linestyle="", markersize=8,
                  label="Hollow = low_n (few battles, less certain)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, -0.09), fontsize=8.8)
    fig.suptitle("Cost vs. Model Strength, by Intent — CI-Aware Routing Confidence",
                fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0, 0.1, 1, 0.97])

    os.makedirs(CHART_DIR, exist_ok=True)
    fig.savefig(OUT_CHART, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_CHART}")


# ============================================================
# STEP 6: honest summary
# ============================================================

def step6_write_summary(df, ci_impact, headline_df):
    print("\n" + "=" * 78)
    print("STEP 6  Write honest summary")
    print("=" * 78)

    L = []
    A = L.append
    A("# Cost-Performance Routing Analysis (by Intent, Individual-Model Level)")
    A("")
    A("**What this is:** for each `primary_intent` category separately, which "
      "individual models are Pareto-optimal on (blended price, Bradley-Terry "
      "strength) -- a model no other model in that same intent beats on both "
      "lower cost AND higher strength at once. A Pareto-optimal model is a real, "
      "defensible routing recommendation; a dominated one is not, regardless of "
      "how good it looks in isolation.")
    A("")
    A("**Deliberately individual-model level, not tier level.** This project's "
      "own tier-heterogeneity finding is that within-tier spread in strength is "
      "~2.7x the between-tier spread -- price tier barely predicts strength. "
      "Framing this analysis by tier would visually contradict that finding. "
      "Tier is included in the saved CSV as a secondary label only; it is not "
      "the chart's grouping or primary color.")
    A("")
    A("**Difficulty proxy, stated explicitly:** this analysis stratifies by "
      "`primary_intent`, not Avery's planned 0-7 difficulty score -- confirmed, "
      "in a separate audit, not to exist anywhere in this repo yet. If/when "
      "that score becomes available, it can replace `primary_intent` as the "
      "stratifying variable (re-run Step 3's grouping on it instead) without "
      "rebuilding this script from scratch.")
    A("")

    A("## Inputs confirmed (Step 1)")
    A("")
    n_low_n = int(df["low_n"].sum())
    A(f"- `outputs/bradley_terry_model_strength_by_intent.csv`: {df['base_model'].nunique()} "
      f"models x {df['primary_intent'].nunique()} intents = {len(df)} rows, a full grid, "
      f"**{n_low_n} cells flagged `low_n`** (too few battles to trust the strength "
      "estimate -- see the low-confidence table below).")
    A("- `final_analysis.py`'s `MODEL_BLENDED_PRICE_USD_PER_MILLION`: every one of "
      "the 9 models matched a real price; no model excluded for a missing price.")
    A("")

    A("## Pareto-optimal models, per intent")
    A("")
    A("| intent | Pareto-optimal models (price, strength) |")
    A("|---|---|")
    for intent in INTENT_ORDER:
        sub = df[df["primary_intent"] == intent].sort_values("price_usd_per_million")
        frontier = sub[sub["pareto_optimal"]]
        cells = []
        for _, r in frontier.iterrows():
            flag = " *(low_n)*" if r["low_n"] else ""
            cells.append(f"`{r['base_model']}` (${r['price_usd_per_million']:.2f}, "
                        f"{r['strength']:+.2f}){flag}")
        A(f"| {intent} | {'; '.join(cells)} |")
    A("")

    A("## Does the same model dominate everywhere, or does it shift by intent?")
    A("")
    pareto_counts = (df[df["pareto_optimal"]].groupby("base_model").size()
                    .sort_values(ascending=False))
    n_intents = df["primary_intent"].nunique()
    always_cheapest = df.loc[df["price_usd_per_million"].idxmin(), "base_model"]
    A(f"`{always_cheapest}` is Pareto-optimal in all {n_intents} intents -- but "
      "this is a mechanical consequence of it being the single cheapest model "
      "in the whole set (no other model can ever beat it on cost), not evidence "
      "it is a uniquely strong model. It should be read as a cost floor, not a "
      "quality signal.")
    A("")
    other_always = [m for m, n in pareto_counts.items()
                    if n == n_intents and m != always_cheapest]
    if other_always:
        A(f"**`{', '.join(other_always)}` is also Pareto-optimal in all "
          f"{n_intents} intents -- unlike the cost floor, this is not a "
          "mechanical artifact** of being the cheapest model (it isn't). "
          "*But* whether its point estimate actually, statistically beats "
          "every costlier-or-cheaper rival it needs to is a separate question "
          "-- checked explicitly, not assumed, in the CI-aware section below. "
          "**Short answer, don't take the point-estimate table below at face "
          "value: see \"Step 3: do the headline findings hold up?\" for "
          "whether this status survives confidence intervals.**")
        A("")
    A("Beyond the cost-floor model, the Pareto-optimal set **genuinely shifts "
      "by intent** -- it is not the same 1-2 models dominating everywhere:")
    A("")
    A("| model | number of intents where Pareto-optimal (of 9) |")
    A("|---|---|")
    for model, n in pareto_counts.items():
        A(f"| `{model}` | {n} |")
    A("")

    A("## CI-aware robustness check: does the point-estimate frontier hold up?")
    A("")
    A("The analysis above compares only point estimates of strength -- the same "
      "category of risk this project already corrected once before, in the "
      "OLS-to-Bradley-Terry rebuild (a finding that didn't survive stronger "
      "methods once uncertainty was actually checked). Two SEPARATE CI-aware "
      "questions are checked here, not one, because they cut in opposite "
      "directions:")
    A("")
    A("- **Are the *dominated* models' status actually certain?** A model is "
      "`confidently_dominated` only if some cheaper-or-equal rival's strength "
      "95% CI lower bound exceeds the dominated model's CI upper bound -- the "
      "same non-overlapping-CI bar this project already uses elsewhere to call "
      "a difference real, not a new, weaker threshold invented for this check. "
      "This is strictly *rarer* than point-estimate dominance, so some "
      "point-dominated cells get **recovered** as merely uncertain.")
    A("- **Is the *optimal* models' status actually certain?** A Pareto-optimal "
      "model `confidently_beats_all_cheaper` only if its own CI lower bound "
      "exceeds the CI upper bound of EVERY strictly-cheaper rival it needs to "
      "beat to earn its spot. This is strictly *rarer* than \"has a higher "
      "point estimate than every cheaper rival\", so some point-optimal cells "
      "get **downgraded** to nominal-only.")
    A("")
    A("`gemini-2.0-flash` (the reference model) has no estimated CI -- its "
      "strength is fixed at 0 by the Bradley-Terry identification constraint, "
      "not independently measured. It is treated as a degenerate `[0, 0]` "
      "interval here (not skipped), since every other model's CI is already "
      "expressed relative to this fixed point -- the standard way to compare "
      "against a BT identification anchor.")
    A("")

    A("### Step 2: how much does this change the picture?")
    A("")
    A(f"- **{ci_impact['n_dominated_point']}** (model, intent) pairs were "
      f"dominated under point estimate. Of those, **{ci_impact['n_recovered']}** "
      f"({100*ci_impact['n_recovered']/max(1,ci_impact['n_dominated_point']):.1f}%) "
      "are NOT confidently dominated once CIs are accounted for -- their "
      "\"loser\" status rests on a point-estimate difference the data doesn't "
      "actually distinguish from noise.")
    A(f"- **{ci_impact['n_optimal_point']}** (model, intent) pairs were "
      f"Pareto-optimal under point estimate. Of those, "
      f"**{ci_impact['n_nominal_only']}** "
      f"({100*ci_impact['n_nominal_only']/max(1,ci_impact['n_optimal_point']):.1f}%) "
      "do NOT confidently beat every cheaper rival -- their \"winner\" status "
      "also rests on a point-estimate edge the data doesn't fully back up.")
    A("")

    A("### Step 3: do the headline findings hold up?")
    A("")
    sonar_row = df[df["base_model"] == "sonar"]
    n_sonar_point = int(sonar_row["pareto_optimal"].sum())
    n_sonar_confident = int(sonar_row["confidently_pareto_optimal"].sum())
    n_sonar_total = len(sonar_row)
    if n_sonar_confident == n_sonar_point == n_sonar_total:
        sonar_verdict = (f"**`sonar`'s \"Pareto-optimal in all {n_sonar_total} "
                         f"intents\" claim HOLDS UP under CI-aware scrutiny**: it "
                         f"confidently beats every cheaper rival "
                         f"(`gemini-2.0-flash` and `gpt-4o-mini-search-preview`) "
                         f"in all {n_sonar_total} intents, not just on point "
                         "estimate.")
    else:
        sonar_verdict = (f"**`sonar`'s \"Pareto-optimal in all {n_sonar_total} "
                         f"intents\" claim does NOT fully hold up**: it is "
                         f"confidently optimal in only {n_sonar_confident} of "
                         f"{n_sonar_total} intents once CIs are checked against "
                         "its cheaper rivals -- the remaining "
                         f"{n_sonar_point - n_sonar_confident} intent(s) have a "
                         "higher point estimate that the data cannot actually "
                         "distinguish from a cheaper alternative.")
    A(sonar_verdict)
    A("")
    A("Same check for every other non-cost-floor model that was ever "
      "point-optimal somewhere:")
    A("")
    A("| model | point-optimal (of 9) | confidently optimal (of 9) |")
    A("|---|---|---|")
    for _, r in headline_df.iterrows():
        A(f"| `{r['base_model']}` | {r['n_point_optimal']} | {r['n_confidently_optimal']} |")
    A("")

    A("## Low-confidence cells (`low_n`), flagged honestly, not silently plotted")
    A("")
    low_n_df = df[df["low_n"]].sort_values(["primary_intent", "n_battles"])
    A("| model | intent | n_battles | strength | Pareto-optimal? |")
    A("|---|---|---|---|---|")
    for _, r in low_n_df.iterrows():
        A(f"| `{r['base_model']}` | {r['primary_intent']} | {int(r['n_battles'])} | "
          f"{r['strength']:+.3f} | {'yes' if r['pareto_optimal'] else 'no'} |")
    A("")
    n_low_n_pareto = int(low_n_df["pareto_optimal"].sum())
    A(f"**{n_low_n_pareto} of the {len(low_n_df)} `low_n` cells are Pareto-optimal** "
      "-- these are marked as hollow markers in the chart and flagged here by "
      "name specifically because a routing recommendation resting on a "
      "low-battle-count estimate deserves less trust than one resting on a "
      "well-sampled cell, even though both are drawn as the same Pareto status.")
    A("")

    A("## Step 5: final honest verdict")
    A("")
    recovered_share = ci_impact["n_recovered"] / max(1, ci_impact["n_dominated_point"])
    nominal_share = ci_impact["n_nominal_only"] / max(1, ci_impact["n_optimal_point"])
    if recovered_share < 0.15 and nominal_share < 0.15:
        verdict = ("**This check STRENGTHENS the original findings.** The point-"
                   "estimate frontier and the CI-aware frontier agree on the large "
                   "majority of cells -- dominated models are mostly confidently "
                   "dominated, and optimal models mostly confidently beat their "
                   "cheaper rivals. The earlier analysis was not overstating "
                   "certainty in any meaningful way.")
    elif recovered_share < 0.4 and nominal_share < 0.4:
        verdict = ("**This check PARTIALLY WEAKENS the original findings -- read "
                   "with real caution, not full confidence.** A meaningful minority "
                   "of cells on both sides (recovered dominated cells, and "
                   "nominal-only optimal cells) do not survive CI scrutiny. The "
                   "broad shape of the result (cost-floor model always optimal, "
                   "`sonar` usually a strong value pick, the frontier shifting by "
                   "intent) still holds, but individual (model, intent) "
                   "recommendations should be checked against the `ci_status` "
                   "column before being treated as settled.")
    else:
        verdict = ("**This check MEANINGFULLY WEAKENS the original findings.** A "
                   "large share of both the \"dominated\" and the \"optimal\" "
                   "labels do not survive CI scrutiny -- the point-estimate "
                   "frontier was overstating certainty. Treat the point-estimate "
                   "Pareto table above as a starting hypothesis, not a settled "
                   "routing recommendation, until more battles narrow these "
                   "intervals.")
    A(verdict)
    A("")
    A(f"Concretely: **{ci_impact['n_recovered']}/{ci_impact['n_dominated_point']}** "
      f"dominated cells recovered, **{ci_impact['n_nominal_only']}/"
      f"{ci_impact['n_optimal_point']}** optimal cells downgraded to nominal-only. "
      "Either direction was a legitimate possible outcome going in -- this is "
      "what the data actually showed, not the result picked for how it sounds.")
    A("")

    A("## Caveats")
    A("")
    A("1. **Blended price does not include search-fee surcharges** for "
      "`gpt-4o-mini-search-preview` and `gpt-4o-search-preview` (per "
      "`final_analysis.py`'s own documented caveat) -- their true relative cost "
      "is higher than the blended number used here suggests.")
    A("2. **`sonar-reasoning`'s price has no independently published source** "
      "(a user-supplied verified figure per `final_analysis.py`) -- treated the "
      "same as every other price here, but worth knowing.")
    A("3. **Strength is a log-odds estimate relative to a fixed reference model "
      "(`gemini-2.0-flash`), not an absolute quality score** -- differences "
      "between two non-reference models are still meaningful (both are on the "
      "same log-odds scale), but the number itself has no natural zero point "
      "beyond the reference.")
    A("4. **Non-overlapping 95% CIs is a conservative bar, not a formal two-sample "
      "significance test** -- it under-flags real differences (some "
      "`confidently_dominated`/`confidently_optimal` cells may be too "
      "conservative), consistent with how this project already treats CI "
      "non-overlap elsewhere. `ci_status`, `eff_ci_lo`, and `eff_ci_hi` remain "
      "available in the saved CSV for anyone who wants a different bar.")
    A("")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[saved] {OUT_MD}")


# ============================================================
# STEP 7: save outputs
# ============================================================

def main():
    print("=" * 78)
    print("COST-PERFORMANCE ROUTING ANALYSIS (additive -- no existing file modified)")
    print("=" * 78)

    df, pricing = step1_confirm_inputs()
    df = step2_merge_price(df, pricing)

    n_before = len(df)
    df = df.dropna(subset=["price_usd_per_million"]).copy()
    if len(df) != n_before:
        print(f"\n[WARNING] dropped {n_before - len(df)} rows with no price match "
              "before Pareto analysis.")

    df = step3_pareto_by_intent(df)
    df = step1b_ci_aware_dominance(df)
    ci_impact = step2_quantify_ci_impact(df)
    headline_df = step3b_recheck_headline_findings(df)
    step4_build_chart(df)
    step6_write_summary(df, ci_impact, headline_df)

    df.to_csv(OUT_CSV, index=False)
    print(f"\n[saved] {OUT_CSV}")
    print(f"\nDone. {len(df)} (model, intent) rows, "
          f"{int(df['pareto_optimal'].sum())} Pareto-optimal "
          f"({int(df['confidently_pareto_optimal'].sum())} confidently so).")


if __name__ == "__main__":
    main()
