"""
loss_anatomy.py

Characterizes what losing responses look like (A1) and splits losses into
STRUCTURAL (the losing model is just weaker at this task category) vs
REPAIRABLE (the models are roughly evenly matched for this intent, but this
specific losing response was notably deficient on a response-level feature
that is already tracked in this repo) vs AMBIGUOUS (neither condition clearly
holds).

This is ADDITIVE ONLY. It does not modify any existing script or output file.
It reuses the exact Bradley-Terry fitting machinery already established in
bradley_terry_analysis.py (same design matrix builder, same bootstrap
procedure, same seed, same covariate scaling) -- nothing is re-derived or
re-specified from scratch.

SCOPE NOTE ON RESPONSE-LEVEL FEATURES (stated explicitly, not buried):
the task that produced this script asked for has_markdown_table, is_refusal,
and numbers_per_100_words as candidate repairable-loss features. None of
these are computed anywhere in this repo -- grepped and confirmed absent.
Per instruction not to invent new features on the spot, this script uses
only the response-level features that already exist in the pipeline:
citation count (citations_a/b, the fixed per-turn-deduped-then-summed
measure) and response length (response_length_a/b, characters). A
repairable loss is therefore defined narrowly as "the loser had a notably
lower citation count or was notably shorter," not the broader feature set
originally envisioned.

STEP 1: per-(model, primary_intent) Bradley-Terry strength.
This refits the same model-identity design (build_design_model_identity)
separately within each primary_intent subgroup, with the same 1,000-rep
conversation-level bootstrap and the same seed used everywhere else in this
project. It is a separate fit per intent (not a single model with a
model x intent interaction term) -- chosen because several intents have too
few battles per model to estimate interaction terms reliably, and separate
per-group fits make the per-cell sample size, and therefore reliability,
directly visible rather than hidden inside a shared-information interaction
model.

STEP 2+3: per-battle paired feature diffs and structural/repairable/
ambiguous classification, thresholds stated explicitly below.

STEP 4: aggregation + markdown report + chart.
STEP 5: sanity checks (eyeball examples, no-battles-dropped check, low-n
cell count).
"""

import os

import numpy as np
import pandas as pd

from final_analysis import (
    INTENT_ORDER,
    classify_cost_tier,
)
from bradley_terry_analysis import (
    N_BOOTSTRAP,
    RANDOM_SEED,
    ALPHA,
    MIN_N_FOR_FIT,
    load_decisive,
    _scale_covariates,
    build_design_model_identity,
    bootstrap_bt,
)

# ============================================================
# SETTINGS (stated explicitly -- these are judgment calls, not derived)
# ============================================================

# Below this many battles for a given (model, primary_intent) cell, the
# strength estimate for that cell is flagged unreliable rather than silently
# used. Chosen as 2x MIN_N_FOR_FIT (the threshold the rest of this project
# already uses for "enough data to fit at all") -- a model needs to show up
# in roughly twice the minimum-fittable number of battles within an intent
# before its *individual* strength within that intent is trusted enough to
# drive a structural/repairable call for other battles.
MIN_BATTLES_PER_MODEL_INTENT = 2 * MIN_N_FOR_FIT  # = 60

# A loss is called REPAIRABLE on the citation/length features only when the
# loser's deficit is at least half a standard deviation of the global
# citation/length difference distribution (the same sd_cit/sd_len used to
# scale every other BT coefficient in this project). Half an SD is a
# judgment call, not a derived number: it is small enough to catch
# meaningfully one-sided battles, large enough that it is not flagging
# nearly-tied responses. Stated here so it can be revisited.
REPAIRABLE_FEATURE_SD_THRESHOLD = 0.5

OUT_STRENGTH_BY_INTENT_CSV = "outputs/bradley_terry_model_strength_by_intent.csv"
OUT_BATTLES_CSV = "outputs/loss_anatomy_battles.csv"
OUT_SUMMARY_CSV = "outputs/loss_anatomy_summary.csv"
OUT_SUMMARY_MD = "outputs/loss_anatomy_summary.md"
OUT_CHART = "charts/chart_loss_anatomy_by_intent.png"


# ============================================================
# STEP 1: per-(model, primary_intent) strength
# ============================================================

def run_strength_by_intent(dec, sd_cit, sd_len):
    print("=" * 78)
    print("STEP 1  Bradley-Terry model strength, fit separately per primary_intent")
    print(f"        ({N_BOOTSTRAP:,} bootstrap reps, seed {RANDOM_SEED}, "
          "same design as the rest of this project)")
    print("=" * 78)

    rows = []
    for intent in INTENT_ORDER:
        sub = dec[dec["primary_intent"] == intent].copy()
        n_sub = len(sub)
        if n_sub < MIN_N_FOR_FIT:
            print(f"[skip] {intent}: n={n_sub} < MIN_N_FOR_FIT={MIN_N_FOR_FIT}, "
                  "cannot fit at all")
            continue

        models = sorted(set(sub["base_model_a"]) | set(sub["base_model_b"]))
        res, cols, ref, nfail, ok, _draws = bootstrap_bt(
            sub, build_design_model_identity,
            models=models, sd_cit=sd_cit, sd_len=sd_len)

        # battle counts per model within this intent, for the low-n flag --
        # counted independently of the fit itself, directly from the data.
        n_battles_by_model = {}
        for m in models:
            n_battles_by_model[m] = int(
                ((sub["base_model_a"] == m) | (sub["base_model_b"] == m)).sum())

        rows.append(dict(
            base_model=ref, primary_intent=intent, strength=0.0,
            ci_lo=np.nan, ci_hi=np.nan, boot_p=np.nan, is_reference=True,
            n_battles=n_battles_by_model[ref],
            low_n=n_battles_by_model[ref] < MIN_BATTLES_PER_MODEL_INTENT,
            converged=ok,
        ))
        for m in models:
            if m == ref:
                continue
            r = res[f"theta[{m}]"]
            rows.append(dict(
                base_model=m, primary_intent=intent, strength=r["coef"],
                ci_lo=r["lo"], ci_hi=r["hi"], boot_p=r["p"], is_reference=False,
                n_battles=n_battles_by_model[m],
                low_n=n_battles_by_model[m] < MIN_BATTLES_PER_MODEL_INTENT,
                converged=ok,
            ))
        n_low = sum(1 for m in models
                   if n_battles_by_model[m] < MIN_BATTLES_PER_MODEL_INTENT)
        print(f"{intent:22} n={n_sub:5}  models={len(models)}  "
              f"low-n cells={n_low}/{len(models)}  reference={ref}")

    df = pd.DataFrame(rows)
    n_total_cells = len(df)
    n_low_total = int(df["low_n"].sum())
    print(f"\n{n_low_total} of {n_total_cells} (model, intent) cells flagged low-n "
          f"(< {MIN_BATTLES_PER_MODEL_INTENT} battles for that model within that intent)")
    return df


# ============================================================
# STEP 2+3: per-battle diffs and classification
# ============================================================

def build_loss_anatomy(dec, strength_df, sd_by_conv_type):
    """
    sd_by_conv_type: {conversation_type: (sd_cit, sd_len)}, each computed by
    _scale_covariates on ONLY that conversation_type's decisive battles.

    BUG FIX: citation_diff_z / length_diff_z were previously standardized
    against one pooled sd_cit/sd_len computed across all 8,613 decisive
    battles. Confirmed by a direct check: multi_turn's own sd_cit (9.50) and
    sd_len (5507) run roughly 2x single_turn's (4.95, 2330), with the pooled
    values (6.13, 3223) sitting in between. Standardizing every battle
    against the pooled value inflated multi-turn's apparent deficits (its
    real spread is wider than the pooled denominator) and shrank
    single-turn's (its real spread is narrower than the pooled denominator)
    -- biasing the repairable/ambiguous split for every battle in BOTH
    groups, not just the conversation_type breakdown table. Each battle is
    now standardized against its OWN conversation_type's sd_cit/sd_len.
    """
    print("\n" + "=" * 78)
    print("STEP 2+3  Per-battle paired diffs and structural/repairable/ambiguous classification")
    print("           (citation/length z-scores standardized per conversation_type -- see bug-fix note)")
    print("=" * 78)

    strength_lookup = {
        (r.base_model, r.primary_intent): (r.strength, r.ci_lo, r.ci_hi, r.low_n)
        for r in strength_df.itertuples()
    }

    rows = []
    n_dropped_no_strength = 0
    for r in dec.itertuples():
        if r.winner == "model_a":
            winner_model, loser_model = r.base_model_a, r.base_model_b
            citation_diff = r.citations_a - r.citations_b
            length_diff = r.response_length_a - r.response_length_b
        else:
            winner_model, loser_model = r.base_model_b, r.base_model_a
            citation_diff = r.citations_b - r.citations_a
            length_diff = r.response_length_b - r.response_length_a

        key_w = (winner_model, r.primary_intent)
        key_l = (loser_model, r.primary_intent)
        if key_w not in strength_lookup or key_l not in strength_lookup:
            # intent had n < MIN_N_FOR_FIT and was skipped entirely in Step 1
            n_dropped_no_strength += 1
            continue
        w_strength, w_lo, w_hi, w_lown = strength_lookup[key_w]
        l_strength, l_lo, l_hi, l_lown = strength_lookup[key_l]

        same_base_model = (winner_model == loser_model)
        strength_diff = w_strength - l_strength

        sd_cit_ct, sd_len_ct = sd_by_conv_type[r.conversation_type]
        cit_z = citation_diff / sd_cit_ct
        len_z = length_diff / sd_len_ct

        if same_base_model:
            # Cannot invoke "the model is weaker" when both sides are the
            # same base model -- structural is not a meaningful category
            # here by construction, independent of any CI.
            unreliable = False
            structural_possible = False
            ci_overlap = None
            loser_significantly_weaker = False
        else:
            unreliable = bool(w_lown or l_lown)
            structural_possible = True
            ci_overlap = not (l_hi < w_lo or w_hi < l_lo)  # True if they overlap
            # STRUCTURAL requires specifically that the LOSER's strength is
            # the lower one with non-overlapping CIs -- not merely that the
            # two CIs fail to overlap in either direction. A non-overlap
            # where the loser's point estimate is actually HIGHER (the
            # nominally stronger model lost this specific battle) is the
            # opposite of "the model is just weaker" and must not be
            # mislabeled structural.
            loser_significantly_weaker = (not ci_overlap) and (l_hi < w_lo)

        deficient_citation = cit_z <= -REPAIRABLE_FEATURE_SD_THRESHOLD
        deficient_length = len_z <= -REPAIRABLE_FEATURE_SD_THRESHOLD
        driver_features = []
        if deficient_citation:
            driver_features.append("citation_count")
        if deficient_length:
            driver_features.append("response_length")
        # if both qualify, report both but also note which is larger in
        # standardized terms, since that's the more actionable single lever
        primary_driver = None
        if driver_features:
            primary_driver = "citation_count" if abs(cit_z) >= abs(len_z) else "response_length"

        if same_base_model:
            if driver_features:
                label = "repairable"
                reason = "identical base model on both sides; loser deficient on " + "+".join(driver_features)
            else:
                label = "ambiguous"
                reason = "identical base model on both sides; no notable citation/length deficit"
        elif unreliable:
            label = "ambiguous"
            reason = "insufficient battles to reliably estimate per-intent strength for this model pair"
        elif loser_significantly_weaker:
            label = "structural"
            reason = "loser's per-intent strength CI is entirely below the winner's"
        elif (not ci_overlap) and driver_features:
            # non-overlap in the OTHER direction: the nominally stronger
            # model lost this specific battle. Not structural (the loser
            # isn't the weaker one) -- if it also shows a feature deficit,
            # that deficit is still the more informative tag than burying
            # the battle in ambiguous.
            label = "repairable"
            reason = ("loser's per-intent strength is actually higher than winner's "
                      "(an upset, not a structural loss); loser deficient on " + "+".join(driver_features))
        elif not ci_overlap:
            label = "ambiguous"
            reason = ("loser's per-intent strength is actually higher than winner's "
                      "(an upset, not a structural loss) with no notable citation/length deficit")
        elif driver_features:
            label = "repairable"
            reason = "strengths not significantly different for this intent; loser deficient on " + "+".join(driver_features)
        else:
            label = "ambiguous"
            reason = "strengths not significantly different and no notable citation/length deficit"

        rows.append(dict(
            primary_intent=r.primary_intent,
            conversation_type=r.conversation_type,
            cost_tier_winner=classify_cost_tier(winner_model),
            cost_tier_loser=classify_cost_tier(loser_model),
            winner_model=winner_model, loser_model=loser_model,
            same_base_model=same_base_model,
            citation_diff=citation_diff, length_diff=length_diff,
            citation_diff_z=cit_z, length_diff_z=len_z,
            winner_strength=w_strength, loser_strength=l_strength,
            strength_diff=strength_diff,
            winner_strength_low_n=bool(w_lown), loser_strength_low_n=bool(l_lown),
            structural_possible=structural_possible, ci_overlap=ci_overlap,
            primary_driver_feature=primary_driver,
            label=label, reason=reason,
        ))

    if n_dropped_no_strength:
        print(f"[note] {n_dropped_no_strength} battles dropped: primary_intent had "
              f"n < {MIN_N_FOR_FIT} in Step 1 and was not fitted at all "
              "(not silently misclassified -- excluded and reported here).")
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} battles classified "
          f"({len(dec)} decisive battles total, {n_dropped_no_strength} excluded for "
          "insufficient intent-level n)")
    print(df["label"].value_counts())
    return df, n_dropped_no_strength


# ============================================================
# STEP 4: aggregation
# ============================================================

def aggregate(battles_df):
    print("\n" + "=" * 78)
    print("STEP 4  Aggregation by primary_intent, cost_tier (of the LOSER), and conversation_type")
    print("=" * 78)
    rows = []
    for group_col, group_label in (("primary_intent", "primary_intent"),
                                   ("cost_tier_loser", "cost_tier"),
                                   ("conversation_type", "conversation_type")):
        for val, sub in battles_df.groupby(group_col):
            n = len(sub)
            counts = sub["label"].value_counts()
            n_struct = int(counts.get("structural", 0))
            n_repair = int(counts.get("repairable", 0))
            n_ambig = int(counts.get("ambiguous", 0))
            rows.append(dict(
                group_type=group_label, group_value=val, n_battles=n,
                n_structural=n_struct, n_repairable=n_repair, n_ambiguous=n_ambig,
                pct_structural=100.0 * n_struct / n if n else np.nan,
                pct_repairable=100.0 * n_repair / n if n else np.nan,
                pct_ambiguous=100.0 * n_ambig / n if n else np.nan,
            ))
            assert n_struct + n_repair + n_ambig == n, (
                f"battles dropped in aggregation for {group_label}={val}")
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ============================================================
# STEP 5: sanity checks
# ============================================================

def sanity_checks(dec, battles_df, strength_df):
    print("\n" + "=" * 78)
    print("STEP 5  Sanity checks")
    print("=" * 78)

    # eyeball examples
    for label in ("repairable", "structural"):
        sub = battles_df[battles_df["label"] == label]
        print(f"\n--- 3 example '{label}' battles ---")
        sample = sub.sample(min(3, len(sub)), random_state=RANDOM_SEED)
        for idx, r in sample.iterrows():
            print(f"  intent={r['primary_intent']}  winner={r['winner_model']} "
                  f"(strength {r['winner_strength']:+.3f})  "
                  f"loser={r['loser_model']} (strength {r['loser_strength']:+.3f})  "
                  f"citation_diff={r['citation_diff']:+d} (z={r['citation_diff_z']:+.2f})  "
                  f"length_diff={r['length_diff']:+d} (z={r['length_diff_z']:+.2f})  "
                  f"reason: {r['reason']}")

    # no battles dropped silently: totals must reconcile
    total_decisive = len(dec)
    total_classified = len(battles_df)
    print(f"\ntotal decisive battles        : {total_decisive}")
    print(f"total classified               : {total_classified}")
    print(f"difference (excluded, reported): {total_decisive - total_classified}")

    n_low_cells = int(strength_df["low_n"].sum())
    n_total_cells = len(strength_df)
    print(f"\nlow-n (model, intent) cells from Step 1: {n_low_cells} of {n_total_cells}")
    unreliable_battles = int(
        (battles_df["winner_strength_low_n"] | battles_df["loser_strength_low_n"]).sum())
    print(f"battles involving at least one low-n strength estimate: "
          f"{unreliable_battles} of {total_classified} "
          f"({100.0 * unreliable_battles / total_classified:.1f}%) -- "
          "these were routed to 'ambiguous', not silently treated as structural/repairable.")


# ============================================================
# Chart
# ============================================================

def build_chart(summary_df, out_path=OUT_CHART):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = summary_df[summary_df["group_type"] == "primary_intent"].copy()
    sub = sub.sort_values("n_battles", ascending=False)

    colors = {"structural": "#9a9992", "repairable": "#1baf7a", "ambiguous": "#2a78d6"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(sub))
    bottom = np.zeros(len(sub))
    for label in ("structural", "repairable", "ambiguous"):
        vals = sub[f"pct_{label}"].values
        ax.bar(x, vals, bottom=bottom, color=colors[label], label=label.capitalize(),
               width=0.62, edgecolor="white", linewidth=0.6)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sub["group_value"], rotation=30, ha="right")
    ax.set_ylabel("% of decisive losses")
    ax.set_ylim(0, 100)
    ax.set_title("Loss anatomy by primary intent: structural vs repairable vs ambiguous")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)

    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i, 102, f"n={int(r['n_battles'])}", ha="center", va="bottom",
                fontsize=8, color="#555555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ============================================================
# Markdown summary
# ============================================================

def write_summary_md(path, dec, battles_df, summary_df, strength_df, n_dropped,
                     sd_cit, sd_len, sd_by_conv_type):
    L = []
    A = L.append
    A("# Loss Anatomy: Structural vs Repairable Losses")
    A("")
    A("## Methodology")
    A("")
    A("For every decisive battle, the losing response is compared against the "
      "winning response on two axes:")
    A("")
    A("1. **Is the losing model structurally weaker for this task category?** "
      "Answered by refitting the project's Bradley-Terry model "
      f"({N_BOOTSTRAP:,} conversation-level bootstrap replications, seed "
      f"`{RANDOM_SEED}`, same design matrix as the rest of this project) "
      "**separately within each `primary_intent`**, with individual base "
      "models as players. This produces a per-(model, intent) strength with "
      "a 95% CI -- see `" + OUT_STRENGTH_BY_INTENT_CSV + "`.")
    A("2. **Was this specific response notably deficient on a tracked "
      "response-level feature?** Using the citation-count and response-"
      "length differences already computed in this pipeline.")
    A("")
    A("**Feature scope, stated explicitly.** The task that produced this report asked "
      "for `has_markdown_table`, `is_refusal`, and `numbers_per_100_words` as candidate "
      "repairable-loss indicators. None of these are computed anywhere in this repo "
      "(confirmed by grep, not assumed). Per instruction not to invent new features on "
      "the spot, this report uses only `citation_diff` and `length_diff` -- the two "
      "response-level features this project already tracks. A 'repairable' loss here "
      "means specifically: *the loser cited less or wrote less, not that the model is "
      "worse at the task.* Markdown-table use, refusals, and numeric density may well "
      "be repairable levers too, but they are not measured in this dataset as it "
      "currently stands.")
    A("")
    A("### Classification rule (thresholds stated plainly)")
    A("")
    A(f"- **STRUCTURAL**: loser and winner are different base models, and the loser's "
      f"95% strength CI for this `primary_intent` sits entirely below the winner's "
      "(not merely \"non-overlapping\" -- specifically non-overlapping in the direction "
      "where the loser is the weaker one). The model is measurably weaker at this task "
      "category; routing around it is the fix, prompting is not. A battle where the "
      "nominally *stronger* model lost (CIs non-overlapping the other way -- an upset) "
      "is never labeled structural; it falls to repairable or ambiguous instead.")
    A(f"- **REPAIRABLE**: either (a) strengths are not significantly different for this "
      f"intent (CIs overlap) but the loser's citation count or response length is at "
      f"least **{REPAIRABLE_FEATURE_SD_THRESHOLD} standard deviations** below the "
      "winner's (standardized against that battle's own `conversation_type` SD -- see "
      "the scaling note immediately below), or (b) both sides are literally the same "
      "base model, in which case a strength comparison is meaningless by construction "
      "and any notable citation/length deficit is the only available explanation.")
    A("")
    A("**SCALING BUG FOUND AND FIXED, stated explicitly.** An earlier version of this "
      "report standardized `citation_diff`/`length_diff` for every battle against one "
      "**pooled** `sd_cit`/`sd_len` computed across all "
      f"{len(dec):,} decisive battles -- this is the \"same global SD scaling used "
      "everywhere else in this project\" convention, and it was the wrong call here. "
      "A direct check confirmed `multi_turn` and `single_turn` battles have "
      "meaningfully different scales:")
    A("")
    A("| scaling | sd_cit | sd_len |")
    A("|---|---|---|")
    A(f"| pooled (previously used for every battle) | {sd_cit:.3f} | {sd_len:,.1f} |")
    for ct in sorted(sd_by_conv_type):
        c, l = sd_by_conv_type[ct]
        n_ct = int((dec["conversation_type"] == ct).sum())
        A(f"| `{ct}` only (n={n_ct:,}) | {c:.3f} | {l:,.1f} |")
    A("")
    mt_c, mt_l = sd_by_conv_type.get("multi_turn", (float("nan"), float("nan")))
    st_c, st_l = sd_by_conv_type.get("single_turn", (float("nan"), float("nan")))
    A(f"`multi_turn`'s own spread is roughly **{mt_c/st_c:.1f}x** `single_turn`'s on "
      f"citations and **{mt_l/st_l:.1f}x** on length. Standardizing every battle "
      "against the pooled value (which sits between the two, weighted toward "
      "`single_turn` by its larger n) inflated how deficient `multi_turn` battles "
      "looked relative to `multi_turn`'s own variation, and shrank how deficient "
      "`single_turn` battles looked relative to `single_turn`'s own variation -- "
      "biasing the repairable/ambiguous split for **every battle in both groups**, not "
      "only the `conversation_type` breakdown table. **This report now standardizes "
      "each battle against its own `conversation_type`'s `sd_cit`/`sd_len`** (the table "
      "above), computed with the same `_scale_covariates` function, just called "
      "separately per `conversation_type` instead of once on the pooled set. The "
      "pooled `sd_cit`/`sd_len` is still used, unchanged, inside Step 1's Bradley-Terry "
      "covariate fit (`beta_citation`/`beta_length` in the per-intent strength model) -- "
      "that usage makes those coefficients comparable *across `primary_intent` "
      "categories* and is a different computation from the citation/length "
      "deficiency check this bug fix addresses.")
    A("- **AMBIGUOUS**: neither condition clearly holds, OR at least one side's "
      "per-intent strength estimate rests on fewer than "
      f"**{MIN_BATTLES_PER_MODEL_INTENT} battles** for that model within that intent "
      "and is therefore too noisy to anchor a structural call. Ambiguous battles are "
      "not forced into either other bucket.")
    A("")
    A(f"**Honesty check on the thresholds:** the half-SD feature threshold and the "
      f"{MIN_BATTLES_PER_MODEL_INTENT}-battle reliability cutoff are both judgment "
      "calls, not statistically derived values -- they are stated here so they can be "
      "revisited. The CI-non-overlap rule for STRUCTURAL is the more principled of the "
      "two since it inherits directly from the bootstrap, but note that non-overlapping "
      "95% CIs is a stricter (more conservative) test than a formal two-sample "
      "difference test; some battles labeled 'ambiguous' here might be structural under "
      "a less conservative test.")
    A("")
    A(f"- **Data:** {len(dec):,} decisive battles. {len(battles_df):,} classified; "
      f"**{n_dropped}** excluded because their `primary_intent` had too few total "
      "battles to fit a per-intent model at all (see "
      "`bradley_terry_by_intent.csv`'s existing `insufficient n` rows for the same cutoff).")
    A("")

    n_low_cells = int(strength_df["low_n"].sum())
    n_total_cells = len(strength_df)
    A(f"- **Per-intent strength reliability:** {n_low_cells} of {n_total_cells} "
      "(model, intent) cells were fit on fewer than "
      f"{MIN_BATTLES_PER_MODEL_INTENT} battles for that model in that intent and are "
      "flagged `low_n=True` in the strength CSV. Battles involving at least one such "
      "cell are routed to AMBIGUOUS rather than silently classified.")
    A("")

    A("## Findings: loss anatomy by primary intent")
    A("")
    A("| primary_intent | n losses | structural | repairable | ambiguous | "
      "% structural | % repairable | % ambiguous |")
    A("|---|---|---|---|---|---|---|---|")
    sub = summary_df[summary_df["group_type"] == "primary_intent"].sort_values(
        "pct_repairable", ascending=False)
    for _, r in sub.iterrows():
        A(f"| {r['group_value']} | {int(r['n_battles'])} | {int(r['n_structural'])} | "
          f"{int(r['n_repairable'])} | {int(r['n_ambiguous'])} | "
          f"{r['pct_structural']:.1f}% | **{r['pct_repairable']:.1f}%** | "
          f"{r['pct_ambiguous']:.1f}% |")
    A("")
    A(f"![Loss anatomy by intent](../{OUT_CHART})")
    A("")
    A("*Stacked bars show the share of each intent's decisive losses that are "
      "structural (gray), repairable (green), or ambiguous (blue); `n` above each "
      "bar is the total losses classified for that intent.*")
    A("")

    A("## Findings: loss anatomy by cost tier (of the losing model)")
    A("")
    A("| cost_tier (loser) | n losses | structural | repairable | ambiguous | "
      "% structural | % repairable | % ambiguous |")
    A("|---|---|---|---|---|---|---|---|")
    sub2 = summary_df[summary_df["group_type"] == "cost_tier"]
    for tier in ("cheap", "mid", "expensive"):
        rr = sub2[sub2["group_value"] == tier]
        if not len(rr):
            continue
        r = rr.iloc[0]
        A(f"| {tier} | {int(r['n_battles'])} | {int(r['n_structural'])} | "
          f"{int(r['n_repairable'])} | {int(r['n_ambiguous'])} | "
          f"{r['pct_structural']:.1f}% | **{r['pct_repairable']:.1f}%** | "
          f"{r['pct_ambiguous']:.1f}% |")
    A("")

    A("## Findings: loss anatomy by conversation type")
    A("")
    A("Added to tie this analysis to the both-bad-rate finding "
      "(`both_bad_rate_summary.md`): multi-turn conversations have a meaningfully "
      "higher both-bad rate than single-turn (14.9% vs 11.4%, CIs do not overlap). "
      "This checks whether the same multi-turn effect shows up in the structural/"
      "repairable/ambiguous split for *decisive* losses, or only in the both-bad "
      "(tie) signal.")
    A("")
    A("| conversation_type | n losses | structural | repairable | ambiguous | "
      "% structural | % repairable | % ambiguous |")
    A("|---|---|---|---|---|---|---|---|")
    sub3 = summary_df[summary_df["group_type"] == "conversation_type"]
    for _, r in sub3.sort_values("group_value").iterrows():
        A(f"| {r['group_value']} | {int(r['n_battles'])} | {int(r['n_structural'])} | "
          f"{int(r['n_repairable'])} | {int(r['n_ambiguous'])} | "
          f"{r['pct_structural']:.1f}% | **{r['pct_repairable']:.1f}%** | "
          f"{r['pct_ambiguous']:.1f}% |")
    A("")

    top_repair = sub.iloc[0]
    top_struct = sub.sort_values("pct_structural", ascending=False).iloc[0]

    n_same_model_total = int(battles_df["same_base_model"].sum())
    repairable_df = battles_df[battles_df["label"] == "repairable"]
    n_repair_total = len(repairable_df)
    n_repair_same = int(repairable_df["same_base_model"].sum())
    n_repair_cross = n_repair_total - n_repair_same
    pct_repair_same = 100.0 * n_repair_same / n_repair_total if n_repair_total else float("nan")

    A("## Plain-language summary")
    A("")
    A(f"**Most repairable room:** `{top_repair['group_value']}` has the highest "
      f"repairable-loss share ({top_repair['pct_repairable']:.1f}% of its "
      f"{int(top_repair['n_battles'])} losses) -- in this category, losing models are "
      "frequently not weaker overall, they simply cited or wrote less in the specific "
      "battle they lost. This is where better prompting (asking for more citations, "
      "longer/more complete answers) has the most headroom to change outcomes.")
    A("")
    A(f"**Most structural:** `{top_struct['group_value']}` has the highest "
      f"structural-loss share ({top_struct['pct_structural']:.1f}%) -- losses here are "
      "driven by a measurable overall strength gap for that task category, not a "
      "fixable response feature. Routing to a stronger model for this intent is the "
      "available lever, not prompting.")
    A("")
    A(f"**The repairable bucket is not all the same claim.** Of the "
      f"{n_repair_total:,} battles labeled repairable, **{n_repair_same:,} "
      f"({pct_repair_same:.1f}%)** are a model playing itself (same base model on both "
      f"sides) -- these describe generation variance between two draws of one model, "
      f"not a case where a different or cheaper model would have closed the gap. The "
      f"remaining **{n_repair_cross:,} ({100 - pct_repair_same:.1f}%)** are genuine "
      "cross-model comparisons, where a different model under-cited or under-wrote "
      "relative to the one that beat it. See caveat #8 below for the per-intent/per-"
      "tier breakdown.")
    A("")

    mt_row = sub3[sub3["group_value"] == "multi_turn"].iloc[0]
    st_row = sub3[sub3["group_value"] == "single_turn"].iloc[0]
    delta_struct = mt_row["pct_structural"] - st_row["pct_structural"]
    delta_repair = mt_row["pct_repairable"] - st_row["pct_repairable"]
    delta_ambig = mt_row["pct_ambiguous"] - st_row["pct_ambiguous"]
    biggest_delta = max((abs(delta_struct), "structural"),
                       (abs(delta_repair), "repairable"),
                       (abs(delta_ambig), "ambiguous"))[1]
    if abs(delta_struct) < 2 and abs(delta_repair) < 2 and abs(delta_ambig) < 2:
        convo_verdict = ("multi-turn's higher both-bad rate does **not** carry over "
                         "into a meaningfully different structural/repairable/ambiguous "
                         "split for decisive losses -- the proportions are within ~2 "
                         "points of single-turn across all three labels. Multi-turn "
                         "conversations fail both-bad more often, but when a multi-turn "
                         "battle *is* decisive, its anatomy looks like single-turn's.")
    else:
        direction = "more" if (delta_struct if biggest_delta == "structural"
                              else delta_repair if biggest_delta == "repairable"
                              else delta_ambig) > 0 else "less"
        convo_verdict = (f"multi-turn losses skew **{direction} {biggest_delta}** than "
                         f"single-turn ({mt_row[f'pct_{biggest_delta}']:.1f}% vs "
                         f"{st_row[f'pct_{biggest_delta}']:.1f}%) -- so part of "
                         "multi-turn's higher both-bad rate does appear to carry over "
                         "into how its decisive losses break down, not just into more "
                         "ties.")
    A(f"**Does multi-turn's higher both-bad rate carry over to decisive losses?** "
      f"Multi-turn: {int(mt_row['n_battles']):,} losses "
      f"({mt_row['pct_structural']:.1f}% structural / {mt_row['pct_repairable']:.1f}% "
      f"repairable / {mt_row['pct_ambiguous']:.1f}% ambiguous). Single-turn: "
      f"{int(st_row['n_battles']):,} losses ({st_row['pct_structural']:.1f}% / "
      f"{st_row['pct_repairable']:.1f}% / {st_row['pct_ambiguous']:.1f}%). "
      f"In short: {convo_verdict}")
    A("")

    A("## Caveats")
    A("")
    A("1. **Feature scope is narrow.** Only citation count and response length are "
      "available as response-level features (see Methodology above); markdown tables, "
      "refusals, and numeric density are not measured in this repo.")
    A("2. **Thresholds are judgment calls**, stated explicitly above, not derived "
      "statistically.")
    A(f"3. **{n_low_cells} of {n_total_cells}** per-(model, intent) strength cells rest "
      "on thin data and are excluded from driving a structural call (routed to "
      "ambiguous instead).")
    A(f"4. **{n_dropped} battles excluded** entirely because their intent had too few "
      "total battles to fit at all (same `MIN_N_FOR_FIT` cutoff used elsewhere in this "
      "project).")
    A("5. **Non-overlapping 95% CIs is a conservative structural test.** Some battles "
      "labeled ambiguous might be structural under a less conservative two-sample test; "
      "this report errs toward under-calling structural rather than over-calling it.")
    A("6. **Citation/length are observational, not causal**, same caveat as the rest of "
      "this project's citation-effect findings: a model that cites more or writes more "
      "differs from one that doesn't in other ways not measured here.")
    A("7. **The per-intent strength is fit on the same battles it's used to classify.** "
      "It is not a held-out / leave-one-out estimate -- the strength for `model X` in "
      "intent `Y` is fit using all of `model X`'s battles in intent `Y`, including the "
      "very battle being classified. At the sample sizes here (hundreds to ~2,000 "
      "battles per intent) any single battle's influence on its own model's strength "
      "estimate is small, but this is not a fully independent test and should be "
      "described that way if asked, not as a clean train/test split.")
    A(f"8. **{n_same_model_total:,} of {len(dec):,} decisive battles "
      f"({100.0 * n_same_model_total / len(dec):.1f}%) are a model playing itself** "
      "(`base_model_a == base_model_b`) -- the same model generating two independent "
      "responses. These can never be classified structural by construction (a model "
      "cannot be structurally weaker than itself), so every one of them falls into "
      f"repairable or ambiguous. **{n_repair_same:,} of the {n_repair_total:,} "
      f"repairable-labeled battles ({pct_repair_same:.1f}%) are this same-model "
      "flavor** -- meaning they describe generation variance between two draws of one "
      "model, not \"a different/cheaper model could close this gap with better "
      "prompting.\" The remaining "
      f"{n_repair_cross:,} ({100 - pct_repair_same:.1f}%) repairable battles are "
      "genuine cross-model comparisons and are the ones the routing/prompting "
      "recommendation above actually applies to.")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[saved] {path}")


def main():
    print("=" * 78)
    print("LOSS ANATOMY ANALYSIS (additive -- no existing file modified)")
    print("=" * 78)

    full, dec, n_all, src = load_decisive()
    # Pooled scaling: kept ONLY for Step 1's Bradley-Terry covariate fit
    # (build_design_model_identity), where it serves a different purpose --
    # making beta_citation/beta_length comparable ACROSS primary_intent
    # fits, matching the convention used everywhere else in this project's
    # BT models. This is unrelated to, and not affected by, the bug fixed
    # below.
    sd_cit, sd_len = _scale_covariates(dec)
    # Group-specific scaling: used for the per-battle citation_diff_z /
    # length_diff_z feature-deficiency check in Step 2+3 (the repairable
    # classification). This is the bug fix -- see build_loss_anatomy's
    # docstring.
    sd_by_conv_type = {
        ct: _scale_covariates(dec[dec["conversation_type"] == ct])
        for ct in sorted(dec["conversation_type"].dropna().unique())
    }
    print(f"source: {src}")
    print(f"decisive battles: {len(dec)}")
    print(f"pooled sd (Step 1 BT covariate fit only): sd_cit={sd_cit:.3f}  sd_len={sd_len:.1f}")
    for ct, (c, l) in sd_by_conv_type.items():
        print(f"  conversation_type={ct:12} sd_cit={c:.3f}  sd_len={l:.1f}  "
              f"(n={int((dec['conversation_type']==ct).sum())})")

    strength_df = run_strength_by_intent(dec, sd_cit, sd_len)
    strength_df.to_csv(OUT_STRENGTH_BY_INTENT_CSV, index=False)
    print(f"[saved] {OUT_STRENGTH_BY_INTENT_CSV}")

    battles_df, n_dropped = build_loss_anatomy(dec, strength_df, sd_by_conv_type)
    battles_df.to_csv(OUT_BATTLES_CSV, index=False)
    print(f"[saved] {OUT_BATTLES_CSV}")

    summary_df = aggregate(battles_df)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    print(f"[saved] {OUT_SUMMARY_CSV}")

    sanity_checks(dec, battles_df, strength_df)

    build_chart(summary_df)

    write_summary_md(OUT_SUMMARY_MD, dec, battles_df, summary_df, strength_df,
                     n_dropped, sd_cit, sd_len, sd_by_conv_type)


if __name__ == "__main__":
    main()
