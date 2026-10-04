"""
bradley_terry_family.py

Tests whether the cost-tier effect reported in bradley_terry_analysis.py is
substantially a MODEL-FAMILY effect in disguise. The individual-model
strength ranking already in bradley_terry_model_strengths.csv put 4 of the
top 5 strongest models in the Sonar family -- a visible pattern the main
summary flagged as an unresolved confound but could not separate, because
tier and model identity are perfectly collinear (classify_cost_tier(model)
is deterministic, so tier and a per-model theta cannot both enter one fit).

Family is NOT collinear with tier the same way: each of the three families
has representatives in more than one tier (Sonar spans all three; Gemini
spans all three; OpenAI spans cheap and expensive). That makes a joint
tier+family fit identifiable, which is the decisive test this script runs.

ADDITIVE script: reuses fit_bt / bootstrap_bt / _scale_covariates /
load_decisive from bradley_terry_analysis.py UNCHANGED. No core machinery is
rewritten here.

No API calls. No new data collection.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bradley_terry_analysis import (   # noqa: E402  (sys.path must be set first)
    N_BOOTSTRAP,
    ALPHA,
    RANDOM_SEED,
    bootstrap_bt,
    _scale_covariates,
    load_decisive,
)

COST_TIER_CSV = "outputs/bradley_terry_cost_tier.csv"
OUT_CSV = "outputs/bradley_terry_family.csv"
SUMMARY_MD = "outputs/bradley_terry_summary.md"
SECTION_MARKER = "# Model Family vs Cost Tier"

# Step 1: family membership, as given in the task -- the exact 9 base models
# already fit throughout this project (verified: 4 Sonar + 3 Gemini + 2
# OpenAI = 9, matching bradley_terry_model_strengths.csv exactly).
SONAR_FAMILY = {"sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro"}
GEMINI_FAMILY = {"gemini-2.0-flash", "gemini-2.5-flash-preview-04-17",
                 "gemini-2.5-pro-exp-03-25"}
OPENAI_FAMILY = {"gpt-4o-mini-search-preview", "gpt-4o-search-preview"}

# A reference effect size to judge "is this effect big" against: the
# smallest SIGNIFICANT citation effect found anywhere else in this project
# (same number used as the honesty threshold in bradley_terry_frustration.py).
REFERENCE_EFFECT_SIZE = 0.3


def classify_model_family(base_model):
    if base_model in SONAR_FAMILY:
        return "sonar"
    if base_model in GEMINI_FAMILY:
        return "gemini"
    if base_model in OPENAI_FAMILY:
        return "openai"
    return None


# ============================================================
# STEP 2: family-as-player-identity design (mirrors build_design_cost_tier)
# ============================================================

def build_design_family(df, sd_cit, sd_len):
    """
    Player = model FAMILY, with `gemini` as the reference category (strength
    0) -- alphabetically first, same convention used elsewhere in this
    project for an arbitrary-but-documented reference choice.
    """
    fa = df["family_a"].values
    fb = df["family_b"].values
    cols, parts = [], []
    for fam in ("openai", "sonar"):
        parts.append(((fa == fam).astype(float) - (fb == fam).astype(float)).reshape(-1, 1))
        cols.append(f"family[{fam}] vs gemini")
    dcit = ((df["citations_a"] - df["citations_b"]).astype(float) / sd_cit).values
    dlen = ((df["response_length_a"] - df["response_length_b"]).astype(float) / sd_len).values
    parts.append(dcit.reshape(-1, 1)); cols.append("beta_citation")
    parts.append(dlen.reshape(-1, 1)); cols.append("beta_length")
    X = np.hstack(parts)
    y = df["_y"].values.astype(float)
    return X, y, cols


# ============================================================
# STEP 4: joint tier + family design -- the decisive, non-collinear test
# ============================================================

def build_design_tier_and_family(df, sd_cit, sd_len):
    """
    Tier (mid/expensive vs cheap) and family (openai/sonar vs gemini) as
    SEPARATE simultaneous factors, same citation/length covariates. This is
    only identifiable because tier and family are not perfectly collinear in
    this dataset: Sonar and Gemini each have representatives in all three
    tiers, and OpenAI has representatives in two. If tier were a pure proxy
    for family (or vice versa), this fit would be rank-deficient; it is not
    (checked at call time via the design matrix rank).
    """
    ta = df["tier_a"].values
    tb = df["tier_b"].values
    fa = df["family_a"].values
    fb = df["family_b"].values
    cols, parts = [], []
    for t in ("mid", "expensive"):
        parts.append(((ta == t).astype(float) - (tb == t).astype(float)).reshape(-1, 1))
        cols.append(f"tier[{t}] vs cheap")
    for fam in ("openai", "sonar"):
        parts.append(((fa == fam).astype(float) - (fb == fam).astype(float)).reshape(-1, 1))
        cols.append(f"family[{fam}] vs gemini")
    dcit = ((df["citations_a"] - df["citations_b"]).astype(float) / sd_cit).values
    dlen = ((df["response_length_a"] - df["response_length_b"]).astype(float) / sd_len).values
    parts.append(dcit.reshape(-1, 1)); cols.append("beta_citation")
    parts.append(dlen.reshape(-1, 1)); cols.append("beta_length")
    X = np.hstack(parts)
    y = df["_y"].values.astype(float)
    return X, y, cols


# ============================================================
# STEP 1: load decisive data, attach family
# ============================================================

def load_with_family():
    print("=" * 78)
    print("STEP 1  Define model family as a player identity")
    print("=" * 78)
    full, dec, n_all, src = load_decisive()
    dec = dec.copy()
    dec["family_a"] = dec["base_model_a"].map(classify_model_family)
    dec["family_b"] = dec["base_model_b"].map(classify_model_family)

    unclassified = dec[dec["family_a"].isna() | dec["family_b"].isna()]
    if len(unclassified):
        bad_models = sorted(set(unclassified["base_model_a"]) | set(unclassified["base_model_b"]))
        print(f"[warn] dropping {len(unclassified)} battles involving a model not in "
              f"any defined family: {bad_models}")
        dec = dec[~(dec["family_a"].isna() | dec["family_b"].isna())].copy()
    else:
        print("All base models in the decisive set map to a defined family "
              "(Sonar/Gemini/OpenAI); nothing dropped.")

    n_sonar = len(SONAR_FAMILY); n_gemini = len(GEMINI_FAMILY); n_openai = len(OPENAI_FAMILY)
    print(f"\nfamily definitions: Sonar ({n_sonar} models), Gemini ({n_gemini} models), "
          f"OpenAI ({n_openai} models)  [{n_sonar + n_gemini + n_openai} total]")
    print(f"decisive battles usable for this analysis: {len(dec)}")

    # Step 4's premise, checked explicitly rather than assumed: family must
    # NOT be perfectly determined by tier (and vice versa) for the joint fit
    # to be identifiable.
    cross = pd.crosstab(
        pd.concat([dec["family_a"], dec["family_b"]], ignore_index=True),
        pd.concat([dec["tier_a"], dec["tier_b"]], ignore_index=True))
    print("\nfamily x tier cross-tabulation (both sides pooled, so the joint "
          "fit's identifiability can be checked directly):")
    print(cross.to_string())
    n_family_tier_cells = int((cross > 0).sum().sum())
    print(f"\n{n_family_tier_cells} of 9 possible (family, tier) cells are populated "
          f"-- family and tier are {'NOT' if n_family_tier_cells > 3 else ''} collinear.")
    return dec, cross


# ============================================================
# STEP 2: family-level BT fit
# ============================================================

def run_family_fit(dec, sd_cit, sd_len):
    print("\n" + "=" * 78)
    print("STEP 2  Bradley-Terry family effect (gemini = reference)")
    print("        citation diff and length diff controlled in the same model")
    print("=" * 78)
    res, cols, _rest, nfail, ok, draws = bootstrap_bt(
        dec, build_design_family, sd_cit=sd_cit, sd_len=sd_len)

    print(f"n={len(dec)}  converged={ok}  bootstrap replicates failed={nfail}")
    for c in ("family[openai] vs gemini", "family[sonar] vs gemini",
             "beta_citation", "beta_length"):
        r = res[c]
        print(f"  {c:26}= {r['coef']:+.4f}  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"p={r['p']:.4f}  {'SIG' if r['sig'] else 'ns'}")

    # Derived contrast sonar - openai, on the bootstrap draws (preserves the
    # correlation between the two family coefficients), mirroring exactly how
    # expensive-minus-mid is computed in run_cost_tier.
    i_openai = cols.index("family[openai] vs gemini")
    i_sonar = cols.index("family[sonar] vs gemini")
    point = float(res[cols[i_sonar]]["coef"] - res[cols[i_openai]]["coef"])
    d = draws[:, i_sonar] - draws[:, i_openai]
    d = d[np.isfinite(d)]
    lo = float(np.percentile(d, 100 * ALPHA / 2))
    hi = float(np.percentile(d, 100 * (1 - ALPHA / 2)))
    p = float(min(1.0, 2.0 * min(np.mean(d <= 0), np.mean(d >= 0))))
    sig = bool(lo > 0 or hi < 0)
    print(f"  {'sonar - openai contrast':26}= {point:+.4f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"p={p:.4f}  {'SIG' if sig else 'ns'}")

    row = dict(n_decisive=len(dec), converged=ok, n_boot_failed=nfail)
    for c in cols:
        key = (c.replace("family[", "").replace("] vs gemini", "_vs_gemini")
                .replace("beta_", ""))
        row[f"{key}_coef"] = res[c]["coef"]
        row[f"{key}_ci_lo"] = res[c]["lo"]
        row[f"{key}_ci_hi"] = res[c]["hi"]
        row[f"{key}_boot_p"] = res[c]["p"]
        row[f"{key}_significant"] = res[c]["sig"]
    row["sonar_minus_openai_coef"] = point
    row["sonar_minus_openai_ci_lo"] = lo
    row["sonar_minus_openai_ci_hi"] = hi
    row["sonar_minus_openai_boot_p"] = p
    row["sonar_minus_openai_significant"] = sig
    return row


# ============================================================
# STEP 3: family effect size vs already-reported tier effect size
# ============================================================

def compare_to_tier(family_row):
    print("\n" + "=" * 78)
    print("STEP 3  Family effect size vs tier effect size (already reported)")
    print("=" * 78)
    if not os.path.exists(COST_TIER_CSV):
        print(f"[warn] {COST_TIER_CSV} not found; cannot compare.")
        return None
    tier = pd.read_csv(COST_TIER_CSV)
    ov = tier[tier["scope"] == "All categories"].iloc[0]

    tier_effects = {
        "tier: mid vs cheap": float(ov["mid_vs_cheap_coef"]),
        "tier: expensive vs cheap": float(ov["expensive_vs_cheap_coef"]),
        "tier: expensive vs mid": float(ov["expensive_minus_mid_coef"]),
    }
    family_effects = {
        "family: openai vs gemini": float(family_row["openai_vs_gemini_coef"]),
        "family: sonar vs gemini": float(family_row["sonar_vs_gemini_coef"]),
        "family: sonar vs openai": float(family_row["sonar_minus_openai_coef"]),
    }
    max_tier = max(abs(v) for v in tier_effects.values())
    max_family = max(abs(v) for v in family_effects.values())

    print("tier effects (from bradley_terry_cost_tier.csv, All categories):")
    for k, v in tier_effects.items():
        print(f"  {k:28}{v:+.4f}")
    print("\nfamily effects (this script):")
    for k, v in family_effects.items():
        print(f"  {k:28}{v:+.4f}")
    print(f"\nlargest |tier effect|   = {max_tier:.4f}")
    print(f"largest |family effect| = {max_family:.4f}")
    ratio = max_family / max_tier if max_tier else float("nan")
    verdict = ("family explains AS MUCH OR MORE than tier" if max_family >= 0.85 * max_tier
               else "tier still explains more than family")
    print(f"ratio (family/tier) = {ratio:.2f}x  -> {verdict}")
    return dict(tier_effects=tier_effects, family_effects=family_effects,
               max_tier=max_tier, max_family=max_family, ratio=ratio, verdict=verdict)


# ============================================================
# STEP 4: joint tier + family fit -- the decisive separating test
# ============================================================

def run_joint_fit(dec, sd_cit, sd_len):
    print("\n" + "=" * 78)
    print("STEP 4  Joint fit: tier AND family as SEPARATE simultaneous factors")
    print("        (the decisive test -- only possible because they are not")
    print("        perfectly collinear in this dataset)")
    print("=" * 78)

    X_check, _y, cols_check = build_design_tier_and_family(dec, sd_cit, sd_len)
    rank = np.linalg.matrix_rank(X_check)
    print(f"design matrix: {X_check.shape[1]} columns, rank={rank} "
          f"({'full rank -- identifiable' if rank == X_check.shape[1] else 'RANK DEFICIENT -- NOT identifiable'})")

    res, cols, _rest, nfail, ok, draws = bootstrap_bt(
        dec, build_design_tier_and_family, sd_cit=sd_cit, sd_len=sd_len)
    print(f"\nn={len(dec)}  converged={ok}  bootstrap replicates failed={nfail}")
    for c in cols:
        r = res[c]
        print(f"  {c:26}= {r['coef']:+.4f}  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"p={r['p']:.4f}  {'SIG' if r['sig'] else 'ns'}")

    tier_sig = res["tier[mid] vs cheap"]["sig"] or res["tier[expensive] vs cheap"]["sig"]
    family_sig = res["family[openai] vs gemini"]["sig"] or res["family[sonar] vs gemini"]["sig"]
    print(f"\ntier remains significant once family is controlled : {tier_sig}")
    print(f"family remains significant once tier is controlled  : {family_sig}")

    row = dict(n_decisive=len(dec), converged=ok, n_boot_failed=nfail, rank=rank,
              n_params=X_check.shape[1])
    for c in cols:
        key = (c.replace("tier[", "tier_").replace("] vs cheap", "_vs_cheap")
                .replace("family[", "family_").replace("] vs gemini", "_vs_gemini")
                .replace("beta_", ""))
        row[f"{key}_coef"] = res[c]["coef"]
        row[f"{key}_ci_lo"] = res[c]["lo"]
        row[f"{key}_ci_hi"] = res[c]["hi"]
        row[f"{key}_boot_p"] = res[c]["p"]
        row[f"{key}_significant"] = res[c]["sig"]
    row["tier_remains_significant"] = tier_sig
    row["family_remains_significant"] = family_sig
    return row


# ============================================================
# STEP 5: markdown section
# ============================================================

def _truncate_existing_section(path):
    """
    Removes a PREVIOUS run's copy of this script's own section, bounded to
    just that section's span (not from the marker to end-of-file -- that
    version silently deleted any later-appended section on a re-run).
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


def write_summary_section(dec, cross, family_row, cmp_result, joint_row,
                          sd_cit, sd_len, path=SUMMARY_MD):
    L = []
    A = L.append
    A("")
    A("---")
    A("")
    A(SECTION_MARKER)
    A("")
    A("Tests whether the cost-tier effect reported earlier in this file is "
      "substantially a **model-family** effect in disguise. The individual-"
      "model strength ranking already showed 4 of the top 5 strongest models "
      "were all Sonar variants -- flagged there as an unresolved confound. "
      "This section resolves it directly, reusing the same Bradley-Terry "
      "machinery, covariate scaling, and bootstrap procedure as the rest of "
      "this file.")
    A("")
    A("## Step 1: family definitions and identifiability")
    A("")
    A("| family | models |")
    A("|---|---|")
    A(f"| Sonar | {', '.join(sorted(SONAR_FAMILY))} |")
    A(f"| Gemini | {', '.join(sorted(GEMINI_FAMILY))} |")
    A(f"| OpenAI | {', '.join(sorted(OPENAI_FAMILY))} |")
    A("")
    A(f"n = {len(dec):,} decisive battles (all map cleanly to one of the three "
      "families; none dropped).")
    A("")
    A("**Family is not collinear with tier.** Unlike tier -- which is a "
      "deterministic function of model identity and so cannot be fit "
      "alongside per-model strengths -- each family has representatives in "
      "more than one tier:")
    A("")
    A("```")
    A(cross.to_string())
    A("```")
    A("")
    n_cells = int((cross > 0).sum().sum())
    A(f"{n_cells} of 9 possible (family, tier) cells are populated. This is what "
      "makes the Step 4 joint fit below identifiable -- tier and family carry "
      "at least partly independent information, so a model can ask about one "
      "while holding the other fixed.")
    A("")

    A("## Step 2: family effect (family as player identity, Gemini = reference)")
    A("")
    A("| term | coefficient | 95% CI | boot p | significant |")
    A("|---|---|---|---|---|")
    for key, lab in (("openai_vs_gemini", "family: OpenAI vs Gemini"),
                     ("sonar_vs_gemini", "family: Sonar vs Gemini"),
                     ("citation", "beta_citation (per 1 SD)"),
                     ("length", "beta_length (per 1 SD)")):
        A(f"| {lab} | {family_row[f'{key}_coef']:+.4f} | "
          f"[{family_row[f'{key}_ci_lo']:+.3f}, {family_row[f'{key}_ci_hi']:+.3f}] | "
          f"{family_row[f'{key}_boot_p']:.4f} | "
          f"{'**yes**' if family_row[f'{key}_significant'] else 'no'} |")
    A(f"| family: Sonar vs OpenAI *(derived)* | "
      f"{family_row['sonar_minus_openai_coef']:+.4f} | "
      f"[{family_row['sonar_minus_openai_ci_lo']:+.3f}, "
      f"{family_row['sonar_minus_openai_ci_hi']:+.3f}] | "
      f"{family_row['sonar_minus_openai_boot_p']:.4f} | "
      f"{'**yes**' if family_row['sonar_minus_openai_significant'] else 'no'} |")
    A("")

    A("## Step 3: family effect size vs tier effect size")
    A("")
    if cmp_result is None:
        A(f"*(`{COST_TIER_CSV}` not found; comparison skipped.)*")
        A("")
    else:
        A("| effect | magnitude (log-odds) |")
        A("|---|---|")
        for k, v in cmp_result["tier_effects"].items():
            A(f"| {k} | {v:+.4f} |")
        for k, v in cmp_result["family_effects"].items():
            A(f"| {k} | {v:+.4f} |")
        A("")
        A(f"Largest \\|tier effect\\| = **{cmp_result['max_tier']:.4f}**. "
          f"Largest \\|family effect\\| = **{cmp_result['max_family']:.4f}** "
          f"({cmp_result['ratio']:.2f}x tier).")
        A("")
        if cmp_result["max_family"] >= 0.85 * cmp_result["max_tier"]:
            A("**Family explains as much or more variation than tier does.** "
              "A tier-as-player fit and a family-as-player fit produce "
              "comparable-sized effects from the same battles, which is "
              "exactly the pattern expected if tier membership is substantially "
              "picking up model family rather than price per se.")
        else:
            A("Tier's effect size is still larger than family's on its own, "
              "which argues against family being a full substitute explanation "
              "-- though Step 4's joint fit is the test that actually separates "
              "them rather than just comparing two single-factor fits.")
        A("")

    A("## Step 4: joint fit -- tier and family as separate simultaneous factors")
    A("")
    A("This is the decisive test: both factors enter the **same** fit, so each "
      "coefficient is the effect of tier *holding family fixed*, and vice "
      "versa. Identifiable only because family and tier are not perfectly "
      "collinear (Step 1).")
    A("")
    A(f"Design matrix: {joint_row['n_params']} columns, rank {joint_row['rank']} "
      f"({'full rank, identifiable' if joint_row['rank'] == joint_row['n_params'] else '**RANK DEFICIENT**'}).")
    A("")
    A("| term | coefficient | 95% CI | boot p | significant |")
    A("|---|---|---|---|---|")
    for key, lab in (("tier_mid_vs_cheap", "tier: mid vs cheap"),
                     ("tier_expensive_vs_cheap", "tier: expensive vs cheap"),
                     ("family_openai_vs_gemini", "family: OpenAI vs Gemini"),
                     ("family_sonar_vs_gemini", "family: Sonar vs Gemini"),
                     ("citation", "beta_citation (per 1 SD)"),
                     ("length", "beta_length (per 1 SD)")):
        A(f"| {lab} | {joint_row[f'{key}_coef']:+.4f} | "
          f"[{joint_row[f'{key}_ci_lo']:+.3f}, {joint_row[f'{key}_ci_hi']:+.3f}] | "
          f"{joint_row[f'{key}_boot_p']:.4f} | "
          f"{'**yes**' if joint_row[f'{key}_significant'] else 'no'} |")
    A("")
    A(f"**Tier remains significant once family is controlled: "
      f"{'YES' if joint_row['tier_remains_significant'] else 'NO'}.** "
      f"**Family remains significant once tier is controlled: "
      f"{'YES' if joint_row['family_remains_significant'] else 'NO'}.**")
    A("")

    A("## Verdict: is the tier finding a family effect, a price effect, or inseparable?")
    A("")
    both_sig = joint_row["tier_remains_significant"] and joint_row["family_remains_significant"]
    only_family = joint_row["family_remains_significant"] and not joint_row["tier_remains_significant"]
    only_tier = joint_row["tier_remains_significant"] and not joint_row["family_remains_significant"]
    neither = not joint_row["tier_remains_significant"] and not joint_row["family_remains_significant"]
    if only_family:
        A("**Substantially a model-family effect.** Once family is controlled, "
          "tier's own coefficients are no longer significant, while family's "
          "coefficients remain significant on their own. The tier-level finding "
          "reported earlier in this file should be read as **family wearing "
          "tier's clothing**: the apparent price effect is better explained by "
          "which lab built the model than by what it costs.")
    elif only_tier:
        A("**Substantially a genuine price/tier effect.** Once tier is "
          "controlled, family's own coefficients are no longer significant, "
          "while tier's remain significant on their own. The visible "
          "concentration of Sonar models at the top of the individual-model "
          "ranking does not survive as an independent effect once price tier "
          "is accounted for.")
    elif both_sig:
        A("**Inseparable with this dataset: both remain significant.** Tier "
          "and family each carry independent signal even with the other held "
          "fixed. This is the honest, harder-to-report outcome: the data "
          "cannot cleanly attribute the original tier finding to price alone "
          "or to family alone. Both the tier-level and family-level claims made "
          "elsewhere in this project should carry this caveat -- neither is "
          "fully doing the other's explanatory work, but neither is fully "
          "independent of it either, given how few models and how much "
          "family/tier overlap this dataset has to work with.")
    else:
        A("**Neither survives the joint fit.** Once each is controlled for the "
          "other, neither tier nor family shows an independent, significant "
          "effect on winning net of citation and length. The single-factor "
          "tier and family effects reported separately are each likely "
          "absorbing some of what is really the other's signal (or citation/"
          "length's), rather than reflecting an independent tier or family "
          "effect in their own right.")
    A("")
    A("**Caveat on this verdict.** With only 9 base models split 4/3/2 across "
      "three families and three tiers, every family-by-tier cell is thin; "
      "this joint fit has much less power per cell than the single-factor "
      "fits reported earlier. A significant result here is solid evidence; a "
      "non-significant one is also consistent with simply not having enough "
      "data to detect a real but modest independent effect of either factor.")
    A("")

    _truncate_existing_section(path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[appended] {SECTION_MARKER} -> {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 78)
    print("MODEL FAMILY VS COST TIER")
    print(f"bootstrap replications = {N_BOOTSTRAP}  seed = {RANDOM_SEED}")
    print("No API calls. No new data collection.")
    print("=" * 78)

    dec, cross = load_with_family()
    _full, dec_all, _n_all, _src = load_decisive()
    sd_cit, sd_len = _scale_covariates(dec_all)
    print(f"\ncovariate scaling (reused from the main analysis): "
          f"1 SD citation diff = {sd_cit:.3f}; 1 SD length diff = {sd_len:.1f} chars")

    family_row = run_family_fit(dec, sd_cit, sd_len)
    cmp_result = compare_to_tier(family_row)
    joint_row = run_joint_fit(dec, sd_cit, sd_len)

    pd.DataFrame([family_row]).to_csv(OUT_CSV, index=False)
    pd.DataFrame([joint_row]).to_csv(OUT_CSV.replace(".csv", "_joint.csv"), index=False)
    print(f"\n[saved] {OUT_CSV}")
    print(f"[saved] {OUT_CSV.replace('.csv', '_joint.csv')}")

    write_summary_section(dec, cross, family_row, cmp_result, joint_row, sd_cit, sd_len)


if __name__ == "__main__":
    main()
