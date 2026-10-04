"""
bradley_terry_english_only.py

English-only replication of the core Bradley-Terry findings (citation effect
by intent, cost-tier effect, and the family-vs-tier joint fit), to check
whether they hold once non-English conversations are excluded.

`languages` is NOT present in final_analysis_derived.csv (verified: its
columns are primary_intent, turn_count, conversation_type, citations_a,
citations_b, response_length_a, response_length_b, base_model_a, base_model_b,
model_a, model_b, winner, is_tie, is_decisive, vote_occurred -- no languages
field). It has to be pulled from train.jsonl directly and merged in by row
index, using the exact same indexed-merge pattern already verified for the
frustration labels (load_decisive()'s df preserves the same 0-based row
index as a line number in train.jsonl; no reset_index is ever called on it).

ADDITIVE script: reuses fit_bt / bootstrap_bt / build_design_model_identity /
build_design_cost_tier / _scale_covariates / load_decisive from
bradley_terry_analysis.py UNCHANGED, and classify_model_family /
build_design_family / build_design_tier_and_family from
bradley_terry_family.py UNCHANGED. No existing script is modified.

No API calls. No new data collection.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bradley_terry_analysis import (   # noqa: E402
    N_BOOTSTRAP,
    ALPHA,
    RANDOM_SEED,
    MIN_N_FOR_FIT,
    LOW_POWER_N,
    bootstrap_bt,
    build_design_model_identity,
    build_design_cost_tier,
    _scale_covariates,
    load_decisive,
    point_of,
)
from bradley_terry_family import (     # noqa: E402
    classify_model_family,
    build_design_family,
    build_design_tier_and_family,
)
from final_analysis import INTENT_ORDER, DATASET_FILE   # noqa: E402

SUMMARY_MD = "outputs/bradley_terry_summary.md"
SECTION_MARKER = "# English-Only Replication"

BY_INTENT_CSV = "outputs/bradley_terry_english_only_by_intent.csv"
COST_TIER_CSV = "outputs/bradley_terry_english_only_cost_tier.csv"
JOINT_CSV = "outputs/bradley_terry_english_only_family_joint.csv"
COMPARISON_CSV = "outputs/bradley_terry_english_only_comparison.csv"

# Sample-size guardrail named explicitly in the task: if the English-only
# subset falls under half of the pooled 8,613, flag it and stop rather than
# silently running a replication that could misread "underpowered" as
# "didn't replicate".
POOLED_N = 8613
POWER_WARNING_FRACTION = 0.5


# ============================================================
# STEP 1: pull `languages` from train.jsonl, confirm its raw format
# ============================================================

def extract_languages(sample_print=10):
    print("=" * 78)
    print("STEP 1  Extract `languages` from train.jsonl, confirm raw format")
    print("=" * 78)
    print(f"`languages` is absent from final_analysis_derived.csv (confirmed columns: "
          "primary_intent, turn_count, conversation_type, citations_a, citations_b, "
          "response_length_a, response_length_b, base_model_a, base_model_b, model_a, "
          "model_b, winner, is_tie, is_decisive, vote_occurred). Pulling it directly "
          f"from {DATASET_FILE} instead, indexed by line number (0-based), the same "
          "indexing load_decisive()'s df already preserves.")
    print()
    print(f"Raw `languages` field for the first {sample_print} records of "
          f"{DATASET_FILE}, exactly as json.loads returns it (no str()/tuple "
          "conversion applied):")

    languages_by_idx = {}
    combo_counts = {}
    n_shown = 0
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            langs = record.get("languages")
            languages_by_idx[i] = langs
            if n_shown < sample_print:
                print(f"  idx={i:6}  type={type(langs).__name__:6}  raw={langs!r}")
                n_shown += 1
            key = tuple(langs) if isinstance(langs, (list, tuple)) else (str(langs),)
            combo_counts[key] = combo_counts.get(key, 0) + 1

    n_total = len(languages_by_idx)
    print(f"\nConfirmed raw format: a JSON array of language-name strings per "
          "conversation (not a stringified Python tuple -- that representation "
          "seen in frustration_sample300.jsonl was introduced later in that "
          "pipeline's own str() conversion, not the source format).")
    print(f"\n{n_total} total conversations scanned.")
    print(f"{len(combo_counts)} distinct language combinations found across the full dataset.")

    top = sorted(combo_counts.items(), key=lambda kv: -kv[1])[:15]
    print("\nmost common combinations:")
    for combo, n in top:
        print(f"  {n:7}  {combo}")

    return languages_by_idx, combo_counts


# ============================================================
# STEP 2: define "English-only" explicitly, justified by the real numbers
# ============================================================

def classify_english_status(langs):
    """
    Returns one of: 'english_only', 'english_plus_other', 'no_english',
    'missing'. A conversation with NO language metadata at all is kept
    distinct from "no English" rather than folded into it, since "unlabeled"
    and "confirmed non-English" are different kinds of missingness.
    """
    if not isinstance(langs, (list, tuple)) or len(langs) == 0:
        return "missing"
    has_english = any(str(l).strip().lower() == "english" for l in langs)
    if not has_english:
        return "no_english"
    if len(langs) == 1:
        return "english_only"
    return "english_plus_other"


def run_step2_definition(languages_by_idx):
    print("\n" + "=" * 78)
    print("STEP 2  Define 'English-only', justified by the actual distribution")
    print("=" * 78)

    status_counts = {"english_only": 0, "english_plus_other": 0,
                     "no_english": 0, "missing": 0}
    status_by_idx = {}
    for idx, langs in languages_by_idx.items():
        status = classify_english_status(langs)
        status_by_idx[idx] = status
        status_counts[status] += 1

    n_total = len(languages_by_idx)
    print("Observed breakdown across ALL conversations (not yet restricted to "
          "decisive battles):")
    for k in ("english_only", "english_plus_other", "no_english", "missing"):
        n = status_counts[k]
        print(f"  {k:20}{n:7}  ({100.0 * n / n_total:5.1f}%)")

    print(f"\nRULE ADOPTED: 'English-only' = the `languages` field lists English "
          "and ONLY English (status == 'english_only'). Conversations tagged "
          "English-plus-other-language(s) are EXCLUDED, not included, because "
          "the assistant's response and/or the user's other turns in a mixed-"
          "language conversation are not reliably English either -- including "
          "them would leave non-English text in a nominally 'English-only' "
          "replication. This is a deliberately conservative definition: it "
          "will under-count 'conversations touched by English' but gives a "
          "cleaner single-language subset for the replication's purpose.")

    return status_by_idx, status_counts


# ============================================================
# STEP 3: merge into the decisive dataset, filter, report counts
# ============================================================

def run_step3_merge_and_filter(status_by_idx):
    print("\n" + "=" * 78)
    print("STEP 3  Merge into the decisive dataset and filter")
    print("=" * 78)
    full, dec, n_all, src = load_decisive()
    print(f"decisive dataset source: {src}")
    n_before = len(dec)
    print(f"decisive battles before English-only filtering: {n_before:,}")

    dec = dec.copy()
    dec["english_status"] = dec.index.map(status_by_idx.get)
    missing_status = dec["english_status"].isna().sum()
    if missing_status:
        print(f"[warn] {missing_status} decisive rows had no matching language "
              "record (index out of range vs train.jsonl's current line count); "
              "dropped before filtering.")
        dec = dec[dec["english_status"].notna()]

    print("\nEnglish-status breakdown restricted to the DECISIVE battles used in "
          "every other analysis in this project:")
    print(dec["english_status"].value_counts().to_string())

    dec_en = dec[dec["english_status"] == "english_only"].copy()
    n_after = len(dec_en)
    pct = 100.0 * n_after / n_before
    print(f"\nBEFORE filtering : {n_before:,} decisive battles")
    print(f"AFTER  filtering : {n_after:,} decisive battles (English-only)")
    print(f"retained         : {pct:.1f}%")

    return dec_en, n_before, n_after, pct


def check_power_and_maybe_stop(n_before, n_after, pct, force_continue):
    print("\n" + "=" * 78)
    print("POWER CHECK before proceeding to Steps 4-9")
    print("=" * 78)
    fraction = n_after / n_before
    warn = fraction < POWER_WARNING_FRACTION
    print(f"English-only n = {n_after:,} of pooled n = {n_before:,} "
          f"({100 * fraction:.1f}% retained)")
    print(f"warning threshold: < {100 * POWER_WARNING_FRACTION:.0f}% retained")
    if warn:
        print(f"\n*** FLAG: English-only sample is under half the pooled n. ***")
        print("A severely underpowered replication can produce misleading "
              "non-significant results that look like \"findings didn't "
              "replicate\" when the real story is \"not enough English-only "
              "data to tell\". Per the task instructions, stopping here unless "
              "--force-continue is passed, so this can be confirmed before "
              "burning the remaining steps on an underpowered run.")
    else:
        print("\nAbove the 50% retention threshold -- proceeding is reasonable "
              "from a power standpoint; low-power flags will still be applied "
              "per-category in Steps 4-6.")
    if warn and not force_continue:
        print("\nSTOPPING after Step 3, as instructed. Re-run with "
              "--force-continue to proceed to Steps 4-9 anyway.")
        sys.exit(0)
    return warn


# ============================================================
# STEP 4: citation effect by intent, English-only
# ============================================================

def run_step4_by_intent(dec_en, sd_cit, sd_len, pooled_intent_csv="outputs/bradley_terry_by_intent.csv"):
    print("\n" + "=" * 78)
    print("STEP 4  Citation effect by primary_intent, ENGLISH-ONLY")
    print("=" * 78)
    rows = []
    print(f"{'primary_intent':22}{'n':>6}{'beta_cit':>10}{'95% CI':>20}{'p':>9}  sig")
    for intent in INTENT_ORDER:
        sub = dec_en[dec_en["primary_intent"] == intent].copy()
        if len(sub) < MIN_N_FOR_FIT:
            rows.append(dict(primary_intent=intent, n_decisive=len(sub),
                             note="insufficient n"))
            print(f"{intent:22}{len(sub):6}  insufficient n for fit")
            continue
        models = sorted(set(sub["base_model_a"]) | set(sub["base_model_b"]))
        res, cols, ref, nfail, ok, _draws = bootstrap_bt(
            sub, build_design_model_identity,
            models=models, sd_cit=sd_cit, sd_len=sd_len)
        c = res["beta_citation"]; l = res["beta_length"]
        rows.append(dict(
            primary_intent=intent, n_decisive=len(sub),
            n_models=len(models), reference_model=ref,
            beta_citation=c["coef"], cit_ci_lo=c["lo"], cit_ci_hi=c["hi"],
            cit_boot_p=c["p"], cit_significant=c["sig"],
            beta_length=l["coef"], len_ci_lo=l["lo"], len_ci_hi=l["hi"],
            len_boot_p=l["p"], len_significant=l["sig"],
            low_power=len(sub) < LOW_POWER_N,
            converged=ok, n_boot_failed=nfail,
        ))
        ci = "[%+.3f, %+.3f]" % (c["lo"], c["hi"])
        print("%-22s%6d%10.4f%20s%9.4f  %s%s" % (
            intent, len(sub), c["coef"], ci, c["p"],
            "YES" if c["sig"] else "no",
            "  (low power, n<%d)" % LOW_POWER_N if len(sub) < LOW_POWER_N else ""))
    df = pd.DataFrame(rows)
    df.to_csv(BY_INTENT_CSV, index=False)
    print(f"\n[saved] {BY_INTENT_CSV}")

    if os.path.exists(pooled_intent_csv):
        pooled = pd.read_csv(pooled_intent_csv).set_index("primary_intent")
        print("\nCOMPARISON vs pooled (bradley_terry_by_intent.csv):")
        print(f"{'primary_intent':22}{'pooled beta':>12}{'pooled sig':>11}"
              f"{'EN beta':>10}{'EN sig':>8}   status")
        comp_rows = []
        for _, r in df.iterrows():
            it = r["primary_intent"]
            if it not in pooled.index:
                continue
            p = pooled.loc[it]
            has_en = np.isfinite(r.get("beta_citation", np.nan))
            p_sig = bool(p["cit_significant"]) if "cit_significant" in p else bool(p.get("significant_at_0.05", False))
            en_sig = bool(r["cit_significant"]) if has_en else None
            if not has_en:
                status = "English-only n too small to fit"
            elif p_sig == en_sig:
                status = "same significance"
            elif p_sig and not en_sig:
                status = "LOST significance (pooled sig, EN-only not)"
            else:
                status = "GAINED significance (EN-only sig, pooled not)"
            print(f"{it:22}{float(p['beta_citation']):12.4f}{str(p_sig):11}"
                  f"{(r['beta_citation'] if has_en else float('nan')):10.4f}"
                  f"{str(en_sig):8}   {status}")
            comp_rows.append(dict(
                primary_intent=it, pooled_n=int(p["n_decisive"]),
                pooled_beta_citation=float(p["beta_citation"]), pooled_significant=p_sig,
                english_only_n=int(r["n_decisive"]),
                english_only_beta_citation=(float(r["beta_citation"]) if has_en else None),
                english_only_significant=en_sig, status=status))
        return df, pd.DataFrame(comp_rows)
    return df, pd.DataFrame()


# ============================================================
# STEP 5: cost tier, English-only
# ============================================================

def run_step5_cost_tier(dec_en, sd_cit, sd_len, pooled_tier_csv="outputs/bradley_terry_cost_tier.csv"):
    print("\n" + "=" * 78)
    print("STEP 5  Cost-tier effect, ENGLISH-ONLY (all categories)")
    print("=" * 78)
    if len(dec_en) < MIN_N_FOR_FIT:
        print(f"n={len(dec_en)} below MIN_N_FOR_FIT={MIN_N_FOR_FIT}; not fit.")
        return None, None

    res, cols, _rest, nfail, ok, draws = bootstrap_bt(
        dec_en, build_design_cost_tier, sd_cit=sd_cit, sd_len=sd_len)
    row = dict(n_decisive=len(dec_en), converged=ok, n_boot_failed=nfail,
              low_power=len(dec_en) < LOW_POWER_N)
    for c in cols:
        key = (c.replace("tier[", "").replace("] vs cheap", "_vs_cheap")
                .replace("beta_", ""))
        row[f"{key}_coef"] = res[c]["coef"]
        row[f"{key}_ci_lo"] = res[c]["lo"]
        row[f"{key}_ci_hi"] = res[c]["hi"]
        row[f"{key}_boot_p"] = res[c]["p"]
        row[f"{key}_significant"] = res[c]["sig"]

    i_mid = cols.index("tier[mid] vs cheap")
    i_exp = cols.index("tier[expensive] vs cheap")
    row["expensive_minus_mid_coef"] = float(point_of(res, cols, i_exp) - point_of(res, cols, i_mid))
    cd = draws[:, i_exp] - draws[:, i_mid]
    cd = cd[np.isfinite(cd)]
    if len(cd) >= 50:
        lo = float(np.percentile(cd, 100 * ALPHA / 2))
        hi = float(np.percentile(cd, 100 * (1 - ALPHA / 2)))
        pp = float(min(1.0, 2.0 * min(np.mean(cd <= 0), np.mean(cd >= 0))))
        row["expensive_minus_mid_ci_lo"] = lo
        row["expensive_minus_mid_ci_hi"] = hi
        row["expensive_minus_mid_boot_p"] = pp
        row["expensive_minus_mid_significant"] = bool(lo > 0 or hi < 0)
    else:
        row["expensive_minus_mid_ci_lo"] = np.nan
        row["expensive_minus_mid_ci_hi"] = np.nan
        row["expensive_minus_mid_boot_p"] = np.nan
        row["expensive_minus_mid_significant"] = False

    print(f"n={len(dec_en)}  converged={ok}  low_power={row['low_power']}")
    for t in ("mid", "expensive"):
        print(f"  tier[{t}] vs cheap : {row[f'{t}_vs_cheap_coef']:+.4f}  "
              f"95% CI [{row[f'{t}_vs_cheap_ci_lo']:+.4f}, {row[f'{t}_vs_cheap_ci_hi']:+.4f}]  "
              f"p={row[f'{t}_vs_cheap_boot_p']:.4f}  "
              f"{'SIG' if row[f'{t}_vs_cheap_significant'] else 'ns'}")
    print(f"  CONTRAST expensive - mid : {row['expensive_minus_mid_coef']:+.4f}  "
          f"95% CI [{row['expensive_minus_mid_ci_lo']:+.4f}, {row['expensive_minus_mid_ci_hi']:+.4f}]  "
          f"p={row['expensive_minus_mid_boot_p']:.4f}  "
          f"{'SIG' if row['expensive_minus_mid_significant'] else 'NOT SIGNIFICANT'}")

    pd.DataFrame([row]).to_csv(COST_TIER_CSV, index=False)
    print(f"\n[saved] {COST_TIER_CSV}")

    pooled_row = None
    if os.path.exists(pooled_tier_csv):
        t = pd.read_csv(pooled_tier_csv)
        ov = t[t["primary_intent"] == "All categories"] if "primary_intent" in t.columns else t[t.get("scope") == "All categories"]
        if len(ov):
            pooled_row = ov.iloc[0].to_dict()
            print("\nCOMPARISON vs pooled (All categories):")
            print(f"  expensive-minus-mid: pooled={pooled_row.get('expensive_minus_mid_coef', float('nan')):+.4f} "
                  f"(sig={pooled_row.get('expensive_minus_mid_significant')})  vs  "
                  f"EN-only={row['expensive_minus_mid_coef']:+.4f} (sig={row['expensive_minus_mid_significant']})")
            same = bool(pooled_row.get("expensive_minus_mid_significant")) == row["expensive_minus_mid_significant"]
            print(f"  null result holds on English-only data: {same}")
    return row, pooled_row


# ============================================================
# STEP 6: family-vs-tier joint fit, English-only
# ============================================================

def run_step6_joint_fit(dec_en, sd_cit, sd_len, pooled_joint_csv="outputs/bradley_terry_family_joint.csv"):
    print("\n" + "=" * 78)
    print("STEP 6  Family-vs-tier joint fit, ENGLISH-ONLY")
    print("=" * 78)
    dec_en = dec_en.copy()
    dec_en["family_a"] = dec_en["base_model_a"].map(classify_model_family)
    dec_en["family_b"] = dec_en["base_model_b"].map(classify_model_family)
    dropped = dec_en["family_a"].isna() | dec_en["family_b"].isna()
    if dropped.any():
        print(f"[warn] dropping {int(dropped.sum())} battles with an unclassified family")
        dec_en = dec_en[~dropped].copy()

    if len(dec_en) < MIN_N_FOR_FIT:
        print(f"n={len(dec_en)} below MIN_N_FOR_FIT={MIN_N_FOR_FIT}; not fit.")
        return None, None

    X_check, _y, cols_check = build_design_tier_and_family(dec_en, sd_cit, sd_len)
    rank = np.linalg.matrix_rank(X_check)
    print(f"design matrix: {X_check.shape[1]} columns, rank={rank} "
          f"({'identifiable' if rank == X_check.shape[1] else 'RANK DEFICIENT'})")

    res, cols, _rest, nfail, ok, draws = bootstrap_bt(
        dec_en, build_design_tier_and_family, sd_cit=sd_cit, sd_len=sd_len)
    row = dict(n_decisive=len(dec_en), converged=ok, n_boot_failed=nfail,
              rank=rank, n_params=X_check.shape[1],
              low_power=len(dec_en) < LOW_POWER_N)
    for c in cols:
        key = (c.replace("tier[", "tier_").replace("] vs cheap", "_vs_cheap")
                .replace("family[", "family_").replace("] vs gemini", "_vs_gemini")
                .replace("beta_", ""))
        row[f"{key}_coef"] = res[c]["coef"]
        row[f"{key}_ci_lo"] = res[c]["lo"]
        row[f"{key}_ci_hi"] = res[c]["hi"]
        row[f"{key}_boot_p"] = res[c]["p"]
        row[f"{key}_significant"] = res[c]["sig"]
    row["tier_remains_significant"] = (res["tier[mid] vs cheap"]["sig"]
                                       or res["tier[expensive] vs cheap"]["sig"])
    row["family_remains_significant"] = (res["family[openai] vs gemini"]["sig"]
                                         or res["family[sonar] vs gemini"]["sig"])

    print(f"n={len(dec_en)}  converged={ok}  low_power={row['low_power']}")
    for c in cols:
        r = res[c]
        print(f"  {c:26}= {r['coef']:+.4f}  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"p={r['p']:.4f}  {'SIG' if r['sig'] else 'ns'}")
    print(f"\ntier remains significant (English-only)   : {row['tier_remains_significant']}")
    print(f"family remains significant (English-only) : {row['family_remains_significant']}")

    pd.DataFrame([row]).to_csv(JOINT_CSV, index=False)
    print(f"\n[saved] {JOINT_CSV}")

    pooled_row = None
    if os.path.exists(pooled_joint_csv):
        pooled_row = pd.read_csv(pooled_joint_csv).iloc[0].to_dict()
        print("\nCOMPARISON vs pooled joint fit:")
        print(f"  tier remains sig   : pooled={pooled_row.get('tier_remains_significant')}  "
              f"EN-only={row['tier_remains_significant']}")
        print(f"  family remains sig : pooled={pooled_row.get('family_remains_significant')}  "
              f"EN-only={row['family_remains_significant']}")
    return row, pooled_row


# ============================================================
# STEP 7: one clear pooled-vs-English-only comparison table
# ============================================================

def build_comparison_table(intent_comp_df, tier_row, tier_pooled, joint_row, joint_pooled):
    rows = []
    for _, r in intent_comp_df.iterrows():
        rows.append(dict(
            section="by_intent", coefficient=f"beta_citation[{r['primary_intent']}]",
            pooled_value=r["pooled_beta_citation"], pooled_significant=r["pooled_significant"],
            english_only_value=r["english_only_beta_citation"],
            english_only_significant=r["english_only_significant"],
            significance_changed=(r["pooled_significant"] != r["english_only_significant"]),
            conclusion_changed=r["status"] not in ("same significance",),
            note=r["status"],
        ))
    if tier_row is not None:
        for key, lab in (("mid_vs_cheap", "tier: mid vs cheap"),
                         ("expensive_vs_cheap", "tier: expensive vs cheap"),
                         ("expensive_minus_mid", "tier: expensive - mid (contrast)")):
            p_sig = bool(tier_pooled.get(f"{key}_significant")) if tier_pooled else None
            e_sig = bool(tier_row[f"{key}_significant"])
            rows.append(dict(
                section="cost_tier", coefficient=lab,
                pooled_value=(float(tier_pooled[f"{key}_coef"]) if tier_pooled else None),
                pooled_significant=p_sig,
                english_only_value=float(tier_row[f"{key}_coef"]),
                english_only_significant=e_sig,
                significance_changed=(p_sig != e_sig) if p_sig is not None else None,
                conclusion_changed=(p_sig != e_sig) if p_sig is not None else None,
                note="",
            ))
    if joint_row is not None:
        for key, lab in (("tier_mid_vs_cheap", "joint: tier mid vs cheap"),
                         ("tier_expensive_vs_cheap", "joint: tier expensive vs cheap"),
                         ("family_openai_vs_gemini", "joint: family OpenAI vs Gemini"),
                         ("family_sonar_vs_gemini", "joint: family Sonar vs Gemini")):
            p_sig = bool(joint_pooled.get(f"{key}_significant")) if joint_pooled else None
            e_sig = bool(joint_row[f"{key}_significant"])
            rows.append(dict(
                section="family_tier_joint", coefficient=lab,
                pooled_value=(float(joint_pooled[f"{key}_coef"]) if joint_pooled else None),
                pooled_significant=p_sig,
                english_only_value=float(joint_row[f"{key}_coef"]),
                english_only_significant=e_sig,
                significance_changed=(p_sig != e_sig) if p_sig is not None else None,
                conclusion_changed=(p_sig != e_sig) if p_sig is not None else None,
                note="",
            ))
    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_CSV, index=False)
    print(f"\n[saved] {COMPARISON_CSV}")
    return df


# ============================================================
# STEP 9: markdown section
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


def write_summary_section(n_before, n_after, pct, status_counts_decisive,
                          intent_df, intent_comp_df, tier_row, tier_pooled,
                          joint_row, joint_pooled, comparison_df,
                          path=SUMMARY_MD):
    L = []
    A = L.append

    n_sig_changed = int(comparison_df["significance_changed"].fillna(False).sum()) if len(comparison_df) else 0
    any_low_power = (bool(intent_df.get("low_power", pd.Series(dtype=bool)).fillna(False).any())
                     or (tier_row is not None and tier_row.get("low_power"))
                     or (joint_row is not None and joint_row.get("low_power")))

    A("")
    A("---")
    A("")
    A(SECTION_MARKER)
    A("")
    if n_sig_changed == 0:
        headline = ("**The project's core findings replicate on English-only data.** "
                    "Every coefficient checked kept the same significance status after "
                    "restricting to English-only conversations, though sample sizes "
                    "shrank and some intervals widened as expected.")
    else:
        headline = (f"**{n_sig_changed} of {len(comparison_df)} checked coefficients "
                   "changed significance status on English-only data.** See the "
                   "comparison table below for specifics -- some of this is a genuine "
                   "replication question and some is the mechanical consequence of a "
                   "smaller sample; both are reported, not collapsed into one verdict.")
    A(headline)
    if any_low_power:
        A("")
        A(f"**Power caveat.** At least one fit below falls under this project's "
          f"n={LOW_POWER_N} low-power threshold once restricted to English-only data. "
          "Its non-significant results should be read as \"not enough data to detect "
          "an effect,\" not as \"no effect\" -- flagged explicitly at each occurrence "
          "below.")
    A("")

    A("## Step 1-2: what `languages` looks like, and the rule adopted")
    A("")
    A("`languages` is not in `final_analysis_derived.csv`; it was pulled directly from "
      "`train.jsonl` and merged in by row index (the same indexed-merge pattern already "
      "verified for the frustration labels). Its raw format is a JSON array of "
      "language-name strings per conversation (e.g. `[\"English\"]`, "
      "`[\"German\", \"English\"]`) -- not the stringified-tuple format seen in "
      "`frustration_sample300.jsonl`, which was introduced later in that pipeline, not "
      "present in the source data.")
    A("")
    A("**Rule adopted: \"English-only\" = the `languages` field lists English and "
      "ONLY English.** Conversations tagged English-plus-other-language(s) are "
      "excluded, not included -- a mixed-language conversation is not reliably English "
      "throughout, so counting it as \"English-only\" would leave non-English text in a "
      "nominally English-only replication.")
    A("")
    A("Breakdown restricted to the decisive battles used in every other analysis in "
      "this project:")
    A("")
    A("| status | n |")
    A("|---|---|")
    for k, lab in (("english_only", "English only"),
                   ("english_plus_other", "English + other language(s)"),
                   ("no_english", "No English"),
                   ("missing", "No language metadata")):
        A(f"| {lab} | {status_counts_decisive.get(k, 0):,} |")
    A("")

    A("## Step 3: filtered sample size")
    A("")
    A(f"| | n | % of pooled |")
    A("|---|---|---|")
    A(f"| Pooled (all languages) | {n_before:,} | 100.0% |")
    A(f"| English-only | {n_after:,} | {pct:.1f}% |")
    A("")

    A("## Step 4: citation effect by intent, English-only")
    A("")
    A("| primary_intent | n | beta_citation | 95% CI | boot p | significant | beta_length |")
    A("|---|---|---|---|---|---|---|")
    for _, r in intent_df.iterrows():
        if not np.isfinite(r.get("beta_citation", np.nan)):
            A(f"| {r['primary_intent']} | {int(r['n_decisive'])} | — | — | — | insufficient n | — |")
            continue
        lp = " *(low power)*" if bool(r.get("low_power", False)) else ""
        A(f"| {r['primary_intent']}{lp} | {int(r['n_decisive'])} | {r['beta_citation']:+.4f} | "
          f"[{r['cit_ci_lo']:+.3f}, {r['cit_ci_hi']:+.3f}] | {r['cit_boot_p']:.4f} | "
          f"{'**yes**' if r['cit_significant'] else 'no'} | {r['beta_length']:+.4f} |")
    A("")
    if len(intent_comp_df):
        A("Comparison against the pooled by-intent results already in this file:")
        A("")
        A("| primary_intent | pooled beta | pooled sig | English-only beta | EN sig | status |")
        A("|---|---|---|---|---|---|")
        for _, r in intent_comp_df.iterrows():
            en_val = ("—" if r["english_only_beta_citation"] is None
                      else f"{r['english_only_beta_citation']:+.4f}")
            A(f"| {r['primary_intent']} | {r['pooled_beta_citation']:+.4f} | "
              f"{r['pooled_significant']} | {en_val} | {r['english_only_significant']} | "
              f"{r['status']} |")
        A("")

    A("## Step 5: cost tier, English-only")
    A("")
    if tier_row is None:
        A("*(Not fit -- insufficient English-only n.)*")
        A("")
    else:
        lp = " *(low power)*" if tier_row.get("low_power") else ""
        A(f"n = {tier_row['n_decisive']:,}{lp}")
        A("")
        A("| term | coefficient | 95% CI | boot p | significant |")
        A("|---|---|---|---|---|")
        for key, lab in (("mid_vs_cheap", "tier: mid vs cheap"),
                         ("expensive_vs_cheap", "tier: expensive vs cheap"),
                         ("expensive_minus_mid", "tier: expensive - mid (contrast)")):
            A(f"| {lab} | {tier_row[f'{key}_coef']:+.4f} | "
              f"[{tier_row[f'{key}_ci_lo']:+.3f}, {tier_row[f'{key}_ci_hi']:+.3f}] | "
              f"{tier_row[f'{key}_boot_p']:.4f} | "
              f"{'**yes**' if tier_row[f'{key}_significant'] else 'no'} |")
        A("")
        if tier_pooled is not None:
            pooled_null_held = (bool(tier_pooled.get("expensive_minus_mid_significant")) ==
                                tier_row["expensive_minus_mid_significant"])
            A(f"**The pooled expensive-vs-mid null result "
              f"({'holds' if pooled_null_held else 'does NOT hold'}) on English-only "
              f"data** (pooled p=0.69, not significant; English-only p="
              f"{tier_row['expensive_minus_mid_boot_p']:.4f}, "
              f"{'not significant' if not tier_row['expensive_minus_mid_significant'] else 'SIGNIFICANT -- a change from pooled'}).")
            A("")

    A("## Step 6: family-vs-tier joint fit, English-only")
    A("")
    if joint_row is None:
        A("*(Not fit -- insufficient English-only n.)*")
        A("")
    else:
        lp = " *(low power)*" if joint_row.get("low_power") else ""
        A(f"n = {joint_row['n_decisive']:,}{lp}  --  design matrix rank "
          f"{joint_row['rank']}/{joint_row['n_params']} "
          f"({'identifiable' if joint_row['rank'] == joint_row['n_params'] else '**RANK DEFICIENT**'})")
        A("")
        A("| term | coefficient | 95% CI | boot p | significant |")
        A("|---|---|---|---|---|")
        for key, lab in (("tier_mid_vs_cheap", "tier: mid vs cheap"),
                         ("tier_expensive_vs_cheap", "tier: expensive vs cheap"),
                         ("family_openai_vs_gemini", "family: OpenAI vs Gemini"),
                         ("family_sonar_vs_gemini", "family: Sonar vs Gemini")):
            A(f"| {lab} | {joint_row[f'{key}_coef']:+.4f} | "
              f"[{joint_row[f'{key}_ci_lo']:+.3f}, {joint_row[f'{key}_ci_hi']:+.3f}] | "
              f"{joint_row[f'{key}_boot_p']:.4f} | "
              f"{'**yes**' if joint_row[f'{key}_significant'] else 'no'} |")
        A("")
        A(f"Tier remains significant once family is controlled: "
          f"**{'YES' if joint_row['tier_remains_significant'] else 'NO'}**. "
          f"Family remains significant once tier is controlled: "
          f"**{'YES' if joint_row['family_remains_significant'] else 'NO'}**.")
        A("")
        if joint_pooled is not None:
            A(f"Pooled fit found both significant "
              f"(tier={joint_pooled.get('tier_remains_significant')}, "
              f"family={joint_pooled.get('family_remains_significant')}). "
              f"English-only: tier={joint_row['tier_remains_significant']}, "
              f"family={joint_row['family_remains_significant']} -- "
              f"{'SAME conclusion' if (bool(joint_pooled.get('tier_remains_significant')) == joint_row['tier_remains_significant'] and bool(joint_pooled.get('family_remains_significant')) == joint_row['family_remains_significant']) else 'CONCLUSION CHANGED'}.")
            A("")

    A("## Step 7: pooled vs English-only, every checked coefficient")
    A("")
    A("| section | coefficient | pooled value | pooled sig | English-only value | EN sig | significance changed |")
    A("|---|---|---|---|---|---|---|")
    for _, r in comparison_df.iterrows():
        pv = "—" if pd.isna(r["pooled_value"]) else f"{r['pooled_value']:+.4f}"
        ev = "—" if pd.isna(r["english_only_value"]) else f"{r['english_only_value']:+.4f}"
        sc = r["significance_changed"]
        sc_str = "—" if pd.isna(sc) else ("**YES**" if sc else "no")
        A(f"| {r['section']} | {r['coefficient']} | {pv} | {r['pooled_significant']} | "
          f"{ev} | {r['english_only_significant']} | {sc_str} |")
    A("")

    A("## Step 8: sample size and power, stated explicitly")
    A("")
    A(f"English-only retained **{pct:.1f}%** of the pooled decisive sample "
      f"({n_after:,} of {n_before:,}). Per-category sample sizes below "
      f"this project's n={LOW_POWER_N} low-power threshold are marked *(low power)* "
      "above wherever they occur, and their non-significant results should be read "
      "as inconclusive, not as evidence of no effect.")
    A("")

    _truncate_existing_section(path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[appended] {SECTION_MARKER} -> {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-continue", action="store_true",
                   help="proceed past Steps 1-3 even if the English-only sample "
                        "falls under the power-warning threshold")
    args = ap.parse_args()

    print("=" * 78)
    print("ENGLISH-ONLY REPLICATION")
    print(f"bootstrap replications = {N_BOOTSTRAP}  seed = {RANDOM_SEED}")
    print("No API calls. No new data collection.")
    print("=" * 78)

    languages_by_idx, combo_counts = extract_languages()
    status_by_idx, status_counts_all = run_step2_definition(languages_by_idx)
    dec_en, n_before, n_after, pct = run_step3_merge_and_filter(status_by_idx)

    # status breakdown restricted to decisive battles, for the report (not
    # the all-conversations breakdown already printed in Step 2)
    _full, dec_all, _n_all, _src = load_decisive()
    dec_all = dec_all.copy()
    dec_all["english_status"] = dec_all.index.map(status_by_idx.get)
    status_counts_decisive = dec_all["english_status"].value_counts().to_dict()

    check_power_and_maybe_stop(n_before, n_after, pct, args.force_continue)

    sd_cit, sd_len = _scale_covariates(dec_all)
    print(f"\ncovariate scaling (reused from the main analysis): "
          f"1 SD citation diff = {sd_cit:.3f}; 1 SD length diff = {sd_len:.1f} chars")

    intent_df, intent_comp_df = run_step4_by_intent(dec_en, sd_cit, sd_len)
    tier_row, tier_pooled = run_step5_cost_tier(dec_en, sd_cit, sd_len)
    joint_row, joint_pooled = run_step6_joint_fit(dec_en, sd_cit, sd_len)

    comparison_df = build_comparison_table(intent_comp_df, tier_row, tier_pooled,
                                           joint_row, joint_pooled)

    write_summary_section(n_before, n_after, pct, status_counts_decisive,
                         intent_df, intent_comp_df, tier_row, tier_pooled,
                         joint_row, joint_pooled, comparison_df)


if __name__ == "__main__":
    main()
