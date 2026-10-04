"""
final_analysis.py

Final consolidated analysis for the Sunday mentor review. Builds on the
fixed per-turn citation aggregation from intent_complexity_analysis_v2.py
and closes the remaining gaps identified after that script:

  1. Response length per side (character count) - a verbosity-bias
     control, since the Search Arena paper reports a verbosity
     coefficient of beta=0.334 that none of our prior analysis
     controlled for.
  2. Citation-winning correlation at the 9 primary_intent-category
     level (not the 3-tier grouping), since tiering was masking
     Creative Generation's known inverted relationship.
  3. Each category's correlation reported 3 ways: raw, model-identity
     controlled, and model-identity + response-length controlled -
     to separate a genuine citation signal from verbosity bias.
  4. The single-turn vs multi-turn split redone at the 9-category
     level (9x2 = 18 cells), each with all 3 correlation variants and
     a low-power flag.
  5. Vote-presence analysis controlled for primary_intent AND base
     model identity simultaneously.
  6. A formal significance test (Fisher z) on the single-turn vs
     multi-turn citation-correlation gap for the three former "High"
     complexity categories.
  7. A single markdown summary with a "confident" vs "uncertain"
     verdict section.

Runs entirely locally against train.jsonl - no LLM/API calls, no paid
services. Uses only numpy, pandas, and the standard library.
"""

import json
import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

from inspect_search_arena import (
    DATASET_FILE,
    get_messages,
    get_assistant_messages,
    get_user_messages,
    get_visible_citation_markers,
)
from intent_complexity_analysis_v2 import (
    citation_stats_for_side,
    strip_model_suffix,
    safe_corr,
    DECISIVE_WINNERS,
    TIE_WINNERS,
)

# ============================================================
# SETTINGS
# ============================================================

DERIVED_CSV = "outputs/final_analysis_derived.csv"
BY_INTENT_CSV = "outputs/final_analysis_by_intent.csv"
STRATIFIED_CSV = "outputs/final_analysis_stratified_9x2.csv"
VOTE_PRESENCE_CSV = "outputs/final_analysis_vote_presence.csv"
SIGNIFICANCE_CSV = "outputs/final_analysis_significance_test.csv"
CITATION_LENGTH_DIAG_CSV = "outputs/final_analysis_citation_length_correlation.csv"
COST_TIER_CSV = "outputs/final_analysis_cost_tier_comparison.csv"
SUMMARY_MD = "outputs/final_analysis_summary.md"

INTENT_ORDER = [
    "Factual Lookup",
    "Text Processing",
    "Other",
    "Explanation",
    "Recommendation",
    "Guidance",
    "Info Synthesis",
    "Analysis",
    "Creative Generation",
]

FORMER_HIGH_TIER_CATEGORIES = ["Info Synthesis", "Analysis", "Creative Generation"]

CONV_TYPE_ORDER = ["single_turn", "multi_turn"]

MIN_DECISIVE_FOR_CONFIDENCE = 200
MIN_GROUP_N_FOR_DIAGNOSTIC = 20
SIGNIFICANCE_ALPHA = 0.05
BORDERLINE_P_LOW = 0.01  # p in [0.01, 0.05) is "borderline" given no multiple-comparisons correction

# Threshold used to call a correlation "substantially changed" once
# response length is controlled for, for the confident/uncertain
# writeup in the final summary.
LENGTH_CONTROL_DROP_THRESHOLD = 0.5  # relative drop (e.g. 0.5 = lost >=50% of magnitude)
NEAR_ZERO_THRESHOLD = 0.03

# Cost-tier mapping (v2, Part B): replaces the earlier generic-pricing-
# pattern approximation with verified real blended input/output pricing
# per million tokens (mid-2026). Base-model identifiers only (post
# strip_model_suffix). Search-fee surcharges (gpt-4o-mini-search-preview,
# gpt-4o-search-preview) are NOT reflected in the blended figure and are
# noted separately - they'd push those two models further up in relative
# cost than the blended number alone suggests.
MODEL_BLENDED_PRICE_USD_PER_MILLION = {
    "gemini-2.0-flash": 0.25,
    "gpt-4o-mini-search-preview": 0.38,  # + $25-30 / 1,000 calls search fee, not reflected here
    "sonar": 1.00,
    "gemini-2.5-flash-preview-04-17": 1.40,
    "sonar-reasoning": 3.00,  # $1.00/Mtok input, $5.00/Mtok output, 50/50 blended; user-supplied verified price
    "sonar-reasoning-pro": 5.00,
    "gemini-2.5-pro-exp-03-25": 5.63,
    "gpt-4o-search-preview": 6.25,  # + $30-50 / 1,000 calls search fee, not reflected here
    "sonar-pro": 9.00,
}

CHEAP_TIER = {"gemini-2.0-flash", "gpt-4o-mini-search-preview", "sonar"}
MID_TIER = {"gemini-2.5-flash-preview-04-17", "sonar-reasoning"}
EXPENSIVE_TIER = {
    "sonar-reasoning-pro",
    "gemini-2.5-pro-exp-03-25",
    "gpt-4o-search-preview",
    "sonar-pro",
}
TIER_PAIRS = [("cheap", "mid"), ("cheap", "expensive"), ("mid", "expensive")]

MIN_H2H_FOR_CONFIDENCE = 100

# Part A/D/E sampling settings
# These three manual-review dumps live in manual_review/ in the repo layout.
# Paths are relative to the repo root, which is the documented working directory
# (see README "Run order"): `python scripts/final_analysis.py` from the root.
ZERO_CITATION_TXT = "manual_review/zero_citation_manual_review.txt"
INTENT_VALIDATION_TXT = "manual_review/intent_manual_validation.txt"
RECOMMENDATION_EXAMPLES_TXT = "manual_review/recommendation_citation_examples.txt"

ZERO_CITATION_SAMPLE_TOTAL = 75
INTENT_VALIDATION_SAMPLE_TOTAL = 40
RECOMMENDATION_EXAMPLE_TARGET = 20
RECOMMENDATION_MIN_CITATION_DIFF = 3
SAMPLING_SEED = 42


# ============================================================
# RESPONSE LENGTH (Step 1)
# ============================================================

def response_length_for_side(record, side):
    """
    Total character count of all assistant turns concatenated for one
    side - same per-turn aggregation pattern as the fixed citation
    logic, just summing len(text) instead of deduped marker count.
    """
    messages = get_messages(record, side)
    assistant_texts = get_assistant_messages(messages)
    return sum(len(t) for t in assistant_texts)


# ============================================================
# PARTIAL CORRELATION VIA OLS RESIDUALIZATION
# ============================================================

def residualize(y, df, categorical_cols=None, continuous_cols=None):
    """
    OLS-residualize y against an intercept, dummy-coded categorical
    columns (fixed effects), and continuous columns. Equivalent to
    simple group-mean-centering when the only control is a single
    categorical column (verified numerically), and generalizes
    cleanly to add continuous controls like response length.
    """
    n = len(y)
    parts = [np.ones((n, 1))]

    if categorical_cols:
        for col in categorical_cols:
            dummies = pd.get_dummies(df[col], drop_first=True).values.astype(float)
            if dummies.shape[1] > 0:
                parts.append(dummies)

    if continuous_cols:
        for col in continuous_cols:
            parts.append(df[col].values.reshape(-1, 1).astype(float))

    X = np.hstack(parts)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def partial_correlation(df, feature_col, target_col, categorical_cols=None, continuous_cols=None):
    feat = df[feature_col].values.astype(float)
    targ = df[target_col].values.astype(float)

    feat_resid = residualize(feat, df, categorical_cols, continuous_cols)
    targ_resid = residualize(targ, df, categorical_cols, continuous_cols)

    return safe_corr(feat_resid, targ_resid)


def three_way_correlations(long_df):
    """
    raw / model-controlled / model+length-controlled citation-winning
    correlations for one subgroup's unrolled response-level table.
    """
    if len(long_df) < 4:
        return float("nan"), float("nan"), float("nan")

    raw = safe_corr(long_df["citation_count"].values, long_df["is_winner"].values)

    model_ctrl = partial_correlation(
        long_df, "citation_count", "is_winner", categorical_cols=["base_model"]
    )

    model_length_ctrl = partial_correlation(
        long_df,
        "citation_count",
        "is_winner",
        categorical_cols=["base_model"],
        continuous_cols=["response_length"],
    )

    return raw, model_ctrl, model_length_ctrl


# ============================================================
# FISHER Z-TEST (Step 6)
# ============================================================

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fisher_z_test(r1, n1, r2, n2):
    """
    Two-tailed Fisher z-test comparing two independent Pearson
    correlations. Returns (z_statistic, p_value).

    NOTE: n1/n2 here should be the number of independent DECISIVE
    RECORDS, not the doubled response-row count used to compute the
    correlations themselves. Each decisive record contributes two
    response rows (one per side) that are not independent draws (their
    citation counts differ, but their win/loss outcomes are perfectly
    complementary by construction). Using the doubled row count as n
    would understate the standard error and overstate significance, so
    we use the more conservative record count as the effective sample
    size.
    """
    if any(np.isnan(v) for v in (r1, r2)) or n1 < 4 or n2 < 4:
        return float("nan"), float("nan")

    r1 = max(min(r1, 0.999999), -0.999999)
    r2 = max(min(r2, 0.999999), -0.999999)

    z1 = math.atanh(r1)
    z2 = math.atanh(r2)

    se = math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z_stat)))

    return z_stat, p_value


def one_sample_corr_significance(r, n_decisive):
    """
    Two-tailed t-test of a Pearson correlation against zero:
    t = r * sqrt(df / (1 - r^2)), df = n - 2.

    NOTE: n_decisive is the number of independent DECISIVE RECORDS,
    not the doubled response-row count the correlation itself was
    computed on (same reasoning as fisher_z_test: the two response
    rows per record are not independent draws). Using the doubled
    count would understate the standard error and overstate
    significance.

    Also NOTE: this treats r as if it were a simple two-variable
    Pearson correlation, when it is actually computed on residuals
    from an OLS fit with several fixed-effect and continuous controls.
    That's an approximation (it ignores the degrees of freedom used up
    by those controls), consistent with the same approximation already
    used for the Fisher z-test.
    """
    if r is None or np.isnan(r) or n_decisive < 4:
        return float("nan"), float("nan"), None

    r = max(min(r, 0.999999), -0.999999)
    df = n_decisive - 2
    denom = 1.0 - r ** 2

    if denom <= 0:
        return float("inf"), 0.0, df

    t_stat = r * math.sqrt(df / denom)
    p_value = 2.0 * stats.t.sf(abs(t_stat), df)

    return t_stat, p_value, df


# ============================================================
# LOAD FULL TABLE
# ============================================================

def load_full_table():
    rows = []
    total = 0
    unmapped_intent_count = 0

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total += 1

            primary_intent = record.get("primary_intent")
            if primary_intent not in INTENT_ORDER:
                unmapped_intent_count += 1

            turn_count = record.get("turn")

            citations_a, n_turns_a, _ = citation_stats_for_side(record, "a")
            citations_b, n_turns_b, _ = citation_stats_for_side(record, "b")

            response_length_a = response_length_for_side(record, "a")
            response_length_b = response_length_for_side(record, "b")

            base_model_a = strip_model_suffix(record.get("model_a"))
            base_model_b = strip_model_suffix(record.get("model_b"))

            winner = record.get("winner")

            if turn_count == 1:
                conv_type = "single_turn"
            elif turn_count is not None and turn_count > 1:
                conv_type = "multi_turn"
            else:
                conv_type = None

            rows.append(
                {
                    "primary_intent": primary_intent,
                    "turn_count": turn_count,
                    "conversation_type": conv_type,
                    "citations_a": citations_a,
                    "citations_b": citations_b,
                    "response_length_a": response_length_a,
                    "response_length_b": response_length_b,
                    "base_model_a": base_model_a,
                    "base_model_b": base_model_b,
                    "model_a": record.get("model_a"),
                    "model_b": record.get("model_b"),
                    "winner": winner,
                    "is_tie": winner in TIE_WINNERS,
                    "is_decisive": winner in DECISIVE_WINNERS,
                    "vote_occurred": winner is not None,
                }
            )

    df = pd.DataFrame(rows)
    return df, total, unmapped_intent_count


def build_long_table(sub_df):
    """
    Unroll decisive records into two response-level rows each:
    base_model, citation_count, response_length, is_winner.
    """
    decisive = sub_df[sub_df["is_decisive"]]

    rows = []
    for _, r in decisive.iterrows():
        rows.append(
            {
                "base_model": r["base_model_a"],
                "citation_count": r["citations_a"],
                "response_length": r["response_length_a"],
                "is_winner": 1 if r["winner"] == "model_a" else 0,
            }
        )
        rows.append(
            {
                "base_model": r["base_model_b"],
                "citation_count": r["citations_b"],
                "response_length": r["response_length_b"],
                "is_winner": 1 if r["winner"] == "model_b" else 0,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STEP 2 + 3: PER-INTENT-CATEGORY ANALYSIS
# ============================================================

def run_by_intent_analysis(df):
    print("\n" + "=" * 100)
    print("STEP 2+3: CITATION-WINNING CORRELATION BY PRIMARY_INTENT (9 categories, 3-way control)")
    print("=" * 100)

    results = []

    for intent in INTENT_ORDER:
        sub_df = df[df["primary_intent"] == intent]
        n_decisive = int(sub_df["is_decisive"].sum())

        long_df = build_long_table(sub_df)
        raw, model_ctrl, model_len_ctrl = three_way_correlations(long_df)

        low_power = n_decisive < MIN_DECISIVE_FOR_CONFIDENCE

        t_stat, p_value, dof = one_sample_corr_significance(model_len_ctrl, n_decisive)
        significant = (not np.isnan(p_value)) and p_value < SIGNIFICANCE_ALPHA
        borderline = (
            (not np.isnan(p_value))
            and BORDERLINE_P_LOW <= p_value < SIGNIFICANCE_ALPHA
        )

        results.append(
            {
                "primary_intent": intent,
                "n_decisive": n_decisive,
                "corr_raw": raw,
                "corr_model_controlled": model_ctrl,
                "corr_model_length_controlled": model_len_ctrl,
                "low_power": low_power,
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant_at_0.05": significant,
                "borderline": borderline,
            }
        )

        flag = "  <-- LOW POWER (n < 200)" if low_power else ""
        sig_note = "SIGNIFICANT" if significant else ("borderline" if borderline else "not significant")
        print(
            f"\n{intent:<22} n_decisive={n_decisive:>5}  "
            f"raw={raw:+.3f}  model_ctrl={model_ctrl:+.3f}  "
            f"model+length_ctrl={model_len_ctrl:+.3f}{flag}"
            f"\n{'':<22} t={t_stat:+.3f}  p={p_value:.4f} -> {sig_note} at alpha=0.05 "
            f"(testing model+length-controlled corr against zero, n={n_decisive})"
        )

    print(
        "\nNOTE: 9 tests were run above without a multiple-comparisons "
        "correction. With alpha=0.05 across 9 independent tests, roughly "
        "0.4-0.5 false positives are expected by chance alone. Treat "
        f"p-values in [{BORDERLINE_P_LOW}, {SIGNIFICANCE_ALPHA}) as borderline, "
        "not confirmed."
    )

    return pd.DataFrame(results)


# ============================================================
# FIX 2: CITATION COUNT vs RESPONSE LENGTH DIAGNOSTIC
# ============================================================

def run_citation_length_diagnostic(df):
    print("\n" + "=" * 100)
    print("DIAGNOSTIC: RAW CORRELATION BETWEEN CITATION COUNT AND RESPONSE LENGTH")
    print("=" * 100)
    print(
        "\nQuantifies how entangled citations and response length are, to "
        "explain why Other/Guidance lost their citation signal once length "
        "was controlled for."
    )

    results = []

    overall_long = build_long_table(df)
    overall_corr = safe_corr(
        overall_long["citation_count"].values, overall_long["response_length"].values
    )
    results.append(
        {
            "scope": "All categories",
            "n_decisive": int(df["is_decisive"].sum()),
            "citation_length_corr": overall_corr,
        }
    )
    print(f"\nOverall (all decisive records): r = {overall_corr:+.3f}")

    for intent in ["Other", "Guidance"]:
        sub_df = df[df["primary_intent"] == intent]
        long_df = build_long_table(sub_df)
        n_decisive = int(sub_df["is_decisive"].sum())

        corr = safe_corr(
            long_df["citation_count"].values, long_df["response_length"].values
        )
        results.append(
            {
                "scope": intent,
                "n_decisive": n_decisive,
                "citation_length_corr": corr,
            }
        )
        print(f"{intent:<22} (n_decisive={n_decisive}): r = {corr:+.3f}")

    return pd.DataFrame(results)


# ============================================================
# PART B + C: THREE-TIER COST COMPARISON WITH SIGNIFICANCE TESTING
# ============================================================

def classify_cost_tier(base_model):
    if base_model in CHEAP_TIER:
        return "cheap"
    if base_model in MID_TIER:
        return "mid"
    if base_model in EXPENSIVE_TIER:
        return "expensive"
    return None


def one_sample_proportion_z_test(p_hat, n):
    """
    Tests whether an observed win rate p_hat (out of n head-to-head
    decisive records) differs from 50%. Framed by the task as a
    "two-proportion z-test", but since a head-to-head win rate and its
    complement (1 - p_hat) always sum to 1 by construction, testing
    "cheaper tier's win rate != 50%" reduces exactly to the standard
    one-sample proportion z-test against a fixed reference of 0.5 - the
    two framings give the identical z-statistic here.
    """
    if p_hat is None or np.isnan(p_hat) or n < 4:
        return float("nan"), float("nan")

    se = math.sqrt(0.25 / n)
    z = (p_hat - 0.5) / se
    p_value = 2.0 * (1.0 - normal_cdf(abs(z)))

    return z, p_value


def run_cost_tier_analysis(df):
    print("\n" + "=" * 100)
    print("PART B+C: THREE-TIER COST COMPARISON WITH SIGNIFICANCE TESTING (routing hypothesis)")
    print("=" * 100)
    print(
        "\nCost tiers use verified real blended pricing (mid-2026), not a "
        "generic approximation. sonar-reasoning has no independently "
        "published price and is placed in the mid tier as an assumption. "
        "Search-fee surcharges on the two OpenAI search-preview models are "
        "not reflected in the blended figures used for tiering."
    )
    print(f"  Cheap:     {sorted(CHEAP_TIER)}")
    print(f"  Mid:       {sorted(MID_TIER)}")
    print(f"  Expensive: {sorted(EXPENSIVE_TIER)}")

    decisive = df[df["is_decisive"]].copy()
    decisive["tier_a"] = decisive["base_model_a"].map(classify_cost_tier)
    decisive["tier_b"] = decisive["base_model_b"].map(classify_cost_tier)

    def build_h2h(sub_decisive, tier_lo, tier_hi):
        mask = (
            ((sub_decisive["tier_a"] == tier_lo) & (sub_decisive["tier_b"] == tier_hi))
            | ((sub_decisive["tier_a"] == tier_hi) & (sub_decisive["tier_b"] == tier_lo))
        )
        h2h = sub_decisive[mask].copy()
        n_h2h = len(h2h)
        if n_h2h == 0:
            return n_h2h, float("nan")

        def cheaper_won(row):
            if row["tier_a"] == tier_lo:
                return row["winner"] == "model_a"
            return row["winner"] == "model_b"

        cheaper_win_rate = h2h.apply(cheaper_won, axis=1).mean()
        return n_h2h, cheaper_win_rate

    results = []
    total_per_category_tests = 0

    scopes = [("All categories", decisive)] + [
        (intent, decisive[decisive["primary_intent"] == intent]) for intent in INTENT_ORDER
    ]

    for scope_name, sub_decisive in scopes:
        print(f"\n{scope_name}:")
        for tier_lo, tier_hi in TIER_PAIRS:
            n_h2h, rate = build_h2h(sub_decisive, tier_lo, tier_hi)
            z_stat, p_value = one_sample_proportion_z_test(rate, n_h2h)

            low_power = n_h2h < MIN_H2H_FOR_CONFIDENCE
            significant = (not np.isnan(p_value)) and p_value < SIGNIFICANCE_ALPHA
            borderline = (
                (not np.isnan(p_value)) and BORDERLINE_P_LOW <= p_value < SIGNIFICANCE_ALPHA
            )

            if scope_name != "All categories":
                total_per_category_tests += 1

            results.append(
                {
                    "primary_intent": scope_name,
                    "comparison": f"{tier_lo}_vs_{tier_hi}",
                    "n_head_to_head": n_h2h,
                    "cheaper_tier_win_rate": rate,
                    "z_statistic": z_stat,
                    "p_value": p_value,
                    "significant_at_0.05": significant,
                    "borderline": borderline,
                    "low_power": low_power,
                }
            )

            rate_str = f"{rate:.1%}" if not np.isnan(rate) else "n/a"
            sig_note = (
                "SIGNIFICANT" if significant else ("borderline" if borderline else "not significant")
            )
            flag = "  <-- LOW POWER (n < 100)" if low_power else ""
            print(
                f"  {tier_lo:<9} vs {tier_hi:<9} n_h2h={n_h2h:>4}  "
                f"cheaper-tier win rate={rate_str:>6}  z={z_stat:+.3f}  "
                f"p={p_value:.4f} -> {sig_note}{flag}"
            )

    print(
        f"\nTotal per-category tests run: {total_per_category_tests} "
        f"({len(INTENT_ORDER)} categories x {len(TIER_PAIRS)} tier "
        "comparisons), plus 3 'All categories' aggregate tests shown for "
        "context but not counted toward the multiple-comparisons total. "
        "No multiple-comparisons correction is applied - at "
        f"alpha={SIGNIFICANCE_ALPHA} across {total_per_category_tests} tests, "
        f"roughly {total_per_category_tests * SIGNIFICANCE_ALPHA:.1f} false "
        "positives are expected by chance alone. Treat p-values in "
        f"[{BORDERLINE_P_LOW}, {SIGNIFICANCE_ALPHA}) as borderline."
    )

    return pd.DataFrame(results), total_per_category_tests


# ============================================================
# SHARED: PER-KEY RESERVOIR SAMPLING (Parts A, D, E)
# ============================================================

def reservoir_add(reservoirs, seen_counts, key, item, capacity, rng):
    """
    Standard per-key reservoir sampling: each key (e.g. a primary_intent
    category) gets its own reservoir of a fixed capacity, filled with a
    uniform random sample of everything seen for that key over a single
    pass, without needing to hold all candidates in memory at once.
    """
    reservoir = reservoirs.setdefault(key, [])
    seen_counts[key] = seen_counts.get(key, 0) + 1
    n_seen = seen_counts[key]

    if len(reservoir) < capacity:
        reservoir.append(item)
    else:
        j = rng.integers(0, n_seen)
        if j < capacity:
            reservoir[j] = item


def split_allocation(total, categories):
    """
    Deterministically split `total` items across `categories` as evenly
    as possible (e.g. 75 across 9 categories -> three categories get 9,
    six get 8; 40 across 9 -> four get 5, five get 4).
    """
    base = total // len(categories)
    remainder = total - base * len(categories)
    allocation = {}
    for i, cat in enumerate(categories):
        allocation[cat] = base + (1 if i < remainder else 0)
    return allocation


# ============================================================
# PART A: ZERO-CITATION MANUAL REVIEW SAMPLE
# Part D: RECOMMENDATION HIGH-CITATION-DIFF EXAMPLES
# Part E: primary_intent MANUAL VALIDATION SAMPLE
# ============================================================
# Combined into a single pass over train.jsonl for efficiency, since all
# three need raw per-turn / per-record text that the lightweight derived
# table doesn't retain.

def collect_manual_review_samples():
    print("\n" + "=" * 100)
    print("PARTS A, D, E: COLLECTING MANUAL-REVIEW SAMPLES (single pass over train.jsonl)")
    print("=" * 100)

    rng_a = np.random.default_rng(SAMPLING_SEED)
    rng_d = np.random.default_rng(SAMPLING_SEED + 1)
    rng_e = np.random.default_rng(SAMPLING_SEED + 2)

    zero_citation_allocation = split_allocation(ZERO_CITATION_SAMPLE_TOTAL, INTENT_ORDER)
    intent_validation_allocation = split_allocation(INTENT_VALIDATION_SAMPLE_TOTAL, INTENT_ORDER)

    zero_citation_reservoirs = {}
    zero_citation_seen = {}

    intent_validation_reservoirs = {}
    intent_validation_seen = {}

    recommendation_reservoir = []
    recommendation_seen = 0

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if record.get("winner") is None:
                continue

            primary_intent = record.get("primary_intent")
            if primary_intent not in INTENT_ORDER:
                continue

            winner = record.get("winner")
            is_decisive = winner in DECISIVE_WINNERS

            # --- Part E: intent validation prompts (any usable record) ---
            user_messages_a = get_user_messages(get_messages(record, "a"))
            first_question = user_messages_a[0] if user_messages_a else ""
            reservoir_add(
                intent_validation_reservoirs,
                intent_validation_seen,
                primary_intent,
                first_question,
                intent_validation_allocation[primary_intent],
                rng_e,
            )

            # --- Part A: zero-citation turns, both sides ---
            for side in ("a", "b"):
                messages = get_messages(record, side)
                assistant_texts = get_assistant_messages(messages)
                user_texts = get_user_messages(messages)
                meta = record.get(f"system_{side}_metadata") or {}
                trace = meta.get("web_search_trace")
                base_model = strip_model_suffix(record.get(f"model_{side}"))

                for turn_idx, answer_text in enumerate(assistant_texts):
                    if len(get_visible_citation_markers(answer_text)) != 0:
                        continue

                    trace_entry = None
                    has_search_activity = None
                    n_sources = None
                    if isinstance(trace, list) and turn_idx < len(trace):
                        trace_entry = trace[turn_idx]
                        if isinstance(trace_entry, list):
                            n_sources = len(trace_entry)
                            has_search_activity = n_sources > 0

                    question = (
                        user_texts[turn_idx] if turn_idx < len(user_texts) else "(no matching user turn found)"
                    )

                    candidate = {
                        "primary_intent": primary_intent,
                        "side": side,
                        "base_model": base_model,
                        "turn_index": turn_idx,
                        "total_turns": len(assistant_texts),
                        "question": question,
                        "answer": answer_text,
                        "has_search_activity": has_search_activity,
                        "n_underlying_sources": n_sources,
                    }

                    reservoir_add(
                        zero_citation_reservoirs,
                        zero_citation_seen,
                        primary_intent,
                        candidate,
                        zero_citation_allocation[primary_intent],
                        rng_a,
                    )

            # --- Part D: Recommendation, winner meaningfully more cited ---
            if primary_intent == "Recommendation" and is_decisive:
                citations_a, _, _ = citation_stats_for_side(record, "a")
                citations_b, _, _ = citation_stats_for_side(record, "b")

                if winner == "model_a":
                    winner_side, loser_side = "a", "b"
                    winner_citations, loser_citations = citations_a, citations_b
                else:
                    winner_side, loser_side = "b", "a"
                    winner_citations, loser_citations = citations_b, citations_a

                citation_diff = winner_citations - loser_citations

                if citation_diff >= RECOMMENDATION_MIN_CITATION_DIFF:
                    winner_answer = "\n\n".join(
                        get_assistant_messages(get_messages(record, winner_side))
                    )
                    loser_answer = "\n\n".join(
                        get_assistant_messages(get_messages(record, loser_side))
                    )

                    candidate = {
                        "question": first_question,
                        "winner_side": winner_side,
                        "winner_model": strip_model_suffix(record.get(f"model_{winner_side}")),
                        "winner_citations": winner_citations,
                        "winner_answer": winner_answer,
                        "loser_side": loser_side,
                        "loser_model": strip_model_suffix(record.get(f"model_{loser_side}")),
                        "loser_citations": loser_citations,
                        "loser_answer": loser_answer,
                        "citation_diff": citation_diff,
                    }

                    recommendation_seen += 1
                    if len(recommendation_reservoir) < RECOMMENDATION_EXAMPLE_TARGET:
                        recommendation_reservoir.append(candidate)
                    else:
                        j = rng_d.integers(0, recommendation_seen)
                        if j < RECOMMENDATION_EXAMPLE_TARGET:
                            recommendation_reservoir[j] = candidate

    zero_citation_total = sum(len(v) for v in zero_citation_reservoirs.values())
    intent_validation_total = sum(len(v) for v in intent_validation_reservoirs.values())

    print(f"\nPart A: collected {zero_citation_total} zero-citation turn samples across {len(zero_citation_reservoirs)} categories")
    print(f"Part D: collected {len(recommendation_reservoir)} Recommendation high-citation-diff examples (from {recommendation_seen} qualifying candidates seen)")
    print(f"Part E: collected {intent_validation_total} intent-validation prompt samples across {len(intent_validation_reservoirs)} categories")

    return zero_citation_reservoirs, recommendation_reservoir, intent_validation_reservoirs


# ============================================================
# PART A OUTPUT: zero_citation_manual_review.txt
# ============================================================

def write_zero_citation_review(zero_citation_reservoirs):
    lines = []
    lines.append("ZERO-CITATION MANUAL REVIEW SAMPLE")
    lines.append("=" * 100)
    lines.append(
        f"{ZERO_CITATION_SAMPLE_TOTAL} turns sampled across the 9 primary_intent "
        "categories where the visible citation count for that specific turn "
        "was zero. Auto-classification was deliberately NOT attempted - "
        "'search activity present but not cited' vs 'deliberately no "
        "citation needed' is a human judgment call. Fill in a label by hand "
        "for each entry below."
    )
    lines.append("=" * 100)
    lines.append("")

    entry_num = 0
    for intent in INTENT_ORDER:
        for item in zero_citation_reservoirs.get(intent, []):
            entry_num += 1
            lines.append(f"=== Entry {entry_num} ===")
            lines.append(f"Primary intent: {item['primary_intent']}")
            lines.append(f"Model: {item['base_model']} (side {item['side']})")
            lines.append(f"Turn: {item['turn_index'] + 1} of {item['total_turns']}")

            if item["has_search_activity"] is None:
                search_note = "UNKNOWN (no trace entry found for this turn)"
            elif item["has_search_activity"]:
                search_note = f"YES - {item['n_underlying_sources']} underlying source(s) found but not visibly cited"
            else:
                search_note = "NO - trace entry for this turn is genuinely empty"
            lines.append(f"Search trace activity for this turn: {search_note}")
            lines.append("")

            lines.append("--- User question (this turn) ---")
            lines.append(item["question"])
            lines.append("")

            lines.append("--- Assistant answer (this turn, full text) ---")
            lines.append(item["answer"])
            lines.append("")

            lines.append("--- MANUAL LABEL (fill in by hand) ---")
            lines.append("[ ] Likely failed/incomplete search")
            lines.append("[ ] Deliberate non-citation (general knowledge, creative task, etc.)")
            lines.append("[ ] Other / unclear")
            lines.append("Notes: ______________________________________________")
            lines.append("")
            lines.append("-" * 100)
            lines.append("")

    with open(ZERO_CITATION_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved -> {ZERO_CITATION_TXT} ({entry_num} entries)")
    return entry_num


# ============================================================
# PART D OUTPUT: recommendation_citation_examples.txt
# ============================================================

def write_recommendation_examples(recommendation_examples):
    lines = []
    lines.append("RECOMMENDATION CATEGORY: HIGH CITATION-DIFFERENCE EXAMPLES")
    lines.append("=" * 100)
    lines.append(
        f"{len(recommendation_examples)} decisive Recommendation-category "
        f"conversations where the winner had at least "
        f"{RECOMMENDATION_MIN_CITATION_DIFF} more (deduped, per-turn-summed) "
        "citations than the loser. For manually judging whether "
        "Recommendation answers naturally list multiple distinct options "
        "(restaurants, products, books, etc.) in a way that makes citation "
        "count a more genuine thoroughness signal here than elsewhere."
    )
    lines.append("=" * 100)
    lines.append("")

    for i, item in enumerate(recommendation_examples, start=1):
        lines.append(f"=== Example {i} ===")
        lines.append(f"Citation difference (winner - loser): +{item['citation_diff']}")
        lines.append("")
        lines.append("--- User question ---")
        lines.append(item["question"])
        lines.append("")
        lines.append(
            f"--- WINNER: {item['winner_model']} (side {item['winner_side']}) "
            f"- {item['winner_citations']} citations ---"
        )
        lines.append(item["winner_answer"])
        lines.append("")
        lines.append(
            f"--- LOSER: {item['loser_model']} (side {item['loser_side']}) "
            f"- {item['loser_citations']} citations ---"
        )
        lines.append(item["loser_answer"])
        lines.append("")
        lines.append("-" * 100)
        lines.append("")

    with open(RECOMMENDATION_EXAMPLES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved -> {RECOMMENDATION_EXAMPLES_TXT} ({len(recommendation_examples)} examples)")


# ============================================================
# PART E OUTPUT: intent_manual_validation.txt
# ============================================================

def write_intent_validation(intent_validation_reservoirs):
    lines = []
    lines.append("PRIMARY_INTENT MANUAL VALIDATION SAMPLE")
    lines.append("=" * 100)
    lines.append(
        f"{INTENT_VALIDATION_SAMPLE_TOTAL} prompts sampled roughly evenly "
        "across the 9 primary_intent categories, for a human spot-check of "
        "whether the assigned label matches what a person would intuitively "
        "call it. No other data included by design - this is not something "
        "the script should try to auto-validate."
    )
    lines.append("=" * 100)
    lines.append("")

    entry_num = 0
    for intent in INTENT_ORDER:
        for prompt in intent_validation_reservoirs.get(intent, []):
            entry_num += 1
            lines.append(f"{entry_num}. Intent: {intent}")
            lines.append(f"   Prompt: {prompt}")
            lines.append("")

    with open(INTENT_VALIDATION_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved -> {INTENT_VALIDATION_TXT} ({entry_num} entries)")
    return entry_num


# ============================================================
# STEP 4: 9x2 STRATIFICATION
# ============================================================

def run_stratified_9x2(df):
    print("\n" + "=" * 100)
    print("STEP 4: SINGLE-TURN VS MULTI-TURN x 9 PRIMARY_INTENT CATEGORIES (9x2 TABLE)")
    print("=" * 100)

    cells = []

    for intent in INTENT_ORDER:
        for conv_type in CONV_TYPE_ORDER:
            cell_df = df[
                (df["primary_intent"] == intent)
                & (df["conversation_type"] == conv_type)
            ]

            n_decisive = int(cell_df["is_decisive"].sum())
            long_df = build_long_table(cell_df)
            raw, model_ctrl, model_len_ctrl = three_way_correlations(long_df)

            low_power = n_decisive < MIN_DECISIVE_FOR_CONFIDENCE

            cells.append(
                {
                    "primary_intent": intent,
                    "conversation_type": conv_type,
                    "n_decisive": n_decisive,
                    "corr_raw": raw,
                    "corr_model_controlled": model_ctrl,
                    "corr_model_length_controlled": model_len_ctrl,
                    "low_power": low_power,
                }
            )

            flag = "  <-- LOW POWER" if low_power else ""
            print(
                f"\n{intent:<22} / {conv_type:<11}  n_decisive={n_decisive:>5}  "
                f"raw={raw:+.3f}  model_ctrl={model_ctrl:+.3f}  "
                f"model+length_ctrl={model_len_ctrl:+.3f}{flag}"
            )

    return pd.DataFrame(cells)


# ============================================================
# STEP 5: VOTE-PRESENCE, INTENT + MODEL CONTROLLED
# ============================================================

def build_vote_presence_long_table(df):
    """
    Unroll all records (voted or not) into two response-level rows:
    primary_intent (shared), base_model (that side's), citation_count
    (that side's), vote_occurred (record-level, shared/duplicated).
    """
    rows = []
    for _, r in df.iterrows():
        vote_int = 1 if r["vote_occurred"] else 0
        rows.append(
            {
                "primary_intent": r["primary_intent"],
                "base_model": r["base_model_a"],
                "citation_count": r["citations_a"],
                "vote_occurred": vote_int,
            }
        )
        rows.append(
            {
                "primary_intent": r["primary_intent"],
                "base_model": r["base_model_b"],
                "citation_count": r["citations_b"],
                "vote_occurred": vote_int,
            }
        )
    return pd.DataFrame(rows)


def run_vote_presence_analysis(df):
    print("\n" + "=" * 100)
    print("STEP 5: DOES CITATION COUNT PREDICT WHETHER A VOTE HAPPENED AT ALL?")
    print("=" * 100)

    analysis_df = df.dropna(subset=["primary_intent"]).copy()
    long_df = build_vote_presence_long_table(analysis_df)

    n_rows = len(long_df)
    n_records = len(analysis_df)
    vote_rate = long_df["vote_occurred"].mean()

    raw_corr = safe_corr(long_df["citation_count"].values, long_df["vote_occurred"].values)

    intent_ctrl = partial_correlation(
        long_df, "citation_count", "vote_occurred", categorical_cols=["primary_intent"]
    )

    intent_model_ctrl = partial_correlation(
        long_df,
        "citation_count",
        "vote_occurred",
        categorical_cols=["primary_intent", "base_model"],
    )

    print(f"\nRecords analyzed: {n_records} (response-level rows: {n_rows})")
    print(f"Overall vote rate: {vote_rate:.1%}")
    print(
        f"\nCitation count vs vote occurring:"
        f"\n  raw correlation:                          {raw_corr:+.4f}"
        f"\n  controlled for primary_intent:            {intent_ctrl:+.4f}"
        f"\n  controlled for primary_intent + base model: {intent_model_ctrl:+.4f}"
    )

    caveat = (
        "CAVEAT: even after controlling for primary_intent and base model "
        "identity, this correlation cannot distinguish between \"high friction "
        "causes abandonment\" and \"a satisfying answer prompts a quick, "
        "casual vote\" - both would produce the same pattern, and the data "
        "cannot separate them. Treat this as descriptive, not causal."
    )
    print(f"\n{caveat}")

    return {
        "n_records": n_records,
        "n_response_rows": n_rows,
        "vote_rate": vote_rate,
        "raw_corr": raw_corr,
        "intent_controlled_corr": intent_ctrl,
        "intent_model_controlled_corr": intent_model_ctrl,
        "caveat": caveat,
    }


# ============================================================
# STEP 6: SIGNIFICANCE TEST ON SINGLE-TURN VS MULTI-TURN GAP
# ============================================================

def run_significance_tests(stratified_df):
    print("\n" + "=" * 100)
    print("STEP 6: SIGNIFICANCE TEST - SINGLE-TURN VS MULTI-TURN GAP (former 'High' tier categories)")
    print("=" * 100)
    print(
        "\nUsing Fisher z-transformation on the RAW citation-winning correlation "
        "(the plain, standard use of this test). Sample size used for the test "
        "is n_decisive (records), not the doubled response-row count, since "
        "the two response rows per record are not independent draws - see "
        "fisher_z_test() docstring."
    )

    results = []

    for intent in FORMER_HIGH_TIER_CATEGORIES:
        single_row = stratified_df[
            (stratified_df["primary_intent"] == intent)
            & (stratified_df["conversation_type"] == "single_turn")
        ].iloc[0]
        multi_row = stratified_df[
            (stratified_df["primary_intent"] == intent)
            & (stratified_df["conversation_type"] == "multi_turn")
        ].iloc[0]

        r1, n1 = single_row["corr_raw"], single_row["n_decisive"]
        r2, n2 = multi_row["corr_raw"], multi_row["n_decisive"]

        z_stat, p_value = fisher_z_test(r1, n1, r2, n2)
        significant = (not np.isnan(p_value)) and p_value < SIGNIFICANCE_ALPHA

        low_power_involved = bool(single_row["low_power"] or multi_row["low_power"])

        results.append(
            {
                "primary_intent": intent,
                "r_single_turn": r1,
                "n_single_turn": int(n1),
                "r_multi_turn": r2,
                "n_multi_turn": int(n2),
                "z_statistic": z_stat,
                "p_value": p_value,
                "significant_at_0.05": significant,
                "low_power_involved": low_power_involved,
            }
        )

        verdict = "SIGNIFICANT" if significant else "not significant"
        power_note = "  (NOTE: at least one side is low-power; interpret cautiously)" if low_power_involved else ""
        print(
            f"\n{intent}:"
            f"\n  single_turn: r={r1:+.3f} (n={n1})   multi_turn: r={r2:+.3f} (n={n2})"
            f"\n  Fisher z = {z_stat:+.3f}, p = {p_value:.4f} -> {verdict} at alpha=0.05{power_note}"
        )

    return pd.DataFrame(results)


# ============================================================
# STEP 7: MARKDOWN SUMMARY
# ============================================================

def classify_length_control_effect(row):
    """
    Did the correlation survive adding the response-length control, or
    was it mostly/entirely verbosity bias?
    """
    raw = row["corr_model_controlled"]
    length_ctrl = row["corr_model_length_controlled"]

    if np.isnan(raw) or np.isnan(length_ctrl):
        return "n/a"

    if abs(length_ctrl) < NEAR_ZERO_THRESHOLD:
        return "collapsed to ~0 - mostly verbosity bias"

    if abs(raw) > 1e-9:
        relative_drop = 1 - (abs(length_ctrl) / abs(raw))
        if relative_drop >= LENGTH_CONTROL_DROP_THRESHOLD:
            return f"shrank substantially ({relative_drop:.0%} drop) - largely verbosity bias"

    if np.sign(raw) != np.sign(length_ctrl) and abs(length_ctrl) > NEAR_ZERO_THRESHOLD:
        return "changed sign after length control - unreliable"

    return "survived length control - likely a real citation signal"


def write_summary_markdown(
    by_intent_df,
    stratified_df,
    vote_result,
    sig_df,
    length_diag_df,
    cost_tier_df,
    cost_tier_total_tests,
    zero_citation_count,
    recommendation_example_count,
    intent_validation_count,
):
    lines = []
    lines.append("# Final Analysis Summary\n")
    lines.append(
        "Consolidated analysis for the mentor review, built on the fixed "
        "per-turn citation aggregation from `intent_complexity_analysis_v2.py`, "
        "extended with response-length (verbosity) controls, 9-category "
        "granularity, and formal significance testing.\n"
    )

    # --- Step 1 ---
    lines.append("## 1. Response Length Control\n")
    lines.append(
        "`response_length_a` / `response_length_b`: total character count of "
        "all assistant turns concatenated per side (same per-turn aggregation "
        "pattern as the fixed citation counts). Used below as a second control "
        "variable alongside base model identity, since the Search Arena paper "
        "reports a verbosity coefficient of beta=0.334 that none of our prior "
        "analysis accounted for.\n"
    )

    # --- Step 2+3 ---
    lines.append("## 2-3. Citation-Winning Correlation by primary_intent (9 categories, 3-way control)\n")
    lines.append(
        f"Categories with fewer than {MIN_DECISIVE_FOR_CONFIDENCE} decisive votes "
        "are flagged as low power.\n"
    )
    lines.append(
        "| Intent | n decisive | Raw | Model-controlled | Model+length-controlled | "
        "t-stat | p-value | Sig. at 0.05? | Length-control verdict | |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, row in by_intent_df.iterrows():
        verdict = classify_length_control_effect(row)
        flag = "LOW POWER" if row["low_power"] else ("BORDERLINE" if row["borderline"] else "")
        sig_str = "Yes" if row["significant_at_0.05"] else ("Borderline" if row["borderline"] else "No")
        lines.append(
            f"| {row['primary_intent']} | {row['n_decisive']} | "
            f"{row['corr_raw']:+.3f} | {row['corr_model_controlled']:+.3f} | "
            f"{row['corr_model_length_controlled']:+.3f} | {row['t_statistic']:+.3f} | "
            f"{row['p_value']:.4f} | {sig_str} | {verdict} | {flag} |"
        )
    lines.append(
        "\n**Fix 1 - significance testing:** the model+length-controlled "
        "correlation for each category was tested against zero via a t-test, "
        "using n_decisive (records, not the doubled response-row count) as "
        "the effective sample size - the same conservative approach already "
        "used for the single-turn/multi-turn Fisher z-test below. **9 tests "
        "were run without a multiple-comparisons correction** - at "
        f"alpha={SIGNIFICANCE_ALPHA}, roughly 0.4-0.5 false positives are "
        f"expected by chance across 9 tests. p-values in "
        f"[{BORDERLINE_P_LOW}, {SIGNIFICANCE_ALPHA}) are marked borderline "
        "and should not be treated as confirmed.\n"
    )

    # --- Fix 2: citation vs response length diagnostic ---
    lines.append("## Diagnostic: Citation Count vs Response Length Correlation\n")
    lines.append(
        "How entangled citation count and response length are - this "
        "explains why some categories' citation signal collapsed once "
        "length was controlled for.\n"
    )
    lines.append("| Scope | n decisive | Citation-Length correlation |")
    lines.append("|---|---|---|")
    for _, row in length_diag_df.iterrows():
        lines.append(
            f"| {row['scope']} | {row['n_decisive']} | {row['citation_length_corr']:+.3f} |"
        )
    lines.append("")

    # --- Parts B+C: three-tier cost comparison with significance testing ---
    lines.append("## Three-Tier Cost Comparison With Significance Testing (Routing Hypothesis Test)\n")
    lines.append(
        "Replaces the earlier generic-pricing-pattern approximation with "
        "verified real blended pricing (mid-2026, USD per million tokens, "
        "input/output blended). `sonar-reasoning` has no independently "
        "published price and is placed in the mid tier as an assumption. "
        "Search-fee surcharges on `gpt-4o-mini-search-preview` ($25-30 / "
        "1,000 calls) and `gpt-4o-search-preview` ($30-50 / 1,000 calls) are "
        "not reflected in the blended figures used for tiering, so those two "
        "models are relatively more expensive than their tier placement "
        "alone suggests.\n"
    )
    lines.append("| Tier | Models |")
    lines.append("|---|---|")
    lines.append(f"| Cheap | {', '.join(sorted(CHEAP_TIER))} |")
    lines.append(f"| Mid | {', '.join(sorted(MID_TIER))} |")
    lines.append(f"| Expensive | {', '.join(sorted(EXPENSIVE_TIER))} |")
    lines.append("")
    lines.append(
        f"For each intent category, three pairwise head-to-head comparisons "
        "are run (cheap-vs-mid, cheap-vs-expensive, mid-vs-expensive). Each "
        "reports the cheaper tier's win rate and a one-sample proportion "
        "z-test of that win rate against 50% (a win rate significantly "
        "below 50% is evidence to escalate that category away from the "
        f"cheaper tier). Comparisons with fewer than {MIN_H2H_FOR_CONFIDENCE} "
        "head-to-head decisive records are flagged as low power.\n"
    )
    lines.append(
        "| Intent | Comparison | n head-to-head | Cheaper-tier win rate | "
        "z | p-value | Sig. at 0.05? | |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, row in cost_tier_df.iterrows():
        flag = "LOW POWER" if row["low_power"] else ("BORDERLINE" if row["borderline"] else "")
        sig_str = "Yes" if row["significant_at_0.05"] else ("Borderline" if row["borderline"] else "No")
        rate_str = f"{row['cheaper_tier_win_rate']:.1%}" if not np.isnan(row["cheaper_tier_win_rate"]) else "n/a"
        lines.append(
            f"| {row['primary_intent']} | {row['comparison']} | {row['n_head_to_head']} | "
            f"{rate_str} | {row['z_statistic']:+.3f} | {row['p_value']:.4f} | {sig_str} | {flag} |"
        )
    lines.append(
        f"\n**Multiple comparisons:** {cost_tier_total_tests} per-category "
        f"tests were run ({len(INTENT_ORDER)} categories x {len(TIER_PAIRS)} "
        "tier comparisons), plus 3 'All categories' aggregate tests shown "
        "for context but excluded from that count. No correction was "
        f"applied - at alpha={SIGNIFICANCE_ALPHA} across "
        f"{cost_tier_total_tests} tests, roughly "
        f"{cost_tier_total_tests * SIGNIFICANCE_ALPHA:.1f} false positives "
        "are expected by chance alone. Treat p-values in "
        f"[{BORDERLINE_P_LOW}, {SIGNIFICANCE_ALPHA}) as borderline.\n"
    )

    # --- Parts A, D, E: manual-review raw outputs (not summarized here) ---
    lines.append("## Manual-Review Raw Outputs (Parts A, D, E)\n")
    lines.append(
        "These need a human to actually read them - they are intentionally "
        "left as raw text files rather than summarized or auto-classified "
        "here.\n"
    )
    lines.append(
        f"- **Part A** - `{ZERO_CITATION_TXT}`: {zero_citation_count} turns "
        "with zero visible citations, sampled across all 9 intent "
        "categories, each with its underlying web_search_trace activity "
        "noted and blank labels for manual classification (failed search "
        "vs. deliberate non-citation vs. other)."
    )
    lines.append(
        f"- **Part D** - `{RECOMMENDATION_EXAMPLES_TXT}`: "
        f"{recommendation_example_count} Recommendation-category examples "
        f"where the winner had at least {RECOMMENDATION_MIN_CITATION_DIFF} "
        "more citations than the loser, both full answers shown side by "
        "side, for judging whether Recommendation answers' natural "
        "list-of-options structure makes citation count a more genuine "
        "thoroughness signal there."
    )
    lines.append(
        f"- **Part E** - `{INTENT_VALIDATION_TXT}`: {intent_validation_count} "
        "prompts sampled roughly evenly across all 9 categories with just "
        "the intent label and prompt text, for a blind human spot-check of "
        "label quality.\n"
    )

    # --- Step 4 ---
    lines.append("## 4. Single-turn vs Multi-turn x 9 Categories (9x2 table)\n")
    lines.append("| Intent | Conversation type | n decisive | Raw | Model-controlled | Model+length-controlled | |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, row in stratified_df.iterrows():
        flag = "LOW POWER" if row["low_power"] else ""
        lines.append(
            f"| {row['primary_intent']} | {row['conversation_type']} | {row['n_decisive']} | "
            f"{row['corr_raw']:+.3f} | {row['corr_model_controlled']:+.3f} | "
            f"{row['corr_model_length_controlled']:+.3f} | {flag} |"
        )
    lines.append("")

    # --- Step 5 ---
    lines.append("## 5. Vote-Presence Analysis (intent + model identity controlled)\n")
    lines.append(f"- Records analyzed: {vote_result['n_records']}")
    lines.append(f"- Overall vote rate: {vote_result['vote_rate']:.1%}")
    lines.append(f"- Raw correlation: {vote_result['raw_corr']:+.4f}")
    lines.append(f"- Controlled for primary_intent: {vote_result['intent_controlled_corr']:+.4f}")
    lines.append(
        f"- Controlled for primary_intent + base model: "
        f"{vote_result['intent_model_controlled_corr']:+.4f}"
    )
    lines.append(f"\n> **{vote_result['caveat']}**\n")

    # --- Step 6 ---
    lines.append("## 6. Significance Test: Single-turn vs Multi-turn Gap (former 'High' tier)\n")
    lines.append(
        "Fisher z-test on the raw citation-winning correlation, using "
        "n_decisive (records) as the effective sample size rather than the "
        "doubled response-row count, since the two rows per record are not "
        "independent observations.\n"
    )
    lines.append("| Intent | r (single-turn) | n | r (multi-turn) | n | z | p-value | Significant at 0.05? | |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, row in sig_df.iterrows():
        power_flag = "LOW POWER SIDE" if row["low_power_involved"] else ""
        lines.append(
            f"| {row['primary_intent']} | {row['r_single_turn']:+.3f} | {row['n_single_turn']} | "
            f"{row['r_multi_turn']:+.3f} | {row['n_multi_turn']} | {row['z_statistic']:+.3f} | "
            f"{row['p_value']:.4f} | {'Yes' if row['significant_at_0.05'] else 'No'} | {power_flag} |"
        )
    lines.append("")

    # --- Step 7: confident vs uncertain ---
    lines.append("## What we can confidently say\n")

    confident_points = []
    survived = by_intent_df[
        by_intent_df.apply(
            lambda r: classify_length_control_effect(r).startswith("survived"), axis=1
        )
        & (~by_intent_df["low_power"])
        & by_intent_df["significant_at_0.05"]
    ]
    for _, row in survived.iterrows():
        borderline_note = (
            " - note: p is in the [0.01, 0.05) borderline range given 9 "
            "uncorrected tests, so treat this one a little more cautiously "
            "than the others in this list"
            if row["borderline"]
            else ""
        )
        confident_points.append(
            f"- **{row['primary_intent']}** (n={row['n_decisive']}): citation-winning "
            f"correlation survives model + response-length control "
            f"({row['corr_model_length_controlled']:+.3f}) AND is statistically "
            f"significant against zero (t={row['t_statistic']:+.3f}, "
            f"p={row['p_value']:.4f}) at the corrected sample size{borderline_note}."
        )

    if not confident_points:
        confident_points.append(
            "- No category's citation-winning correlation both (a) has adequate "
            "sample size, (b) clearly survives the response-length control, and "
            "(c) is statistically significant - see the exploratory section below."
        )

    lines.extend(confident_points)

    sig_yes = sig_df[sig_df["significant_at_0.05"] & (~sig_df["low_power_involved"])]
    if not sig_yes.empty:
        for _, row in sig_yes.iterrows():
            lines.append(
                f"- The single-turn vs multi-turn citation-correlation gap for "
                f"**{row['primary_intent']}** is statistically significant "
                f"(p={row['p_value']:.4f}), not plausibly noise."
            )

    lines.append(
        "\n- conv_id uniqueness and the citation-aggregation fix (established in "
        "`intent_complexity_analysis_v2.py`) remain solid - both were verified "
        "against real, full-scan data rather than assumed."
    )

    lines.append("\n## What remains uncertain or exploratory\n")

    uncertain_points = []

    collapsed_or_dropped = by_intent_df[
        by_intent_df.apply(
            lambda r: not classify_length_control_effect(r).startswith("survived")
            and not classify_length_control_effect(r) == "n/a",
            axis=1,
        )
    ]
    for _, row in collapsed_or_dropped.iterrows():
        verdict = classify_length_control_effect(row)
        low_power_note = " (also low power)" if row["low_power"] else ""
        uncertain_points.append(
            f"- **{row['primary_intent']}**: {verdict}{low_power_note}."
        )

    # Categories that survived the length control but didn't clear the bar
    # for "confident" (either low power or not statistically significant) -
    # these are a distinct middle ground, not full collapses.
    survived_but_not_confident = by_intent_df[
        by_intent_df.apply(
            lambda r: classify_length_control_effect(r).startswith("survived"), axis=1
        )
        & (by_intent_df["low_power"] | ~by_intent_df["significant_at_0.05"])
    ]
    for _, row in survived_but_not_confident.iterrows():
        if row["low_power"]:
            reason = "sample size is below the 200-decisive-vote threshold"
        elif row["borderline"]:
            reason = (
                f"p={row['p_value']:.4f} is borderline (between "
                f"{BORDERLINE_P_LOW} and {SIGNIFICANCE_ALPHA}) given 9 "
                "uncorrected tests"
            )
        else:
            reason = f"p={row['p_value']:.4f} does not clear alpha={SIGNIFICANCE_ALPHA}"
        uncertain_points.append(
            f"- **{row['primary_intent']}**: correlation survived the length "
            f"control ({row['corr_model_length_controlled']:+.3f}) but {reason} "
            "- not yet confirmed."
        )

    low_power_cells = stratified_df[stratified_df["low_power"]]
    if not low_power_cells.empty:
        cells_str = ", ".join(
            f"{r['primary_intent']}/{r['conversation_type']} (n={r['n_decisive']})"
            for _, r in low_power_cells.iterrows()
        )
        uncertain_points.append(f"- Low-power cells in the 9x2 table: {cells_str}.")

    sig_no_or_lowpower = sig_df[(~sig_df["significant_at_0.05"]) | sig_df["low_power_involved"]]
    for _, row in sig_no_or_lowpower.iterrows():
        reason = (
            "not statistically significant" if not row["significant_at_0.05"]
            else "significant but involves a low-power side"
        )
        uncertain_points.append(
            f"- The single-turn vs multi-turn gap for **{row['primary_intent']}** "
            f"is {reason} (p={row['p_value']:.4f}) - do not present this as a "
            "confirmed effect."
        )

    uncertain_points.append(
        "- The vote-presence finding (Step 5) is exploratory regardless of "
        "control variables: it is fundamentally correlational and the stated "
        "friction-vs-satisfaction ambiguity cannot be resolved with this data."
    )
    uncertain_points.append(
        "- **Multiple comparisons**: the 9 per-category significance tests in "
        "Section 2-3 were run with no correction for multiple comparisons. At "
        f"alpha={SIGNIFICANCE_ALPHA} across 9 tests, roughly 0.4-0.5 false "
        "positives are expected by chance alone even if every category's true "
        "effect were zero. Treat any single category's significance as "
        "suggestive, not proof, until replicated."
    )
    uncertain_points.append(
        "- The cost-tier comparison now uses verified real blended pricing "
        "(Part B), which is more solid than the earlier generic-pattern "
        "approximation, but two caveats remain: `sonar-reasoning`'s price "
        "isn't independently published (assumed mid-tier), and the "
        "OpenAI search-preview models' per-call search fees aren't folded "
        "into the blended figures used for tiering - so treat tier "
        "placement as approximately, not exactly, cost-ordered."
    )
    uncertain_points.append(
        "- The mid-tier beating the expensive tier in several categories "
        "(e.g. Other 60.2%, Explanation 59.3%, Info Synthesis 55.5%, and "
        "54.7% overall - see the three-tier table) is a genuinely "
        "surprising result worth flagging on its own: it suggests "
        "'expensive' does not mean 'better' in this data, independent of "
        "the cheap-tier routing question. Worth a follow-up look before "
        "presenting as a settled finding."
    )
    uncertain_points.append(
        "- All partial correlations here use OLS residualization (fixed effects "
        "for categorical controls, linear for response length) with Pearson "
        "correlation on the residuals - not a formal significance-tested "
        "regression model. The Step 6 single-turn/multi-turn gap and the Fix 1 "
        "by-intent tests do have real p-values attached (both using n_decisive "
        "as the effective sample size), but every other 'controlled' "
        "correlation here (Step 4's 9x2 table, Step 5's vote-presence result) "
        "is still an untested directional effect-size estimate."
    )

    lines.extend(uncertain_points)

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved final written summary -> {SUMMARY_MD}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("FINAL ANALYSIS - Search Arena")
    print("=" * 100)

    print(f"\nLoading full dataset from {DATASET_FILE} (all rows)...")
    df, total, unmapped_intent_count = load_full_table()
    print(f"Loaded {total} total records.")
    if unmapped_intent_count:
        print(
            f"NOTE: {unmapped_intent_count} record(s) have a primary_intent "
            "outside the 9 known categories and are excluded from "
            "intent-level analysis."
        )

    df.to_csv(DERIVED_CSV, index=False)
    print(f"Saved full per-record derived table -> {DERIVED_CSV} ({len(df)} rows)")

    by_intent_df = run_by_intent_analysis(df)
    by_intent_df.to_csv(BY_INTENT_CSV, index=False)
    print(f"\nSaved by-intent summary -> {BY_INTENT_CSV}")

    length_diag_df = run_citation_length_diagnostic(df)
    length_diag_df.to_csv(CITATION_LENGTH_DIAG_CSV, index=False)
    print(f"\nSaved citation-length diagnostic -> {CITATION_LENGTH_DIAG_CSV}")

    cost_tier_df, cost_tier_total_tests = run_cost_tier_analysis(df)
    cost_tier_df.to_csv(COST_TIER_CSV, index=False)
    print(f"\nSaved cost-tier comparison -> {COST_TIER_CSV}")

    zero_citation_reservoirs, recommendation_examples, intent_validation_reservoirs = (
        collect_manual_review_samples()
    )
    zero_citation_count = write_zero_citation_review(zero_citation_reservoirs)
    write_recommendation_examples(recommendation_examples)
    intent_validation_count = write_intent_validation(intent_validation_reservoirs)

    stratified_df = run_stratified_9x2(df)
    stratified_df.to_csv(STRATIFIED_CSV, index=False)
    print(f"\nSaved 9x2 stratified summary -> {STRATIFIED_CSV}")

    vote_result = run_vote_presence_analysis(df)
    pd.DataFrame([vote_result]).drop(columns=["caveat"]).to_csv(VOTE_PRESENCE_CSV, index=False)
    print(f"\nSaved vote-presence summary -> {VOTE_PRESENCE_CSV}")

    sig_df = run_significance_tests(stratified_df)
    sig_df.to_csv(SIGNIFICANCE_CSV, index=False)
    print(f"\nSaved significance test results -> {SIGNIFICANCE_CSV}")

    write_summary_markdown(
        by_intent_df,
        stratified_df,
        vote_result,
        sig_df,
        length_diag_df,
        cost_tier_df,
        cost_tier_total_tests,
        zero_citation_count,
        len(recommendation_examples),
        intent_validation_count,
    )


if __name__ == "__main__":
    main()
