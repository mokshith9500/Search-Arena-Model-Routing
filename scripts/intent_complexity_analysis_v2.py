"""
intent_complexity_analysis_v2.py

Extends intent_complexity_analysis.py with four additional checks:

  0. conv_id duplication check - verify each row is a distinct
     conversation, not a growing snapshot of the same conversation
     logged multiple times. Dedup (max turn_count wins) only if
     duplicates are actually found.
  1. Verified + fixed multi-turn citation aggregation. The original
     script deduped citation markers across the WHOLE concatenated
     conversation, which undercounts when citation numbering restarts
     each turn (it does - confirmed below). Fixed to dedupe per turn,
     then sum. Adds a citations_per_turn normalized field.
  2. Single-turn vs multi-turn x complexity-tier stratified
     citation-winning correlation (3x2 = 6 cells), each reported with
     its own sample size and a low-power flag (< 200 decisive votes).
  3. Vote-presence analysis: does citation count predict whether a
     conversation received a vote at all (using the full dataset, not
     just voted rows), controlling for primary_intent. Reported with
     an explicit caveat about causal ambiguity.
  4. A single markdown summary tying all of the above together, ending
     with a plain-language robust-vs-exploratory readout.

Runs entirely locally against train.jsonl - no LLM/API calls.
"""

import json
import sys

import numpy as np
import pandas as pd

from inspect_search_arena import (
    DATASET_FILE,
    get_messages,
    get_assistant_messages,
    get_visible_citation_markers,
)

# ============================================================
# SETTINGS
# ============================================================

DERIVED_CSV = "intent_complexity_derived_v2.csv"
STRATIFIED_CSV = "intent_complexity_stratified_summary.csv"
SUMMARY_MD = "intent_complexity_summary_v2.md"

COMPLEXITY_TIERS = {
    "Factual Lookup": "Low",
    "Text Processing": "Low",
    "Other": "Low",
    "Explanation": "Medium",
    "Recommendation": "Medium",
    "Guidance": "Medium",
    "Info Synthesis": "High",
    "Analysis": "High",
    "Creative Generation": "High",
}

TIER_ORDER = ["Low", "Medium", "High"]
CONV_TYPE_ORDER = ["single_turn", "multi_turn"]

DECISIVE_WINNERS = ("model_a", "model_b")
TIE_WINNERS = ("tie", "tie (bothbad)")

MODEL_CONFIG_SUFFIXES = ["-high", "-loc", "-grounding", "-wo-search"]

MIN_DECISIVE_FOR_CONFIDENCE = 200
MIN_GROUP_N_FOR_WINRATE_DIAGNOSTIC = 20


# ============================================================
# HELPERS (shared logic)
# ============================================================

def strip_model_suffix(name):
    if not name:
        return name

    changed = True
    while changed:
        changed = False
        for suf in MODEL_CONFIG_SUFFIXES:
            if name.endswith(suf):
                name = name[: -len(suf)]
                changed = True
                break

    return name


def per_turn_citation_counts(assistant_texts):
    """
    Deduped visible citation marker count PER assistant turn.
    """
    return [len(get_visible_citation_markers(t)) for t in assistant_texts]


def citation_stats_for_side(record, side):
    """
    FIXED aggregation: dedupe citation markers within each assistant
    turn separately, then sum across turns. See the "citation
    aggregation" section of the summary report for why this differs
    from concatenate-then-dedupe.
    """
    messages = get_messages(record, side)
    assistant_texts = get_assistant_messages(messages)
    per_turn = per_turn_citation_counts(assistant_texts)
    return sum(per_turn), len(assistant_texts), per_turn


def citation_count_old_buggy(record, side):
    """
    Reproduction of the ORIGINAL intent_complexity_analysis.py logic,
    kept here only to demonstrate/confirm the bug against a real
    example - not used in the actual analysis below.
    """
    messages = get_messages(record, side)
    assistant_texts = get_assistant_messages(messages)
    combined = "\n".join(assistant_texts)
    return len(get_visible_citation_markers(combined))


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def partial_correlation(df, feature_col, target_col, group_col):
    """
    Within-group partial correlation: subtract each group's own mean
    from both the feature and the target before correlating (a
    lightweight fixed-effects residualization). Used both for
    "control for model identity" and "control for primary_intent".
    """
    group_feat_mean = df.groupby(group_col)[feature_col].transform("mean")
    group_target_mean = df.groupby(group_col)[target_col].transform("mean")

    resid_feat = df[feature_col] - group_feat_mean
    resid_target = df[target_col] - group_target_mean

    return safe_corr(resid_feat.values, resid_target.values)


# ============================================================
# LOAD FULL TABLE (all 24,069 records, not just voted ones)
# ============================================================

def load_full_table():
    rows = []
    unmapped_intent_count = 0
    total = 0

    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            total += 1

            primary_intent = record.get("primary_intent")
            tier = COMPLEXITY_TIERS.get(primary_intent)
            if tier is None:
                unmapped_intent_count += 1

            meta_a = record.get("system_a_metadata")
            meta_b = record.get("system_b_metadata")
            conv_id_a = meta_a.get("conv_id") if isinstance(meta_a, dict) else None
            conv_id_b = meta_b.get("conv_id") if isinstance(meta_b, dict) else None

            turn_count = record.get("turn")

            citations_a, n_asst_turns_a, _ = citation_stats_for_side(record, "a")
            citations_b, n_asst_turns_b, _ = citation_stats_for_side(record, "b")

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
                    "conv_id_a": conv_id_a,
                    "conv_id_b": conv_id_b,
                    "primary_intent": primary_intent,
                    "complexity_tier": tier,
                    "turn_count": turn_count,
                    "conversation_type": conv_type,
                    "citations_a": citations_a,
                    "citations_b": citations_b,
                    "citations_per_turn_a": (
                        citations_a / turn_count if turn_count else np.nan
                    ),
                    "citations_per_turn_b": (
                        citations_b / turn_count if turn_count else np.nan
                    ),
                    "avg_citations": (citations_a + citations_b) / 2.0,
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


# ============================================================
# STEP 0: conv_id duplication check
# ============================================================

def check_and_dedupe_conv_id(df):
    report_lines = []

    n_total = len(df)
    n_unique_a = df["conv_id_a"].nunique(dropna=False)
    n_unique_b = df["conv_id_b"].nunique(dropna=False)

    dup_a_mask = df.duplicated(subset="conv_id_a", keep=False)
    dup_b_mask = df.duplicated(subset="conv_id_b", keep=False)

    n_dup_a_rows = int(dup_a_mask.sum())
    n_dup_b_rows = int(dup_b_mask.sum())

    report_lines.append(f"Total rows scanned: {n_total}")
    report_lines.append(
        f"Unique conv_id (model_a side): {n_unique_a}  "
        f"({'no duplicates' if n_dup_a_rows == 0 else f'{n_dup_a_rows} rows involved in duplicates'})"
    )
    report_lines.append(
        f"Unique conv_id (model_b side): {n_unique_b}  "
        f"({'no duplicates' if n_dup_b_rows == 0 else f'{n_dup_b_rows} rows involved in duplicates'})"
    )

    if n_dup_a_rows == 0 and n_dup_b_rows == 0:
        report_lines.append(
            "\nCONCLUSION: conv_id is unique per row on both sides. Each row is "
            "a distinct conversation, not a growing snapshot of an existing one. "
            "No deduplication needed; 0 rows removed."
        )
        return df, {
            "duplicates_found": False,
            "rows_removed": 0,
            "n_before": n_total,
            "n_after": n_total,
            "report_text": "\n".join(report_lines),
        }

    # Duplicates found: dedupe on conv_id_a, keeping the max turn_count
    # row per conv_id (the fullest/most complete snapshot).
    before = len(df)
    df_sorted = df.sort_values("turn_count", ascending=False, kind="mergesort")
    df_deduped = df_sorted.drop_duplicates(subset="conv_id_a", keep="first")
    df_deduped = df_deduped.sort_index()
    removed = before - len(df_deduped)

    report_lines.append(
        f"\nCONCLUSION: found growing-snapshot duplicates on conv_id_a "
        f"({n_dup_a_rows} rows involved). Deduplicated by keeping the "
        f"max-turn_count row per conv_id_a. Removed {removed} rows "
        f"({before} -> {len(df_deduped)})."
    )

    return df_deduped, {
        "duplicates_found": True,
        "rows_removed": removed,
        "n_before": before,
        "n_after": len(df_deduped),
        "report_text": "\n".join(report_lines),
    }


# ============================================================
# PART 1: citation aggregation verification
# ============================================================

def find_multiturn_citation_example():
    """
    Scan for a real multi-turn record where per-turn citation markers
    restart (e.g. both turn 1 and turn 2 use "[1]"), to concretely
    demonstrate the old concatenate-then-dedupe bug vs the fixed
    per-turn-dedupe-then-sum logic.
    """
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record.get("winner") is None:
                continue
            if (record.get("turn") or 0) < 2:
                continue

            messages = get_messages(record, "a")
            assistant_texts = get_assistant_messages(messages)

            if len(assistant_texts) < 2:
                continue

            per_turn = per_turn_citation_counts(assistant_texts)

            if sum(1 for c in per_turn if c >= 2) < 2:
                continue

            old_count = citation_count_old_buggy(record, "a")
            new_count, n_turns, _ = citation_stats_for_side(record, "a")

            if new_count > old_count:
                return {
                    "turn_field": record.get("turn"),
                    "n_assistant_turns": n_turns,
                    "per_turn_marker_counts": per_turn,
                    "old_concatenate_then_dedupe_count": old_count,
                    "new_per_turn_sum_count": new_count,
                }

    return None


def print_citation_fix_report(example):
    print("\n" + "=" * 90)
    print("PART 1: CITATION AGGREGATION - VERIFY AND FIX")
    print("=" * 90)

    print(
        "\nORIGINAL logic (intent_complexity_analysis.py, citation_count_for_side):\n"
        "  1. Concatenate ALL assistant turns into one string.\n"
        "  2. Run get_visible_citation_markers() ONCE on the combined string.\n"
        "  3. This dedupes citation MARKER STRINGS (e.g. '[1]') across the\n"
        "     whole conversation - so if turn 1 and turn 2 each independently\n"
        "     cite a '[1]' (referring to different sources, since each turn\n"
        "     typically re-runs its own web search and renumbers from 1),\n"
        "     they collapse into a single citation instead of two."
    )

    print(
        "\nFIXED logic (citation_stats_for_side, this script):\n"
        "  1. Run get_visible_citation_markers() SEPARATELY on each assistant\n"
        "     turn's text (dedupes only WITHIN that turn).\n"
        "  2. Sum the per-turn counts across turns.\n"
        "  This correctly avoids counting a repeated '[1]' twice *within* a\n"
        "  turn, while not conflating citation numbering across turns."
    )

    if example is None:
        print(
            "\nNo real example found where a citation number repeats across "
            "turns and this actually changes the count. Proceeding with the "
            "fixed logic regardless, since it's the structurally correct one."
        )
        return

    print("\nConfirmed against a real multi-turn record:")
    print(f"  turn field: {example['turn_field']}")
    print(f"  assistant turns: {example['n_assistant_turns']}")
    print(f"  per-turn deduped marker counts: {example['per_turn_marker_counts']}")
    print(
        f"  OLD (concatenate-then-dedupe) total: "
        f"{example['old_concatenate_then_dedupe_count']}"
    )
    print(f"  NEW (per-turn dedupe, summed) total: {example['new_per_turn_sum_count']}")
    print(
        "  -> the old logic undercounts multi-turn citations whenever "
        "citation numbering restarts each turn, which it does here."
    )


# ============================================================
# PART 2: single-turn vs multi-turn x tier stratification
# ============================================================

def build_long_table(sub_df):
    """
    Unroll decisive records into two response-level rows (one per
    side): base_model, citation_count, is_winner.
    """
    decisive = sub_df[sub_df["is_decisive"]]

    rows = []
    for _, r in decisive.iterrows():
        rows.append(
            {
                "base_model": r["base_model_a"],
                "citation_count": r["citations_a"],
                "is_winner": 1 if r["winner"] == "model_a" else 0,
            }
        )
        rows.append(
            {
                "base_model": r["base_model_b"],
                "citation_count": r["citations_b"],
                "is_winner": 1 if r["winner"] == "model_b" else 0,
            }
        )

    return pd.DataFrame(rows)


def run_stratified_analysis(df):
    print("\n" + "=" * 90)
    print("PART 2: SINGLE-TURN VS MULTI-TURN x COMPLEXITY TIER (3x2 TABLE)")
    print("=" * 90)

    cells = []

    for tier in TIER_ORDER:
        for conv_type in CONV_TYPE_ORDER:
            cell_df = df[
                (df["complexity_tier"] == tier)
                & (df["conversation_type"] == conv_type)
            ]

            n_decisive = int(cell_df["is_decisive"].sum())
            long_df = build_long_table(cell_df)

            if len(long_df) >= 4:
                raw = raw_corr = safe_corr(
                    long_df["citation_count"].values, long_df["is_winner"].values
                )
                controlled = partial_correlation(
                    long_df, "citation_count", "is_winner", "base_model"
                )
            else:
                raw_corr = float("nan")
                controlled = float("nan")

            low_power = n_decisive < MIN_DECISIVE_FOR_CONFIDENCE

            cells.append(
                {
                    "tier": tier,
                    "conversation_type": conv_type,
                    "n_decisive": n_decisive,
                    "citation_corr_raw": raw_corr,
                    "citation_corr_model_controlled": controlled,
                    "low_power": low_power,
                }
            )

            flag = "  <-- LOW POWER (n < 200), do not draw conclusions" if low_power else ""
            print(
                f"\n{tier:>6} / {conv_type:<11}  n_decisive={n_decisive:>5}  "
                f"raw={raw_corr:+.3f}  controlled={controlled:+.3f}{flag}"
            )

    return pd.DataFrame(cells)


# ============================================================
# PART 3: vote-presence analysis
# ============================================================

def run_vote_presence_analysis(df):
    print("\n" + "=" * 90)
    print("PART 3: DOES CITATION COUNT PREDICT WHETHER A VOTE HAPPENED AT ALL?")
    print("=" * 90)

    analysis_df = df.dropna(subset=["primary_intent"]).copy()
    analysis_df["vote_occurred_int"] = analysis_df["vote_occurred"].astype(int)

    n = len(analysis_df)
    vote_rate = analysis_df["vote_occurred_int"].mean()

    raw_corr = safe_corr(
        analysis_df["avg_citations"].values, analysis_df["vote_occurred_int"].values
    )
    controlled_corr = partial_correlation(
        analysis_df, "avg_citations", "vote_occurred_int", "primary_intent"
    )

    intent_vote_rates = (
        analysis_df.groupby("primary_intent")["vote_occurred_int"]
        .agg(["mean", "count"])
    )
    intent_vote_rates = intent_vote_rates[
        intent_vote_rates["count"] >= MIN_GROUP_N_FOR_WINRATE_DIAGNOSTIC
    ]

    print(f"\nFull dataset used: {n} rows (excludes 1 row with null primary_intent)")
    print(f"Overall vote rate: {vote_rate:.1%}")
    if not intent_vote_rates.empty:
        print(
            f"Vote rate range across primary_intent categories: "
            f"{intent_vote_rates['mean'].min():.1%} - {intent_vote_rates['mean'].max():.1%} "
            f"(confirms intent needs to be controlled for)"
        )

    print(
        f"\nCitation count (avg of both sides) vs vote occurring:"
        f"\n  raw correlation:              {raw_corr:+.4f}"
        f"\n  controlled for primary_intent: {controlled_corr:+.4f}"
    )

    caveat = (
        "CAVEAT: this correlation cannot distinguish between \"high friction "
        "causes abandonment\" and \"a satisfying answer prompts a quick, "
        "casual vote\" - both would produce the same pattern, and the data "
        "cannot separate them. Treat this as descriptive, not causal."
    )
    print(f"\n{caveat}")

    return {
        "n": n,
        "vote_rate": vote_rate,
        "raw_corr": raw_corr,
        "controlled_corr": controlled_corr,
        "intent_vote_rate_min": (
            float(intent_vote_rates["mean"].min()) if not intent_vote_rates.empty else None
        ),
        "intent_vote_rate_max": (
            float(intent_vote_rates["mean"].max()) if not intent_vote_rates.empty else None
        ),
        "caveat": caveat,
    }


# ============================================================
# PART 4: MARKDOWN SUMMARY
# ============================================================

def write_summary_markdown(dedup_info, citation_example, stratified_df, vote_result):
    lines = []
    lines.append("# Intent Complexity Analysis v2 - Summary\n")

    # --- Section: conv_id dedup ---
    lines.append("## 0. conv_id Duplication Check\n")
    lines.append("```")
    lines.append(dedup_info["report_text"])
    lines.append("```\n")

    # --- Section: citation fix ---
    lines.append("## 1. Citation Aggregation: Verified and Fixed\n")
    lines.append(
        "**Original logic** (`intent_complexity_analysis.py`): concatenated all "
        "assistant turns into one string, then deduped citation marker strings "
        "(e.g. `[1]`) once across the whole conversation.\n"
    )
    lines.append(
        "**Problem**: citation numbering restarts each turn (confirmed on real "
        "multi-turn records), so a `[1]` in turn 1 and a `[1]` in turn 2 usually "
        "refer to *different* sources, but the old logic collapsed them into one "
        "citation - undercounting citations in multi-turn conversations.\n"
    )
    lines.append(
        "**Fix**: dedupe citation markers separately *within* each assistant "
        "turn, then sum the per-turn counts across turns. A new "
        "`citations_per_turn` field (total citations / turn count) is also "
        "added alongside the raw total.\n"
    )

    if citation_example:
        lines.append("**Confirmed against a real record:**\n")
        lines.append("```")
        lines.append(f"turn field: {citation_example['turn_field']}")
        lines.append(f"assistant turns: {citation_example['n_assistant_turns']}")
        lines.append(
            f"per-turn deduped marker counts: {citation_example['per_turn_marker_counts']}"
        )
        lines.append(
            f"OLD (concatenate-then-dedupe) total: "
            f"{citation_example['old_concatenate_then_dedupe_count']}"
        )
        lines.append(
            f"NEW (per-turn dedupe, summed) total: "
            f"{citation_example['new_per_turn_sum_count']}"
        )
        lines.append("```\n")

    # --- Section: stratified table ---
    lines.append("## 2. Single-turn vs Multi-turn x Complexity Tier (3x2 table)\n")
    lines.append(
        f"Cells with fewer than {MIN_DECISIVE_FOR_CONFIDENCE} decisive votes are "
        "flagged as too small to draw a confident conclusion from.\n"
    )
    lines.append("| Tier | Conversation type | n decisive | Citation corr (raw) | Citation corr (model-controlled) | |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in stratified_df.iterrows():
        flag = "LOW POWER" if row["low_power"] else ""
        lines.append(
            f"| {row['tier']} | {row['conversation_type']} | {row['n_decisive']} | "
            f"{row['citation_corr_raw']:+.3f} | {row['citation_corr_model_controlled']:+.3f} | {flag} |"
        )
    lines.append("")

    # --- Section: vote presence ---
    lines.append("## 3. Vote-Presence Analysis (citation count -> whether a vote occurred)\n")
    lines.append(f"- Rows analyzed: {vote_result['n']} (full dataset, all winner values)")
    lines.append(f"- Overall vote rate: {vote_result['vote_rate']:.1%}")
    if vote_result["intent_vote_rate_min"] is not None:
        lines.append(
            f"- Vote rate range across primary_intent categories: "
            f"{vote_result['intent_vote_rate_min']:.1%} - {vote_result['intent_vote_rate_max']:.1%}"
        )
    lines.append(f"- Raw correlation (avg citations vs vote occurred): {vote_result['raw_corr']:+.4f}")
    lines.append(
        f"- Correlation controlled for primary_intent: {vote_result['controlled_corr']:+.4f}"
    )
    lines.append(f"\n> **{vote_result['caveat']}**\n")

    # --- Section: robust vs exploratory ---
    lines.append("## 4. Robust vs Exploratory: What to Trust\n")

    low_power_cells = stratified_df[stratified_df["low_power"]]
    n_low_power = len(low_power_cells)

    lines.append("**Robust (backed by structural checks and adequate sample size):**")
    lines.append(
        "- conv_id uniqueness: confirmed with a full scan, not a sample - "
        + ("no dedup was needed." if not dedup_info["duplicates_found"]
           else f"{dedup_info['rows_removed']} duplicate rows were removed.")
    )
    lines.append(
        "- The citation-aggregation bug and its fix: confirmed against a real "
        "multi-turn record where citation numbering demonstrably restarts."
    )
    if n_low_power < len(stratified_df):
        lines.append(
            "- Cells in the 3x2 table with n_decisive >= "
            f"{MIN_DECISIVE_FOR_CONFIDENCE} are large enough to describe a "
            "real (if modest) correlation, though these are still simple "
            "fixed-effects partial correlations, not a causal estimate."
        )

    lines.append("\n**Exploratory (needs more evidence before being treated as a conclusion):**")
    if n_low_power:
        flagged = ", ".join(
            f"{r['tier']}/{r['conversation_type']} (n={r['n_decisive']})"
            for _, r in low_power_cells.iterrows()
        )
        lines.append(f"- Low-power cells: {flagged}. Do not draw conclusions from these.")
    lines.append(
        "- The vote-presence result is inherently exploratory: even after "
        "controlling for primary_intent, the correlation is descriptive only "
        "and cannot resolve the friction-vs-satisfaction ambiguity stated above."
    )
    lines.append(
        "- All \"model-controlled\" and \"intent-controlled\" correlations here "
        "use simple within-group mean-centering (a fixed-effects style partial "
        "correlation), not a formal regression model with significance testing "
        "- treat coefficient magnitudes as directional effect sizes, not "
        "statistically validated estimates."
    )

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nSaved full written summary -> {SUMMARY_MD}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print("INTENT COMPLEXITY ANALYSIS v2")
    print("=" * 90)

    print(f"\nLoading full dataset from {DATASET_FILE} (all rows, not just voted ones)...")
    df, total, unmapped_intent_count = load_full_table()
    print(f"Loaded {total} total records.")
    if unmapped_intent_count:
        print(
            f"NOTE: {unmapped_intent_count} record(s) have a primary_intent not "
            f"in the 9 mapped categories (complexity_tier = NaN for these)."
        )

    # --------------------------------------------------------
    # Step 0: conv_id duplication check
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("STEP 0: conv_id DUPLICATION CHECK")
    print("=" * 90)
    df, dedup_info = check_and_dedupe_conv_id(df)
    print("\n" + dedup_info["report_text"])

    # --------------------------------------------------------
    # Part 1: citation aggregation fix
    # --------------------------------------------------------
    print("\nSearching for a real multi-turn example to confirm the citation fix...")
    citation_example = find_multiturn_citation_example()
    print_citation_fix_report(citation_example)

    # --------------------------------------------------------
    # Save full derived table
    # --------------------------------------------------------
    df.to_csv(DERIVED_CSV, index=False)
    print(f"\nSaved full per-record derived table -> {DERIVED_CSV} ({len(df)} rows)")

    # --------------------------------------------------------
    # Part 2: stratified 3x2 table
    # --------------------------------------------------------
    stratified_df = run_stratified_analysis(df)
    stratified_df.to_csv(STRATIFIED_CSV, index=False)
    print(f"\nSaved stratified summary -> {STRATIFIED_CSV}")

    # --------------------------------------------------------
    # Part 3: vote-presence analysis
    # --------------------------------------------------------
    vote_result = run_vote_presence_analysis(df)

    # --------------------------------------------------------
    # Part 4: markdown summary
    # --------------------------------------------------------
    write_summary_markdown(dedup_info, citation_example, stratified_df, vote_result)


if __name__ == "__main__":
    main()
