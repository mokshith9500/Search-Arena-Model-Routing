"""
bh_correction.py

Applies the Benjamini-Hochberg false discovery rate correction to the
p-values already reported in bradley_terry_by_intent.csv and
bradley_terry_cost_tier.csv -- both flagged throughout this project as
uncorrected for multiple comparisons, but never actually corrected.

SCOPING DECISION (stated explicitly since the task left it to be decided):
the correction is applied PER FILE, pooling every `*_boot_p` column in that
file into one family for the BH procedure. This is the literal reading of
"the full set of p-values already reported in bradley_terry_by_intent.csv
and bradley_terry_cost_tier.csv" -- it does not split citation-vs-length
tests, or All-categories-vs-per-intent tier tests, into separate families.
If a narrower grouping is wanted later, the per-test p-values and family
sizes are printed and saved so it can be redone.

No re-fitting, no new analysis: this is a correction applied to p-values
that already exist in files already on disk.
"""

import numpy as np
import pandas as pd

BY_INTENT_CSV = "outputs/bradley_terry_by_intent.csv"
COST_TIER_CSV = "outputs/bradley_terry_cost_tier.csv"
SUMMARY_MD = "outputs/bradley_terry_summary.md"
SECTION_MARKER = "# Multiple-Comparisons Correction (Benjamini-Hochberg)"
ALPHA = 0.05


def bh_adjust(pvals):
    """
    Standard Benjamini-Hochberg step-up adjusted p-values ("q-values").
    For m p-values sorted ascending p_(1)<=...<=p_(m):
        q_(m) = p_(m)
        q_(i) = min(q_(i+1), p_(i) * m / i)   for i = m-1 .. 1
    then clipped to [0, 1]. A test is significant at level alpha iff its
    adjusted p-value <= alpha, which is equivalent to the usual BH step-up
    rejection rule.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.empty(m)
    q_ranked[-1] = ranked[-1]
    for i in range(m - 2, -1, -1):
        q_ranked[i] = min(q_ranked[i + 1], ranked[i] * m / (i + 1))
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty(m)
    q[order] = q_ranked
    return q


def correct_file(path, p_columns, label):
    print("=" * 78)
    print(f"{label}  ({path})")
    print("=" * 78)
    df = pd.read_csv(path)

    all_p = []
    locations = []   # (row_idx, col) for every p-value in the pooled family
    for col in p_columns:
        if col not in df.columns:
            continue
        for row_idx, val in df[col].items():
            if pd.notna(val):
                all_p.append(float(val))
                locations.append((row_idx, col))

    m = len(all_p)
    print(f"pooled family size: {m} p-values across columns {p_columns}")
    q = bh_adjust(all_p)

    for col in p_columns:
        if col not in df.columns:
            continue
        bh_col = col.replace("_boot_p", "_boot_p_bh")
        sig_col = col.replace("_boot_p", "_significant_bh")
        df[bh_col] = np.nan
        df[sig_col] = False

    n_changed = 0
    print(f"\n{'row':28}{'column':24}{'p (orig)':>10}{'p (BH)':>10}  sig(orig)->sig(BH)")
    report_rows = []
    for (row_idx, col), p_orig, q_val in zip(locations, all_p, q):
        bh_col = col.replace("_boot_p", "_boot_p_bh")
        sig_col = col.replace("_boot_p", "_significant_bh")
        sig_orig_col = col.replace("_boot_p", "_significant")
        df.at[row_idx, bh_col] = q_val
        sig_bh = bool(q_val <= ALPHA)
        df.at[row_idx, sig_col] = sig_bh
        sig_orig = bool(df.at[row_idx, sig_orig_col]) if sig_orig_col in df.columns else (p_orig <= ALPHA)
        changed = sig_orig != sig_bh
        n_changed += int(changed)
        row_label = df.at[row_idx, df.columns[0]]
        print(f"{str(row_label):28}{col:24}{p_orig:10.4f}{q_val:10.4f}  "
              f"{sig_orig!s:>5} -> {sig_bh!s:<5}{'  *** CHANGED ***' if changed else ''}")
        report_rows.append(dict(row=row_label, column=col, p_original=p_orig,
                                p_bh=q_val, significant_original=sig_orig,
                                significant_bh=sig_bh, changed=changed))

    print(f"\n{n_changed} of {m} tests changed significance status under BH correction.")
    df.to_csv(path, index=False)
    print(f"[saved] {path} (added *_boot_p_bh / *_significant_bh columns)")
    return pd.DataFrame(report_rows), m, n_changed


def _truncate_existing_section(path):
    """
    Removes a PREVIOUS run's copy of this script's own section, bounded to
    just that section's span (from its own "---" separator to the start of
    the NEXT top-level section, or EOF if it was last) -- not from the
    marker to end-of-file. Cutting to EOF was a real bug: if another
    script's section had been appended after this one, a later re-run of
    this script would silently delete that later content too. Fixed so each
    script only ever touches its own section, regardless of what has been
    appended after it since.
    """
    import os
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


def write_summary_note(intent_report, intent_m, intent_changed,
                       tier_report, tier_m, tier_changed, path=SUMMARY_MD):
    L = []
    A = L.append
    A("")
    A("---")
    A("")
    A(SECTION_MARKER)
    A("")
    A("Every p-value reported in `bradley_terry_by_intent.csv` and "
      "`bradley_terry_cost_tier.csv` was previously flagged as uncorrected for "
      "multiple comparisons (see Limitations #4 above) but never actually "
      "corrected. This section applies the standard Benjamini-Hochberg false "
      "discovery rate procedure and reports exactly what changes.")
    A("")
    A("**Scoping decision, stated explicitly:** the correction pools every "
      "`*_boot_p` column within a file into one family for that file "
      f"(`bradley_terry_by_intent.csv`: {intent_m} p-values across `cit_boot_p` "
      f"and `len_boot_p`; `bradley_terry_cost_tier.csv`: {tier_m} p-values "
      "across `mid_vs_cheap_boot_p`, `expensive_vs_cheap_boot_p`, "
      "`citation_boot_p`, `length_boot_p`, and `expensive_minus_mid_boot_p` "
      "over all 10 scope rows). This is the literal reading of \"the full set "
      "of p-values already reported in\" each file. A new `*_boot_p_bh` and "
      "`*_significant_bh` column was added to both CSVs alongside the "
      "original columns -- nothing original was overwritten or removed.")
    A("")
    A(f"**Result: {intent_changed} of {intent_m} by-intent tests and "
      f"{tier_changed} of {tier_m} cost-tier tests changed significance "
      "status under correction.**")
    A("")

    A("## By-intent citation and length tests")
    A("")
    A("| row | test | p (original) | p (BH-adjusted) | sig before | sig after | changed |")
    A("|---|---|---|---|---|---|---|")
    for _, r in intent_report.iterrows():
        A(f"| {r['row']} | {r['column']} | {r['p_original']:.4f} | {r['p_bh']:.4f} | "
          f"{r['significant_original']} | {r['significant_bh']} | "
          f"{'**YES**' if r['changed'] else 'no'} |")
    A("")

    A("## Cost-tier contrast tests")
    A("")
    A("| row | test | p (original) | p (BH-adjusted) | sig before | sig after | changed |")
    A("|---|---|---|---|---|---|---|")
    for _, r in tier_report.iterrows():
        A(f"| {r['row']} | {r['column']} | {r['p_original']:.4f} | {r['p_bh']:.4f} | "
          f"{r['significant_original']} | {r['significant_bh']} | "
          f"{'**YES**' if r['changed'] else 'no'} |")
    A("")

    survived = pd.concat([intent_report, tier_report])
    survived = survived[survived["significant_original"] & survived["significant_bh"]]
    lost = pd.concat([intent_report, tier_report])
    lost = lost[lost["significant_original"] & ~lost["significant_bh"]]

    A("## What this means for the project's claims")
    A("")
    if len(lost) == 0:
        A("**Every finding this project called significant survives Benjamini-Hochberg "
          "correction.** None of the originally-significant tests lost significance "
          "once corrected for the full set of comparisons run across both files. "
          "These findings are no longer just caveated as \"uncorrected\" -- they have "
          "actually been tested against the multiple-comparisons problem and held up.")
    else:
        A(f"**{len(lost)} previously-significant test(s) do NOT survive correction:**")
        A("")
        for _, r in lost.iterrows():
            A(f"- {r['row']} / {r['column']}: p={r['p_original']:.4f} -> "
              f"BH-adjusted p={r['p_bh']:.4f}")
        A("")
        A(f"The remaining **{len(survived)}** originally-significant tests do survive "
          "correction and can now be stated without the uncorrected-multiple-"
          "comparisons caveat.")
    A("")

    _truncate_existing_section(path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[appended] {SECTION_MARKER} -> {path}")


def main():
    print("=" * 78)
    print("BENJAMINI-HOCHBERG CORRECTION")
    print("No API calls. No new analysis -- correcting p-values already on disk.")
    print("=" * 78)
    intent_report, intent_m, intent_changed = correct_file(
        BY_INTENT_CSV, ["cit_boot_p", "len_boot_p"], "By-intent citation/length tests")
    print()
    tier_report, tier_m, tier_changed = correct_file(
        COST_TIER_CSV,
        ["mid_vs_cheap_boot_p", "expensive_vs_cheap_boot_p", "citation_boot_p",
         "length_boot_p", "expensive_minus_mid_boot_p"],
        "Cost-tier contrast tests")

    write_summary_note(intent_report, intent_m, intent_changed,
                      tier_report, tier_m, tier_changed)


if __name__ == "__main__":
    main()
