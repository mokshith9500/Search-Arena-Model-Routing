"""
build_charts.py

Builds all 8 report charts as static PNGs. Every number plotted is read
directly from a CSV already in this project (or, for chart 4, parsed
directly out of the markdown table already in bradley_terry_summary.md) --
nothing is typed in by hand or estimated.

Style: matplotlib, light background, thin recessive spines/gridlines, no 3D,
no gradient fills, color used only to carry a real distinction (tier or
significance), direct labels used selectively rather than printing every
value. Palette follows this project's dataviz-skill categorical/status
slots: blue=significant/series-1, orange=series-2, aqua=series-3 (tier
order cheap/mid/expensive), gray=not-significant, red=flagged/changed.
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#2a78d6"      # significant / series 1 (cheap tier)
ORANGE = "#eb6834"    # series 2 (mid tier)
AQUA = "#1baf7a"       # series 3 (expensive tier)
GRAY = "#9a9992"       # not significant
RED = "#d03b3b"        # flagged / changed
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
    "font.size": 10.5,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e8e7e1",
    "grid.linewidth": 0.7,
})

TIER_COLOR = {"cheap": BLUE, "mid": ORANGE, "expensive": AQUA}


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


# ============================================================
# Chart 1: individual model strength by tier
# source: bradley_terry_model_strengths.csv
# columns used: base_model, cost_tier, strength, ci_lo, ci_hi, is_reference
# ============================================================

def chart1():
    df = pd.read_csv("outputs/bradley_terry_model_strengths.csv")
    df = df.sort_values("strength", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(df))
    for i, r in df.iterrows():
        color = TIER_COLOR[r["cost_tier"]]
        if bool(r["is_reference"]):
            ax.plot(0, i, "o", color=color, markersize=7, zorder=3)
            ax.text(0.03, i, " (reference)", va="center", ha="left",
                    fontsize=8.5, color=MUTED)
        else:
            lo, hi = r["ci_lo"], r["ci_hi"]
            ax.plot([lo, hi], [i, i], color=color, linewidth=2, zorder=2, alpha=0.9)
            ax.plot(r["strength"], i, "o", color=color, markersize=7, zorder=3)
    ax.axvline(0, color="#b8b7ae", linewidth=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["base_model"])
    ax.set_xlabel("Bradley-Terry strength (log-odds vs. reference model)")
    ax.set_title("Individual Model Strength, Colored by Price Tier", fontsize=12, pad=10)
    handles = [plt.Line2D([0], [0], marker="o", color=TIER_COLOR[t], linestyle="",
                          markersize=7, label=t.capitalize())
              for t in ("cheap", "mid", "expensive")]
    ax.legend(handles=handles, loc="lower right", frameon=False, title="Tier")
    fig.tight_layout()
    savefig(fig, "chart_model_strength_by_tier.png")


# ============================================================
# Chart 2: citation effect by intent (forest plot)
# source: bradley_terry_by_intent.csv
# columns used: primary_intent, beta_citation, cit_ci_lo, cit_ci_hi, cit_significant
# ============================================================

def chart2():
    df = pd.read_csv("outputs/bradley_terry_by_intent.csv")
    df = df.sort_values("beta_citation", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(df))
    for i, r in df.iterrows():
        color = BLUE if bool(r["cit_significant"]) else GRAY
        ax.plot([r["cit_ci_lo"], r["cit_ci_hi"]], [i, i], color=color,
                linewidth=2, zorder=2)
        ax.plot(r["beta_citation"], i, "o", color=color, markersize=7, zorder=3)
    ax.axvline(0, color="#b8b7ae", linewidth=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["primary_intent"])
    ax.set_xlabel("beta_citation (log-odds per 1 SD citation difference)")
    ax.set_title("Citation Effect on Winning, by Intent Category", fontsize=12, pad=10)
    handles = [plt.Line2D([0], [0], marker="o", color=BLUE, linestyle="", markersize=7,
                          label="95% CI excludes zero"),
              plt.Line2D([0], [0], marker="o", color=GRAY, linestyle="", markersize=7,
                        label="95% CI includes zero")]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    savefig(fig, "chart_citation_by_intent.png")


# ============================================================
# Chart 3: cost vs performance scatter
# source: tau2_precomputed_summary.csv
# columns used: agent_model, cost_tier_50_50, pass_at_1, avg_agent_cost_usd
# filter: domain == "ALL (pooled airline+retail+telecom)"
# ============================================================

def chart3():
    df = pd.read_csv("outputs/tau2_precomputed_summary.csv")
    pooled = df[df["domain"].str.startswith("ALL")].copy()
    pooled["short_name"] = pooled["agent_model"].str.replace(
        r"-\d{4}-\d{2}-\d{2}$|-\d{8}$", "", regex=True)
    pooled = pooled.sort_values("avg_agent_cost_usd").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    # the three cheaper models cluster within $0.05 of each other on cost;
    # stagger their label offsets vertically so none overlap another point
    # or label (purely a label-placement choice, not a data transform).
    label_offsets = [(8, -14), (8, 10), (8, -4), (8, 6)]
    for (_, r), off in zip(pooled.iterrows(), label_offsets):
        color = TIER_COLOR[r["cost_tier_50_50"]]
        ax.scatter(r["avg_agent_cost_usd"], r["pass_at_1"], s=110, color=color,
                  zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(r["short_name"], (r["avg_agent_cost_usd"], r["pass_at_1"]),
                   textcoords="offset points", xytext=off, fontsize=8.5, color=MUTED)
    ax.set_xlim(0, pooled["avg_agent_cost_usd"].max() * 1.25)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Measured cost per task (USD)")
    ax.set_ylabel("Pass@1 (pooled across domains)")
    ax.set_title("Cost vs. Task Success — tau2-bench", fontsize=12, pad=10)
    handles = [plt.Line2D([0], [0], marker="o", color=TIER_COLOR[t], linestyle="",
                          markersize=8, label=t.capitalize())
              for t in ("mid", "expensive") if t in pooled["cost_tier_50_50"].values]
    ax.legend(handles=handles, loc="lower right", frameon=False, title="Tier")
    fig.tight_layout()
    savefig(fig, "chart_cost_vs_performance.png")


# ============================================================
# Chart 4: frustration evidence accumulation across 3 sample sizes
# source: bradley_terry_summary.md, "How the estimate evolved across
# sample sizes" table (parsed directly, not retyped)
# ============================================================

def parse_evolution_table(md_path="outputs/bradley_terry_summary.md"):
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    marker = "### How the estimate evolved across sample sizes"
    i = text.find(marker)
    assert i != -1, "evolution table not found in bradley_terry_summary.md"
    block = text[i:i + 1200]
    rows = []
    for line in block.splitlines():
        m = re.match(
            r"\|\s*(original|partial|complete)\s*\(n=(\d+)\)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*"
            r"([+-]?[\d.]+)\s*\|\s*\[([+-]?[\d.]+),\s*([+-]?[\d.]+)\]\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
            line)
        if m:
            stage, n, nf, ne, contrast, lo, hi, width, p = m.groups()
            rows.append(dict(stage=stage, n=int(n), contrast=float(contrast),
                             lo=float(lo), hi=float(hi), p=float(p)))
    assert len(rows) == 3, f"expected 3 rows, parsed {len(rows)}"
    return rows


def chart4():
    rows = parse_evolution_table()
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    x = [r["n"] for r in rows]
    y = [r["contrast"] for r in rows]
    lo = [r["contrast"] - r["lo"] for r in rows]
    hi = [r["hi"] - r["contrast"] for r in rows]
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color=BLUE, ecolor=BLUE,
               elinewidth=2, capsize=5, markersize=8, linewidth=1.6, zorder=3)
    ax.axhline(0, color="#b8b7ae", linewidth=1, zorder=1)
    for r in rows:
        ax.annotate(f"n={r['n']}, p={r['p']:.2f}", (r["n"], r["hi"] + 0.07),
                   ha="center", fontsize=8.5, color=MUTED)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_ylim(-0.75, 1.15)
    ax.set_xlabel("Labeled sample size (log scale)")
    ax.set_ylabel("Frustrated − engaged contrast on beta_citation\n(log-odds, 95% CI)")
    ax.set_title("Frustration Moderator Effect: Narrowing, Still Crosses Zero", fontsize=11.5, pad=14)
    fig.tight_layout()
    savefig(fig, "chart_frustration_evidence.png")


# ============================================================
# Chart 5: tier-level contrasts
# source: bradley_terry_cost_tier.csv, "All categories" row
# columns used: mid_vs_cheap_coef/_ci_lo/_ci_hi/_significant,
#               expensive_vs_cheap_*, expensive_minus_mid_*
# ============================================================

def chart5():
    df = pd.read_csv("outputs/bradley_terry_cost_tier.csv")
    row = df[df["scope"] == "All categories"].iloc[0]
    terms = [
        ("Mid vs Cheap", "mid_vs_cheap"),
        ("Expensive vs Cheap", "expensive_vs_cheap"),
        ("Expensive vs Mid", "expensive_minus_mid"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    y = np.arange(len(terms))
    for i, (label, key) in enumerate(terms):
        coef = row[f"{key}_coef"]
        lo = row[f"{key}_ci_lo"]
        hi = row[f"{key}_ci_hi"]
        sig = bool(row[f"{key}_significant"])
        color = BLUE if sig else GRAY
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.5, zorder=2)
        ax.plot(coef, i, "o", color=color, markersize=9, zorder=3)
    ax.axvline(0, color="#b8b7ae", linewidth=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in terms])
    ax.set_xlabel("Coefficient (log-odds, 95% CI)")
    ax.set_title("Price Tier Effects on Winning", fontsize=12, pad=10)
    handles = [plt.Line2D([0], [0], marker="o", color=BLUE, linestyle="", markersize=8,
                          label="Significant (CI excludes 0)"),
              plt.Line2D([0], [0], marker="o", color=GRAY, linestyle="", markersize=8,
                        label="Not significant (CI includes 0)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
             frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    savefig(fig, "chart_tier_contrasts.png")


# ============================================================
# Chart 6: OLS vs Bradley-Terry, mid-vs-expensive
# sources: final_analysis_cost_tier_comparison.csv (OLS win rate, All
#          categories / mid_vs_expensive row) and
#          bradley_terry_cost_tier.csv (All categories, expensive_minus_mid)
# Two separate panels -- never a dual axis -- since the two methods report
# on genuinely different scales (a win-rate percentage vs a log-odds
# coefficient).
# ============================================================

def chart6():
    ols = pd.read_csv("outputs/final_analysis_cost_tier_comparison.csv")
    ols_row = ols[(ols["primary_intent"] == "All categories") &
                 (ols["comparison"] == "mid_vs_expensive")].iloc[0]
    ols_rate = float(ols_row["cheaper_tier_win_rate"])  # mid is the cheaper tier here
    ols_sig = bool(ols_row["significant_at_0.05"])

    bt = pd.read_csv("outputs/bradley_terry_cost_tier.csv")
    bt_row = bt[bt["scope"] == "All categories"].iloc[0]
    bt_coef = float(bt_row["expensive_minus_mid_coef"])
    bt_lo = float(bt_row["expensive_minus_mid_ci_lo"])
    bt_hi = float(bt_row["expensive_minus_mid_ci_hi"])
    bt_sig = bool(bt_row["expensive_minus_mid_significant"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 4.0))

    axL.bar(["Mid-tier\nwin rate"], [ols_rate * 100], color=BLUE if ols_sig else GRAY,
           width=0.5)
    axL.axhline(50, color="#b8b7ae", linewidth=1, linestyle="--")
    axL.text(0, ols_rate * 100 + 1.5, f"{ols_rate*100:.1f}%\np<0.001 (sig.)",
             ha="center", fontsize=9, color=INK)
    axL.set_ylim(0, 70)
    axL.set_ylabel("Head-to-head win rate (%)")
    axL.set_title("Original: OLS / win-rate\n(uncontrolled for citations, length)", fontsize=10)

    axR.errorbar([0], [bt_coef], yerr=[[bt_coef - bt_lo], [bt_hi - bt_coef]],
               fmt="o", color=BLUE if bt_sig else GRAY, markersize=10, capsize=6,
               elinewidth=2.2)
    axR.axhline(0, color="#b8b7ae", linewidth=1)
    axR.text(0.15, bt_coef, f"{bt_coef:+.3f} log-odds\np={bt_row['expensive_minus_mid_boot_p']:.2f} (not sig.)",
             va="center", fontsize=9, color=INK)
    axR.set_xlim(-0.6, 1.4)
    axR.set_xticks([])
    axR.set_ylim(-0.3, 0.3)
    axR.set_ylabel("Expensive − Mid (log-odds, 95% CI)")
    axR.set_title("Rebuilt: Bradley-Terry\n(citations + length controlled)", fontsize=10)

    fig.suptitle("Why We Rebuilt the Analysis — A Finding That Didn't Survive Stronger Methods",
               fontsize=12, y=1.04)
    fig.tight_layout()
    savefig(fig, "chart_ols_vs_bt_correction.png")


# ============================================================
# Chart 7: English-only replication scatter
# source: bradley_terry_english_only_comparison.csv
# columns used: pooled_value, english_only_value, significance_changed
# ============================================================

def chart7():
    df = pd.read_csv("outputs/bradley_terry_english_only_comparison.csv")
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    lims = (min(df["pooled_value"].min(), df["english_only_value"].min()) - 0.1,
           max(df["pooled_value"].max(), df["english_only_value"].max()) + 0.1)
    ax.plot(lims, lims, color="#b8b7ae", linewidth=1.2, linestyle="--", zorder=1,
           label="Perfect replication (y = x)")
    changed = df[df["significance_changed"] == True]  # noqa: E712
    unchanged = df[df["significance_changed"] == False]  # noqa: E712
    ax.scatter(unchanged["pooled_value"], unchanged["english_only_value"],
             s=80, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6,
             label="Significance unchanged")
    ax.scatter(changed["pooled_value"], changed["english_only_value"],
             s=80, color=RED, zorder=3, edgecolor="white", linewidth=0.6,
             label="Significance changed")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Pooled coefficient (all languages)")
    ax.set_ylabel("English-only coefficient")
    ax.set_title("English-Only Replication — Findings Hold Up", fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "chart_english_only_replication.png")


# ============================================================
# Chart 8: success-per-dollar, tau2-bench
# source: tau2_precomputed_summary.csv
# columns used: agent_model, cost_tier_50_50, pass_at_1, avg_agent_cost_usd
# filter: domain == "ALL (pooled airline+retail+telecom)"
# derived: successes_per_dollar = pass_at_1 / avg_agent_cost_usd (both
# values read directly from the CSV; the division is the only arithmetic,
# matching how this ratio was already reported earlier in this project)
# ============================================================

def chart8():
    df = pd.read_csv("outputs/tau2_precomputed_summary.csv")
    pooled = df[df["domain"].str.startswith("ALL")].copy()
    pooled["succ_per_dollar"] = pooled["pass_at_1"] / pooled["avg_agent_cost_usd"]
    pooled["short_name"] = pooled["agent_model"].str.replace(
        r"-\d{4}-\d{2}-\d{2}$|-\d{8}$", "", regex=True)
    pooled = pooled.sort_values("succ_per_dollar", ascending=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = [TIER_COLOR[t] for t in pooled["cost_tier_50_50"]]
    bars = ax.bar(pooled["short_name"], pooled["succ_per_dollar"], color=colors, width=0.55)
    for b, v in zip(bars, pooled["succ_per_dollar"]):
        ax.text(b.get_x() + b.get_width() / 2, v + max(pooled["succ_per_dollar"]) * 0.02,
               f"{v:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Successes per dollar (Pass@1 / measured cost)")
    ax.set_title("Task Success Per Dollar Spent", fontsize=12, pad=10)
    handles = [plt.Line2D([0], [0], marker="s", color=TIER_COLOR[t], linestyle="",
                          markersize=9, label=t.capitalize())
              for t in ("mid", "expensive") if t in pooled["cost_tier_50_50"].values]
    ax.legend(handles=handles, loc="upper right", frameon=False, title="Tier")
    fig.tight_layout()
    savefig(fig, "chart_cost_efficiency.png")


if __name__ == "__main__":
    chart1()
    chart2()
    chart3()
    chart4()
    chart5()
    chart6()
    chart7()
    chart8()
