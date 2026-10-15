"""Multicollinearity diagnostics for Search-Arena-Model-Routing.

Place this file at scripts/multicollinearity_diagnostics.py and run it from the
repository root. Reuses the SAME design matrices as the Bradley-Terry scripts.

Typical usage:
    python scripts/multicollinearity_diagnostics.py
    python scripts/multicollinearity_diagnostics.py --frustration-labels frustration_full990.jsonl
    python scripts/multicollinearity_diagnostics.py --self-test

Requirements: numpy, pandas, and the repo's existing Python modules.

NO model refits, API calls, or modifications to the repo's analysis outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Approximate thresholds for *screening*, not formal hypothesis tests.
VIF_MODERATE = 5.0
VIF_HIGH = 10.0
CORR_HIGH = 0.8
SINGULAR_TOLERANCE = 1e-10


def _matrix_rank_condition(X: np.ndarray) -> tuple[int, float]:
    """Rank and condition number after scaling columns to unit Euclidean norm.

    Do not center the original BT design: its models contain NO intercept.
    Scaling avoids confounding ill-conditioning with citation/length units.
    """
    if X.shape[1] == 0:
        return 0, np.nan
    norms = np.linalg.norm(X, axis=0)
    Xn = X / np.where(norms > 0, norms, 1.0)
    s = np.linalg.svd(Xn, compute_uv=False)
    if len(s) == 0 or s[0] == 0:
        return 0, np.inf
    rank = int(np.sum(s > SINGULAR_TOLERANCE * s[0]))
    condition = float(s[0] / s[-1]) if len(s) >= Xn.shape[1] and s[-1] > SINGULAR_TOLERANCE * s[0] else np.inf
    return rank, condition


def diagnose_matrix(scope: str, X: np.ndarray, cols: list[str], *, has_intercept: bool = False):
    """Compute per-predictor VIFs and pairwise Pearson correlations.

    For no-intercept BT models, auxiliary regressions ALSO omit an intercept:
      VIF_j = 1 / (1 - ||projection of x_j onto X_-j||^2 / ||x_j||^2).
    For a design with an explicit intercept (frustration direct-effect fit),
    calculate conventional centered VIFs for NON-intercept predictors only.
    The full original design is always used for the rank check.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(cols):
        raise ValueError(f"{scope}: X and column names have incompatible shapes")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{scope}: non-finite design matrix; clean missing entries first")
    if len(X) == 0:
        raise ValueError(f"{scope}: no observations")

    n, p = X.shape
    rank, cond = _matrix_rank_condition(X)
    if has_intercept:
        keep = [j for j, c in enumerate(cols) if c != "intercept"]
        names = [cols[j] for j in keep]
        raw = X[:, keep]
        # Intercept in main model -> ordinary *centered* auxiliary regressions.
        Z = raw - raw.mean(axis=0, keepdims=True)
        method = "centered (intercept in model)"
    else:
        names = list(cols)
        Z = X
        method = "uncentered (no intercept in model)"

    vif_rows: list[dict] = []
    for j, name in enumerate(names):
        y = Z[:, j]
        ss = float(y @ y)
        if ss < 1e-20:
            vif, r2, status = np.inf, 1.0, "ZERO / CONSTANT COLUMN"
        elif Z.shape[1] == 1:
            vif, r2, status = 1.0, 0.0, "OK"
        else:
            others = np.delete(Z, j, axis=1)
            coefficients = np.linalg.lstsq(others, y, rcond=None)[0]
            residual = y - others @ coefficients
            residual_fraction = float(residual @ residual) / ss
            r2 = float(np.clip(1.0 - residual_fraction, 0.0, 1.0))
            if residual_fraction <= SINGULAR_TOLERANCE:
                vif, status = np.inf, "PERFECT / NEAR-PERFECT ALIAS"
            else:
                vif = 1.0 / residual_fraction
                status = ("HIGH (>=10)" if vif >= VIF_HIGH else
                          "MODERATE (>=5)" if vif >= VIF_MODERATE else "OK")
        vif_rows.append({
            "scope": scope,
            "predictor": name,
            "n_conversations": n,
            "n_design_columns": p,
            "design_rank": rank,
            "full_rank": rank == p,
            "scaled_condition_number": cond,
            "vif": vif,
            "auxiliary_r_squared": r2,
            "vif_method": method,
            "flag": status,
        })

    corr_rows: list[dict] = []
    for j in range(len(names)):
        for k in range(j + 1, len(names)):
            a = Z[:, j]
            b = Z[:, k]
            # Pearson correlations CENTER both columns even for a no-intercept
            # model. VIFs / matrix rank are more relevant for the BT fit itself.
            if np.std(a) == 0 or np.std(b) == 0:
                correlation = np.nan
            else:
                correlation = float(np.corrcoef(a, b)[0, 1])
            corr_rows.append({
                "scope": scope,
                "predictor_1": names[j],
                "predictor_2": names[k],
                "pearson_r": correlation,
                "high_abs_correlation": (bool(abs(correlation) >= CORR_HIGH)
                                         if np.isfinite(correlation) else False),
            })
    return vif_rows, corr_rows, {"scope": scope, "n": n, "p": p, "rank": rank,
                                 "condition_number": cond, "full_rank": rank == p}


def _test_diagnostics():
    rng = np.random.default_rng(1)
    n = 400
    x1 = rng.normal(size=n)
    x2 = 1.5 * x1  # exact alias
    x3 = rng.normal(size=n)
    rows, correlations, meta = diagnose_matrix(
        "self_test", np.column_stack((x1, x2, x3)), ["x1", "x2", "x3"])
    assert meta["rank"] == 2 and not meta["full_rank"]
    assert np.isinf(rows[0]["vif"]) and np.isinf(rows[1]["vif"])
    assert abs(correlations[0]["pearson_r"] - 1.0) < 1e-12
    print("PASS: exact alias detected (rank 2/3; VIF(x1) = VIF(x2) = infinity)")
    x2 = 1.5 * x1 + rng.normal(size=n, scale=0.6)
    rows, _, meta = diagnose_matrix(
        "self_test_noisy", np.column_stack((x1, x2, x3)), ["x1", "x2", "x3"])
    assert meta["full_rank"] and np.isfinite(rows[0]["vif"])
    print("PASS: full-rank design has finite VIFs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/multicollinearity",
                        help="Directory for results (default: outputs/multicollinearity)")
    parser.add_argument("--skip-intents", action="store_true",
                        help="Only run the pooled model specifications")
    parser.add_argument("--frustration-labels", default=None,
                        help="Optionally check the frustrated/engaged models and the direct-effect fit")
    parser.add_argument("--self-test", action="store_true",
                        help="Test numerical diagnostics without repo data")
    args = parser.parse_args()

    if args.self_test:
        _test_diagnostics()
        return

    # Safety: don't trigger an unexpected 1.2 GB JSONL rebuild if the cached
    # conversation-level analysis table is missing.
    if not Path("outputs/final_analysis_derived.csv").exists():
        parser.error("outputs/final_analysis_derived.csv is missing. "
                     "Run 'python scripts/final_analysis.py' first from the repo root.")

    # Reuse exactly the same functions and scaling as existing regressions.
    from bradley_terry_analysis import (  # pylint: disable=import-outside-toplevel
        INTENT_ORDER, MIN_N_FOR_FIT, _scale_covariates,
        build_design_model_identity, build_design_cost_tier, load_decisive,
    )
    from bradley_terry_family import (  # pylint: disable=import-outside-toplevel
        classify_model_family, build_design_family, build_design_tier_and_family,
    )

    _, dec, _, source = load_decisive()
    sd_cit, sd_len = _scale_covariates(dec)
    print(f"Loaded {len(dec):,} decisive battles from {source}")
    print(f"BT covariate scales: citation={sd_cit:.3f}, length={sd_len:.3f}")
    all_vifs, all_correlations, all_specs = [], [], []

    def add(scope, subset, builder, **kwargs):
        if len(subset) < 2:
            print(f"[skip] {scope}: fewer than two observations")
            return
        X, _y, cols, *_extras = builder(subset, sd_cit=sd_cit, sd_len=sd_len, **kwargs)
        has_intercept = "intercept" in cols
        vif_rows, corr_rows, spec = diagnose_matrix(scope, X, list(cols), has_intercept=has_intercept)
        all_vifs.extend(vif_rows)
        all_correlations.extend(corr_rows)
        all_specs.append(spec)
        highest = max((r["vif"] for r in vif_rows), default=np.nan)
        print(f"{scope:36} n={spec['n']:5}  rank={spec['rank']}/{spec['p']}  "
              f"max VIF={highest:.2f}  condition={spec['condition_number']:.2g}")

    def models_present(df):
        return sorted(set(df["base_model_a"]) | set(df["base_model_b"]))

    add("pooled_model_identity", dec, build_design_model_identity, models=models_present(dec))
    add("pooled_cost_tier", dec, build_design_cost_tier)

    family_dec = dec.copy()
    family_dec["family_a"] = family_dec["base_model_a"].map(classify_model_family)
    family_dec["family_b"] = family_dec["base_model_b"].map(classify_model_family)
    unknown = family_dec["family_a"].isna() | family_dec["family_b"].isna()
    if unknown.any():
        print(f"[warn] family models: {unknown.sum()} rows with unknown family excluded")
        family_dec = family_dec.loc[~unknown].copy()
    if len(family_dec):
        add("pooled_model_family", family_dec, build_design_family)
        add("pooled_tier_plus_family", family_dec, build_design_tier_and_family)

    if not args.skip_intents:
        for intent in INTENT_ORDER:
            sub = dec.loc[dec["primary_intent"] == intent]
            if len(sub) < MIN_N_FOR_FIT:
                print(f"[skip] intent_{intent}: n={len(sub)} < {MIN_N_FOR_FIT}")
                continue
            add("intent_" + str(intent), sub, build_design_model_identity,
                models=models_present(sub))

    if args.frustration_labels:
        from bradley_terry_frustration import (  # pylint: disable=import-outside-toplevel
            load_frustration_labels, merge_with_decisive, build_design_frustration_direct,
        )
        label_df = load_frustration_labels(args.frustration_labels)
        matched, unmatched = merge_with_decisive(dec, label_df)
        print(f"Frustration labels: {len(matched)} matched, {len(unmatched)} unmatched")
        two = matched.loc[matched["frustration_label"].isin(["frustrated", "engaged"])].copy()
        if len(two) >= MIN_N_FOR_FIT:
            add("frustration_direct_effect", two, build_design_frustration_direct)
        for label in ("frustrated", "engaged"):
            sub = two.loc[two["frustration_label"] == label]
            if len(sub) >= MIN_N_FOR_FIT:
                add("frustration_group_" + label, sub, build_design_model_identity,
                    models=models_present(sub))
            else:
                print(f"[skip] frustration_group_{label}: n={len(sub)} < {MIN_N_FOR_FIT}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vif = pd.DataFrame(all_vifs)
    correlations = pd.DataFrame(all_correlations)
    specs = pd.DataFrame(all_specs)
    vif.to_csv(out / "vif_by_model.csv", index=False)
    correlations.to_csv(out / "pairwise_correlations.csv", index=False)
    specs.to_csv(out / "design_rank_and_condition.csv", index=False)

    top = vif.loc[vif["vif"] >= VIF_MODERATE].sort_values("vif", ascending=False)
    strong = correlations.loc[correlations["high_abs_correlation"]].copy()
    strong["absolute_r"] = strong["pearson_r"].abs()
    strong = strong.sort_values("absolute_r", ascending=False)
    lines = [
        "# Search Arena multicollinearity diagnostics", "",
        f"Source: `{source}`. Decisive battles only. Same design matrices and "
        "global citation/length scaling as the repository's Bradley–Terry fits.", "",
        "This is a DIAGNOSTIC, not a model fit. No coefficients or results were overwritten.", "",
        "## Design matrix rank and condition number", "",
        "Columns are standardized by Euclidean norm for condition-number reporting. "
        "An infinite condition number signals rank deficiency or near-dependency "
        "at the diagnostic tolerance.", "",
        "| Specification | N | Rank / columns | Condition number |", "|---|---:|---:|---:|",
    ]
    for _, s in specs.iterrows():
        c = s["condition_number"]
        ctxt = "infinite" if not np.isfinite(c) else f"{c:.2f}"
        lines.append(f"| {s['scope']} | {s['n']:,} | {s['rank']} / {s['p']} | {ctxt} |")
    lines += ["", "## Predictors flagged by VIF", "",
              "VIF >= 5 is a screening flag; VIF >= 10 is a stronger warning, "
              "not an automatic reason to remove a variable. Infinite VIF means "
              "(near-)perfect linear dependence. Most BT specifications have NO "
              "intercept, so their VIFs use uncentered auxiliary regressions.", "",
              "| Specification | Predictor | VIF | Flag |", "|---|---|---:|---|"]
    if len(top) == 0:
        lines.append("| *No VIF values >= 5* | — | — | — |")
    else:
        for _, r in top.iterrows():
            lines.append(f"| {r['scope']} | `{r['predictor']}` | "
                         f"{'infinite' if not np.isfinite(r['vif']) else format(r['vif'], '.2f')} | "
                         f"{r['flag']} |")
    lines += ["", "## High pairwise Pearson correlations", "",
              "| Specification | Predictor 1 | Predictor 2 | r |", "|---|---|---|---:|"]
    if len(strong) == 0:
        lines.append("| *No absolute correlations >= 0.80* | — | — | — |")
    else:
        for _, r in strong.iterrows():
            lines.append(f"| {r['scope']} | `{r['predictor_1']}` | "
                         f"`{r['predictor_2']}` | {r['pearson_r']:+.3f} |")
    lines += [
        "", "## Interpretation notes", "",
        "- **Check the VIF for `beta_citation` and `beta_length` first.** "
        "These measure how redundant the two main behavioral predictors are "
        "*given the model identity/tier/family controls in that specific fit*.",
        "- **Full matrix rank is necessary, not sufficient for stable estimation.** "
        "Large finite VIFs and condition numbers can still mean imprecise coefficients.",
        "- **High correlation != causation and != guaranteed multicollinearity.** "
        "Low pairwise correlations can coexist with multivariable linear dependencies.",
        "- **Do not put cost tier and full model-identity indicators together.** "
        "Cost tier is deterministically derived from identity, giving an exact alias.",
        "- **Do not drop citation or length controls automatically.** If high VIFs "
        "occur, report them, compare sensitivity fits, and inspect bootstrap CIs.",
        "- **Small subgroup fits need additional caution.** Low sample size, "
        "rare model matchups, and quasi-separation can produce wide intervals "
        "even if VIFs are low.",
    ]
    report = out / "diagnostic_summary.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSaved:")
    for name in ("vif_by_model.csv", "pairwise_correlations.csv", "design_rank_and_condition.csv", "diagnostic_summary.md"):
        print(" -", out / name)
    if len(top):
        print(f"\nFLAG: {len(top)} predictors across fits have VIF >= {VIF_MODERATE}")
    else:
        print("\nNo predictors with VIF >= 5 in the tested designs.")


if __name__ == "__main__":
    main()
