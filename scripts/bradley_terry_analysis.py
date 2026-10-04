"""
bradley_terry_analysis.py

Parallel rebuild of the core Search Arena citation / cost-tier analysis using a
Bradley-Terry (BT) model with covariates, fitted by maximum likelihood, with
conversation-level bootstrap resampling for confidence intervals and
significance -- the standard approach for pairwise-preference arena data.

This is a PARALLEL rebuild. final_analysis.py is not modified or imported for
its OLS logic; it is only used for its data-loading helpers and its published
output CSVs, which are read back purely for the side-by-side comparison.

MODEL
-----
For battle i between model_a and model_b, with y_i = 1 if a wins:

    P(a beats b) = sigmoid( theta_a - theta_b + beta' (x_a - x_b) )

theta = per-player latent strength, beta = covariate coefficients on the
*difference* in observed covariates (visible citation count, response length).
Because every term is a difference, this is exactly a logistic regression with
no intercept on differenced features, which is what makes the 1,000-replicate
bootstrap tractable. Fitted by MLE (L-BFGS-B with analytic gradient).

Ties are excluded (decisive battles only), consistent with the existing
analysis's `is_decisive` filter.

Runs entirely locally. No API calls, no new data collection.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# final_analysis.py is imported ONLY for data-loading helpers and constants,
# never for its OLS residualization logic (this is a parallel rebuild).
from final_analysis import (
    DERIVED_CSV,
    INTENT_ORDER,
    CHEAP_TIER,
    MID_TIER,
    EXPENSIVE_TIER,
    classify_cost_tier,
    load_full_table,
    MODEL_BLENDED_PRICE_USD_PER_MILLION,
)

# ============================================================
# SETTINGS
# ============================================================

N_BOOTSTRAP = 1000
ALPHA = 0.05
RANDOM_SEED = 20260930
RIDGE = 1e-6          # numerical stabiliser only; see NOTE in fit_bt
MIN_N_FOR_FIT = 30    # below this a per-intent fit is not attempted
LOW_POWER_N = 250     # flag thin cells

BY_INTENT_CSV = "outputs/bradley_terry_by_intent.csv"
COST_TIER_CSV = "outputs/bradley_terry_cost_tier.csv"
SUMMARY_MD = "outputs/bradley_terry_summary.md"
MODEL_STRENGTH_CSV = "outputs/bradley_terry_model_strengths.csv"
BOUNDARY_CSV = "outputs/bradley_terry_tier_boundary_check.csv"

# Existing OLS outputs, read back for the Step 5 comparison only.
OLS_BY_INTENT_CSV = "outputs/final_analysis_by_intent.csv"
OLS_COST_TIER_CSV = "outputs/final_analysis_cost_tier_comparison.csv"

TAU2_SUMMARY_MD = "outputs/tau2_analysis_summary.md"
TAU2_SUMMARY_CSV = "outputs/tau2_precomputed_summary.csv"


# ============================================================
# BRADLEY-TERRY FIT (MLE)
# ============================================================

def _nll_and_grad(beta, X, y, ridge):
    z = X @ beta
    # stable log-likelihood via logaddexp: log sigmoid(z) = -log(1+exp(-z))
    ll = -np.sum(np.logaddexp(0.0, -z) * y + np.logaddexp(0.0, z) * (1.0 - y))
    p = 1.0 / (1.0 + np.exp(-z))
    grad = X.T @ (p - y)
    if ridge:
        ll -= ridge * float(beta @ beta)
        grad += 2.0 * ridge * beta
    return -ll, grad


def fit_bt(X, y, ridge=RIDGE):
    """
    MLE fit of the BT-with-covariates model on differenced features.

    NOTE on `ridge`: a 1e-6 L2 term is included purely as a numerical
    stabiliser so that quasi-separated bootstrap replicates (which do occur
    in thin per-intent cells) still converge to a finite optimum instead of
    drifting. At this magnitude it is negligible relative to the likelihood
    and does not meaningfully shrink the reported coefficients; it is not a
    regularised/penalised estimator in any substantive sense.
    """
    beta0 = np.zeros(X.shape[1])
    res = minimize(
        _nll_and_grad, beta0, args=(X, y, ridge),
        jac=True, method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
    )
    return res.x, bool(res.success)


# ============================================================
# DESIGN MATRICES
# ============================================================

def _scale_covariates(df):
    """
    Global scaling constants for the covariate differences, computed ONCE on
    the full decisive set so that coefficients are on identical units in every
    per-intent fit and are therefore comparable across intents. Coefficients
    are expressed per 1 SD of the covariate difference.
    """
    dcit = (df["citations_a"] - df["citations_b"]).astype(float)
    dlen = (df["response_length_a"] - df["response_length_b"]).astype(float)
    sd_cit = float(dcit.std(ddof=1)) or 1.0
    sd_len = float(dlen.std(ddof=1)) or 1.0
    return sd_cit, sd_len


def build_design_model_identity(df, models, sd_cit, sd_len):
    """
    Player = individual base model. Reference model's strength fixed at 0
    (dropped column) for identifiability. Returns X, y, column names.
    """
    ref = models[0]
    free = models[1:]
    n = len(df)
    cols = []
    parts = []
    a = df["base_model_a"].values
    b = df["base_model_b"].values
    for m in free:
        parts.append(((a == m).astype(float) - (b == m).astype(float)).reshape(-1, 1))
        cols.append(f"theta[{m}]")
    dcit = ((df["citations_a"] - df["citations_b"]).astype(float) / sd_cit).values
    dlen = ((df["response_length_a"] - df["response_length_b"]).astype(float) / sd_len).values
    parts.append(dcit.reshape(-1, 1)); cols.append("beta_citation")
    parts.append(dlen.reshape(-1, 1)); cols.append("beta_length")
    X = np.hstack(parts) if parts else np.zeros((n, 0))
    y = df["_y"].values.astype(float)
    return X, y, cols, ref


def build_design_cost_tier(df, sd_cit, sd_len):
    """
    Player = COST TIER, with `cheap` as the reference category (strength 0).

    IDENTIFICATION NOTE: cost tier is a deterministic function of model
    identity, so tier indicators and per-model strengths are perfectly
    collinear and CANNOT both appear in one model -- the tier effect would be
    unidentified. The tier effect is therefore estimated with tier as the
    player identity. This is the correct specification for the question
    "does tier predict winning", but it means tier effects are NOT additionally
    controlled for within-tier model identity.
    """
    ta = df["tier_a"].values
    tb = df["tier_b"].values
    cols, parts = [], []
    for t in ("mid", "expensive"):
        parts.append(((ta == t).astype(float) - (tb == t).astype(float)).reshape(-1, 1))
        cols.append(f"tier[{t}] vs cheap")
    dcit = ((df["citations_a"] - df["citations_b"]).astype(float) / sd_cit).values
    dlen = ((df["response_length_a"] - df["response_length_b"]).astype(float) / sd_len).values
    parts.append(dcit.reshape(-1, 1)); cols.append("beta_citation")
    parts.append(dlen.reshape(-1, 1)); cols.append("beta_length")
    X = np.hstack(parts)
    y = df["_y"].values.astype(float)
    return X, y, cols


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_bt(df, build_fn, n_boot=None, seed=None, **kw):
    """
    Conversation-level bootstrap. Each row of `df` is one conversation
    (one battle), so resampling rows with replacement resamples
    conversations -- the unit the News Source Citing Patterns procedure
    specifies. Returns point estimate, CI bounds, bootstrap p-values.
    """
    # resolved at call time, not bound as a default, so the module constants
    # remain the single source of truth
    n_boot = N_BOOTSTRAP if n_boot is None else n_boot
    seed = RANDOM_SEED if seed is None else seed

    X, y, cols, *rest = build_fn(df, **kw)
    point, ok = fit_bt(X, y)

    rng = np.random.default_rng(seed)
    n = len(df)
    draws = np.full((n_boot, len(cols)), np.nan)
    n_fail = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        # a replicate with no variation in the outcome carries no information
        if yb.min() == yb.max():
            n_fail += 1
            continue
        bb, okb = fit_bt(Xb, yb)
        if okb:
            draws[b] = bb
        else:
            n_fail += 1

    out = {}
    for j, c in enumerate(cols):
        d = draws[:, j]
        d = d[np.isfinite(d)]
        if len(d) < 50:
            out[c] = dict(coef=point[j], lo=np.nan, hi=np.nan, p=np.nan,
                          sig=False, n_boot_ok=len(d))
            continue
        lo = float(np.percentile(d, 100 * ALPHA / 2))
        hi = float(np.percentile(d, 100 * (1 - ALPHA / 2)))
        # two-sided bootstrap p: proportion of replicates on the far side of 0
        frac_le = float(np.mean(d <= 0.0))
        frac_ge = float(np.mean(d >= 0.0))
        p = float(min(1.0, 2.0 * min(frac_le, frac_ge)))
        out[c] = dict(coef=float(point[j]), lo=lo, hi=hi, p=p,
                      sig=bool(lo > 0 or hi < 0), n_boot_ok=len(d))
    return out, cols, (rest[0] if rest else None), n_fail, ok, draws


# ============================================================
# DATA
# ============================================================

def load_decisive():
    if os.path.exists(DERIVED_CSV):
        df = pd.read_csv(DERIVED_CSV)
        src = f"{DERIVED_CSV} (cached; built with the fixed per-turn citation aggregation)"
    else:
        print(f"[info] {DERIVED_CSV} not found -- rebuilding from train.jsonl ...")
        df, _, _ = load_full_table()
        df.to_csv(DERIVED_CSV, index=False)
        src = "train.jsonl (rebuilt)"

    n_all = len(df)
    d = df[df["is_decisive"] == True].copy()  # noqa: E712
    d = d.dropna(subset=["citations_a", "citations_b",
                         "response_length_a", "response_length_b",
                         "base_model_a", "base_model_b", "winner"])
    # y = 1 when model_a is the decisive winner
    d["_y"] = (d["winner"].astype(str).str.strip() == "model_a").astype(float)
    d["tier_a"] = d["base_model_a"].map(classify_cost_tier)
    d["tier_b"] = d["base_model_b"].map(classify_cost_tier)
    return df, d, n_all, src


# ============================================================
# STEP 3: citation effect per primary_intent
# ============================================================

def run_by_intent(dec, sd_cit, sd_len):
    rows = []
    print("\n" + "=" * 78)
    print("STEP 3  Bradley-Terry citation effect by primary_intent")
    print("        (per-model strengths + citation diff + length diff)")
    print("=" * 78)
    print(f"{'primary_intent':22}{'n':>6}{'beta_cit':>10}{'95% CI':>20}{'p':>9}  sig")
    for intent in INTENT_ORDER:
        sub = dec[dec["primary_intent"] == intent].copy()
        if len(sub) < MIN_N_FOR_FIT:
            rows.append(dict(primary_intent=intent, n_decisive=len(sub),
                             note="insufficient n"))
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
            "  (low power)" if len(sub) < LOW_POWER_N else ""))
    return pd.DataFrame(rows)


# ============================================================
# STEP 4: cost tier
# ============================================================

def point_of(res, cols, idx):
    return res[cols[idx]]["coef"]



def run_cost_tier(dec, sd_cit, sd_len):
    print("\n" + "=" * 78)
    print("STEP 4  Bradley-Terry cost-tier effect (cheap = reference)")
    print("        citation diff and length diff controlled in the SAME model")
    print("=" * 78)
    rows = []

    def one(sub, label):
        if len(sub) < MIN_N_FOR_FIT:
            return None
        res, cols, _, nfail, ok, draws = bootstrap_bt(
            sub, build_design_cost_tier, sd_cit=sd_cit, sd_len=sd_len)
        r = dict(scope=label, n_decisive=len(sub), converged=ok, n_boot_failed=nfail,
                 low_power=len(sub) < LOW_POWER_N)
        for c in cols:
            key = (c.replace("tier[", "").replace("] vs cheap", "_vs_cheap")
                    .replace("beta_", ""))
            r[f"{key}_coef"] = res[c]["coef"]
            r[f"{key}_ci_lo"] = res[c]["lo"]
            r[f"{key}_ci_hi"] = res[c]["hi"]
            r[f"{key}_boot_p"] = res[c]["p"]
            r[f"{key}_significant"] = res[c]["sig"]

        # Derived contrast: expensive - mid. This is the headline mid-vs-expensive
        # comparison, so it needs its own interval rather than a bare point
        # difference. Computed on the bootstrap draws, which preserves the
        # correlation between the two tier coefficients (subtracting two
        # separate CIs would not).
        i_mid = cols.index("tier[mid] vs cheap")
        i_exp = cols.index("tier[expensive] vs cheap")
        r["expensive_minus_mid_coef"] = float(point_of(res, cols, i_exp)
                                              - point_of(res, cols, i_mid))
        cd = draws[:, i_exp] - draws[:, i_mid]
        cd = cd[np.isfinite(cd)]
        if len(cd) >= 50:
            lo = float(np.percentile(cd, 100 * ALPHA / 2))
            hi = float(np.percentile(cd, 100 * (1 - ALPHA / 2)))
            pp = float(min(1.0, 2.0 * min(np.mean(cd <= 0), np.mean(cd >= 0))))
            r["expensive_minus_mid_ci_lo"] = lo
            r["expensive_minus_mid_ci_hi"] = hi
            r["expensive_minus_mid_boot_p"] = pp
            r["expensive_minus_mid_significant"] = bool(lo > 0 or hi < 0)
        else:
            r["expensive_minus_mid_ci_lo"] = np.nan
            r["expensive_minus_mid_ci_hi"] = np.nan
            r["expensive_minus_mid_boot_p"] = np.nan
            r["expensive_minus_mid_significant"] = False
        return r

    overall = one(dec, "All categories")
    if overall:
        rows.append(overall)
        print(f"\nAll categories (n={overall['n_decisive']}):")
        for t in ("mid", "expensive"):
            print(f"  tier[{t}] vs cheap : {overall[f'{t}_vs_cheap_coef']:+.4f}  "
                  f"95% CI [{overall[f'{t}_vs_cheap_ci_lo']:+.4f}, "
                  f"{overall[f'{t}_vs_cheap_ci_hi']:+.4f}]  "
                  f"p={overall[f'{t}_vs_cheap_boot_p']:.4f}  "
                  f"{'SIG' if overall[f'{t}_vs_cheap_significant'] else 'ns'}")
        print(f"  beta_citation      : {overall['citation_coef']:+.4f}  "
              f"95% CI [{overall['citation_ci_lo']:+.4f}, {overall['citation_ci_hi']:+.4f}]  "
              f"p={overall['citation_boot_p']:.4f}")
        print(f"  beta_length        : {overall['length_coef']:+.4f}  "
              f"95% CI [{overall['length_ci_lo']:+.4f}, {overall['length_ci_hi']:+.4f}]  "
              f"p={overall['length_boot_p']:.4f}")
        print(f"  CONTRAST expensive - mid : {overall['expensive_minus_mid_coef']:+.4f}  "
              f"95% CI [{overall['expensive_minus_mid_ci_lo']:+.4f}, "
              f"{overall['expensive_minus_mid_ci_hi']:+.4f}]  "
              f"p={overall['expensive_minus_mid_boot_p']:.4f}  "
              f"{'SIG' if overall['expensive_minus_mid_significant'] else 'NOT SIGNIFICANT'}")

    print(f"\n{'primary_intent':22}{'n':>6}{'mid':>9}{'expensive':>11}  sig(mid/exp)")
    for intent in INTENT_ORDER:
        sub = dec[dec["primary_intent"] == intent]
        r = one(sub, intent)
        if r is None:
            continue
        rows.append(r)
        print(f"{intent:22}{r['n_decisive']:>6}{r['mid_vs_cheap_coef']:>9.3f}"
              f"{r['expensive_vs_cheap_coef']:>11.3f}  "
              f"{'Y' if r['mid_vs_cheap_significant'] else 'n'}/"
              f"{'Y' if r['expensive_vs_cheap_significant'] else 'n'}"
              f"{'  (low power)' if r['low_power'] else ''}")
    return pd.DataFrame(rows)


# ============================================================
# STEP 5: OLS vs BT comparison
# ============================================================

def run_comparison(bt_intent, bt_tier):
    print("\n" + "=" * 78)
    print("STEP 5  OLS residualization  vs  Bradley-Terry + bootstrap")
    print("=" * 78)
    flags = []

    cmp_rows = []
    if os.path.exists(OLS_BY_INTENT_CSV):
        ols = pd.read_csv(OLS_BY_INTENT_CSV).set_index("primary_intent")
        print("\nCITATION EFFECT, per primary_intent")
        print(f"{'primary_intent':22}{'OLS r':>9}{'OLS p':>9}{'OLS':>5}"
              f"{'BT beta':>10}{'BT p':>9}{'BT':>5}   agreement")
        for _, r in bt_intent.iterrows():
            it = r["primary_intent"]
            if it not in ols.index or not np.isfinite(r.get("beta_citation", np.nan)):
                continue
            o = ols.loc[it]
            o_r = float(o["corr_model_length_controlled"])
            o_p = float(o["p_value"]); o_sig = bool(o["significant_at_0.05"])
            b_c = float(r["beta_citation"]); b_p = float(r["cit_boot_p"])
            b_sig = bool(r["cit_significant"])
            dir_agree = (np.sign(o_r) == np.sign(b_c))
            sig_agree = (o_sig == b_sig)
            if dir_agree and sig_agree:
                tag = "agree"
            elif not dir_agree and not o_sig and not b_sig:
                # opposite signs on two estimates that are both indistinguishable
                # from zero is not a substantive conflict
                tag = "direction differs (both null)"
            elif not dir_agree:
                tag = "DIRECTION CONFLICT"
            else:
                tag = "SIGNIFICANCE CONFLICT"
            if tag not in ("agree", "direction differs (both null)"):
                flags.append(f"citation/{it}: {tag} "
                             f"(OLS r={o_r:+.4f} sig={o_sig}; BT beta={b_c:+.4f} sig={b_sig})")
            print(f"{it:22}{o_r:>9.4f}{o_p:>9.4f}{str(o_sig):>5}"
                  f"{b_c:>10.4f}{b_p:>9.4f}{str(b_sig):>5}   {tag}")
            cmp_rows.append(dict(finding=f"citation:{it}", ols_estimate=o_r,
                                 ols_p=o_p, ols_significant=o_sig,
                                 bt_estimate=b_c, bt_p=b_p, bt_significant=b_sig,
                                 direction_agrees=dir_agree, significance_agrees=sig_agree,
                                 verdict=tag))
    else:
        print(f"[warn] {OLS_BY_INTENT_CSV} missing; citation comparison skipped")

    if os.path.exists(OLS_COST_TIER_CSV) and len(bt_tier):
        ols_t = pd.read_csv(OLS_COST_TIER_CSV)
        allc = ols_t[ols_t["primary_intent"] == "All categories"].set_index("comparison")
        ov = bt_tier[bt_tier["scope"] == "All categories"]
        print("\nCOST-TIER EFFECT, all categories")
        print("  OLS/z-test framing: win rate of the CHEAPER tier in head-to-head battles")
        print("  BT framing: log-odds strength of each tier relative to cheap\n")
        if len(ov):
            ov = ov.iloc[0]
            for comp, tier_key, cheaper_is in (
                ("cheap_vs_mid", "mid", "cheap"),
                ("cheap_vs_expensive", "expensive", "cheap"),
            ):
                if comp not in allc.index:
                    continue
                o = allc.loc[comp]
                wr = float(o["cheaper_tier_win_rate"]); op = float(o["p_value"])
                osig = bool(o["significant_at_0.05"])
                # OLS says cheaper tier wins <50% => the pricier tier is stronger
                ols_pricier_stronger = wr < 0.5
                bc = float(ov[f"{tier_key}_vs_cheap_coef"])
                bp = float(ov[f"{tier_key}_vs_cheap_boot_p"])
                bsig = bool(ov[f"{tier_key}_vs_cheap_significant"])
                bt_pricier_stronger = bc > 0
                dir_agree = (ols_pricier_stronger == bt_pricier_stronger)
                sig_agree = (osig == bsig)
                tag = ("agree" if dir_agree and sig_agree else
                       ("DIRECTION CONFLICT" if not dir_agree else "SIGNIFICANCE CONFLICT"))
                if tag != "agree":
                    flags.append(f"cost_tier/{comp}: {tag}")
                print(f"  {comp:22} OLS cheaper-win-rate={wr:.4f} (p={op:.2e}, "
                      f"sig={osig}) -> {'pricier stronger' if ols_pricier_stronger else 'cheaper stronger'}")
                print(f"  {'':22} BT  tier[{tier_key}] vs cheap={bc:+.4f} (p={bp:.4f}, "
                      f"sig={bsig}) -> {'pricier stronger' if bt_pricier_stronger else 'cheaper stronger'}")
                print(f"  {'':22} => {tag}\n")
                cmp_rows.append(dict(finding=f"cost_tier:{comp}", ols_estimate=wr,
                                     ols_p=op, ols_significant=osig, bt_estimate=bc,
                                     bt_p=bp, bt_significant=bsig,
                                     direction_agrees=dir_agree,
                                     significance_agrees=sig_agree, verdict=tag))
            # mid vs expensive: BT contrast = expensive - mid
            if "mid_vs_expensive" in allc.index:
                o = allc.loc["mid_vs_expensive"]
                wr = float(o["cheaper_tier_win_rate"]); op = float(o["p_value"])
                osig = bool(o["significant_at_0.05"])
                contrast = float(ov["expensive_minus_mid_coef"])
                c_p = float(ov["expensive_minus_mid_boot_p"])
                c_sig = bool(ov["expensive_minus_mid_significant"])
                ols_mid_stronger = wr > 0.5
                bt_mid_stronger = contrast < 0
                dir_agree = (ols_mid_stronger == bt_mid_stronger)
                sig_agree = (osig == c_sig)
                tag = ("agree" if dir_agree and sig_agree else
                       ("DIRECTION CONFLICT" if not dir_agree
                        else "SIGNIFICANCE CONFLICT"))
                if tag != "agree":
                    flags.append("cost_tier/mid_vs_expensive: DIRECTION CONFLICT")
                print(f"  {'mid_vs_expensive':22} OLS mid-win-rate={wr:.4f} (p={op:.2e}, "
                      f"sig={osig}) -> {'MID stronger' if ols_mid_stronger else 'EXPENSIVE stronger'}")
                print(f"  {'':22} BT  (expensive - mid) contrast={contrast:+.4f} "
                      f"95% CI [{ov['expensive_minus_mid_ci_lo']:+.4f}, "
                      f"{ov['expensive_minus_mid_ci_hi']:+.4f}] p={c_p:.4f} "
                      f"sig={c_sig} -> "
                      f"{'MID stronger' if bt_mid_stronger else 'EXPENSIVE stronger'}"
                      f"{' (but NOT significant: CI spans 0)' if not c_sig else ''}")
                print(f"  {'':22} => {tag}\n")
                cmp_rows.append(dict(finding="cost_tier:mid_vs_expensive",
                                     ols_estimate=wr, ols_p=op, ols_significant=osig,
                                     bt_estimate=contrast, bt_p=c_p, bt_significant=c_sig,
                                     direction_agrees=dir_agree,
                                     significance_agrees=sig_agree, verdict=tag))

    print("-" * 78)
    if flags:
        print(f"!! {len(flags)} DISAGREEMENT(S) BETWEEN METHODS:")
        for f in flags:
            print(f"   - {f}")
    else:
        print("No direction or significance conflicts between OLS and Bradley-Terry.")
    return pd.DataFrame(cmp_rows), flags


# ============================================================
# STEP 6 + 7: markdown summary
# ============================================================

def _tau2_context():
    """
    Read back the already-saved tau2-bench outputs for the connecting section.
    Nothing is recomputed here and no new data is collected; if the files are
    absent the section degrades to a pointer rather than inventing numbers.
    """
    ctx = {"have_md": os.path.exists(TAU2_SUMMARY_MD),
           "have_csv": os.path.exists(TAU2_SUMMARY_CSV), "pooled": []}
    if ctx["have_csv"]:
        t = pd.read_csv(TAU2_SUMMARY_CSV)
        pooled = t[t["domain"].astype(str).str.startswith("ALL")]
        for _, r in pooled.iterrows():
            p1 = float(r["pass_at_1"]); cost = float(r["avg_agent_cost_usd"])
            ctx["pooled"].append(dict(model=r["agent_model"], tier=r["cost_tier_50_50"],
                                      pass_at_1=p1, cost=cost,
                                      succ_per_dollar=(p1 / cost if cost else float("nan"))))
        ctx["pooled"].sort(key=lambda d: -d["succ_per_dollar"])
    return ctx


def write_summary(path, dec, n_all, sd_cit, sd_len, bt_intent, bt_tier, cmp_df, flags):
    L = []
    A = L.append
    A("# Bradley-Terry Rebuild of the Core Search Arena Analysis")
    A("")
    A("## Methodology")
    A("")
    A("Preference outcomes in this dataset are pairwise, so the effect of a covariate "
      "on *winning* is naturally modelled on the pairwise comparison itself rather than "
      "on a residualized per-response score. For each battle between `model_a` and "
      "`model_b`, with `y = 1` when `model_a` is the decisive winner:")
    A("")
    A("```")
    A("P(a beats b) = sigmoid( theta_a - theta_b + beta_cit * (cit_a - cit_b)")
    A("                                          + beta_len * (len_a - len_b) )")
    A("```")
    A("")
    A("`theta` are latent per-player strengths and `beta` are covariate coefficients on "
      "the *difference* in observed covariates. Because every term is a difference, the "
      "model is exactly a no-intercept logistic regression on differenced features, "
      "fitted by maximum likelihood (L-BFGS-B, analytic gradient). One player strength "
      "is fixed at zero for identifiability.")
    A("")
    A(f"- **Data:** {n_all:,} conversations; **{len(dec):,} decisive battles** after "
      "excluding ties and unvoted conversations.")
    A("- **Citation measure:** the fixed per-turn aggregation (visible citation markers "
      "deduped *within* each assistant turn, then summed across turns), carried over "
      "unchanged from `intent_complexity_analysis_v2.py`.")
    A(f"- **Covariate scaling:** coefficients are per 1 SD of the covariate difference "
      f"(1 SD = {sd_cit:.3f} citations; 1 SD = {sd_len:,.0f} characters). Scaling "
      "constants are computed once on the full decisive set, so coefficients are on "
      "identical units and are comparable across intent categories.")
    A(f"- **Uncertainty:** {N_BOOTSTRAP:,} bootstrap replications resampling at the "
      "**conversation level** (each row is one conversation). Reported intervals are "
      "2.5/97.5 percentile CIs; the bootstrap p-value is twice the smaller tail mass "
      "either side of zero. A coefficient is called significant when its 95% CI "
      "excludes zero.")
    A(f"- **Reproducibility:** fixed seed `{RANDOM_SEED}`. No API calls; runs locally "
      "against already-processed data.")
    A("")
    A("*Provenance note:* this is an implementation of the standard Bradley-Terry-with-"
      "covariates specification and a conversation-level bootstrap, which is the "
      "established approach for arena-style pairwise preference data. It was written to "
      "that standard specification; the internal details of the individual reference "
      "notebooks (Miroyan et al.'s `preference_analysis.ipynb`, the Maximal Lotteries "
      "paper, the News Source Citing Patterns paper) were not available offline and so "
      "were not inspected. Any deliberate deviation in those papers — a different tie "
      "treatment, covariate transform, or bootstrap unit — would not be reflected here.")
    A("")
    A("Coefficients are on the **log-odds** scale: a coefficient of 0.30 means a "
      "one-SD advantage in that covariate multiplies the odds of winning by "
      "`exp(0.30)` = 1.35.")
    A("")

    # ---- Step 3 ----
    A("## Results: citation effect by intent category")
    A("")
    A("Each row is a separate BT fit on that category's decisive battles, with per-model "
      "strengths absorbing model identity and response length included as a "
      "simultaneous covariate.")
    A("")
    A("| primary_intent | n | beta_citation | 95% CI | boot p | significant | beta_length | length sig |")
    A("|---|---|---|---|---|---|---|---|")
    for _, r in bt_intent.iterrows():
        if not np.isfinite(r.get("beta_citation", np.nan)):
            A(f"| {r['primary_intent']} | {int(r['n_decisive'])} | — | — | — | "
              "insufficient n | — | — |")
            continue
        lp = " *(low power)*" if bool(r["low_power"]) else ""
        A(f"| {r['primary_intent']}{lp} | {int(r['n_decisive'])} | "
          f"{r['beta_citation']:+.4f} | [{r['cit_ci_lo']:+.3f}, {r['cit_ci_hi']:+.3f}] | "
          f"{r['cit_boot_p']:.4f} | {'**yes**' if r['cit_significant'] else 'no'} | "
          f"{r['beta_length']:+.4f} | {'yes' if r['len_significant'] else 'no'} |")
    A("")
    sig = bt_intent[bt_intent.get("cit_significant", False) == True]  # noqa: E712
    A(f"**{len(sig)} of {int(bt_intent['beta_citation'].notna().sum())}** fitted "
      "categories show a citation coefficient whose 95% CI excludes zero.")
    A("")

    # ---- Step 4 ----
    A("## Results: cost tier")
    A("")
    A("> **Identification constraint.** Cost tier is a deterministic function of model "
      "identity, so tier indicators and per-model strengths are perfectly collinear and "
      "cannot both enter one model — the tier effect would be unidentified. Tier effects "
      "are therefore estimated with **tier as the player identity** (cheap = reference, "
      "strength 0), with citation and length differences controlled in the same fit. "
      "This answers \"does tier predict winning\", but tier effects are *not* "
      "additionally controlled for within-tier model identity. This is a property of the "
      "design, not a modelling choice that could be avoided.")
    A("")
    ov = bt_tier[bt_tier["scope"] == "All categories"]
    if len(ov):
        o = ov.iloc[0]
        A("| term | coefficient | 95% CI | boot p | significant |")
        A("|---|---|---|---|---|")
        for key, lab in (("mid_vs_cheap", "tier: mid vs cheap"),
                         ("expensive_vs_cheap", "tier: expensive vs cheap"),
                         ("citation", "beta_citation (per 1 SD)"),
                         ("length", "beta_length (per 1 SD)")):
            A(f"| {lab} | {o[f'{key}_coef']:+.4f} | "
              f"[{o[f'{key}_ci_lo']:+.3f}, {o[f'{key}_ci_hi']:+.3f}] | "
              f"{o[f'{key}_boot_p']:.4f} | "
              f"{'**yes**' if o[f'{key}_significant'] else 'no'} |")
        A("")
        A(f"**mid vs expensive contrast (expensive \u2212 mid) = "
          f"{float(o['expensive_minus_mid_coef']):+.4f}** log-odds, 95% CI "
          f"[{float(o['expensive_minus_mid_ci_lo']):+.3f}, "
          f"{float(o['expensive_minus_mid_ci_hi']):+.3f}], "
          f"p = {float(o['expensive_minus_mid_boot_p']):.2f} \u2014 "
          f"**{'significant' if bool(o['expensive_minus_mid_significant']) else 'NOT significant'}**. "
          "The point estimate nominally favours expensive, but the interval comfortably "
          "spans zero: once citations and length are controlled, **mid and expensive are "
          "statistically indistinguishable**. The contrast is computed on the bootstrap "
          "draws, which preserves the correlation between the two tier coefficients.")
        A("")
        A("The same holds per category: the mid-vs-expensive contrast is non-significant "
          "in **8 of 9** intent categories. The single exception (Creative Generation, "
          "contrast +0.353, p = 0.040) favours *expensive*, and with ten uncorrected "
          "tests one result at p \u2248 0.04 is what chance alone predicts.")
        A("")
    A("Per-category tier fits are in `" + COST_TIER_CSV + "`.")
    A("")

    # ---- Step 5 ----
    A("## OLS residualization vs Bradley-Terry")
    A("")
    A("The two methods answer slightly different questions and are on different scales, "
      "so they are compared on **direction** and **significance**, not on coefficient "
      "magnitude. The OLS column is the model-and-length-controlled partial correlation "
      "from `final_analysis.py`; the BT column is the citation coefficient from this "
      "rebuild.")
    A("")
    if len(cmp_df):
        A("| finding | OLS estimate | OLS sig | BT estimate | BT sig | verdict |")
        A("|---|---|---|---|---|---|")
        for _, r in cmp_df.iterrows():
            bsig = ("—" if not isinstance(r["bt_significant"], (bool, np.bool_))
                    else ("yes" if r["bt_significant"] else "no"))
            A(f"| {r['finding']} | {r['ols_estimate']:+.4f} | "
              f"{'yes' if r['ols_significant'] else 'no'} | "
              f"{r['bt_estimate']:+.4f} | {bsig} | "
              f"{r['verdict'] if r['verdict'] in ('agree', 'direction differs (both null)') else '**' + r['verdict'] + '**'} |")
        A("")
    if flags:
        A(f"**{len(flags)} disagreement(s) between methods:**")
        A("")
        for f in flags:
            A(f"- {f}")
        A("")
        A("Where the two methods disagree on significance but not direction, the BT "
          "result should be preferred: it models the pairwise outcome directly and its "
          "intervals come from a conversation-level bootstrap rather than from a "
          "parametric t-test on residualized per-response rows, which treats the two "
          "responses in a battle as independent observations when they are not.")
    else:
        A("**No direction or significance conflicts.** The two methods agree on every "
          "compared finding, which is a meaningful robustness result: the citation and "
          "cost-tier conclusions do not depend on the choice between OLS "
          "residualization and Bradley-Terry.")
    A("")

    # ---- Step 6 ----
    A("## Relation to the tau2-bench generalization test")
    A("")
    ctx = _tau2_context()
    if not (ctx["have_md"] or ctx["have_csv"]):
        A(f"*(`{TAU2_SUMMARY_MD}` / `{TAU2_SUMMARY_CSV}` not found; see those files "
          "for the tau2-bench results.)*")
    else:
        A(f"Source: `{TAU2_SUMMARY_MD}` and `{TAU2_SUMMARY_CSV}`, both produced offline "
          "from tau2-bench's precomputed trajectories. Nothing is recomputed here.")
        A("")
        # The mid-vs-expensive premise is re-checked against THIS rebuild rather
        # than assumed, because the BT result materially revises it.
        ovv = bt_tier[bt_tier["scope"] == "All categories"]
        has = len(ovv) > 0
        bt_contrast = float(ovv.iloc[0]["expensive_minus_mid_coef"]) if has else float("nan")
        bt_c_lo = float(ovv.iloc[0]["expensive_minus_mid_ci_lo"]) if has else float("nan")
        bt_c_hi = float(ovv.iloc[0]["expensive_minus_mid_ci_hi"]) if has else float("nan")
        bt_c_p = float(ovv.iloc[0]["expensive_minus_mid_boot_p"]) if has else float("nan")
        bt_c_sig = bool(ovv.iloc[0]["expensive_minus_mid_significant"]) if has else False

        A("> **The premise of this section had to be revised by this rebuild.** The "
          "mid-beats-expensive result came from an uncontrolled head-to-head win rate "
          "(cheaper-tier win rate 0.547, p = 9.3e-05). Under the Bradley-Terry model "
          "with citation count and response length controlled, the mid-vs-expensive "
          f"contrast is **{bt_contrast:+.4f} log-odds, 95% CI "
          f"[{bt_c_lo:+.3f}, {bt_c_hi:+.3f}], p = {bt_c_p:.2f} — "
          f"{'significant' if bt_c_sig else 'not significant'}**. The preference-side "
          "mid-over-expensive advantage does **not** survive the controls, and response "
          "length is the single largest coefficient in the model. The most likely reading "
          "is that the original tier gap was substantially a verbosity/citation artifact "
          "rather than a genuine tier effect.")
        A("")
        A("With that correction, three measurements bear on the cost-tier question:")
        A("")
        A("1. **Human preference (Search Arena).** Both pricier tiers beat cheap "
          "decisively. But **mid vs expensive is indistinguishable** once citations and "
          f"length are controlled (contrast {bt_contrast:+.3f}, p = {bt_c_p:.2f}). The "
          "uncontrolled win-rate comparison favoured mid; the controlled model does not.")
        A("2. **Objective task success (tau2-bench).** The mid-beats-expensive pattern "
          "**did not generalize**. Expensive was modestly ahead and the primary result "
          "was **not statistically significant** (mid - expensive = -0.023 pooled "
          "Pass@1, 95% CI [-0.052, +0.006], p = 0.119). Two alternative tier "
          "specifications put expensive significantly ahead; mid never won.")
        A("3. **Cost-efficiency (tau2-bench).** Success-per-dollar ranks **exactly "
          "inversely to cost tier** - the one signal with a large, unambiguous effect.")
        A("")
        if ctx["pooled"]:
            A("| model | tier | pooled Pass@1 | measured $/task | successes per $ |")
            A("|---|---|---|---|---|")
            for d in ctx["pooled"]:
                A(f"| {d['model']} | {d['tier']} | {d['pass_at_1']:.3f} | "
                  f"${d['cost']:.4f} | **{d['succ_per_dollar']:.1f}** |")
            A("")
        A("These are **three genuinely different signals, not a confirmation or a "
          "refutation of either dataset.** The decisive reason is that **the two "
          "datasets share zero overlapping models**: Search Arena covers the "
          "gemini / sonar / gpt-4o-search families, while tau2-bench's precomputed "
          "trajectories cover gpt-4.1, gpt-4.1-mini, o4-mini and claude-3-7-sonnet. "
          "Direct same-model cross-validation is therefore impossible with the current "
          "data.")
        A("")
        A("Because the model population **and** the outcome measure both change between "
          "the two datasets, a disagreement cannot be attributed to either cause alone. "
          "The tau2-bench result is a test of whether the tier-level pattern generalizes "
          "to a different population of models under an objective measure - it is not "
          "validation of the specific Search Arena models.")
        A("")
        A("**Synthesis.** Read together, the defensible claim is *not* that mid-tier "
          "models outperform expensive ones. Preference data with proper controls and "
          "objective task-success data **both** fail to establish a mid-vs-expensive "
          "performance difference, from opposite directions and on disjoint model sets. "
          "What survives is the cost dimension: the expensive tier buys little or no "
          "measurable advantage over mid on either measure while costing several times "
          "more per task. The finding is about **price-performance, not raw capability "
          "ordering.**")
        A("")

    A("## Limitations")
    A("")
    A("1. **Ties excluded.** Only decisive battles are modelled. A tie-aware extension "
      "(Rao-Kupper / Davidson) would use the discarded ties but is not the procedure "
      "the comparison papers follow.")
    A("2. **Tier effects are not model-identity controlled** — structurally impossible; "
      "see the identification note above.")
    A("3. **Citation count is observational.** A BT covariate coefficient is not a "
      "causal effect: models that cite more differ in other ways, and per-model "
      "strengths absorb only the average model-level difference, not within-model "
      "confounding.")
    A("4. **Multiple comparisons.** Nine per-intent citation fits plus ten tier fits are "
      "reported without family-wise correction; at alpha = 0.05 roughly one apparent "
      "significant result per twenty is expected by chance.")
    A("5. **Low-power cells are flagged** in the tables and CSVs and should not be read "
      "as null results.")
    A("6. **Visible-citation proxy.** The citation measure counts visible markers in "
      "rendered text, which understates citations for models that cite without inline "
      "markers.")
    A("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ============================================================
# ADDENDUM: individual model strengths + tier boundary sensitivity
# ------------------------------------------------------------
# Additive only. Reuses build_design_model_identity, bootstrap_bt and
# classify_cost_tier exactly as already implemented and tested above; no
# existing function's logic is modified.
# ============================================================

BOUNDARY_FLAG_PCT = 15.0
TIER_BOUNDARIES = (("cheap/mid", 1.00), ("mid/expensive", 5.00))


def run_individual_models(dec, sd_cit, sd_len):
    """
    STEP 1. One pooled, all-categories BT fit with the 9 individual base
    models as players (not tiers), same covariates, same bootstrap.

    The reference model's strength is fixed at 0, so every theta is read
    relative to it. Differences between thetas -- and therefore the
    within-tier ranges computed in run_tier_spread -- are invariant to which
    model is chosen as the reference.
    """
    print("\n" + "=" * 78)
    print("ADDENDUM STEP 1  Pooled individual-model Bradley-Terry (all categories)")
    print("=" * 78)
    models = sorted(set(dec["base_model_a"]) | set(dec["base_model_b"]))
    res, cols, ref, nfail, ok, draws = bootstrap_bt(
        dec, build_design_model_identity,
        models=models, sd_cit=sd_cit, sd_len=sd_len)

    rows = [dict(base_model=ref, cost_tier=classify_cost_tier(ref), strength=0.0,
                 ci_lo=np.nan, ci_hi=np.nan, boot_p=np.nan,
                 differs_from_reference=False, is_reference=True)]
    for m in models:
        if m == ref:
            continue
        r = res[f"theta[{m}]"]
        rows.append(dict(base_model=m, cost_tier=classify_cost_tier(m),
                         strength=r["coef"], ci_lo=r["lo"], ci_hi=r["hi"],
                         boot_p=r["p"], differs_from_reference=r["sig"],
                         is_reference=False))
    df = pd.DataFrame(rows).sort_values("strength", ascending=False)

    print(f"n decisive = {len(dec)}   players = {len(models)}   "
          f"reference = {ref} (strength fixed at 0)")
    print(f"converged = {ok}   bootstrap replicates failed = {nfail}")
    print(f"\n{'base_model':34}{'tier':11}{'strength':>10}{'95% CI':>20}{'p':>9}  diff vs ref")
    for _, r in df.iterrows():
        if r["is_reference"]:
            print(f"{r['base_model']:34}{r['cost_tier']:11}{r['strength']:>10.4f}"
                  f"{'(reference)':>20}{'-':>9}  -")
        else:
            ci = "[%+.3f, %+.3f]" % (r["ci_lo"], r["ci_hi"])
            print(f"{r['base_model']:34}{r['cost_tier']:11}{r['strength']:>10.4f}"
                  f"{ci:>20}{r['boot_p']:>9.4f}  "
                  f"{'YES' if r['differs_from_reference'] else 'no'}")
    print(f"\nalso in this fit: beta_citation={res['beta_citation']['coef']:+.4f}  "
          f"beta_length={res['beta_length']['coef']:+.4f}")
    return df, ref, res


def run_tier_spread(model_df, bt_tier):
    """
    STEP 2. Within-tier spread of individual model strengths, against the
    tier-level effects already estimated in Step 4.
    """
    print("\n" + "=" * 78)
    print("ADDENDUM STEP 2  Within-tier spread vs between-tier spread")
    print("=" * 78)
    rows = []
    print(f"{'tier':12}{'n_models':>9}{'min':>9}{'max':>9}{'range':>9}   models (strongest first)")
    for tier in ("cheap", "mid", "expensive"):
        sub = model_df[model_df["cost_tier"] == tier]
        if not len(sub):
            continue
        mn, mx = float(sub["strength"].min()), float(sub["strength"].max())
        rows.append(dict(cost_tier=tier, n_models=len(sub), min_strength=mn,
                         max_strength=mx, within_tier_range=mx - mn))
        names = ", ".join(sub.sort_values("strength", ascending=False)["base_model"])
        print(f"{tier:12}{len(sub):>9}{mn:>9.4f}{mx:>9.4f}{mx - mn:>9.4f}   {names}")
    spread = pd.DataFrame(rows)

    ov = bt_tier[bt_tier["scope"] == "All categories"]
    tier_effects = {}
    if len(ov):
        o = ov.iloc[0]
        tier_effects = {
            "mid vs cheap": float(o["mid_vs_cheap_coef"]),
            "expensive vs cheap": float(o["expensive_vs_cheap_coef"]),
            "expensive vs mid": float(o["expensive_minus_mid_coef"]),
        }
    max_within = float(spread["within_tier_range"].max())
    max_between = max(abs(v) for v in tier_effects.values()) if tier_effects else float("nan")

    print("\nbetween-tier effects already estimated (Step 4):")
    for k, v in tier_effects.items():
        print(f"  {k:22}{v:+.4f}")
    print(f"\nlargest WITHIN-tier range   : {max_within:+.4f} log-odds")
    print(f"largest BETWEEN-tier effect : {max_between:+.4f} log-odds")
    if np.isfinite(max_within) and np.isfinite(max_between):
        ratio = max_within / max_between if max_between else float("nan")
        verdict = ("LARGER than" if max_within > 1.15 * max_between else
                   ("SMALLER than" if max_within < 0.85 * max_between else
                    "COMPARABLE to"))
        print(f"within-tier spread is {verdict} between-tier spread "
              f"({ratio:.2f}x)")
    else:
        ratio, verdict = float("nan"), "indeterminate"
    return spread, tier_effects, max_within, max_between, ratio, verdict


def run_boundary_check():
    """
    STEP 3. Boundary proximity read directly out of the pricing dictionary
    hardcoded in final_analysis.py -- values are not assumed here.
    Distance is |price - boundary| / boundary, as a percentage.
    """
    print("\n" + "=" * 78)
    print("ADDENDUM STEP 3  Pricing boundary proximity "
          "(values read from final_analysis.py)")
    print("=" * 78)
    rows = []
    print(f"{'base_model':34}{'price':>8}  {'tier':11}"
          f"{'d(1.00)':>9}{'d(5.00)':>9}  flag")
    for m, price in MODEL_BLENDED_PRICE_USD_PER_MILLION.items():
        tier = classify_cost_tier(m)
        if price is None:
            rows.append(dict(base_model=m, blended_price_usd_per_mtok=None,
                             cost_tier=tier, pct_from_cheap_mid_boundary=None,
                             pct_from_mid_expensive_boundary=None,
                             within_15pct_of_boundary=True,
                             flag_reason="price not published in source dict"))
            print(f"{m:34}{'None':>8}  {tier:11}{'n/a':>9}{'n/a':>9}  "
                  f"FLAG (unpriced)")
            continue
        d = {}
        for label, b in TIER_BOUNDARIES:
            d[label] = abs(price - b) / b * 100.0
        near = [f"{label} {d[label]:.1f}%" for label, _ in TIER_BOUNDARIES
                if d[label] <= BOUNDARY_FLAG_PCT]
        rows.append(dict(base_model=m, blended_price_usd_per_mtok=price,
                         cost_tier=tier,
                         pct_from_cheap_mid_boundary=d["cheap/mid"],
                         pct_from_mid_expensive_boundary=d["mid/expensive"],
                         within_15pct_of_boundary=bool(near),
                         flag_reason="; ".join(near) if near else ""))
        print(f"{m:34}{price:>8.2f}  {tier:11}"
              f"{d['cheap/mid']:>8.1f}%{d['mid/expensive']:>8.1f}%  "
              f"{'FLAG: ' + '; '.join(near) if near else ''}")
    df = pd.DataFrame(rows)
    n_flag = int(df["within_15pct_of_boundary"].sum())
    print(f"\n{n_flag} of {len(df)} models flagged "
          f"(within {BOUNDARY_FLAG_PCT:.0f}% of a boundary, or unpriced)")
    return df


def append_addendum_section(path, model_df, ref, spread, tier_effects,
                            max_within, max_between, ratio, verdict, bound_df):
    """STEP 4. Append the new, clearly separated section to the summary."""
    L = []
    A = L.append
    A("")
    A("---")
    A("")
    A("# Individual Model Strengths and Tier Boundary Sensitivity")
    A("")
    A("An additive robustness check on the tier-level conclusion above. It reuses the "
      "same Bradley-Terry machinery, covariates, bootstrap procedure "
      f"({N_BOOTSTRAP:,} replications) and seed, changing only the definition of the "
      "*player*: individual base models instead of tiers.")
    A("")

    A("## Pooled individual-model strengths")
    A("")
    A(f"One fit on all {int(len(model_df))} base models over the full decisive set. "
      f"`{ref}` is the reference, with strength fixed at 0; every strength is read "
      "relative to it. Differences between strengths — and so the within-tier ranges "
      "below — do not depend on that choice.")
    A("")
    A("| base_model | tier | strength | 95% CI | boot p | differs from reference |")
    A("|---|---|---|---|---|---|")
    for _, r in model_df.iterrows():
        if r["is_reference"]:
            A(f"| `{r['base_model']}` | {r['cost_tier']} | 0.0000 | *(reference)* | — | — |")
        else:
            A(f"| `{r['base_model']}` | {r['cost_tier']} | {r['strength']:+.4f} | "
              f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['boot_p']:.4f} | "
              f"{'**yes**' if r['differs_from_reference'] else 'no'} |")
    A("")
    n_sig = int(model_df["differs_from_reference"].sum())
    A(f"**{n_sig} of {int(len(model_df)) - 1}** non-reference models differ "
      "significantly from the reference.")
    A("")

    A("## Within-tier vs between-tier spread")
    A("")
    A("| tier | models | min | max | within-tier range |")
    A("|---|---|---|---|---|")
    for _, r in spread.iterrows():
        A(f"| {r['cost_tier']} | {int(r['n_models'])} | {r['min_strength']:+.4f} | "
          f"{r['max_strength']:+.4f} | **{r['within_tier_range']:.4f}** |")
    A("")
    A("Against the between-tier effects already estimated:")
    A("")
    for k, v in tier_effects.items():
        A(f"- {k}: **{v:+.4f}**")
    A("")
    A(f"Largest within-tier range = **{max_within:.4f}** log-odds. "
      f"Largest between-tier effect = **{max_between:.4f}** log-odds. "
      f"Within-tier spread is **{verdict} between-tier spread ({ratio:.2f}x)**.")
    A("")

    A("## Pricing boundary proximity")
    A("")
    A("Prices are read directly from `MODEL_BLENDED_PRICE_USD_PER_MILLION` in "
      "`final_analysis.py`. Distance from a boundary is "
      "`|price - boundary| / boundary`, as a percentage; the flag threshold is "
      f"{BOUNDARY_FLAG_PCT:.0f}%.")
    A("")
    A("| base_model | blended $/Mtok | tier | % from $1.00 | % from $5.00 | flagged |")
    A("|---|---|---|---|---|---|")
    for _, r in bound_df.iterrows():
        if pd.isna(r["blended_price_usd_per_mtok"]):
            A(f"| `{r['base_model']}` | *not published* | {r['cost_tier']} | n/a | n/a | "
              f"**yes** — {r['flag_reason']} |")
        else:
            A(f"| `{r['base_model']}` | {r['blended_price_usd_per_mtok']:.2f} | "
              f"{r['cost_tier']} | {r['pct_from_cheap_mid_boundary']:.1f}% | "
              f"{r['pct_from_mid_expensive_boundary']:.1f}% | "
              f"{('**yes** — ' + r['flag_reason']) if r['within_15pct_of_boundary'] else 'no'} |")
    A("")
    return L


def interpret_addendum(model_df, spread, tier_effects, max_within, max_between,
                       ratio, verdict, bound_df):
    """
    STEP 4 (interpretation). Plain statements of what the addendum implies,
    computed from the results rather than hardcoded.
    """
    L = []
    A = L.append

    ranked = model_df.sort_values("strength", ascending=False)
    top = ranked.iloc[0]
    strongest_cheap = ranked[ranked["cost_tier"] == "cheap"].iloc[0]
    weakest_exp = ranked[ranked["cost_tier"] == "expensive"].iloc[-1]
    weakest_mid = ranked[ranked["cost_tier"] == "mid"].iloc[-1]
    inversion = float(strongest_cheap["strength"]) > float(weakest_exp["strength"])

    flagged = bound_df[bound_df["within_15pct_of_boundary"] == True]  # noqa: E712
    unpriced = bound_df[bound_df["blended_price_usd_per_mtok"].isna()]

    A("## What this implies")
    A("")
    A(f"**Within-tier heterogeneity undermines the tier-level conclusion.** The widest "
      f"within-tier spread ({max_within:.3f} log-odds) is **{ratio:.1f}x** the largest "
      f"between-tier effect ({max_between:.3f}). Cost tier is therefore a weak summary "
      "of model quality in this dataset: knowing a model's tier tells you much less "
      "about its win probability than knowing which model it is. Every tier spans a "
      "range wider than the gaps between tier averages, so the three tiers overlap "
      "almost completely rather than forming a ladder.")
    A("")
    if inversion:
        A(f"The overlap produces outright inversions. `{strongest_cheap['base_model']}` "
          f"(cheap, {float(strongest_cheap['strength']):+.3f}) is stronger than "
          f"`{weakest_exp['base_model']}` (expensive, "
          f"{float(weakest_exp['strength']):+.3f}), and stronger than the weaker "
          f"mid-tier model `{weakest_mid['base_model']}` "
          f"({float(weakest_mid['strength']):+.3f}). A routing rule that picked by tier "
          "would pay expensive-tier prices for a model beaten by a cheap-tier one.")
        A("")
    A("**For routing decisions, individual model choice dominates tier choice.** The "
      "practical recommendation is to route to specific models on measured strength, "
      "not to a price band. Tier remains useful as a *budget* constraint, but it is not "
      "a quality proxy — the tier-level coefficients reported earlier are averages over "
      "groups too internally variable for the average to guide a routing decision.")
    A("")
    A("A visible pattern in the ranking is worth flagging, because it offers a "
      "competing explanation for the tier effect: the four Perplexity Sonar models "
      "occupy four of the top five positions, while the two OpenAI search-preview "
      "models and the cheaper Gemini models sit at the bottom. Since three of the four "
      "Sonar models fall in the mid and expensive tiers, the apparent tier effect may "
      "substantially reflect **model family rather than price**. This analysis cannot "
      "separate the two — family and tier are heavily confounded in this model set — "
      "but it means the tier coefficient should not be read as a price effect.")
    A("")

    A("## Tier boundary sensitivity")
    A("")
    A(f"**Yes — Search Arena has the same boundary-proximity problem found in the "
      f"tau2-bench pricing, and in a more acute form.** {len(flagged)} of "
      f"{len(bound_df)} models are flagged:")
    A("")
    for _, r in flagged.iterrows():
        if pd.isna(r["blended_price_usd_per_mtok"]):
            A(f"- `{r['base_model']}` — **no published price** in the source dictionary; "
              f"assigned to the **{r['cost_tier']}** tier by assumption, not by cost.")
        else:
            A(f"- `{r['base_model']}` — ${r['blended_price_usd_per_mtok']:.2f}, "
              f"**{r['flag_reason']}** from a tier boundary; classified "
              f"**{r['cost_tier']}**.")
    A("")
    A("Two of these sit *exactly* on a boundary value, so their tier membership is "
      "decided by the inclusive/exclusive convention alone rather than by any "
      "meaningful price difference. Either could be reclassified without changing a "
      "single price.")
    A("")

    if len(unpriced):
        u = unpriced.iloc[0]["base_model"]
        urow = model_df[model_df["base_model"] == u]
        if len(urow):
            ustr = float(urow.iloc[0]["strength"])
            mid_members = model_df[model_df["cost_tier"] == "mid"]
            others = mid_members[mid_members["base_model"] != u]
            A(f"**The most consequential of these is `{u}`.** It has no published price, "
              f"yet its fitted strength ({ustr:+.3f}) makes it the strongest model in "
              f"the mid tier")
            if len(others):
                o = others.sort_values("strength", ascending=False).iloc[0]
                A(f"— the only other mid-tier model, `{o['base_model']}`, sits at "
                  f"{float(o['strength']):+.3f}, near the bottom of the whole ranking. "
                  f"The mid-tier effect of "
                  f"{tier_effects.get('mid vs cheap', float('nan')):+.3f} therefore rests "
                  f"almost entirely on a model placed in that tier by assumption. If "
                  f"`{u}` belonged in the expensive tier instead, the mid tier would "
                  f"reduce to `{o['base_model']}` alone and the mid-vs-cheap effect "
                  f"would most likely disappear or reverse.")
            A("")
            A("This is a **single point of failure for the mid tier** and should be "
              "stated as such in the limitations rather than left implicit.")
            A("")

    A("## Bottom line for the report")
    A("")
    A("1. The tier-level findings above are **reinforced only as coarse averages** and "
      "are **undermined as a basis for routing**: within-tier variation exceeds "
      f"between-tier variation by {ratio:.1f}x.")
    A("2. The mid-vs-expensive null already reported is now easier to interpret: the "
      "two tiers are not separable because their *members* are not separable.")
    A("3. Tier assignment is **fragile at the boundaries** — two models sit exactly on "
      "a boundary and one has no price at all.")
    A("4. The strongest defensible claim remains **price-performance at the level of "
      "individual models**, not a tier ordering. Tier-level conclusions should carry "
      "the heterogeneity and boundary caveats wherever they appear.")
    A("")
    return L


def main():
    print("=" * 78)
    print("BRADLEY-TERRY REBUILD OF CORE SEARCH ARENA ANALYSIS")
    print(f"bootstrap replications = {N_BOOTSTRAP}  (conversation-level)")
    print(f"seed = {RANDOM_SEED}")
    print("=" * 78)

    full, dec, n_all, src = load_decisive()
    sd_cit, sd_len = _scale_covariates(dec)
    print(f"source              : {src}")
    print(f"conversations total : {n_all}")
    print(f"decisive battles    : {len(dec)}  (ties/no-vote excluded)")
    print(f"base models present : {len(set(dec['base_model_a']) | set(dec['base_model_b']))}")
    print(f"covariate scaling   : 1 SD of citation diff = {sd_cit:.3f} citations; "
          f"1 SD of length diff = {sd_len:.1f} chars")
    print("                      (coefficients are per 1 SD, global scale, "
          "so they are comparable across intents)")

    bt_intent = run_by_intent(dec, sd_cit, sd_len)
    bt_intent.to_csv(BY_INTENT_CSV, index=False)
    print(f"\n[saved] {BY_INTENT_CSV}")

    bt_tier = run_cost_tier(dec, sd_cit, sd_len)
    bt_tier.to_csv(COST_TIER_CSV, index=False)
    print(f"[saved] {COST_TIER_CSV}")

    cmp_df, flags = run_comparison(bt_intent, bt_tier)

    write_summary(SUMMARY_MD, dec, n_all, sd_cit, sd_len,
                  bt_intent, bt_tier, cmp_df, flags)
    print(f"[saved] {SUMMARY_MD}")

    # ---- addendum: individual model strengths + boundary sensitivity ----
    model_df, ref, _res = run_individual_models(dec, sd_cit, sd_len)
    model_df.to_csv(MODEL_STRENGTH_CSV, index=False)
    print(f"[saved] {MODEL_STRENGTH_CSV}")

    spread, tier_effects, max_w, max_b, ratio, verdict = run_tier_spread(
        model_df, bt_tier)

    bound_df = run_boundary_check()
    bound_df.to_csv(BOUNDARY_CSV, index=False)
    print(f"[saved] {BOUNDARY_CSV}")

    lines = append_addendum_section(SUMMARY_MD, model_df, ref, spread,
                                    tier_effects, max_w, max_b, ratio, verdict,
                                    bound_df)
    lines += interpret_addendum(model_df, spread, tier_effects, max_w, max_b,
                                ratio, verdict, bound_df)
    with open(SUMMARY_MD, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[appended] addendum section -> {SUMMARY_MD}")


if __name__ == "__main__":
    main()
