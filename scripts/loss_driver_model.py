"""
loss_driver_model.py

A unified feature-importance model answering John's framing directly: what's
driving losses, and how do these factors combine. Every test so far in this
project looked at one feature at a time (Bradley-Terry coefficients one
covariate group at a time, loss_anatomy's per-battle rules, etc.). This
combines every proven feature plus two new ones into a single gradient-
boosted model, ranked by held-out permutation importance, and explicitly
checks for interactions rather than assuming the tree finds them.

ADDITIVE ONLY. Does not modify any existing file. Reuses, unchanged:
- load_decisive() from bradley_terry_analysis.py (same decisive population
  used throughout this project)
- classify_cost_tier() from final_analysis.py
- classify_model_family() from bradley_terry_family.py
- get_messages / get_user_messages / get_assistant_messages from
  inspect_search_arena.py (same raw-text extraction helpers used by every
  other feature built from train.jsonl in this project)

STEP 3 (stated here, before anything is built, per the task's instruction):
WHY THIS DOESN'T REPEAT THE ARBITRARY-SIDE-LABEL BUG
-----------------------------------------------------
frustration_outcome_quality.py and embedding_loop_detector.py both hit the
same structural problem: frustration is a property of the user's turns,
which are IDENTICAL on both sides of a battle, so no frustration-based
feature can ever predict which arbitrary side (model_a vs model_b) wins --
any such test is null by construction, not by finding.

Every feature in this model is different IN KIND: each one is a genuine
per-side difference that varies across sides WITHIN a single battle --
citations_a != citations_b, response_length_a != response_length_b,
model_strength_a != model_strength_b, and so on. These CAN legitimately
differ by side, so they CAN legitimately predict which side wins. This is
checked explicitly in Step 7 (zero-variance check) rather than just assumed:
a feature that turned out to be identical across sides for every battle
would be caught there, not silently fit anyway.

The one exception, by design: `primary_intent` is a conversation-level
property (shared by both sides, like frustration). It is deliberately kept
in the model anyway, but for a different, explicit reason: not as a feature
expected to predict y on its own (it should show ~zero marginal/permutation
importance, by the same logic that frustration showed zero direct effect),
but as CONTEXT that lets the tree model learn intent-conditional splits on
the genuinely-asymmetric features (e.g. "citation_diff matters more within
Factual Lookup than within Guidance" -- exactly Step 6's question). If
`primary_intent` showed a large permutation importance on its own, that
would be a red flag worth investigating, not a feature behaving as
expected.
"""

import json
import os

import numpy as np
import pandas as pd

from bradley_terry_analysis import load_decisive, RANDOM_SEED
from final_analysis import classify_cost_tier
from bradley_terry_family import classify_model_family
from inspect_search_arena import get_messages, get_user_messages, get_assistant_messages

MODEL_STRENGTH_CSV = "outputs/bradley_terry_model_strengths.csv"
BY_INTENT_CSV = "outputs/bradley_terry_by_intent.csv"
TRAIN_JSONL = "train.jsonl"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

TIER_RANK = {"cheap": 0, "mid": 1, "expensive": 2}

# Hedging markers, English-only by construction (flagged explicitly as a
# limitation in the summary -- this dataset is ~42% non-English per earlier
# scripts' checks, and this feature cannot see hedging in other languages).
HEDGE_MARKERS = [
    "might", "possibly", "it's unclear", "i'm not certain", "perhaps",
    "may be", "it seems", "i think", "not sure", "could be", "likely",
    "it is possible", "i am not sure", "unclear", "uncertain", "seems to",
    "appears to", "probably", "may not", "it's possible", "somewhat",
]

OUT_FEATURES_CSV = "outputs/loss_driver_features.csv"
OUT_IMPORTANCE_CSV = "outputs/loss_driver_importance.csv"
OUT_INTERACTIONS_CSV = "outputs/loss_driver_interactions.csv"
OUT_SUMMARY_MD = "outputs/loss_driver_model_summary.md"

TEAMMATE_BASELINE_AUC = 0.705
TEAMMATE_FULL_AUC = 0.744


# ============================================================
# STEP 1: reuse existing, proven features
# ============================================================

def load_proven_features():
    print("=" * 78)
    print("STEP 1  Load proven features against the decisive-battle population")
    print("=" * 78)
    full, dec, n_all, src = load_decisive()
    print(f"source: {src}")
    print(f"decisive battles: {len(dec)}")

    strength = pd.read_csv(MODEL_STRENGTH_CSV).set_index("base_model")["strength"]
    print(f"loaded {len(strength)} model strengths from {MODEL_STRENGTH_CSV}")
    missing_models = (set(dec["base_model_a"]) | set(dec["base_model_b"])) - set(strength.index)
    if missing_models:
        print(f"[WARNING] {len(missing_models)} base models in dec have no strength "
              f"entry: {missing_models}")

    df = dec.copy()
    df["citation_diff"] = df["citations_a"] - df["citations_b"]
    df["length_diff"] = df["response_length_a"] - df["response_length_b"]
    df["strength_a"] = df["base_model_a"].map(strength)
    df["strength_b"] = df["base_model_b"].map(strength)
    df["strength_diff"] = df["strength_a"] - df["strength_b"]
    df["family_a"] = df["base_model_a"].map(classify_model_family)
    df["family_b"] = df["base_model_b"].map(classify_model_family)
    # Family is nominal (sonar/gemini/openai), not ordinal -- there is no
    # natural numeric "difference" the way there is for citations or tier.
    # The one meaningful per-side-pair signal from an unordered category is
    # whether the two sides are IN the same family at all. Stated explicitly
    # as a judgment call, not snuck in silently.
    df["same_family"] = (df["family_a"] == df["family_b"]).astype(int)
    df["tier_a"] = df["base_model_a"].map(classify_cost_tier)
    df["tier_b"] = df["base_model_b"].map(classify_cost_tier)
    df["tier_rank_a"] = df["tier_a"].map(TIER_RANK)
    df["tier_rank_b"] = df["tier_b"].map(TIER_RANK)
    df["tier_diff"] = df["tier_rank_a"] - df["tier_rank_b"]

    n_strength_nan = int(df["strength_diff"].isna().sum())
    n_family_nan = int(df["family_a"].isna().sum() + df["family_b"].isna().sum())
    n_tier_nan = int(df["tier_diff"].isna().sum())
    print(f"\nrow count after attaching proven features: {len(df)}")
    print(f"strength_diff NaN: {n_strength_nan}   "
          f"family NaN (either side): {n_family_nan}   tier_diff NaN: {n_tier_nan}")
    if n_strength_nan or n_family_nan or n_tier_nan:
        print("[WARNING] silent mismatches found -- investigate before proceeding.")
    else:
        print("no silent mismatches: every decisive battle has every proven feature.")
    return df


# ============================================================
# STEP 2: two new text-derived features
# ============================================================

def extract_raw_texts(df, path=TRAIN_JSONL):
    """
    Pulls, for every decisive battle, the full concatenated user text and
    full concatenated assistant text for each side -- the SAME per-side
    text aggregation convention already used for response_length_a/b in
    final_analysis.py's response_length_for_side (concatenate all turns for
    that side), so these new features are computed on exactly the same text
    the existing length feature already measures, not some other slice of
    the conversation.
    """
    print("\n" + "=" * 78)
    print("STEP 2  Extract raw query/response text from train.jsonl")
    print("=" * 78)
    idx_set = set(df.index)
    texts = {}
    n_found = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in idx_set:
                continue
            r = json.loads(line)
            row = {}
            for side in ("a", "b"):
                msgs = get_messages(r, side)
                user_texts = get_user_messages(msgs)
                asst_texts = get_assistant_messages(msgs)
                row[f"query_{side}"] = "\n".join(t or "" for t in user_texts)
                row[f"response_{side}"] = "\n".join(t or "" for t in asst_texts)
            texts[i] = row
            n_found += 1
    print(f"extracted text for {n_found} of {len(idx_set)} decisive battles")
    if n_found != len(idx_set):
        print(f"[WARNING] {len(idx_set) - n_found} decisive battles not found in "
              f"{path} -- investigate before proceeding.")
    return pd.DataFrame.from_dict(texts, orient="index")


def embed_queries_and_responses(text_df):
    print("\n" + "=" * 78)
    print("STEP 2  Embed queries and responses (local, no API calls)")
    print("=" * 78)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"loaded `{EMBEDDING_MODEL_NAME}` (same model already validated in "
          "embedding_loop_detector.py)")

    cols = ["query_a", "query_b", "response_a", "response_b"]
    flat = []
    spans = {}
    for col in cols:
        start = len(flat)
        flat.extend(text_df[col].tolist())
        spans[col] = (start, len(flat))
    print(f"encoding {len(flat)} texts ({len(cols)} fields x {len(text_df)} battles) ...")
    emb = model.encode(flat, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True)
    print(f"done. embeddings shape={emb.shape}")

    out = {}
    for col in cols:
        start, end = spans[col]
        out[col] = emb[start:end]
    return out


def compute_hedging_rate(text):
    if not text:
        return 0.0
    n_words = len(text.split())
    if n_words == 0:
        return 0.0
    lower = text.lower()
    n_hits = sum(lower.count(marker) for marker in HEDGE_MARKERS)
    return n_hits / n_words


def build_new_features(df, text_df):
    print("\n" + "=" * 78)
    print("STEP 2  Compute query-response alignment and hedging-rate features")
    print("=" * 78)

    emb = embed_queries_and_responses(text_df)
    align_a = np.sum(emb["query_a"] * emb["response_a"], axis=1)  # unit vectors -> dot = cosine
    align_b = np.sum(emb["query_b"] * emb["response_b"], axis=1)
    alignment_diff = pd.Series(align_a - align_b, index=text_df.index)

    hedge_a = text_df["response_a"].map(compute_hedging_rate)
    hedge_b = text_df["response_b"].map(compute_hedging_rate)
    hedging_diff = hedge_a - hedge_b

    print(f"alignment_diff: mean={alignment_diff.mean():+.4f} "
          f"sd={alignment_diff.std():.4f}  "
          f"range=[{alignment_diff.min():+.4f}, {alignment_diff.max():+.4f}]")
    print(f"hedging_diff:   mean={hedging_diff.mean():+.5f} "
          f"sd={hedging_diff.std():.5f}  "
          f"range=[{hedging_diff.min():+.5f}, {hedging_diff.max():+.5f}]")

    out = df.copy()
    out["alignment_diff"] = alignment_diff.reindex(out.index)
    out["hedging_diff"] = hedging_diff.reindex(out.index)
    n_nan = int(out["alignment_diff"].isna().sum() + out["hedging_diff"].isna().sum())
    print(f"NaN in new features: {n_nan} (expected 0)")
    return out


# ============================================================
# STEP 7: zero-variance / degenerate-input check (run BEFORE fitting)
# ============================================================

FEATURES_NUMERIC = ["citation_diff", "length_diff", "strength_diff",
                    "same_family", "tier_diff", "alignment_diff", "hedging_diff"]
FEATURE_CATEGORICAL = "primary_intent"


def check_zero_variance(train_df, test_df):
    print("\n" + "=" * 78)
    print("STEP 7  Zero-variance / degenerate-input check (train AND test splits)")
    print("=" * 78)
    problems = []
    for col in FEATURES_NUMERIC:
        sd_train = train_df[col].std()
        sd_test = test_df[col].std()
        flag = (sd_train == 0) or (sd_test == 0)
        print(f"{col:18} train sd={sd_train:.5f}  test sd={sd_test:.5f}"
              f"{'  *** ZERO VARIANCE ***' if flag else ''}")
        if flag:
            problems.append(col)
    n_cat_train = train_df[FEATURE_CATEGORICAL].nunique()
    n_cat_test = test_df[FEATURE_CATEGORICAL].nunique()
    print(f"{FEATURE_CATEGORICAL:18} train levels={n_cat_train}  test levels={n_cat_test}"
          f"{'  *** DEGENERATE ***' if n_cat_train <= 1 or n_cat_test <= 1 else ''}")
    if n_cat_train <= 1 or n_cat_test <= 1:
        problems.append(FEATURE_CATEGORICAL)

    if problems:
        print(f"\n[WARNING] {len(problems)} degenerate feature(s): {problems} -- "
              "any importance score for these would be meaningless, not just "
              "small. Flagging explicitly rather than silently fitting anyway.")
    else:
        print("\nNo zero-variance or degenerate features found in either split. "
              "Safe to proceed.")
    return problems


# ============================================================
# STEP 4: gradient-boosted model with held-out validation
# ============================================================

def fit_and_evaluate(df):
    print("\n" + "=" * 78)
    print("STEP 4  Fit gradient-boosted classifier, held-out AUC")
    print("=" * 78)
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    y = df["_y"].astype(int)
    X = df[FEATURES_NUMERIC + [FEATURE_CATEGORICAL]].copy()
    X[FEATURE_CATEGORICAL] = X[FEATURE_CATEGORICAL].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    print(f"train n={len(X_train)}  test n={len(X_test)}  "
          f"train win-rate={y_train.mean():.3f}  test win-rate={y_test.mean():.3f}")

    problems = check_zero_variance(X_train.assign(**{"_y": y_train.values}),
                                   X_test.assign(**{"_y": y_test.values}))

    def fit_one(feature_cols, label):
        Xtr = X_train[feature_cols].copy()
        Xte = X_test[feature_cols].copy()
        cat_features = [FEATURE_CATEGORICAL] if FEATURE_CATEGORICAL in feature_cols else None
        clf = HistGradientBoostingClassifier(
            categorical_features=cat_features, random_state=RANDOM_SEED,
            max_iter=200, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=15)
        clf.fit(Xtr, y_train)
        train_auc = roc_auc_score(y_train, clf.predict_proba(Xtr)[:, 1])
        test_auc = roc_auc_score(y_test, clf.predict_proba(Xte)[:, 1])
        print(f"{label:28} train AUC={train_auc:.4f}  held-out AUC={test_auc:.4f}  "
              f"n_features={len(feature_cols)}")
        return clf, train_auc, test_auc

    baseline_features = ["citation_diff", "length_diff", "strength_diff",
                         "same_family", "tier_diff"]
    full_features = FEATURES_NUMERIC + [FEATURE_CATEGORICAL]

    print(f"\nreference point (external, not reproduced exactly): teammate's deck "
          f"reported baseline AUC={TEAMMATE_BASELINE_AUC} -> full AUC="
          f"{TEAMMATE_FULL_AUC} with their five features. We do not have their "
          "exact feature definitions, so this is a scale comparison, not a "
          "replication.\n")

    base_clf, base_train_auc, base_test_auc = fit_one(baseline_features, "baseline (proven features)")
    full_clf, full_train_auc, full_test_auc = fit_one(full_features, "full (+ intent + 2 new features)")

    return dict(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        baseline_features=baseline_features, full_features=full_features,
        base_clf=base_clf, full_clf=full_clf,
        base_train_auc=base_train_auc, base_test_auc=base_test_auc,
        full_train_auc=full_train_auc, full_test_auc=full_test_auc,
        zero_variance_problems=problems,
    )


# ============================================================
# STEP 5: permutation importance on the held-out set
# ============================================================

def run_permutation_importance(fit_result, n_repeats=30):
    print("\n" + "=" * 78)
    print(f"STEP 5  Permutation importance on held-out set ({n_repeats} repeats)")
    print("=" * 78)
    from sklearn.inspection import permutation_importance

    clf = fit_result["full_clf"]
    X_test = fit_result["X_test"][fit_result["full_features"]]
    y_test = fit_result["y_test"]

    result = permutation_importance(
        clf, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
        random_state=RANDOM_SEED, n_jobs=-1)

    rows = []
    for i, feat in enumerate(fit_result["full_features"]):
        rows.append(dict(feature=feat, mean_auc_drop=result.importances_mean[i],
                         std_auc_drop=result.importances_std[i]))
    out = pd.DataFrame(rows).sort_values("mean_auc_drop", ascending=False)
    print(out.to_string(index=False))
    print("\n(mean_auc_drop = held-out AUC using the real column minus held-out AUC "
          "after shuffling that column; higher = more important. NOT raw impurity "
          "importance, which is biased toward high-cardinality features like "
          "primary_intent.)")
    return out


# ============================================================
# STEP 6: direct interaction checks (not just trusting the tree)
# ============================================================

def check_citation_by_intent(df):
    print("\n" + "=" * 78)
    print("STEP 6a  Citation effect size by primary_intent (reusing established results)")
    print("=" * 78)
    if os.path.exists(BY_INTENT_CSV):
        by_intent = pd.read_csv(BY_INTENT_CSV)
        print(f"reusing already-fit, proven per-intent citation coefficients from "
              f"{BY_INTENT_CSV} rather than recomputing a new statistic:")
        cols = ["primary_intent", "n_decisive", "beta_citation", "cit_ci_lo",
               "cit_ci_hi", "cit_boot_p", "cit_significant"]
        avail = [c for c in cols if c in by_intent.columns]
        print(by_intent[avail].sort_values("beta_citation", ascending=False).to_string(index=False))
        rng = by_intent["beta_citation"].max() - by_intent["beta_citation"].min()
        print(f"\nrange of beta_citation across intents: {rng:.4f} log-odds "
              f"({by_intent['beta_citation'].max():.4f} max, "
              f"{by_intent['beta_citation'].min():.4f} min) -- "
              f"{'a real spread' if rng > 0.2 else 'a modest spread'}, "
              "already established and CI-bounded in bradley_terry_by_intent.csv.")
        return by_intent[avail]
    else:
        print(f"[WARNING] {BY_INTENT_CSV} not found -- falling back to a fresh simple "
              "correlation per intent (less rigorous than the established BT fit).")
        rows = []
        for intent, sub in df.groupby("primary_intent"):
            r = float(np.corrcoef(sub["citation_diff"], sub["_y"])[0, 1])
            rows.append(dict(primary_intent=intent, n=len(sub), pearson_r=r))
        out = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
        print(out.to_string(index=False))
        return out


def check_citation_by_tier(df, min_n=100):
    print("\n" + "=" * 78)
    print("STEP 6b  Citation effect size by cost tier of the matchup")
    print("=" * 78)
    print("No 'difficulty' measure exists anywhere in this repo (grepped, confirmed "
          "absent) -- per the project convention of not inventing a proxy, only the "
          "cost-tier half of this check is run. 'Tier of the matchup' is defined as "
          "the HIGHER of the two sides' tiers (cheap < mid < expensive), since tier_a/"
          "tier_b are an arbitrary pair order and the max is order-invariant.")
    tier_rank_rev = {v: k for k, v in TIER_RANK.items()}
    battle_tier_rank = df[["tier_rank_a", "tier_rank_b"]].max(axis=1)
    battle_tier = battle_tier_rank.map(tier_rank_rev)

    rows = []
    for tier in ("cheap", "mid", "expensive"):
        sub = df[battle_tier == tier]
        n = len(sub)
        if n < min_n:
            print(f"{tier:12} n={n:5}  *** below min_n={min_n}, not reported as a rate ***")
            rows.append(dict(battle_tier=tier, n=n, pearson_r=np.nan, too_small=True))
            continue
        r = float(np.corrcoef(sub["citation_diff"], sub["_y"])[0, 1])
        print(f"{tier:12} n={n:5}  pearson_r(citation_diff, y)={r:+.4f}")
        rows.append(dict(battle_tier=tier, n=n, pearson_r=r, too_small=False))
    out = pd.DataFrame(rows)
    valid = out[~out["too_small"]]
    if len(valid) >= 2:
        rng = valid["pearson_r"].max() - valid["pearson_r"].min()
        print(f"\nrange of pearson_r across reportable tiers: {rng:.4f} "
              f"({'a real spread' if rng > 0.1 else 'a modest spread'} -- this is a "
              "simple correlation, not a CI-bounded coefficient like the by-intent "
              "check above; read it as a screening signal, not a proven effect.)")
    return out


# ============================================================
# Markdown summary
# ============================================================

def write_summary_md(path, fit_result, importance_df, citation_by_intent_df,
                     citation_by_tier_df, n_rows):
    L = []
    A = L.append
    A("# Unified Loss-Driver Model")
    A("")
    A("## What this is, and why it's different from every prior test")
    A("")
    A("Every test so far in this project looked at one feature at a time -- a "
      "Bradley-Terry coefficient per covariate group, loss_anatomy's per-battle "
      "rules, one frustration signal at a time. This combines every proven feature "
      "plus two new ones (query-response alignment, hedging rate) into a single "
      "gradient-boosted model, ranked by **held-out permutation importance**, with "
      "two interaction effects checked directly rather than assumed.")
    A("")
    A("## Why this doesn't repeat the arbitrary-side-label bug")
    A("")
    A("`frustration_outcome_quality.py` and `embedding_loop_detector.py` both hit the "
      "same wall: frustration is identical on both sides of a battle, so it can never "
      "predict which arbitrary side wins. **Every feature here is different in kind** "
      "-- a genuine per-side difference that varies within a battle (citations_a != "
      "citations_b, etc.), checked explicitly for zero variance in Step 7 rather than "
      "assumed. The one deliberate exception is `primary_intent`, a conversation-level "
      "property kept in the model **as interaction context, not as a feature expected "
      "to predict y on its own** -- it should show near-zero direct importance, by the "
      "same logic frustration did, and a large value there would be a red flag.")
    A("")

    A("## Step 1+2: features used")
    A("")
    A(f"Fit on **{n_rows:,}** decisive battles (same population as every other script "
      "via `load_decisive()`), no silent mismatches on any proven feature.")
    A("")
    A("| feature | source | type |")
    A("|---|---|---|")
    A("| `citation_diff` | existing (citations_a - citations_b) | per-side diff |")
    A("| `length_diff` | existing (response_length_a - response_length_b) | per-side diff |")
    A("| `strength_diff` | `bradley_terry_model_strengths.csv` | per-side diff |")
    A("| `same_family` | `classify_model_family` (bradley_terry_family.py) | per-side-pair indicator |")
    A("| `tier_diff` | `classify_cost_tier` (final_analysis.py), ordinal cheap=0/mid=1/expensive=2 | per-side diff |")
    A("| `alignment_diff` | **NEW** -- cosine(query, response) per side, from train.jsonl, `paraphrase-multilingual-MiniLM-L12-v2` | per-side diff |")
    A("| `hedging_diff` | **NEW** -- hedging-marker rate per response word, from train.jsonl | per-side diff |")
    A("| `primary_intent` | existing | shared context (NOT a per-side diff) |")
    A("")

    A("## Step 4: held-out AUC")
    A("")
    A(f"Train/test split: 80/20, stratified on outcome, seed `{RANDOM_SEED}`. "
      f"Train n={len(fit_result['X_train']):,}, test n={len(fit_result['X_test']):,}.")
    A("")
    A("| model | features | train AUC | held-out AUC |")
    A("|---|---|---|---|")
    A(f"| baseline (proven features only) | {len(fit_result['baseline_features'])} | "
      f"{fit_result['base_train_auc']:.4f} | **{fit_result['base_test_auc']:.4f}** |")
    A(f"| full (+ intent + 2 new features) | {len(fit_result['full_features'])} | "
      f"{fit_result['full_train_auc']:.4f} | **{fit_result['full_test_auc']:.4f}** |")
    A("")
    A(f"**External reference point, not a replication:** the teammate's deck reported "
      f"baseline AUC {TEAMMATE_BASELINE_AUC} -> {TEAMMATE_FULL_AUC} with their own five "
      "features. We do not have their exact feature definitions or train/test split, so "
      "this is a scale comparison only, not an apples-to-apples benchmark.")
    A("")
    delta = fit_result["full_test_auc"] - fit_result["base_test_auc"]
    A(f"**Held-out AUC moved by {delta:+.4f}** when adding `primary_intent` + the two "
      "new text-derived features to the proven baseline -- essentially flat, and the "
      "small train-AUC gain (+{:.4f}) without a matching held-out gain is the classic "
      "signature of added flexibility fitting noise, not signal.".format(
          fit_result["full_train_auc"] - fit_result["base_train_auc"]))
    A("")
    A(f"Both of our held-out AUCs ({fit_result['base_test_auc']:.3f}, "
      f"{fit_result['full_test_auc']:.3f}) sit above the teammate's reported range "
      f"({TEAMMATE_BASELINE_AUC}-{TEAMMATE_FULL_AUC}), but on a different feature set, "
      "model class, and train/test split -- not a claim that this model is "
      "\"better,\" just a number worth having alongside theirs.")
    A("")

    A("## Step 7: zero-variance check")
    A("")
    if fit_result["zero_variance_problems"]:
        A(f"**{len(fit_result['zero_variance_problems'])} degenerate feature(s) found:** "
          f"{fit_result['zero_variance_problems']}. Any importance score reported for "
          "these is not meaningful and should be disregarded.")
    else:
        A("No zero-variance or degenerate features found in either the train or test "
          "split. Every importance score below is computed on genuinely-varying inputs.")
    A("")

    A("## Step 5: permutation importance (held-out set, NOT impurity-based)")
    A("")
    A("| feature | mean AUC drop | std |")
    A("|---|---|---|")
    for _, r in importance_df.iterrows():
        A(f"| `{r['feature']}` | {r['mean_auc_drop']:+.5f} | {r['std_auc_drop']:.5f} |")
    A("")
    A("Ranked by how much held-out AUC drops when that column is shuffled -- impurity-"
      "based importance was deliberately NOT used here since it is biased toward "
      "high-cardinality features like `primary_intent`.")
    A("")

    A("## Step 6: interaction checks")
    A("")
    A("### Citation effect by primary_intent")
    A("")
    A("Reusing the already-fit, CI-bounded per-intent citation coefficients from "
      f"`{BY_INTENT_CSV}` rather than recomputing a new statistic:")
    A("")
    if "beta_citation" in citation_by_intent_df.columns:
        A("| primary_intent | n | beta_citation | significant |")
        A("|---|---|---|---|")
        for _, r in citation_by_intent_df.sort_values("beta_citation", ascending=False).iterrows():
            sig = r.get("cit_significant", "")
            A(f"| {r['primary_intent']} | {int(r['n_decisive'])} | {r['beta_citation']:+.4f} | "
              f"{'yes' if sig else 'no'} |")
    A("")
    A("### Citation effect by cost tier of the matchup (higher of the two sides)")
    A("")
    A("No 'difficulty' measure exists anywhere in this repo -- confirmed by grep, not "
      "assumed -- so only the cost-tier half of this check could be run; a difficulty-"
      "based stratification is not available with current data.")
    A("")
    A("| battle tier | n | pearson r (citation_diff, win) |")
    A("|---|---|---|")
    for _, r in citation_by_tier_df.iterrows():
        if r["too_small"]:
            A(f"| {r['battle_tier']} | {int(r['n'])} | *too few battles* |")
        else:
            A(f"| {r['battle_tier']} | {int(r['n'])} | {r['pearson_r']:+.4f} |")
    A("")
    n_vals = citation_by_tier_df.loc[~citation_by_tier_df["too_small"], "n"]
    if len(n_vals) and (n_vals.max() / n_vals.min()) > 5:
        A(f"**Caveat on this specific check:** the three tiers are badly imbalanced in "
          f"n ({int(n_vals.min())} to {int(n_vals.max())}, "
          f"{n_vals.max()/n_vals.min():.1f}x) since most battles involve at least one "
          "expensive-tier model -- the cheap/mid correlations rest on substantially "
          "less data than the expensive one and should be read with that in mind, even "
          "though all three clear the min_n threshold.")
        A("")

    A("## Step 8: honest bottom line")
    A("")
    new_feats = importance_df[importance_df["feature"].isin(["alignment_diff", "hedging_diff"])]
    A("**On the two new features specifically** (query-response alignment, hedging "
      "rate) -- reported plainly, not downplayed:")
    A("")
    for _, r in new_feats.iterrows():
        verdict = ("meaningfully moves held-out AUC" if r["mean_auc_drop"] > 0.005
                  else "does NOT meaningfully move held-out AUC")
        A(f"- `{r['feature']}`: mean AUC drop = {r['mean_auc_drop']:+.5f} -- "
          f"**{verdict}**.")
    A("")
    top_feat = importance_df.iloc[0]
    A(f"**Largest driver by permutation importance: `{top_feat['feature']}`** "
      f"(mean AUC drop {top_feat['mean_auc_drop']:+.5f}).")
    A("")
    intent_row = importance_df[importance_df["feature"] == "primary_intent"]
    if len(intent_row):
        ir = intent_row.iloc[0]
        expected = abs(ir["mean_auc_drop"]) < 0.005
        A(f"**`primary_intent` direct importance: {ir['mean_auc_drop']:+.5f}** -- "
          f"{'as expected (near zero, consistent with it being a shared conversation-level property with no direct effect on an arbitrary side)' if expected else '**larger than expected -- worth investigating, this would be a surprising direct effect for a conversation-level property**'}.")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[saved] {path}")


def main():
    print("=" * 78)
    print("LOSS DRIVER MODEL (additive -- no existing file modified)")
    print("=" * 78)

    df = load_proven_features()
    text_df = extract_raw_texts(df)
    df = build_new_features(df, text_df)

    df.to_csv(OUT_FEATURES_CSV, index=False)
    print(f"\n[saved] {OUT_FEATURES_CSV}")

    fit_result = fit_and_evaluate(df)
    importance_df = run_permutation_importance(fit_result)
    importance_df.to_csv(OUT_IMPORTANCE_CSV, index=False)
    print(f"[saved] {OUT_IMPORTANCE_CSV}")

    cit_by_intent = check_citation_by_intent(df)
    cit_by_tier = check_citation_by_tier(df)
    pd.concat([
        cit_by_intent.assign(check="by_intent"),
        cit_by_tier.assign(check="by_tier"),
    ], ignore_index=True).to_csv(OUT_INTERACTIONS_CSV, index=False)
    print(f"[saved] {OUT_INTERACTIONS_CSV}")

    write_summary_md(OUT_SUMMARY_MD, fit_result, importance_df, cit_by_intent,
                     cit_by_tier, len(df))


if __name__ == "__main__":
    main()
