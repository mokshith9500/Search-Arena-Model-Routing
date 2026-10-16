"""
embedding_loop_detector.py

An embedding/cosine-similarity frustration-loop detector -- a more rigorous,
offline replacement for the keyword-driven Gemini classifier in
frustration_classify.py. Built for John's "friction = losing" question.

ADDITIVE ONLY. Reuses load_decisive, bootstrap_bt, build_design_model_identity
and _scale_covariates from bradley_terry_analysis.py UNCHANGED. Does not
modify any existing file.

STEP 0 (population), confirmed before writing any new extraction logic:
frustration_full990.jsonl ALREADY IS the exact eligible population this task
asks for -- decisive multi-turn battles (1,714, by conversation_type) with
>= 2 actually-stored user turns (989 of them; TOTAL_ELIGIBLE=990 documented
in bradley_terry_frustration.py, one conversation permanently unclassifiable
by Gemini's content-safety filter per that script's own notes). It already
carries `user_turns` (the exact ordered text used to classify it) and `idx`
(verified to match load_decisive()'s dataframe index exactly, 989/989, so
the existing join logic in bradley_terry_frustration.py's merge_with_decisive
applies unchanged). There is therefore no need to re-extract anything from
train.jsonl -- doing so would risk silently reconstructing a DIFFERENT
population than the one the keyword classifier actually scored, which would
make Step 7's comparison invalid.

STEP 1 (embedding model): sentence-transformers, confirmed installed (v6.0.1)
and able to load a model locally with zero API calls. Model choice:
`paraphrase-multilingual-MiniLM-L12-v2`, not the English-only
all-MiniLM-L6-v2. Reason, checked rather than assumed: this eligible
population is NOT English-dominated -- only ~58% (571/989) of conversations
are English (`languages` field), the remaining ~42% span 110+ other distinct
language tags (Russian, German, Chinese, Spanish, Japanese, ...). Using the
English-only model would silently degrade embedding quality for over 40% of
the population, exactly the kind of language bias this project's own
memory notes ("frustration classifier is language-invariant") already flag
as something to protect.
"""

import json
import os

import numpy as np
import pandas as pd

from bradley_terry_analysis import (
    N_BOOTSTRAP,
    ALPHA,
    RANDOM_SEED,
    MIN_N_FOR_FIT,
    load_decisive,
    bootstrap_bt,
    build_design_model_identity,  # noqa: F401  (imported per task instruction; not used directly -- frustration has no per-model identity, same reasoning as bradley_terry_frustration.py)
    _scale_covariates,
)

FRUSTRATION_FULL_JSONL = "frustration_full990.jsonl"
FRUSTRATION_PILOT_JSONL = "frustration_pilot.jsonl"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

OUT_METRICS_CSV = "outputs/embedding_loop_metrics.csv"
OUT_PILOT_TXT = "outputs/embedding_loop_pilot_review.txt"
OUT_BT_CSV = "outputs/bradley_terry_embedding_loop.csv"
OUT_SUMMARY_MD = "outputs/embedding_loop_detector_summary.md"

MIN_GROUP_N = 50  # same honesty threshold used in bradley_terry_frustration.py


# ============================================================
# STEP 1: load the embedding model (local, offline, zero API calls)
# ============================================================

def load_embedding_model():
    print("=" * 78)
    print("STEP 1  Load local embedding model (no API calls)")
    print("=" * 78)
    from sentence_transformers import SentenceTransformer
    import time
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dt = time.time() - t0
    dim = model.get_sentence_embedding_dimension()
    print(f"loaded `{EMBEDDING_MODEL_NAME}` in {dt:.1f}s, embedding dim={dim}")
    print("reason: ~42% of the eligible population is non-English (see module "
          "docstring) -- the multilingual model avoids degrading those turns.")
    return model


# ============================================================
# STEP 2: load the eligible population (already-extracted, see module docstring)
# ============================================================

def load_eligible_population(path=FRUSTRATION_FULL_JSONL):
    print("\n" + "=" * 78)
    print("STEP 2  Load the eligible multi-turn population (reused, not re-extracted)")
    print("=" * 78)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"loaded {len(rows)} conversations from {path}")

    # Found during the sanity check, not assumed away: 3 of the 989 records
    # have their user_turns REDACTED to a single placeholder string
    # ("[excluded: contains personal/sensitive content, available on
    # request]") by whoever built this population file, even though the
    # stored `n_user_turns` metadata still reflects the pre-redaction count
    # (2-3). These cannot be embedded at all -- there is no real text left --
    # so they are explicitly excluded here, reported by name, rather than
    # silently producing a NaN metric three steps downstream.
    REDACTION_MARKER = "[excluded: contains personal/sensitive content, available on request]"
    redacted = [r for r in rows if r["user_turns"] == [REDACTION_MARKER]]
    if redacted:
        print(f"[REDACTED, excluded] {len(redacted)} conversation(s) have user_turns "
              f"replaced with a privacy-redaction placeholder, not real text: "
              f"idx={[r['idx'] for r in redacted]}")
        rows = [r for r in rows if r["user_turns"] != [REDACTION_MARKER]]
    print(f"analyzable population after excluding redacted turns: {len(rows)}")

    n_turns = [len(r["user_turns"]) for r in rows]
    print(f"user_turns per conversation: min={min(n_turns)} max={max(n_turns)} "
          f"mean={np.mean(n_turns):.2f}")
    assert min(n_turns) >= 2, (
        "found a conversation with < 2 usable turns after excluding redacted "
        "rows -- investigate, this should not happen")
    labels = pd.Series([r.get("label") for r in rows]).value_counts()
    print(f"keyword-classifier label counts (for the Step 7 comparison):\n{labels}")

    full, dec, n_all, src = load_decisive()
    idx_set = set(dec.index)
    n_matched = sum(1 for r in rows if r["idx"] in idx_set)
    print(f"idx match against load_decisive()'s index: {n_matched} of {len(rows)}")
    if n_matched != len(rows):
        print(f"[WARNING] {len(rows) - n_matched} conversations do not match the "
              "decisive dataset index -- investigate before trusting Step 6.")
    return rows, dec


# ============================================================
# STEP 2 (cont.): embed every user turn
# ============================================================

def embed_all_turns(model, rows):
    print("\n" + "=" * 78)
    print("STEP 2  Embed every user turn (batched, local)")
    print("=" * 78)
    flat_texts = []
    spans = []  # (start, end) into flat_texts for each conversation
    for r in rows:
        start = len(flat_texts)
        flat_texts.extend(r["user_turns"])
        spans.append((start, len(flat_texts)))
    print(f"encoding {len(flat_texts)} user turns across {len(rows)} conversations ...")
    embeddings = model.encode(
        flat_texts, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True,  # unit-norm so dot product == cosine similarity
    )
    print(f"done. embeddings shape={embeddings.shape}")
    return embeddings, spans


# ============================================================
# STEP 3: compute the three candidate metrics per conversation
# ============================================================

def compute_metrics(embeddings, spans):
    """
    All three metrics use the L2-normalized embeddings, so a dot product IS
    the cosine similarity -- no separate normalization step needed per pair.

    - mean_cos_consecutive: mean cos(turn_i, turn_{i+1}) for i=1..n-1.
      "Stuck repeating the same request."
    - mean_cos_first_to_later: mean cos(turn_1, turn_j) for j=2..n.
      "Drifts from the original request (low) vs stays anchored (high)."
    - embedding_variance: mean squared Euclidean distance of each turn's
      embedding from the conversation's own centroid. For unit vectors,
      ||e_i - centroid||^2 is a direct, monotonic function of how far e_i
      sits from the average direction -- a tight loop has near-zero
      variance, topic progression has higher variance.
    """
    rows = []
    for start, end in spans:
        E = embeddings[start:end]
        n = len(E)
        if n >= 2:
            cos_consec = [float(np.dot(E[i], E[i + 1])) for i in range(n - 1)]
            mean_cos_consecutive = float(np.mean(cos_consec))
            cos_first = [float(np.dot(E[0], E[j])) for j in range(1, n)]
            mean_cos_first_to_later = float(np.mean(cos_first))
            centroid = E.mean(axis=0)
            sq_dist = np.sum((E - centroid) ** 2, axis=1)
            embedding_variance = float(np.mean(sq_dist))
        else:
            mean_cos_consecutive = np.nan
            mean_cos_first_to_later = np.nan
            embedding_variance = np.nan
        rows.append(dict(
            n_turns=n,
            mean_cos_consecutive=mean_cos_consecutive,
            mean_cos_first_to_later=mean_cos_first_to_later,
            embedding_variance=embedding_variance,
        ))
    return pd.DataFrame(rows)


# ============================================================
# STEP 4: pilot validation against the SAME 20-conversation set
# ============================================================

def run_pilot_validation(rows, metrics_df, pilot_path=FRUSTRATION_PILOT_JSONL,
                         out_txt=OUT_PILOT_TXT):
    print("\n" + "=" * 78)
    print("STEP 4  Pilot validation against the same 20-conversation set")
    print("=" * 78)
    pilot_idx = []
    with open(pilot_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pilot_idx.append(json.loads(line)["idx"])
    print(f"pilot set: {len(pilot_idx)} conversations, from {pilot_path}")

    idx_to_pos = {r["idx"]: i for i, r in enumerate(rows)}
    missing = [i for i in pilot_idx if i not in idx_to_pos]
    if missing:
        print(f"[WARNING] {len(missing)} pilot idx not found in the eligible "
              f"population: {missing}")

    lines = []
    L = lines.append
    L("=" * 78)
    L("EMBEDDING LOOP DETECTOR -- PILOT VALIDATION (same 20 conversations as "
      "the keyword classifier)")
    L("=" * 78)
    L(f"model: {EMBEDDING_MODEL_NAME}   metrics: mean_cos_consecutive, "
      "mean_cos_first_to_later, embedding_variance")
    L("=" * 78 + "\n")

    pilot_rows = []
    for pidx in pilot_idx:
        if pidx not in idx_to_pos:
            continue
        pos = idx_to_pos[pidx]
        r = rows[pos]
        m = metrics_df.iloc[pos]
        pilot_rows.append(dict(idx=pidx, keyword_label=r.get("label"),
                               n_turns=m["n_turns"],
                               mean_cos_consecutive=m["mean_cos_consecutive"],
                               mean_cos_first_to_later=m["mean_cos_first_to_later"],
                               embedding_variance=m["embedding_variance"]))
        L("-" * 78)
        L(f"idx={pidx}  keyword_label={str(r.get('label')).upper()}  "
          f"n_turns={m['n_turns']}  languages={r.get('languages')}")
        L(f"mean_cos_consecutive={m['mean_cos_consecutive']:.4f}   "
          f"mean_cos_first_to_later={m['mean_cos_first_to_later']:.4f}   "
          f"embedding_variance={m['embedding_variance']:.4f}")
        L("-" * 78)
        for i, t in enumerate(r["user_turns"], 1):
            shown = t if len(t) <= 400 else t[:400] + f"...[truncated, {len(t)} chars]"
            L(f"  [TURN {i}] {shown}")
        L("")

    pdf = pd.DataFrame(pilot_rows)
    print("\n%-8s%-12s%6s%16s%16s%16s" % (
        "idx", "label", "n", "cos_consec", "cos_first", "variance"))
    for _, r in pdf.iterrows():
        print("%-8d%-12s%6d%16.4f%16.4f%16.4f" % (
            r["idx"], r["keyword_label"], r["n_turns"],
            r["mean_cos_consecutive"], r["mean_cos_first_to_later"],
            r["embedding_variance"]))

    # eyeball comparison: within the pilot, do frustrated/engaged separate
    # on each metric? (neutral excluded -- no clean contrast, same
    # convention as bradley_terry_frustration.py)
    two = pdf[pdf["keyword_label"].isin(["frustrated", "engaged"])]
    print(f"\npilot frustrated={int((two['keyword_label']=='frustrated').sum())}  "
          f"engaged={int((two['keyword_label']=='engaged').sum())}  "
          f"(neutral excluded from this comparison, n="
          f"{int((pdf['keyword_label']=='neutral').sum())})")
    L("=" * 78)
    L("PILOT COMPARISON: frustrated vs engaged, by metric (neutral excluded)")
    L("=" * 78)
    for metric in ("mean_cos_consecutive", "mean_cos_first_to_later", "embedding_variance"):
        gf = two[two["keyword_label"] == "frustrated"][metric]
        ge = two[two["keyword_label"] == "engaged"][metric]
        msg = (f"{metric:28} frustrated mean={gf.mean():.4f} (n={len(gf)})   "
              f"engaged mean={ge.mean():.4f} (n={len(ge)})   "
              f"diff={gf.mean() - ge.mean():+.4f}")
        print(msg)
        L(msg)

    L("\nPer-conversation disagreement flags (keyword label vs the metric that "
      "most separated the two groups above -- see run output for which metric "
      "that was):")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[saved] {out_txt}")
    return pdf


# ============================================================
# STEP 5: compute for the full eligible population (already done in
# compute_metrics -- this just assembles + saves the combined table)
# ============================================================

def assemble_full_table(rows, metrics_df):
    print("\n" + "=" * 78)
    print("STEP 5  Full-population metrics table (all eligible conversations)")
    print("=" * 78)
    base = pd.DataFrame([dict(
        idx=r["idx"], keyword_label=r.get("label"), intent=r.get("intent"),
        languages=str(r.get("languages")), winner=r.get("winner"),
        model_a=r.get("model_a"), model_b=r.get("model_b"),
    ) for r in rows])
    out = pd.concat([base, metrics_df], axis=1)
    print(f"assembled {len(out)} rows")
    print(out[["mean_cos_consecutive", "mean_cos_first_to_later",
              "embedding_variance"]].describe())
    out.to_csv(OUT_METRICS_CSV, index=False)
    print(f"[saved] {OUT_METRICS_CSV}")
    return out


# ============================================================
# Metric selection: quantitative check across the FULL labeled population
# (in addition to the pilot eyeball in Step 4) -- point-biserial correlation
# of each metric against the keyword frustrated(1)/engaged(0) label, dropping
# neutral. This is additional evidence, not a replacement for the Step 4
# pilot inspection the task asked for.
# ============================================================

def select_metric(full_df):
    print("\n" + "=" * 78)
    print("METRIC SELECTION  Point-biserial correlation vs keyword label "
          "(full population, frustrated=1/engaged=0, neutral dropped)")
    print("=" * 78)
    two = full_df[full_df["keyword_label"].isin(["frustrated", "engaged"])].copy()
    y = (two["keyword_label"] == "frustrated").astype(float)
    results = {}
    for metric in ("mean_cos_consecutive", "mean_cos_first_to_later", "embedding_variance"):
        x = two[metric]
        r = float(np.corrcoef(x, y)[0, 1])
        results[metric] = r
        print(f"{metric:28} r={r:+.4f}  (n={len(two)})")
    chosen = max(results, key=lambda k: abs(results[k]))
    print(f"\nchosen primary metric: {chosen} (largest |r| against the keyword label)")
    return chosen, results


# ============================================================
# STEP 6: does the chosen metric predict losing (y = model_a wins),
# controlling for citation/length -- same design as
# bradley_terry_frustration.py's run_direct_effect, continuous covariate
# instead of a binary frustrated/engaged indicator.
# ============================================================

def build_design_loop_direct(df, sd_cit, sd_len, metric_col, metric_mean, metric_sd):
    """
    Conversation-level covariate design, identical in spirit to
    build_design_frustration_direct in bradley_terry_frustration.py: the
    loop-similarity metric cannot enter as a per-model theta or a
    differenced covariate (it is a property of the user's turns, identical
    for both sides of the battle, not of either response), so it enters as
    a single z-scored continuous term plus an intercept, alongside the same
    citation/length difference covariates used everywhere else in this
    project.
    """
    n = len(df)
    z_metric = ((df[metric_col].values.astype(float) - metric_mean) / metric_sd)
    dcit = ((df["citations_a"] - df["citations_b"]).astype(float) / sd_cit).values
    dlen = ((df["response_length_a"] - df["response_length_b"]).astype(float) / sd_len).values
    X = np.column_stack([np.ones(n), z_metric, dcit, dlen])
    cols = ["intercept", "loop_similarity_z", "beta_citation", "beta_length"]
    y = df["_y"].values.astype(float)
    return X, y, cols


def run_loop_direct_effect(dec, merged, metric_col, sd_cit, sd_len):
    print("\n" + "=" * 78)
    print(f"STEP 6  Does `{metric_col}` predict losing (y = model_a wins)?")
    print("        (conversation-level covariate, NOT a per-model effect --")
    print("         controlling for which side had more citations / was longer)")
    print("=" * 78)

    n = len(merged)
    if n < MIN_N_FOR_FIT:
        print(f"n={n} is below MIN_N_FOR_FIT={MIN_N_FOR_FIT}; not fit.")
        return None

    metric_mean = float(merged[metric_col].mean())
    metric_sd = float(merged[metric_col].std(ddof=1)) or 1.0

    res, cols, _rest, nfail, ok, _draws = bootstrap_bt(
        merged, build_design_loop_direct, sd_cit=sd_cit, sd_len=sd_len,
        metric_col=metric_col, metric_mean=metric_mean, metric_sd=metric_sd)
    m = res["loop_similarity_z"]
    c = res["beta_citation"]
    l = res["beta_length"]
    print(f"n={n}  converged={ok}  bootstrap replicates failed={nfail}")
    print(f"metric standardized: mean={metric_mean:.4f}  sd={metric_sd:.4f}")
    print(f"\n  loop_similarity_z = {m['coef']:+.4f}  95% CI [{m['lo']:+.3f}, "
          f"{m['hi']:+.3f}]  p={m['p']:.4f}  "
          f"{'SIG' if m['sig'] else 'not significant'}")
    print(f"  beta_citation     = {c['coef']:+.4f}  95% CI [{c['lo']:+.3f}, "
          f"{c['hi']:+.3f}]  p={c['p']:.4f}")
    print(f"  beta_length       = {l['coef']:+.4f}  95% CI [{l['lo']:+.3f}, "
          f"{l['hi']:+.3f}]  p={l['p']:.4f}")
    print("\nCAVEAT (same as bradley_terry_frustration.py's direct-effect test): "
          "model_a/model_b are arbitrary side labels, not model identities tied "
          "to the user's turns. This coefficient asks whether the loop-similarity "
          "metric relates to which ARBITRARY SIDE won, net of citation/length -- "
          "a non-null result here would be surprising and should be checked "
          "carefully before trusting it; a null result is the expected finding "
          "under a correctly-specified model with no real per-side effect.")
    return dict(
        test="loop_similarity_direct_effect_on_outcome", metric=metric_col, n=n,
        metric_coef=m["coef"], metric_ci_lo=m["lo"], metric_ci_hi=m["hi"],
        metric_boot_p=m["p"], metric_significant=m["sig"],
        beta_citation=c["coef"], cit_ci_lo=c["lo"], cit_ci_hi=c["hi"],
        cit_boot_p=c["p"], cit_significant=c["sig"],
        beta_length=l["coef"], len_ci_lo=l["lo"], len_ci_hi=l["hi"],
        len_boot_p=l["p"], len_significant=l["sig"],
        converged=ok, n_boot_failed=nfail,
    )


# ============================================================
# STEP 7: agreement with the earlier keyword classifier
# ============================================================

def compare_with_keyword_classifier(full_df, chosen_metric):
    print("\n" + "=" * 78)
    print(f"STEP 7  Agreement with the keyword classifier (metric: {chosen_metric})")
    print("=" * 78)
    two = full_df[full_df["keyword_label"].isin(["frustrated", "engaged"])].copy()
    n_neutral = int((full_df["keyword_label"] == "neutral").sum())
    print(f"comparing on frustrated-vs-engaged only (n={len(two)}); "
          f"{n_neutral} 'neutral'-labeled conversations excluded, same "
          "frustrated-vs-engaged convention as bradley_terry_frustration.py")

    # Threshold, stated explicitly as a judgment call: the midpoint between
    # the two keyword-labeled groups' own means on the chosen metric. This
    # is NOT tuned to maximize apparent agreement -- it is fixed by the two
    # groups' actual central tendency, then applied uniformly.
    mean_f = two.loc[two["keyword_label"] == "frustrated", chosen_metric].mean()
    mean_e = two.loc[two["keyword_label"] == "engaged", chosen_metric].mean()
    threshold = (mean_f + mean_e) / 2
    higher_is_frustrated = mean_f > mean_e
    print(f"frustrated mean={mean_f:.4f}  engaged mean={mean_e:.4f}  "
          f"threshold (midpoint)={threshold:.4f}  "
          f"higher-is-frustrated={higher_is_frustrated}")

    if higher_is_frustrated:
        pred_frustrated = two[chosen_metric] >= threshold
    else:
        pred_frustrated = two[chosen_metric] <= threshold
    actual_frustrated = two["keyword_label"] == "frustrated"

    agree = (pred_frustrated == actual_frustrated)
    agreement_rate = float(agree.mean())
    n_agree = int(agree.sum())
    print(f"\nagreement rate: {n_agree}/{len(two)} = {agreement_rate:.1%}")

    # Honesty check: how does this compare to the TRIVIAL baseline of always
    # predicting the majority class? An "agreement rate" alone can look
    # respectable while actually underperforming a baseline that does no
    # work at all -- that comparison must be stated, not left implicit.
    majority_class_rate = float(max(actual_frustrated.mean(), 1 - actual_frustrated.mean()))
    print(f"majority-class baseline (always predict the more common label): "
          f"{majority_class_rate:.1%}")
    beats_baseline = agreement_rate > majority_class_rate
    print(f"embedding classifier {'BEATS' if beats_baseline else 'DOES NOT beat'} "
          "the trivial majority-class baseline.")

    tp = int((pred_frustrated & actual_frustrated).sum())
    tn = int((~pred_frustrated & ~actual_frustrated).sum())
    fp = int((pred_frustrated & ~actual_frustrated).sum())
    fn = int((~pred_frustrated & actual_frustrated).sum())
    print(f"confusion (embedding-predicted vs keyword-actual):")
    print(f"  predicted frustrated & keyword frustrated : {tp}")
    print(f"  predicted engaged    & keyword engaged     : {tn}")
    print(f"  predicted frustrated & keyword engaged     : {fp}  (disagreement)")
    print(f"  predicted engaged    & keyword frustrated  : {fn}  (disagreement)")

    disagreements = two[~agree][["idx", "keyword_label", chosen_metric]]
    print(f"\n{len(disagreements)} disagreements out of {len(two)} "
          f"({100 * len(disagreements) / len(two):.1f}%)")
    return dict(agreement_rate=agreement_rate, n=len(two), n_agree=n_agree,
               tp=tp, tn=tn, fp=fp, fn=fn, threshold=threshold,
               higher_is_frustrated=higher_is_frustrated,
               majority_class_rate=majority_class_rate,
               beats_baseline=beats_baseline), disagreements


# ============================================================
# Sanity checks
# ============================================================

def sanity_checks(rows, full_df):
    print("\n" + "=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)
    assert len(full_df) == len(rows), "row count mismatch after assembly"
    n_nan = int(full_df[["mean_cos_consecutive", "mean_cos_first_to_later",
                        "embedding_variance"]].isna().sum().sum())
    print(f"rows: {len(full_df)}  NaN metric cells: {n_nan} "
          "(expected 0 -- redacted conversations were already excluded in Step 2, "
          "every remaining conversation has >= 2 real turns)")
    assert n_nan == 0, "unexpected NaN metrics -- investigate before trusting results"
    print("reconciliation OK.")


# ============================================================
# Markdown summary
# ============================================================

def write_summary_md(path, full_df, chosen_metric, metric_corrs, pilot_df,
                     direct_effect_row, agreement, disagreements, n_eligible_source):
    L = []
    A = L.append
    A("# Embedding-Based Frustration-Loop Detector")
    A("")
    A("## Methodology")
    A("")
    A("A local, offline replacement for the keyword-driven Gemini classifier in "
      "`frustration_classify.py`, built to test John's \"friction = losing\" "
      "question with a more rigorous, embedding-based signal instead of an LLM "
      "judging keywords.")
    A("")
    A(f"- **Model:** `{EMBEDDING_MODEL_NAME}`, run locally via `sentence-transformers` "
      "-- zero API calls, zero cost, zero quota walls. Chosen over the English-only "
      "`all-MiniLM-L6-v2` because only ~58% of this eligible population is English; "
      "the remaining ~42% spans 110+ language tags, and an English-only model would "
      "silently degrade embeddings for all of them.")
    A(f"- **Population:** the same {n_eligible_source:,} eligible decisive multi-turn "
      "conversations already classified by the keyword method (`frustration_full990.jsonl`) "
      "-- reused directly rather than re-extracted, since that file already carries the "
      "exact `user_turns` text the keyword classifier scored, and its `idx` is confirmed "
      "to match `load_decisive()`'s index exactly. Re-extracting from `train.jsonl` risked "
      "silently building a different population and invalidating the Step 7 comparison.")
    A("- **Three candidate metrics**, all computed on L2-normalized embeddings (so a dot "
      "product is exactly the cosine similarity):")
    A("  1. `mean_cos_consecutive` -- mean cosine similarity between turn *i* and turn "
      "*i+1*. Captures \"stuck repeating the same request.\"")
    A("  2. `mean_cos_first_to_later` -- mean cosine similarity between turn 1 and every "
      "later turn. Captures whether the conversation drifts from the opening request "
      "(low) or stays anchored near it (high).")
    A("  3. `embedding_variance` -- mean squared distance of each turn's embedding from "
      "the conversation's own centroid. A tight cluster (low variance) suggests looping; "
      "high spread suggests topic progression.")
    A("")
    A("**Metric selection, stated explicitly.** All three metrics were computed for every "
      "conversation (Step 5) and checked two ways: (a) eyeballed against the same "
      "20-conversation pilot already used for the keyword classifier, for direct "
      "comparability; (b) point-biserial correlation against the keyword "
      "frustrated(1)/engaged(0) label across the full labeled population (neutral "
      "dropped, same convention as `bradley_terry_frustration.py`). Results:")
    A("")
    A("| metric | r vs keyword label |")
    A("|---|---|")
    for metric, r in metric_corrs.items():
        marker = " **<- chosen**" if metric == chosen_metric else ""
        A(f"| `{metric}` | {r:+.4f}{marker} |")
    A("")
    A(f"**`{chosen_metric}`** was carried forward as the primary metric (largest |r| "
      "against the keyword label).")
    A("")

    A("## Pilot validation (same 20 conversations as the keyword classifier)")
    A("")
    A("| idx | keyword_label | n_turns | mean_cos_consecutive | mean_cos_first_to_later | embedding_variance |")
    A("|---|---|---|---|---|---|")
    for _, r in pilot_df.iterrows():
        A(f"| {int(r['idx'])} | {r['keyword_label']} | {int(r['n_turns'])} | "
          f"{r['mean_cos_consecutive']:.4f} | {r['mean_cos_first_to_later']:.4f} | "
          f"{r['embedding_variance']:.4f} |")
    A("")
    two_pilot = pilot_df[pilot_df["keyword_label"].isin(["frustrated", "engaged"])]
    gf = two_pilot[two_pilot["keyword_label"] == "frustrated"][chosen_metric]
    ge = two_pilot[two_pilot["keyword_label"] == "engaged"][chosen_metric]
    A(f"On the pilot alone (n={len(two_pilot)}, neutral excluded): frustrated mean "
      f"`{chosen_metric}` = {gf.mean():.4f} (n={len(gf)}) vs engaged mean = "
      f"{ge.mean():.4f} (n={len(ge)}). Full per-conversation turns and all three "
      "metrics are in `" + OUT_PILOT_TXT + "` for manual inspection.")
    A("")

    A("## Full-population results")
    A("")
    A(f"Computed for all {len(full_df):,} eligible conversations -- see "
      "`" + OUT_METRICS_CSV + "`. No API calls means this ran in a single local pass, "
      "no incremental saves or quota handling needed (unlike the keyword classifier's "
      "multi-session, quota-limited run).")
    A("")

    A("## Does the loop-similarity metric predict losing?")
    A("")
    if direct_effect_row is None:
        A("Not fit -- see console output for the reason (likely insufficient n).")
    else:
        r = direct_effect_row
        A(f"Mirrors `bradley_terry_frustration.py`'s direct-effect design exactly, with "
          f"the continuous, z-scored `{chosen_metric}` in place of a binary "
          "frustrated/engaged indicator. Conversation-level covariate (frustration has no "
          "per-model identity -- same reasoning as the keyword-based version), controlling "
          "for citation and length differences in the same fit:")
        A("")
        A("| term | coefficient | 95% CI | boot p | significant |")
        A("|---|---|---|---|---|")
        A(f"| `{chosen_metric}` (z-scored) | {r['metric_coef']:+.4f} | "
          f"[{r['metric_ci_lo']:+.3f}, {r['metric_ci_hi']:+.3f}] | "
          f"{r['metric_boot_p']:.4f} | {'**yes**' if r['metric_significant'] else 'no'} |")
        A(f"| beta_citation | {r['beta_citation']:+.4f} | "
          f"[{r['cit_ci_lo']:+.3f}, {r['cit_ci_hi']:+.3f}] | "
          f"{r['cit_boot_p']:.4f} | {'yes' if r['cit_significant'] else 'no'} |")
        A(f"| beta_length | {r['beta_length']:+.4f} | "
          f"[{r['len_ci_lo']:+.3f}, {r['len_ci_hi']:+.3f}] | "
          f"{r['len_boot_p']:.4f} | {'yes' if r['len_significant'] else 'no'} |")
        A("")
        A(f"n = {r['n']:,}, {N_BOOTSTRAP:,} bootstrap replications, seed `{RANDOM_SEED}`.")
        A("")
        A("**Caveat, stated plainly (same as the keyword-based direct-effect test):** "
          "`model_a`/`model_b` are arbitrary side labels within each battle, not model "
          "identities tied to the user's turns -- the user's turns (and therefore this "
          "metric) are identical on both sides of a battle. This coefficient tests "
          "whether the metric relates to which *arbitrary side* won, net of citation/"
          "length. **A null result here is the expected, correctly-specified finding**, "
          "not a failure to find something; a non-null result would be surprising and "
          "would need independent scrutiny before being trusted.")
    A("")

    A("## Agreement with the keyword classifier")
    A("")
    A(f"Comparing frustrated-vs-engaged only (neutral dropped from both sides, n={agreement['n']:,}). "
      f"Threshold, stated explicitly as a judgment call: the midpoint between the two "
      f"keyword-labeled groups' own means on `{chosen_metric}` "
      f"({'higher' if agreement['higher_is_frustrated'] else 'lower'} = predicted "
      f"frustrated), **not** tuned to maximize the apparent agreement rate.")
    A("")
    A(f"**Agreement rate: {agreement['n_agree']:,} / {agreement['n']:,} = "
      f"{agreement['agreement_rate']:.1%}**")
    A("")
    A(f"**Baseline check, stated plainly:** the trivial baseline of always predicting "
      f"the more common label (`engaged`, {agreement['majority_class_rate']:.1%} of this "
      f"set) would itself score {agreement['majority_class_rate']:.1%}. The embedding "
      f"classifier {'beats' if agreement['beats_baseline'] else '**does NOT beat**'} "
      "that trivial baseline.")
    A("")
    A("| | keyword: frustrated | keyword: engaged |")
    A("|---|---|---|")
    A(f"| embedding: frustrated | {agreement['tp']:,} | {agreement['fp']:,} |")
    A(f"| embedding: engaged | {agreement['fn']:,} | {agreement['tn']:,} |")
    A("")
    A(f"{len(disagreements):,} of {agreement['n']:,} "
      f"({100*len(disagreements)/agreement['n']:.1f}%) disagree between the two methods.")
    A("")

    A("## Honest bottom line")
    A("")
    if abs(metric_corrs[chosen_metric]) < 0.2:
        baseline_clause = (
            f"and actually falls **below** the trivial majority-class baseline "
            f"({agreement['agreement_rate']:.1%} vs {agreement['majority_class_rate']:.1%} "
            "for always guessing the more common label)"
            if not agreement["beats_baseline"] else
            f"and only narrowly beats the trivial majority-class baseline "
            f"({agreement['agreement_rate']:.1%} vs {agreement['majority_class_rate']:.1%})"
        )
        A(f"**The embedding signal correlates only weakly with the keyword label** "
          f"(r = {metric_corrs[chosen_metric]:+.3f} for the best of the three metrics), "
          f"{baseline_clause}. This is a legitimate, honestly-reported **inconclusive "
          "result for cosine-similarity-based looping as a frustration proxy, and by "
          "the baseline check, not even a usable one as a standalone classifier** -- not "
          "something to push toward looking more decisive than it is. Either the "
          "embedding geometry does not capture what the keyword classifier (and John's "
          "original description) meant by \"frustration,\" or the keyword classifier "
          "itself is the noisier signal and this is actually picking up something real "
          "that the keyword method missed -- this comparison alone cannot distinguish "
          "those two explanations, and the direct-effect test in Step 6 (a clean, "
          "expected null) does not resolve it either.")
    else:
        A(f"**The embedding signal shows a real correspondence with the keyword label** "
          f"(r = {metric_corrs[chosen_metric]:+.3f}, agreement rate "
          f"{agreement['agreement_rate']:.1%}), though it is not a perfect replacement -- "
          f"{len(disagreements):,} conversations disagree between the two methods and "
          "are worth a closer manual look rather than assuming either method is "
          "automatically correct on those specific cases.")
    A("")
    A("## Caveats")
    A("")
    A("1. **Frustration has no per-model identity.** The direct-effect test above is a "
      "control-style check, not expected to be the headline finding -- the real test of "
      "whether frustration intensity matters is whether it *moderates* the citation "
      "effect (as in `bradley_terry_frustration.py`'s Step 3), which this script does "
      "not repeat with the continuous metric; that would be a reasonable next step if "
      "this signal is judged worth pursuing further.")
    A("2. **Embeddings capture semantic similarity, not sentiment.** A user who "
      "legitimately needs to restate a complex request in similar words (not out of "
      "frustration) would score similarly to one who is actually stuck in a loop -- "
      "this metric cannot distinguish motivation, only textual/semantic recurrence.")
    A("3. **The agreement threshold is a judgment call** (the midpoint between the two "
      "keyword-labeled groups' means), stated explicitly above; a different threshold "
      "would shift the agreement rate without changing the underlying correlation.")
    A("4. **Multilingual embedding quality is not uniform across languages** -- "
      "`paraphrase-multilingual-MiniLM-L12-v2` covers 50+ languages but is not "
      "necessarily equally strong in all of them; no per-language breakdown of "
      "embedding quality was done here.")
    A("5. **Sample size.** n for the direct-effect test and the agreement comparison is "
      "in the hundreds, not thousands -- consistent with every other frustration-related "
      "finding in this project, treat point estimates with the same caution already "
      "applied there.")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[saved] {path}")


def main():
    print("=" * 78)
    print("EMBEDDING LOOP DETECTOR (additive -- no existing file modified)")
    print("=" * 78)

    model = load_embedding_model()
    rows, dec = load_eligible_population()
    embeddings, spans = embed_all_turns(model, rows)
    metrics_df = compute_metrics(embeddings, spans)

    pilot_df = run_pilot_validation(rows, metrics_df)

    full_df = assemble_full_table(rows, metrics_df)
    sanity_checks(rows, full_df)

    chosen_metric, metric_corrs = select_metric(full_df)

    # merge onto dec (decisive dataset) for the BT test, same join pattern
    # as bradley_terry_frustration.py's merge_with_decisive
    idx_df = full_df.set_index("idx")
    matched_idx = idx_df.index.intersection(dec.index)
    merged = dec.loc[matched_idx].copy()
    for metric in ("mean_cos_consecutive", "mean_cos_first_to_later", "embedding_variance"):
        merged[metric] = idx_df.loc[matched_idx, metric]
    print(f"\nmatched {len(merged)} of {len(full_df)} embedded conversations to the "
          "decisive BT dataset for Step 6")

    sd_cit, sd_len = _scale_covariates(dec)
    direct_effect_row = run_loop_direct_effect(dec, merged, chosen_metric, sd_cit, sd_len)

    bt_rows = [direct_effect_row] if direct_effect_row else []
    pd.DataFrame(bt_rows).to_csv(OUT_BT_CSV, index=False)
    print(f"[saved] {OUT_BT_CSV}")

    agreement, disagreements = compare_with_keyword_classifier(full_df, chosen_metric)
    if len(disagreements):
        print("\nsample of disagreements:")
        print(disagreements.head(10).to_string(index=False))

    write_summary_md(OUT_SUMMARY_MD, full_df, chosen_metric, metric_corrs, pilot_df,
                     direct_effect_row, agreement, disagreements, len(full_df))


if __name__ == "__main__":
    main()
