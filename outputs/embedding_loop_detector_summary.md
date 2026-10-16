# Embedding-Based Frustration-Loop Detector

## Methodology

A local, offline replacement for the keyword-driven Gemini classifier in `frustration_classify.py`, built to test John's "friction = losing" question with a more rigorous, embedding-based signal instead of an LLM judging keywords.

- **Model:** `paraphrase-multilingual-MiniLM-L12-v2`, run locally via `sentence-transformers` -- zero API calls, zero cost, zero quota walls. Chosen over the English-only `all-MiniLM-L6-v2` because only ~58% of this eligible population is English; the remaining ~42% spans 110+ language tags, and an English-only model would silently degrade embeddings for all of them.
- **Population:** the same 986 eligible decisive multi-turn conversations already classified by the keyword method (`frustration_full990.jsonl`) -- reused directly rather than re-extracted, since that file already carries the exact `user_turns` text the keyword classifier scored, and its `idx` is confirmed to match `load_decisive()`'s index exactly. Re-extracting from `train.jsonl` risked silently building a different population and invalidating the Step 7 comparison.
- **Three candidate metrics**, all computed on L2-normalized embeddings (so a dot product is exactly the cosine similarity):
  1. `mean_cos_consecutive` -- mean cosine similarity between turn *i* and turn *i+1*. Captures "stuck repeating the same request."
  2. `mean_cos_first_to_later` -- mean cosine similarity between turn 1 and every later turn. Captures whether the conversation drifts from the opening request (low) or stays anchored near it (high).
  3. `embedding_variance` -- mean squared distance of each turn's embedding from the conversation's own centroid. A tight cluster (low variance) suggests looping; high spread suggests topic progression.

**Metric selection, stated explicitly.** All three metrics were computed for every conversation (Step 5) and checked two ways: (a) eyeballed against the same 20-conversation pilot already used for the keyword classifier, for direct comparability; (b) point-biserial correlation against the keyword frustrated(1)/engaged(0) label across the full labeled population (neutral dropped, same convention as `bradley_terry_frustration.py`). Results:

| metric | r vs keyword label |
|---|---|
| `mean_cos_consecutive` | +0.1937 **<- chosen** |
| `mean_cos_first_to_later` | +0.1760 |
| `embedding_variance` | -0.0732 |

**`mean_cos_consecutive`** was carried forward as the primary metric (largest |r| against the keyword label).

## Pilot validation (same 20 conversations as the keyword classifier)

| idx | keyword_label | n_turns | mean_cos_consecutive | mean_cos_first_to_later | embedding_variance |
|---|---|---|---|---|---|
| 22000 | engaged | 7 | 0.3972 | 0.3913 | 0.5394 |
| 7134 | engaged | 2 | 0.7335 | 0.7335 | 0.1333 |
| 11942 | neutral | 2 | 0.4172 | 0.4172 | 0.2914 |
| 11365 | engaged | 2 | 0.3781 | 0.3781 | 0.3109 |
| 11326 | engaged | 2 | 0.3216 | 0.3216 | 0.3392 |
| 4448 | frustrated | 4 | 0.4088 | 0.5177 | 0.3888 |
| 3719 | neutral | 3 | 0.5436 | 0.6787 | 0.2445 |
| 14164 | frustrated | 2 | 0.6561 | 0.6561 | 0.1720 |
| 21009 | engaged | 3 | 0.7238 | 0.6124 | 0.2499 |
| 10255 | engaged | 2 | 0.1639 | 0.1639 | 0.4180 |
| 12786 | engaged | 2 | 0.0597 | 0.0597 | 0.4701 |
| 12617 | frustrated | 2 | 0.5511 | 0.5511 | 0.2245 |
| 15468 | neutral | 2 | 0.1178 | 0.1178 | 0.4411 |
| 14218 | engaged | 3 | 0.4333 | 0.3974 | 0.3738 |
| 4659 | frustrated | 2 | 0.6392 | 0.6392 | 0.1804 |
| 4833 | engaged | 3 | 0.2109 | 0.3487 | 0.4967 |
| 11723 | engaged | 2 | 0.5125 | 0.5125 | 0.2438 |
| 22161 | frustrated | 3 | 0.4282 | 0.3846 | 0.3842 |

On the pilot alone (n=15, neutral excluded): frustrated mean `mean_cos_consecutive` = 0.5367 (n=5) vs engaged mean = 0.3934 (n=10). Full per-conversation turns and all three metrics are in `outputs/embedding_loop_pilot_review.txt` for manual inspection.

## Full-population results

Computed for all 986 eligible conversations -- see `outputs/embedding_loop_metrics.csv`. No API calls means this ran in a single local pass, no incremental saves or quota handling needed (unlike the keyword classifier's multi-session, quota-limited run).

## Does the loop-similarity metric predict losing?

Mirrors `bradley_terry_frustration.py`'s direct-effect design exactly, with the continuous, z-scored `mean_cos_consecutive` in place of a binary frustrated/engaged indicator. Conversation-level covariate (frustration has no per-model identity -- same reasoning as the keyword-based version), controlling for citation and length differences in the same fit:

| term | coefficient | 95% CI | boot p | significant |
|---|---|---|---|---|
| `mean_cos_consecutive` (z-scored) | +0.0435 | [-0.092, +0.170] | 0.4880 | no |
| beta_citation | +0.2223 | [+0.106, +0.361] | 0.0000 | yes |
| beta_length | +0.3688 | [+0.214, +0.544] | 0.0000 | yes |

n = 986, 1,000 bootstrap replications, seed `20260930`.

**Caveat, stated plainly (same as the keyword-based direct-effect test):** `model_a`/`model_b` are arbitrary side labels within each battle, not model identities tied to the user's turns -- the user's turns (and therefore this metric) are identical on both sides of a battle. This coefficient tests whether the metric relates to which *arbitrary side* won, net of citation/length. **A null result here is the expected, correctly-specified finding**, not a failure to find something; a non-null result would be surprising and would need independent scrutiny before being trusted.

## Agreement with the keyword classifier

Comparing frustrated-vs-engaged only (neutral dropped from both sides, n=835). Threshold, stated explicitly as a judgment call: the midpoint between the two keyword-labeled groups' own means on `mean_cos_consecutive` (higher = predicted frustrated), **not** tuned to maximize the apparent agreement rate.

**Agreement rate: 482 / 835 = 57.7%**

**Baseline check, stated plainly:** the trivial baseline of always predicting the more common label (`engaged`, 66.6% of this set) would itself score 66.6%. The embedding classifier **does NOT beat** that trivial baseline.

| | keyword: frustrated | keyword: engaged |
|---|---|---|
| embedding: frustrated | 142 | 216 |
| embedding: engaged | 137 | 340 |

353 of 835 (42.3%) disagree between the two methods.

## Honest bottom line

**The embedding signal correlates only weakly with the keyword label** (r = +0.194 for the best of the three metrics), and actually falls **below** the trivial majority-class baseline (57.7% vs 66.6% for always guessing the more common label). This is a legitimate, honestly-reported **inconclusive result for cosine-similarity-based looping as a frustration proxy, and by the baseline check, not even a usable one as a standalone classifier** -- not something to push toward looking more decisive than it is. Either the embedding geometry does not capture what the keyword classifier (and John's original description) meant by "frustration," or the keyword classifier itself is the noisier signal and this is actually picking up something real that the keyword method missed -- this comparison alone cannot distinguish those two explanations, and the direct-effect test in Step 6 (a clean, expected null) does not resolve it either.

## Caveats

1. **Frustration has no per-model identity.** The direct-effect test above is a control-style check, not expected to be the headline finding -- the real test of whether frustration intensity matters is whether it *moderates* the citation effect (as in `bradley_terry_frustration.py`'s Step 3), which this script does not repeat with the continuous metric; that would be a reasonable next step if this signal is judged worth pursuing further.
2. **Embeddings capture semantic similarity, not sentiment.** A user who legitimately needs to restate a complex request in similar words (not out of frustration) would score similarly to one who is actually stuck in a loop -- this metric cannot distinguish motivation, only textual/semantic recurrence.
3. **The agreement threshold is a judgment call** (the midpoint between the two keyword-labeled groups' means), stated explicitly above; a different threshold would shift the agreement rate without changing the underlying correlation.
4. **Multilingual embedding quality is not uniform across languages** -- `paraphrase-multilingual-MiniLM-L12-v2` covers 50+ languages but is not necessarily equally strong in all of them; no per-language breakdown of embedding quality was done here.
5. **Sample size.** n for the direct-effect test and the agreement comparison is in the hundreds, not thousands -- consistent with every other frustration-related finding in this project, treat point estimates with the same caution already applied there.

