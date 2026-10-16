# Unified Loss-Driver Model

## What this is, and why it's different from every prior test

Every test so far in this project looked at one feature at a time -- a Bradley-Terry coefficient per covariate group, loss_anatomy's per-battle rules, one frustration signal at a time. This combines every proven feature plus two new ones (query-response alignment, hedging rate) into a single gradient-boosted model, ranked by **held-out permutation importance**, with two interaction effects checked directly rather than assumed.

## Why this doesn't repeat the arbitrary-side-label bug

`frustration_outcome_quality.py` and `embedding_loop_detector.py` both hit the same wall: frustration is identical on both sides of a battle, so it can never predict which arbitrary side wins. **Every feature here is different in kind** -- a genuine per-side difference that varies within a battle (citations_a != citations_b, etc.), checked explicitly for zero variance in Step 7 rather than assumed. The one deliberate exception is `primary_intent`, a conversation-level property kept in the model **as interaction context, not as a feature expected to predict y on its own** -- it should show near-zero direct importance, by the same logic frustration did, and a large value there would be a red flag.

## Step 1+2: features used

Fit on **8,613** decisive battles (same population as every other script via `load_decisive()`), no silent mismatches on any proven feature.

| feature | source | type |
|---|---|---|
| `citation_diff` | existing (citations_a - citations_b) | per-side diff |
| `length_diff` | existing (response_length_a - response_length_b) | per-side diff |
| `strength_diff` | `bradley_terry_model_strengths.csv` | per-side diff |
| `same_family` | `classify_model_family` (bradley_terry_family.py) | per-side-pair indicator |
| `tier_diff` | `classify_cost_tier` (final_analysis.py), ordinal cheap=0/mid=1/expensive=2 | per-side diff |
| `alignment_diff` | **NEW** -- cosine(query, response) per side, from train.jsonl, `paraphrase-multilingual-MiniLM-L12-v2` | per-side diff |
| `hedging_diff` | **NEW** -- hedging-marker rate per response word, from train.jsonl | per-side diff |
| `primary_intent` | existing | shared context (NOT a per-side diff) |

## Step 4: held-out AUC

Train/test split: 80/20, stratified on outcome, seed `20260930`. Train n=6,890, test n=1,723.

| model | features | train AUC | held-out AUC |
|---|---|---|---|
| baseline (proven features only) | 5 | 0.7824 | **0.7517** |
| full (+ intent + 2 new features) | 8 | 0.8134 | **0.7509** |

**External reference point, not a replication:** the teammate's deck reported baseline AUC 0.705 -> 0.744 with their own five features. We do not have their exact feature definitions or train/test split, so this is a scale comparison only, not an apples-to-apples benchmark.

**Held-out AUC moved by -0.0007** when adding `primary_intent` + the two new text-derived features to the proven baseline -- essentially flat, and the small train-AUC gain (+0.0310) without a matching held-out gain is the classic signature of added flexibility fitting noise, not signal.

Both of our held-out AUCs (0.752, 0.751) sit above the teammate's reported range (0.705-0.744), but on a different feature set, model class, and train/test split -- not a claim that this model is "better," just a number worth having alongside theirs.

## Step 7: zero-variance check

No zero-variance or degenerate features found in either the train or test split. Every importance score below is computed on genuinely-varying inputs.

## Step 5: permutation importance (held-out set, NOT impurity-based)

| feature | mean AUC drop | std |
|---|---|---|
| `length_diff` | +0.09299 | 0.00706 |
| `strength_diff` | +0.05457 | 0.00745 |
| `citation_diff` | +0.03148 | 0.00538 |
| `primary_intent` | +0.00297 | 0.00231 |
| `alignment_diff` | +0.00239 | 0.00209 |
| `hedging_diff` | +0.00129 | 0.00128 |
| `tier_diff` | +0.00101 | 0.00067 |
| `same_family` | +0.00060 | 0.00026 |

Ranked by how much held-out AUC drops when that column is shuffled -- impurity-based importance was deliberately NOT used here since it is biased toward high-cardinality features like `primary_intent`.

## Step 6: interaction checks

### Citation effect by primary_intent

Reusing the already-fit, CI-bounded per-intent citation coefficients from `outputs/bradley_terry_by_intent.csv` rather than recomputing a new statistic:

| primary_intent | n | beta_citation | significant |
|---|---|---|---|
| Text Processing | 184 | +0.6861 | yes |
| Recommendation | 1026 | +0.4706 | yes |
| Explanation | 901 | +0.4549 | yes |
| Creative Generation | 604 | +0.4210 | yes |
| Info Synthesis | 1892 | +0.3956 | yes |
| Factual Lookup | 1974 | +0.3445 | yes |
| Analysis | 901 | +0.3116 | yes |
| Guidance | 680 | +0.1975 | no |
| Other | 451 | -0.0623 | no |

### Citation effect by cost tier of the matchup (higher of the two sides)

No 'difficulty' measure exists anywhere in this repo -- confirmed by grep, not assumed -- so only the cost-tier half of this check could be run; a difficulty-based stratification is not available with current data.

| battle tier | n | pearson r (citation_diff, win) |
|---|---|---|
| cheap | 354 | +0.3070 |
| mid | 617 | +0.1966 |
| expensive | 7642 | +0.2563 |

**Caveat on this specific check:** the three tiers are badly imbalanced in n (354 to 7642, 21.6x) since most battles involve at least one expensive-tier model -- the cheap/mid correlations rest on substantially less data than the expensive one and should be read with that in mind, even though all three clear the min_n threshold.

## Step 8: honest bottom line

**On the two new features specifically** (query-response alignment, hedging rate) -- reported plainly, not downplayed:

- `alignment_diff`: mean AUC drop = +0.00239 -- **does NOT meaningfully move held-out AUC**.
- `hedging_diff`: mean AUC drop = +0.00129 -- **does NOT meaningfully move held-out AUC**.

**Largest driver by permutation importance: `length_diff`** (mean AUC drop +0.09299).

**`primary_intent` direct importance: +0.00297** -- as expected (near zero, consistent with it being a shared conversation-level property with no direct effect on an arbitrary side).

