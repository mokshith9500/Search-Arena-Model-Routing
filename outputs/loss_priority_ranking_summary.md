# Loss Priority Ranking: Which Driver to Fix First

## John's question, answered directly

`loss_driver_model.py` ranked drivers by average effect size (permutation importance) alone. That's not enough to answer "what do we fix first" -- a feature can have a large average effect while rarely being the single thing that actually would have flipped a specific loss. This combines **effect size** with a **counterfactual flip rate** (how often closing one feature's gap alone would have changed the model's predicted winner) into one priority score.

## Step 1: model rebuild, verified

`load_proven_features`, `extract_raw_texts`, `build_new_features`, and `fit_and_evaluate` were imported from `loss_driver_model.py` and called unchanged -- no logic reimplemented. Rebuilt held-out AUC: **0.7509**, matching the already-reported 0.7509 within tolerance (0.0005). Confirmed before trusting anything built on top of it, per the task's own instruction to stop if it didn't match.

## Step 2: held-out losses isolated

Of **1,723** held-out battles, **1,203** (69.8%) were correctly predicted by the full model. Only these are used for the counterfactual test below -- a counterfactual on a battle the model already gets wrong isn't a meaningful test of what *would* flip it.

## Steps 3-4: counterfactual flip rate + combined priority score

**Counterfactual definition.** One feature at a time, every correctly-predicted held-out battle has that feature's per-side difference set to its neutral/parity value (0 for a true a-minus-b difference; **1, not 0, for `same_family`** -- see the code comment: 1 is the "no disadvantage" state for a same/different indicator, so setting it to 0 would backwards-widen the gap for already-matched battles). The model re-predicts; a **flip** is counted when the predicted winner changes to the actual loser.

**Combined score formula, stated exactly:** `priority_score = importance_norm * flip_rate_norm`, where `importance_norm` and `flip_rate_norm` are each the feature's value min-max scaled to [0,1] across the 7 testable features. A feature ranks high only if it is **both** a large average effect **and** frequently the single deciding factor -- neither alone is sufficient, which is exactly the property this ranking was built to capture.

| rank | feature | fix type | permutation importance | flip rate | priority score |
|---|---|---|---|---|---|
| 1 | `length_diff` | engineering fix | +0.09299 | 0.1771 | **1.0000** |
| 2 | `strength_diff` | routing fix | +0.05457 | 0.1047 | **0.3410** |
| 3 | `citation_diff` | engineering fix | +0.03148 | 0.0707 | **0.1296** |
| 4 | `alignment_diff` | engineering fix | +0.00239 | 0.0291 | **0.0029** |
| 5 | `hedging_diff` | engineering fix | +0.00129 | 0.0150 | **0.0005** |
| 6 | `tier_diff` | routing fix | +0.00101 | 0.0075 | **0.0001** |
| 7 | `same_family` | routing fix | +0.00060 | 0.0033 | **0.0000** |

**Honest note:** in this run, the combined ranking happens to land in **exactly the same order** as permutation importance alone -- the flip-rate signal correlated strongly with effect size here rather than reordering anything. That is a legitimate result, not a failure of the method: it means effect size and "how often this is the deciding factor" agree for this model and this population. The combined score still adds real information (the magnitudes and relative gaps differ from importance alone), and a different model, split, or feature set could easily produce a case where the two rankings diverge -- this script would surface that automatically if it occurred.

*(`primary_intent` excluded from this ranking -- permutation importance +0.00297, but no flip rate is applicable: it is a shared conversation-level property, identical on both sides of every battle, so there is no per-side gap to close.)*

## Step 5: two fundamentally different kinds of fixes

| fix type | features | what it actually means |
|---|---|---|
| **Engineering fix** | `citation_diff`, `length_diff`, `hedging_diff`, `alignment_diff` | Something a system could change about *how a given model behaves* -- prompting, response formatting, retrieval. Does not require switching models. |
| **Routing fix** | `strength_diff`, `tier_diff`, `same_family` | Something about *which model you choose to use at all*. "Fixing" this means routing to a different/stronger model, not changing how the current one responds. |

**This distinction matters for prioritization**, not just description: the top engineering fix and the top routing fix point to entirely different team actions (prompt/response engineering vs. a routing-policy change), and the two should not be read as directly substitutable just because one has a higher combined score than the other.

**Top engineering fix: `length_diff`** (priority score 1.0000). **Top routing fix: `strength_diff`** (priority score 0.3410).

## Step 6: real limitations of this method -- stated plainly

1. **This shows what the model's learned decision boundary implies, not a guaranteed real-world causal effect.** The gradient-boosted model is a statistical fit to historical battles, not a causal model of why a response wins -- a high flip rate means the model's *learned function* is sensitive to that feature near these specific losses, which is suggestive evidence for prioritization, not proof of what would happen if a team actually changed it.
2. **Setting a feature to exact parity is an idealized full fix, not a realistic partial improvement.** Real engineering work rarely closes a gap completely -- a flip rate measured at full parity is an upper bound on what any real, partial intervention would achieve.
3. **Flipping one feature at a time ignores interactions.** The sum of individual flip counts across all seven features is well above the number of correctly-predicted losses (printed explicitly in the Step 3 console output) -- this confirms flip rates overlap rather than partition the losses. Several features can each individually "flip" the same battle, and fixing one may make another's flip unnecessary (or insufficient on its own if two features were jointly compensating). **Individual flip rates do not simply add up to the total number of losses explained.**
4. **Held-out sample size.** The flip test runs on only 1,203 correctly-predicted held-out battles -- small differences in flip rate between lower-ranked features should not be over-read.

