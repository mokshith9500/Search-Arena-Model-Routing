# Cost-Performance Routing Analysis (by Intent, Individual-Model Level)

**What this is:** for each `primary_intent` category separately, which individual models are Pareto-optimal on (blended price, Bradley-Terry strength) -- a model no other model in that same intent beats on both lower cost AND higher strength at once. A Pareto-optimal model is a real, defensible routing recommendation; a dominated one is not, regardless of how good it looks in isolation.

**Deliberately individual-model level, not tier level.** This project's own tier-heterogeneity finding is that within-tier spread in strength is ~2.7x the between-tier spread -- price tier barely predicts strength. Framing this analysis by tier would visually contradict that finding. Tier is included in the saved CSV as a secondary label only; it is not the chart's grouping or primary color.

**Difficulty proxy, stated explicitly:** this analysis stratifies by `primary_intent`, not Avery's planned 0-7 difficulty score -- confirmed, in a separate audit, not to exist anywhere in this repo yet. If/when that score becomes available, it can replace `primary_intent` as the stratifying variable (re-run Step 3's grouping on it instead) without rebuilding this script from scratch.

## Inputs confirmed (Step 1)

- `outputs/bradley_terry_model_strength_by_intent.csv`: 9 models x 9 intents = 81 rows, a full grid, **9 cells flagged `low_n`** (too few battles to trust the strength estimate -- see the low-confidence table below).
- `final_analysis.py`'s `MODEL_BLENDED_PRICE_USD_PER_MILLION`: every one of the 9 models matched a real price; no model excluded for a missing price.

## Pareto-optimal models, per intent

| intent | Pareto-optimal models (price, strength) |
|---|---|
| Factual Lookup | `gemini-2.0-flash` ($0.25, +0.00); `sonar` ($1.00, +0.45); `sonar-reasoning-pro` ($5.00, +0.74) |
| Text Processing | `gemini-2.0-flash` ($0.25, +0.00) *(low_n)*; `sonar` ($1.00, +0.54) *(low_n)*; `sonar-reasoning` ($3.00, +0.72) *(low_n)*; `sonar-reasoning-pro` ($5.00, +1.10) *(low_n)* |
| Other | `gemini-2.0-flash` ($0.25, +0.00); `gpt-4o-mini-search-preview` ($0.38, +0.09) *(low_n)*; `sonar` ($1.00, +0.88) |
| Explanation | `gemini-2.0-flash` ($0.25, +0.00); `sonar` ($1.00, +0.45); `sonar-reasoning` ($3.00, +0.79); `gemini-2.5-pro-exp-03-25` ($5.63, +0.84) |
| Recommendation | `gemini-2.0-flash` ($0.25, +0.00); `gpt-4o-mini-search-preview` ($0.38, +0.06); `sonar` ($1.00, +0.67); `sonar-pro` ($9.00, +0.93) |
| Guidance | `gemini-2.0-flash` ($0.25, +0.00); `sonar` ($1.00, +0.30); `sonar-reasoning` ($3.00, +0.96); `gemini-2.5-pro-exp-03-25` ($5.63, +0.97) |
| Info Synthesis | `gemini-2.0-flash` ($0.25, +0.00); `sonar` ($1.00, +0.27); `sonar-reasoning` ($3.00, +0.51); `sonar-reasoning-pro` ($5.00, +0.70) |
| Analysis | `gemini-2.0-flash` ($0.25, +0.00); `sonar` ($1.00, +0.79); `sonar-reasoning` ($3.00, +0.91); `sonar-reasoning-pro` ($5.00, +1.11) |
| Creative Generation | `gemini-2.0-flash` ($0.25, +0.00); `sonar` ($1.00, +0.44); `sonar-reasoning` ($3.00, +0.57); `sonar-reasoning-pro` ($5.00, +0.78) |

## Does the same model dominate everywhere, or does it shift by intent?

`gemini-2.0-flash` is Pareto-optimal in all 9 intents -- but this is a mechanical consequence of it being the single cheapest model in the whole set (no other model can ever beat it on cost), not evidence it is a uniquely strong model. It should be read as a cost floor, not a quality signal.

**`sonar` is also Pareto-optimal in all 9 intents -- unlike the cost floor, this is not a mechanical artifact** of being the cheapest model (it isn't). *But* whether its point estimate actually, statistically beats every costlier-or-cheaper rival it needs to is a separate question -- checked explicitly, not assumed, in the CI-aware section below. **Short answer, don't take the point-estimate table below at face value: see "Step 3: do the headline findings hold up?" for whether this status survives confidence intervals.**

Beyond the cost-floor model, the Pareto-optimal set **genuinely shifts by intent** -- it is not the same 1-2 models dominating everywhere:

| model | number of intents where Pareto-optimal (of 9) |
|---|---|
| `gemini-2.0-flash` | 9 |
| `sonar` | 9 |
| `sonar-reasoning` | 6 |
| `sonar-reasoning-pro` | 5 |
| `gemini-2.5-pro-exp-03-25` | 2 |
| `gpt-4o-mini-search-preview` | 2 |
| `sonar-pro` | 1 |

## CI-aware robustness check: does the point-estimate frontier hold up?

The analysis above compares only point estimates of strength -- the same category of risk this project already corrected once before, in the OLS-to-Bradley-Terry rebuild (a finding that didn't survive stronger methods once uncertainty was actually checked). Two SEPARATE CI-aware questions are checked here, not one, because they cut in opposite directions:

- **Are the *dominated* models' status actually certain?** A model is `confidently_dominated` only if some cheaper-or-equal rival's strength 95% CI lower bound exceeds the dominated model's CI upper bound -- the same non-overlapping-CI bar this project already uses elsewhere to call a difference real, not a new, weaker threshold invented for this check. This is strictly *rarer* than point-estimate dominance, so some point-dominated cells get **recovered** as merely uncertain.
- **Is the *optimal* models' status actually certain?** A Pareto-optimal model `confidently_beats_all_cheaper` only if its own CI lower bound exceeds the CI upper bound of EVERY strictly-cheaper rival it needs to beat to earn its spot. This is strictly *rarer* than "has a higher point estimate than every cheaper rival", so some point-optimal cells get **downgraded** to nominal-only.

`gemini-2.0-flash` (the reference model) has no estimated CI -- its strength is fixed at 0 by the Bradley-Terry identification constraint, not independently measured. It is treated as a degenerate `[0, 0]` interval here (not skipped), since every other model's CI is already expressed relative to this fixed point -- the standard way to compare against a BT identification anchor.

### Step 2: how much does this change the picture?

- **47** (model, intent) pairs were dominated under point estimate. Of those, **42** (89.4%) are NOT confidently dominated once CIs are accounted for -- their "loser" status rests on a point-estimate difference the data doesn't actually distinguish from noise.
- **34** (model, intent) pairs were Pareto-optimal under point estimate. Of those, **24** (70.6%) do NOT confidently beat every cheaper rival -- their "winner" status also rests on a point-estimate edge the data doesn't fully back up.

### Step 3: do the headline findings hold up?

**`sonar`'s "Pareto-optimal in all 9 intents" claim does NOT fully hold up**: it is confidently optimal in only 1 of 9 intents once CIs are checked against its cheaper rivals -- the remaining 8 intent(s) have a higher point estimate that the data cannot actually distinguish from a cheaper alternative.

Same check for every other non-cost-floor model that was ever point-optimal somewhere:

| model | point-optimal (of 9) | confidently optimal (of 9) |
|---|---|---|
| `sonar` | 9 | 1 |
| `sonar-reasoning` | 6 | 0 |
| `sonar-reasoning-pro` | 5 | 0 |
| `gemini-2.5-pro-exp-03-25` | 2 | 0 |
| `gpt-4o-mini-search-preview` | 2 | 0 |
| `sonar-pro` | 1 | 0 |

## Low-confidence cells (`low_n`), flagged honestly, not silently plotted

| model | intent | n_battles | strength | Pareto-optimal? |
|---|---|---|---|---|
| `gemini-2.5-flash-preview-04-17` | Other | 23 | +0.627 | no |
| `gpt-4o-mini-search-preview` | Other | 54 | +0.094 | yes |
| `gemini-2.5-flash-preview-04-17` | Text Processing | 12 | +0.178 | no |
| `gpt-4o-mini-search-preview` | Text Processing | 13 | -1.830 | no |
| `sonar-reasoning` | Text Processing | 28 | +0.723 | yes |
| `sonar` | Text Processing | 33 | +0.539 | yes |
| `sonar-reasoning-pro` | Text Processing | 34 | +1.103 | yes |
| `gemini-2.0-flash` | Text Processing | 35 | +0.000 | yes |
| `gemini-2.5-pro-exp-03-25` | Text Processing | 38 | +0.884 | no |

**5 of the 9 `low_n` cells are Pareto-optimal** -- these are marked as hollow markers in the chart and flagged here by name specifically because a routing recommendation resting on a low-battle-count estimate deserves less trust than one resting on a well-sampled cell, even though both are drawn as the same Pareto status.

## Step 5: final honest verdict

**This check MEANINGFULLY WEAKENS the original findings.** A large share of both the "dominated" and the "optimal" labels do not survive CI scrutiny -- the point-estimate frontier was overstating certainty. Treat the point-estimate Pareto table above as a starting hypothesis, not a settled routing recommendation, until more battles narrow these intervals.

Concretely: **42/47** dominated cells recovered, **24/34** optimal cells downgraded to nominal-only. Either direction was a legitimate possible outcome going in -- this is what the data actually showed, not the result picked for how it sounds.

## Caveats

1. **Blended price does not include search-fee surcharges** for `gpt-4o-mini-search-preview` and `gpt-4o-search-preview` (per `final_analysis.py`'s own documented caveat) -- their true relative cost is higher than the blended number used here suggests.
2. **`sonar-reasoning`'s price has no independently published source** (a user-supplied verified figure per `final_analysis.py`) -- treated the same as every other price here, but worth knowing.
3. **Strength is a log-odds estimate relative to a fixed reference model (`gemini-2.0-flash`), not an absolute quality score** -- differences between two non-reference models are still meaningful (both are on the same log-odds scale), but the number itself has no natural zero point beyond the reference.
4. **Non-overlapping 95% CIs is a conservative bar, not a formal two-sample significance test** -- it under-flags real differences (some `confidently_dominated`/`confidently_optimal` cells may be too conservative), consistent with how this project already treats CI non-overlap elsewhere. `ci_status`, `eff_ci_lo`, and `eff_ci_hi` remain available in the saved CSV for anyone who wants a different bar.

