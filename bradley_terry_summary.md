# Bradley-Terry Rebuild of the Core Search Arena Analysis

## Methodology

Preference outcomes in this dataset are pairwise, so the effect of a covariate on *winning* is naturally modelled on the pairwise comparison itself rather than on a residualized per-response score. For each battle between `model_a` and `model_b`, with `y = 1` when `model_a` is the decisive winner:

```
P(a beats b) = sigmoid( theta_a - theta_b + beta_cit * (cit_a - cit_b)
                                          + beta_len * (len_a - len_b) )
```

`theta` are latent per-player strengths and `beta` are covariate coefficients on the *difference* in observed covariates. Because every term is a difference, the model is exactly a no-intercept logistic regression on differenced features, fitted by maximum likelihood (L-BFGS-B, analytic gradient). One player strength is fixed at zero for identifiability.

- **Data:** 24,069 conversations; **8,613 decisive battles** after excluding ties and unvoted conversations.
- **Citation measure:** the fixed per-turn aggregation (visible citation markers deduped *within* each assistant turn, then summed across turns), carried over unchanged from `intent_complexity_analysis_v2.py`.
- **Covariate scaling:** coefficients are per 1 SD of the covariate difference (1 SD = 6.126 citations; 1 SD = 3,223 characters). Scaling constants are computed once on the full decisive set, so coefficients are on identical units and are comparable across intent categories.
- **Uncertainty:** 1,000 bootstrap replications resampling at the **conversation level** (each row is one conversation). Reported intervals are 2.5/97.5 percentile CIs; the bootstrap p-value is twice the smaller tail mass either side of zero. A coefficient is called significant when its 95% CI excludes zero.
- **Reproducibility:** fixed seed `20260930`. No API calls; runs locally against already-processed data.

*Provenance note:* this is an implementation of the standard Bradley-Terry-with-covariates specification and a conversation-level bootstrap, which is the established approach for arena-style pairwise preference data. It was written to that standard specification; the internal details of the individual reference notebooks (Miroyan et al.'s `preference_analysis.ipynb`, the Maximal Lotteries paper, the News Source Citing Patterns paper) were not available offline and so were not inspected. Any deliberate deviation in those papers — a different tie treatment, covariate transform, or bootstrap unit — would not be reflected here.

Coefficients are on the **log-odds** scale: a coefficient of 0.30 means a one-SD advantage in that covariate multiplies the odds of winning by `exp(0.30)` = 1.35.

## Results: citation effect by intent category

Each row is a separate BT fit on that category's decisive battles, with per-model strengths absorbing model identity and response length included as a simultaneous covariate.

| primary_intent | n | beta_citation | 95% CI | boot p | significant | beta_length | length sig |
|---|---|---|---|---|---|---|---|
| Factual Lookup | 1974 | +0.3445 | [+0.163, +0.543] | 0.0000 | **yes** | +0.4306 | yes |
| Text Processing *(low power)* | 184 | +0.6861 | [+0.033, +1.556] | 0.0380 | **yes** | +1.2970 | yes |
| Other | 451 | -0.0623 | [-0.495, +0.485] | 0.7980 | no | +0.4512 | no |
| Explanation | 901 | +0.4549 | [+0.220, +0.744] | 0.0020 | **yes** | +0.4837 | yes |
| Recommendation | 1026 | +0.4706 | [+0.299, +0.652] | 0.0000 | **yes** | +0.3981 | yes |
| Guidance | 680 | +0.1975 | [-0.023, +0.451] | 0.0840 | no | +0.4617 | yes |
| Info Synthesis | 1892 | +0.3956 | [+0.236, +0.590] | 0.0000 | **yes** | +0.8389 | yes |
| Analysis | 901 | +0.3116 | [+0.094, +0.577] | 0.0000 | **yes** | +0.8982 | yes |
| Creative Generation | 604 | +0.4210 | [+0.112, +0.736] | 0.0060 | **yes** | +0.1922 | yes |

**7 of 9** fitted categories show a citation coefficient whose 95% CI excludes zero.

## Results: cost tier

> **Identification constraint.** Cost tier is a deterministic function of model identity, so tier indicators and per-model strengths are perfectly collinear and cannot both enter one model — the tier effect would be unidentified. Tier effects are therefore estimated with **tier as the player identity** (cheap = reference, strength 0), with citation and length differences controlled in the same fit. This answers "does tier predict winning", but tier effects are *not* additionally controlled for within-tier model identity. This is a property of the design, not a modelling choice that could be avoided.

| term | coefficient | 95% CI | boot p | significant |
|---|---|---|---|---|
| tier: mid vs cheap | +0.2702 | [+0.160, +0.381] | 0.0000 | **yes** |
| tier: expensive vs cheap | +0.2919 | [+0.220, +0.372] | 0.0000 | **yes** |
| beta_citation (per 1 SD) | +0.5022 | [+0.414, +0.582] | 0.0000 | **yes** |
| beta_length (per 1 SD) | +0.7248 | [+0.562, +0.872] | 0.0000 | **yes** |

**mid vs expensive contrast (expensive − mid) = +0.0217** log-odds, 95% CI [-0.072, +0.114], p = 0.69 — **NOT significant**. The point estimate nominally favours expensive, but the interval comfortably spans zero: once citations and length are controlled, **mid and expensive are statistically indistinguishable**. The contrast is computed on the bootstrap draws, which preserves the correlation between the two tier coefficients.

The same holds per category: the mid-vs-expensive contrast is non-significant in **8 of 9** intent categories. The single exception (Creative Generation, contrast +0.353, p = 0.040) favours *expensive*, and with ten uncorrected tests one result at p ≈ 0.04 is what chance alone predicts.

Per-category tier fits are in `bradley_terry_cost_tier.csv`.

## OLS residualization vs Bradley-Terry

The two methods answer slightly different questions and are on different scales, so they are compared on **direction** and **significance**, not on coefficient magnitude. The OLS column is the model-and-length-controlled partial correlation from `final_analysis.py`; the BT column is the citation coefficient from this rebuild.

| finding | OLS estimate | OLS sig | BT estimate | BT sig | verdict |
|---|---|---|---|---|---|
| citation:Factual Lookup | +0.0554 | yes | +0.3445 | yes | agree |
| citation:Text Processing | +0.0800 | no | +0.6861 | yes | **SIGNIFICANCE CONFLICT** |
| citation:Other | +0.0239 | no | -0.0623 | no | direction differs (both null) |
| citation:Explanation | +0.0641 | no | +0.4549 | yes | **SIGNIFICANCE CONFLICT** |
| citation:Recommendation | +0.1181 | yes | +0.4706 | yes | agree |
| citation:Guidance | +0.0176 | no | +0.1975 | no | agree |
| citation:Info Synthesis | +0.0540 | yes | +0.3956 | yes | agree |
| citation:Analysis | +0.0694 | yes | +0.3116 | yes | agree |
| citation:Creative Generation | +0.0905 | yes | +0.4210 | yes | agree |
| cost_tier:cheap_vs_mid | +0.3605 | yes | +0.2702 | yes | agree |
| cost_tier:cheap_vs_expensive | +0.4280 | yes | +0.2919 | yes | agree |
| cost_tier:mid_vs_expensive | +0.5471 | yes | +0.0217 | no | **DIRECTION CONFLICT** |

**3 disagreement(s) between methods:**

- citation/Text Processing: SIGNIFICANCE CONFLICT (OLS r=+0.0800 sig=False; BT beta=+0.6861 sig=True)
- citation/Explanation: SIGNIFICANCE CONFLICT (OLS r=+0.0641 sig=False; BT beta=+0.4549 sig=True)
- cost_tier/mid_vs_expensive: DIRECTION CONFLICT

Where the two methods disagree on significance but not direction, the BT result should be preferred: it models the pairwise outcome directly and its intervals come from a conversation-level bootstrap rather than from a parametric t-test on residualized per-response rows, which treats the two responses in a battle as independent observations when they are not.

## Relation to the tau2-bench generalization test

Source: `tau2_analysis_summary.md` and `tau2_precomputed_summary.csv`, both produced offline from tau2-bench's precomputed trajectories. Nothing is recomputed here.

> **The premise of this section had to be revised by this rebuild.** The mid-beats-expensive result came from an uncontrolled head-to-head win rate (cheaper-tier win rate 0.547, p = 9.3e-05). Under the Bradley-Terry model with citation count and response length controlled, the mid-vs-expensive contrast is **+0.0217 log-odds, 95% CI [-0.072, +0.114], p = 0.69 — not significant**. The preference-side mid-over-expensive advantage does **not** survive the controls, and response length is the single largest coefficient in the model. The most likely reading is that the original tier gap was substantially a verbosity/citation artifact rather than a genuine tier effect.

With that correction, three measurements bear on the cost-tier question:

1. **Human preference (Search Arena).** Both pricier tiers beat cheap decisively. But **mid vs expensive is indistinguishable** once citations and length are controlled (contrast +0.022, p = 0.69). The uncontrolled win-rate comparison favoured mid; the controlled model does not.
2. **Objective task success (tau2-bench).** The mid-beats-expensive pattern **did not generalize**. Expensive was modestly ahead and the primary result was **not statistically significant** (mid - expensive = -0.023 pooled Pass@1, 95% CI [-0.052, +0.006], p = 0.119). Two alternative tier specifications put expensive significantly ahead; mid never won.
3. **Cost-efficiency (tau2-bench).** Success-per-dollar ranks **exactly inversely to cost tier** - the one signal with a large, unambiguous effect.

| model | tier | pooled Pass@1 | measured $/task | successes per $ |
|---|---|---|---|---|
| gpt-4.1-mini-2025-04-14 | mid | 0.541 | $0.0268 | **20.2** |
| o4-mini-2025-04-16 | mid | 0.572 | $0.0704 | **8.1** |
| gpt-4.1-2025-04-14 | expensive | 0.545 | $0.0786 | **6.9** |
| claude-3-7-sonnet-20250219 | expensive | 0.615 | $0.4385 | **1.4** |

These are **three genuinely different signals, not a confirmation or a refutation of either dataset.** The decisive reason is that **the two datasets share zero overlapping models**: Search Arena covers the gemini / sonar / gpt-4o-search families, while tau2-bench's precomputed trajectories cover gpt-4.1, gpt-4.1-mini, o4-mini and claude-3-7-sonnet. Direct same-model cross-validation is therefore impossible with the current data.

Because the model population **and** the outcome measure both change between the two datasets, a disagreement cannot be attributed to either cause alone. The tau2-bench result is a test of whether the tier-level pattern generalizes to a different population of models under an objective measure - it is not validation of the specific Search Arena models.

**Synthesis.** Read together, the defensible claim is *not* that mid-tier models outperform expensive ones. Preference data with proper controls and objective task-success data **both** fail to establish a mid-vs-expensive performance difference, from opposite directions and on disjoint model sets. What survives is the cost dimension: the expensive tier buys little or no measurable advantage over mid on either measure while costing several times more per task. The finding is about **price-performance, not raw capability ordering.**

## Limitations

1. **Ties excluded.** Only decisive battles are modelled. A tie-aware extension (Rao-Kupper / Davidson) would use the discarded ties but is not the procedure the comparison papers follow.
2. **Tier effects are not model-identity controlled** — structurally impossible; see the identification note above.
3. **Citation count is observational.** A BT covariate coefficient is not a causal effect: models that cite more differ in other ways, and per-model strengths absorb only the average model-level difference, not within-model confounding.
4. **Multiple comparisons.** Nine per-intent citation fits plus ten tier fits are reported without family-wise correction; at alpha = 0.05 roughly one apparent significant result per twenty is expected by chance.
5. **Low-power cells are flagged** in the tables and CSVs and should not be read as null results.
6. **Visible-citation proxy.** The citation measure counts visible markers in rendered text, which understates citations for models that cite without inline markers.


---

# Individual Model Strengths and Tier Boundary Sensitivity

An additive robustness check on the tier-level conclusion above. It reuses the same Bradley-Terry machinery, covariates, bootstrap procedure (1,000 replications) and seed, changing only the definition of the *player*: individual base models instead of tiers.

## Pooled individual-model strengths

One fit on all 9 base models over the full decisive set. `gemini-2.0-flash` is the reference, with strength fixed at 0; every strength is read relative to it. Differences between strengths — and so the within-tier ranges below — do not depend on that choice.

| base_model | tier | strength | 95% CI | boot p | differs from reference |
|---|---|---|---|---|---|
| `sonar-reasoning-pro` | expensive | +0.7109 | [+0.564, +0.869] | 0.0000 | **yes** |
| `sonar-pro` | expensive | +0.6392 | [+0.506, +0.782] | 0.0000 | **yes** |
| `sonar-reasoning` | mid | +0.6122 | [+0.458, +0.767] | 0.0000 | **yes** |
| `gemini-2.5-pro-exp-03-25` | expensive | +0.5773 | [+0.410, +0.748] | 0.0000 | **yes** |
| `sonar` | cheap | +0.4688 | [+0.304, +0.631] | 0.0000 | **yes** |
| `gpt-4o-search-preview` | expensive | +0.0091 | [-0.116, +0.142] | 0.9240 | no |
| `gemini-2.0-flash` | cheap | 0.0000 | *(reference)* | — | — |
| `gemini-2.5-flash-preview-04-17` | mid | -0.0552 | [-0.261, +0.151] | 0.5820 | no |
| `gpt-4o-mini-search-preview` | cheap | -0.3059 | [-0.479, -0.118] | 0.0000 | **yes** |

**6 of 8** non-reference models differ significantly from the reference.

## Within-tier vs between-tier spread

| tier | models | min | max | within-tier range |
|---|---|---|---|---|
| cheap | 3 | -0.3059 | +0.4688 | **0.7747** |
| mid | 2 | -0.0552 | +0.6122 | **0.6674** |
| expensive | 4 | +0.0091 | +0.7109 | **0.7018** |

Against the between-tier effects already estimated:

- mid vs cheap: **+0.2702**
- expensive vs cheap: **+0.2919**
- expensive vs mid: **+0.0217**

Largest within-tier range = **0.7747** log-odds. Largest between-tier effect = **0.2919** log-odds. Within-tier spread is **LARGER than between-tier spread (2.65x)**.

## Pricing boundary proximity

Prices are read directly from `MODEL_BLENDED_PRICE_USD_PER_MILLION` in `final_analysis.py`. Distance from a boundary is `|price - boundary| / boundary`, as a percentage; the flag threshold is 15%.

| base_model | blended $/Mtok | tier | % from $1.00 | % from $5.00 | flagged |
|---|---|---|---|---|---|
| `gemini-2.0-flash` | 0.25 | cheap | 75.0% | 95.0% | no |
| `gpt-4o-mini-search-preview` | 0.38 | cheap | 62.0% | 92.4% | no |
| `sonar` | 1.00 | cheap | 0.0% | 80.0% | **yes** — cheap/mid 0.0% |
| `gemini-2.5-flash-preview-04-17` | 1.40 | mid | 40.0% | 72.0% | no |
| `sonar-reasoning` | *not published* | mid | n/a | n/a | **yes** — price not published in source dict |
| `sonar-reasoning-pro` | 5.00 | expensive | 400.0% | 0.0% | **yes** — mid/expensive 0.0% |
| `gemini-2.5-pro-exp-03-25` | 5.63 | expensive | 463.0% | 12.6% | **yes** — mid/expensive 12.6% |
| `gpt-4o-search-preview` | 6.25 | expensive | 525.0% | 25.0% | no |
| `sonar-pro` | 9.00 | expensive | 800.0% | 80.0% | no |

## What this implies

**Within-tier heterogeneity undermines the tier-level conclusion.** The widest within-tier spread (0.775 log-odds) is **2.7x** the largest between-tier effect (0.292). Cost tier is therefore a weak summary of model quality in this dataset: knowing a model's tier tells you much less about its win probability than knowing which model it is. Every tier spans a range wider than the gaps between tier averages, so the three tiers overlap almost completely rather than forming a ladder.

The overlap produces outright inversions. `sonar` (cheap, +0.469) is stronger than `gpt-4o-search-preview` (expensive, +0.009), and stronger than the weaker mid-tier model `gemini-2.5-flash-preview-04-17` (-0.055). A routing rule that picked by tier would pay expensive-tier prices for a model beaten by a cheap-tier one.

**For routing decisions, individual model choice dominates tier choice.** The practical recommendation is to route to specific models on measured strength, not to a price band. Tier remains useful as a *budget* constraint, but it is not a quality proxy — the tier-level coefficients reported earlier are averages over groups too internally variable for the average to guide a routing decision.

A visible pattern in the ranking is worth flagging, because it offers a competing explanation for the tier effect: the four Perplexity Sonar models occupy four of the top five positions, while the two OpenAI search-preview models and the cheaper Gemini models sit at the bottom. Since three of the four Sonar models fall in the mid and expensive tiers, the apparent tier effect may substantially reflect **model family rather than price**. This analysis cannot separate the two — family and tier are heavily confounded in this model set — but it means the tier coefficient should not be read as a price effect.

## Tier boundary sensitivity

**Yes — Search Arena has the same boundary-proximity problem found in the tau2-bench pricing, and in a more acute form.** 4 of 9 models are flagged:

- `sonar` — $1.00, **cheap/mid 0.0%** from a tier boundary; classified **cheap**.
- `sonar-reasoning` — **no published price** in the source dictionary; assigned to the **mid** tier by assumption, not by cost.
- `sonar-reasoning-pro` — $5.00, **mid/expensive 0.0%** from a tier boundary; classified **expensive**.
- `gemini-2.5-pro-exp-03-25` — $5.63, **mid/expensive 12.6%** from a tier boundary; classified **expensive**.

Two of these sit *exactly* on a boundary value, so their tier membership is decided by the inclusive/exclusive convention alone rather than by any meaningful price difference. Either could be reclassified without changing a single price.

**The most consequential of these is `sonar-reasoning`.** It has no published price, yet its fitted strength (+0.612) makes it the strongest model in the mid tier
— the only other mid-tier model, `gemini-2.5-flash-preview-04-17`, sits at -0.055, near the bottom of the whole ranking. The mid-tier effect of +0.270 therefore rests almost entirely on a model placed in that tier by assumption. If `sonar-reasoning` belonged in the expensive tier instead, the mid tier would reduce to `gemini-2.5-flash-preview-04-17` alone and the mid-vs-cheap effect would most likely disappear or reverse.

This is a **single point of failure for the mid tier** and should be stated as such in the limitations rather than left implicit.

## Bottom line for the report

1. The tier-level findings above are **reinforced only as coarse averages** and are **undermined as a basis for routing**: within-tier variation exceeds between-tier variation by 2.7x.
2. The mid-vs-expensive null already reported is now easier to interpret: the two tiers are not separable because their *members* are not separable.
3. Tier assignment is **fragile at the boundaries** — two models sit exactly on a boundary and one has no price at all.
4. The strongest defensible claim remains **price-performance at the level of individual models**, not a tier ordering. Tier-level conclusions should carry the heterogeneity and boundary caveats wherever they appear.

