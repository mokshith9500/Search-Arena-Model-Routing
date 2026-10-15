# Search Arena multicollinearity diagnostics

Source: `outputs/final_analysis_derived.csv (cached; built with the fixed per-turn citation aggregation)`. Decisive battles only. Same design matrices and global citation/length scaling as the repository's Bradley–Terry fits.

This is a DIAGNOSTIC, not a model fit. No coefficients or results were overwritten.

## Design matrix rank and condition number

Columns are standardized by Euclidean norm for condition-number reporting. An infinite condition number signals rank deficiency or near-dependency at the diagnostic tolerance.

| Specification | N | Rank / columns | Condition number |
|---|---:|---:|---:|
| pooled_model_identity | 8,613 | 10 / 10 | 4.67 |
| pooled_cost_tier | 8,613 | 4 / 4 | 1.99 |
| pooled_model_family | 8,613 | 4 / 4 | 2.66 |
| pooled_tier_plus_family | 8,613 | 6 / 6 | 3.05 |
| intent_Factual Lookup | 1,974 | 10 / 10 | 5.14 |
| intent_Text Processing | 184 | 10 / 10 | 4.33 |
| intent_Other | 451 | 10 / 10 | 4.90 |
| intent_Explanation | 901 | 10 / 10 | 5.29 |
| intent_Recommendation | 1,026 | 10 / 10 | 4.87 |
| intent_Guidance | 680 | 10 / 10 | 4.76 |
| intent_Info Synthesis | 1,892 | 10 / 10 | 4.95 |
| intent_Analysis | 901 | 10 / 10 | 4.55 |
| intent_Creative Generation | 604 | 10 / 10 | 4.51 |
| frustration_direct_effect | 838 | 4 / 4 | 1.94 |
| frustration_group_frustrated | 280 | 10 / 10 | 4.37 |
| frustration_group_engaged | 558 | 10 / 10 | 4.77 |

## Predictors flagged by VIF

VIF >= 5 is a screening flag; VIF >= 10 is a stronger warning, not an automatic reason to remove a variable. Infinite VIF means (near-)perfect linear dependence. Most BT specifications have NO intercept, so their VIFs use uncentered auxiliary regressions.

| Specification | Predictor | VIF | Flag |
|---|---|---:|---|
| *No VIF values >= 5* | — | — | — |

## High pairwise Pearson correlations

| Specification | Predictor 1 | Predictor 2 | r |
|---|---|---|---:|
| *No absolute correlations >= 0.80* | — | — | — |

## Interpretation notes

- **Check the VIF for `beta_citation` and `beta_length` first.** These measure how redundant the two main behavioral predictors are *given the model identity/tier/family controls in that specific fit*.
- **Full matrix rank is necessary, not sufficient for stable estimation.** Large finite VIFs and condition numbers can still mean imprecise coefficients.
- **High correlation != causation and != guaranteed multicollinearity.** Low pairwise correlations can coexist with multivariable linear dependencies.
- **Do not put cost tier and full model-identity indicators together.** Cost tier is deterministically derived from identity, giving an exact alias.
- **Do not drop citation or length controls automatically.** If high VIFs occur, report them, compare sensitivity fits, and inspect bootstrap CIs.
- **Small subgroup fits need additional caution.** Low sample size, rare model matchups, and quasi-separation can produce wide intervals even if VIFs are low.
