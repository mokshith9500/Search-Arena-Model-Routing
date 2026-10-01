# Final Analysis Summary

Consolidated analysis for the mentor review, built on the fixed per-turn citation aggregation from `intent_complexity_analysis_v2.py`, extended with response-length (verbosity) controls, 9-category granularity, and formal significance testing.

## 1. Response Length Control

`response_length_a` / `response_length_b`: total character count of all assistant turns concatenated per side (same per-turn aggregation pattern as the fixed citation counts). Used below as a second control variable alongside base model identity, since the Search Arena paper reports a verbosity coefficient of beta=0.334 that none of our prior analysis accounted for.

## 2-3. Citation-Winning Correlation by primary_intent (9 categories, 3-way control)

Categories with fewer than 200 decisive votes are flagged as low power.

| Intent | n decisive | Raw | Model-controlled | Model+length-controlled | t-stat | p-value | Sig. at 0.05? | Length-control verdict | |
|---|---|---|---|---|---|---|---|---|---|
| Factual Lookup | 1974 | +0.145 | +0.065 | +0.055 | +2.463 | 0.0139 | Yes | survived length control - likely a real citation signal | BORDERLINE |
| Text Processing | 184 | +0.139 | +0.127 | +0.080 | +1.082 | 0.2805 | No | survived length control - likely a real citation signal | LOW POWER |
| Other | 451 | +0.077 | +0.046 | +0.024 | +0.506 | 0.6129 | No | collapsed to ~0 - mostly verbosity bias |  |
| Explanation | 901 | +0.190 | +0.076 | +0.064 | +1.927 | 0.0543 | No | survived length control - likely a real citation signal |  |
| Recommendation | 1026 | +0.193 | +0.129 | +0.118 | +3.807 | 0.0001 | Yes | survived length control - likely a real citation signal |  |
| Guidance | 680 | +0.107 | +0.034 | +0.018 | +0.458 | 0.6471 | No | collapsed to ~0 - mostly verbosity bias |  |
| Info Synthesis | 1892 | +0.172 | +0.082 | +0.054 | +2.350 | 0.0189 | Yes | survived length control - likely a real citation signal | BORDERLINE |
| Analysis | 901 | +0.176 | +0.080 | +0.069 | +2.087 | 0.0372 | Yes | survived length control - likely a real citation signal | BORDERLINE |
| Creative Generation | 604 | +0.149 | +0.104 | +0.091 | +2.230 | 0.0261 | Yes | survived length control - likely a real citation signal | BORDERLINE |

**Fix 1 - significance testing:** the model+length-controlled correlation for each category was tested against zero via a t-test, using n_decisive (records, not the doubled response-row count) as the effective sample size - the same conservative approach already used for the single-turn/multi-turn Fisher z-test below. **9 tests were run without a multiple-comparisons correction** - at alpha=0.05, roughly 0.4-0.5 false positives are expected by chance across 9 tests. p-values in [0.01, 0.05) are marked borderline and should not be treated as confirmed.

## Diagnostic: Citation Count vs Response Length Correlation

How entangled citation count and response length are - this explains why some categories' citation signal collapsed once length was controlled for.

| Scope | n decisive | Citation-Length correlation |
|---|---|---|
| All categories | 8613 | +0.412 |
| Other | 451 | +0.684 |
| Guidance | 680 | +0.316 |

## Three-Tier Cost Comparison With Significance Testing (Routing Hypothesis Test)

Replaces the earlier generic-pricing-pattern approximation with verified real blended pricing (mid-2026, USD per million tokens, input/output blended). `sonar-reasoning` has no independently published price and is placed in the mid tier as an assumption. Search-fee surcharges on `gpt-4o-mini-search-preview` ($25-30 / 1,000 calls) and `gpt-4o-search-preview` ($30-50 / 1,000 calls) are not reflected in the blended figures used for tiering, so those two models are relatively more expensive than their tier placement alone suggests.

| Tier | Models |
|---|---|
| Cheap | gemini-2.0-flash, gpt-4o-mini-search-preview, sonar |
| Mid | gemini-2.5-flash-preview-04-17, sonar-reasoning |
| Expensive | gemini-2.5-pro-exp-03-25, gpt-4o-search-preview, sonar-pro, sonar-reasoning-pro |

For each intent category, three pairwise head-to-head comparisons are run (cheap-vs-mid, cheap-vs-expensive, mid-vs-expensive). Each reports the cheaper tier's win rate and a one-sample proportion z-test of that win rate against 50% (a win rate significantly below 50% is evidence to escalate that category away from the cheaper tier). Comparisons with fewer than 100 head-to-head decisive records are flagged as low power.

| Intent | Comparison | n head-to-head | Cheaper-tier win rate | z | p-value | Sig. at 0.05? | |
|---|---|---|---|---|---|---|---|
| All categories | cheap_vs_mid | 552 | 36.1% | -6.555 | 0.0000 | Yes |  |
| All categories | cheap_vs_expensive | 2549 | 42.8% | -7.269 | 0.0000 | Yes |  |
| All categories | mid_vs_expensive | 1718 | 54.7% | +3.908 | 0.0001 | Yes |  |
| Factual Lookup | cheap_vs_mid | 125 | 37.6% | -2.773 | 0.0056 | Yes |  |
| Factual Lookup | cheap_vs_expensive | 582 | 43.1% | -3.316 | 0.0009 | Yes |  |
| Factual Lookup | mid_vs_expensive | 405 | 55.8% | +2.335 | 0.0195 | Yes | BORDERLINE |
| Text Processing | cheap_vs_mid | 8 | 12.5% | -2.121 | 0.0339 | Yes | LOW POWER |
| Text Processing | cheap_vs_expensive | 56 | 41.1% | -1.336 | 0.1814 | No | LOW POWER |
| Text Processing | mid_vs_expensive | 32 | 43.8% | -0.707 | 0.4795 | No | LOW POWER |
| Other | cheap_vs_mid | 32 | 50.0% | +0.000 | 1.0000 | No | LOW POWER |
| Other | cheap_vs_expensive | 151 | 45.7% | -1.058 | 0.2901 | No |  |
| Other | mid_vs_expensive | 93 | 60.2% | +1.970 | 0.0488 | Yes | LOW POWER |
| Explanation | cheap_vs_mid | 56 | 32.1% | -2.673 | 0.0075 | Yes | LOW POWER |
| Explanation | cheap_vs_expensive | 252 | 38.9% | -3.528 | 0.0004 | Yes |  |
| Explanation | mid_vs_expensive | 216 | 59.3% | +2.722 | 0.0065 | Yes |  |
| Recommendation | cheap_vs_mid | 45 | 42.2% | -1.043 | 0.2967 | No | LOW POWER |
| Recommendation | cheap_vs_expensive | 297 | 43.1% | -2.379 | 0.0174 | Yes | BORDERLINE |
| Recommendation | mid_vs_expensive | 196 | 55.6% | +1.571 | 0.1161 | No |  |
| Guidance | cheap_vs_mid | 60 | 38.3% | -1.807 | 0.0707 | No | LOW POWER |
| Guidance | cheap_vs_expensive | 205 | 41.5% | -2.445 | 0.0145 | Yes | BORDERLINE |
| Guidance | mid_vs_expensive | 118 | 52.5% | +0.552 | 0.5807 | No |  |
| Info Synthesis | cheap_vs_mid | 146 | 32.9% | -4.138 | 0.0000 | Yes |  |
| Info Synthesis | cheap_vs_expensive | 547 | 45.0% | -2.352 | 0.0187 | Yes | BORDERLINE |
| Info Synthesis | mid_vs_expensive | 364 | 55.5% | +2.097 | 0.0360 | Yes | BORDERLINE |
| Analysis | cheap_vs_mid | 53 | 30.2% | -2.885 | 0.0039 | Yes | LOW POWER |
| Analysis | cheap_vs_expensive | 294 | 43.2% | -2.333 | 0.0197 | Yes | BORDERLINE |
| Analysis | mid_vs_expensive | 165 | 50.3% | +0.078 | 0.9379 | No |  |
| Creative Generation | cheap_vs_mid | 27 | 40.7% | -0.962 | 0.3359 | No | LOW POWER |
| Creative Generation | cheap_vs_expensive | 165 | 38.8% | -2.880 | 0.0040 | Yes |  |
| Creative Generation | mid_vs_expensive | 129 | 46.5% | -0.792 | 0.4281 | No |  |

**Multiple comparisons:** 27 per-category tests were run (9 categories x 3 tier comparisons), plus 3 'All categories' aggregate tests shown for context but excluded from that count. No correction was applied - at alpha=0.05 across 27 tests, roughly 1.4 false positives are expected by chance alone. Treat p-values in [0.01, 0.05) as borderline.

## Manual-Review Raw Outputs (Parts A, D, E)

These need a human to actually read them - they are intentionally left as raw text files rather than summarized or auto-classified here.

- **Part A** - `zero_citation_manual_review.txt`: 75 turns with zero visible citations, sampled across all 9 intent categories, each with its underlying web_search_trace activity noted and blank labels for manual classification (failed search vs. deliberate non-citation vs. other).
- **Part D** - `recommendation_citation_examples.txt`: 20 Recommendation-category examples where the winner had at least 3 more citations than the loser, both full answers shown side by side, for judging whether Recommendation answers' natural list-of-options structure makes citation count a more genuine thoroughness signal there.
- **Part E** - `intent_manual_validation.txt`: 40 prompts sampled roughly evenly across all 9 categories with just the intent label and prompt text, for a blind human spot-check of label quality.

## 4. Single-turn vs Multi-turn x 9 Categories (9x2 table)

| Intent | Conversation type | n decisive | Raw | Model-controlled | Model+length-controlled | |
|---|---|---|---|---|---|---|
| Factual Lookup | single_turn | 1602 | +0.182 | +0.083 | +0.067 |  |
| Factual Lookup | multi_turn | 372 | +0.127 | +0.074 | +0.068 |  |
| Text Processing | single_turn | 140 | +0.154 | +0.141 | +0.124 | LOW POWER |
| Text Processing | multi_turn | 44 | +0.140 | +0.158 | +0.069 | LOW POWER |
| Other | single_turn | 329 | +0.131 | +0.082 | +0.025 |  |
| Other | multi_turn | 122 | +0.077 | +0.042 | +0.021 | LOW POWER |
| Explanation | single_turn | 737 | +0.225 | +0.102 | +0.086 |  |
| Explanation | multi_turn | 164 | +0.197 | +0.036 | +0.050 | LOW POWER |
| Recommendation | single_turn | 829 | +0.217 | +0.150 | +0.130 |  |
| Recommendation | multi_turn | 197 | +0.166 | +0.100 | +0.098 | LOW POWER |
| Guidance | single_turn | 512 | +0.115 | +0.042 | +0.024 |  |
| Guidance | multi_turn | 168 | +0.111 | +0.025 | +0.018 | LOW POWER |
| Info Synthesis | single_turn | 1509 | +0.214 | +0.123 | +0.101 |  |
| Info Synthesis | multi_turn | 383 | +0.124 | +0.035 | -0.008 |  |
| Analysis | single_turn | 737 | +0.195 | +0.107 | +0.084 |  |
| Analysis | multi_turn | 164 | +0.176 | +0.060 | +0.059 | LOW POWER |
| Creative Generation | single_turn | 504 | +0.158 | +0.117 | +0.107 |  |
| Creative Generation | multi_turn | 100 | +0.144 | +0.068 | +0.042 | LOW POWER |

## 5. Vote-Presence Analysis (intent + model identity controlled)

- Records analyzed: 24068
- Overall vote rate: 52.6%
- Raw correlation: -0.0519
- Controlled for primary_intent: -0.0689
- Controlled for primary_intent + base model: -0.0689

> **CAVEAT: even after controlling for primary_intent and base model identity, this correlation cannot distinguish between "high friction causes abandonment" and "a satisfying answer prompts a quick, casual vote" - both would produce the same pattern, and the data cannot separate them. Treat this as descriptive, not causal.**

## 6. Significance Test: Single-turn vs Multi-turn Gap (former 'High' tier)

Fisher z-test on the raw citation-winning correlation, using n_decisive (records) as the effective sample size rather than the doubled response-row count, since the two rows per record are not independent observations.

| Intent | r (single-turn) | n | r (multi-turn) | n | z | p-value | Significant at 0.05? | |
|---|---|---|---|---|---|---|---|---|
| Info Synthesis | +0.214 | 1509 | +0.124 | 383 | +1.602 | 0.1092 | No |  |
| Analysis | +0.195 | 737 | +0.176 | 164 | +0.231 | 0.8169 | No | LOW POWER SIDE |
| Creative Generation | +0.158 | 504 | +0.144 | 100 | +0.137 | 0.8912 | No | LOW POWER SIDE |

## What we can confidently say

- **Factual Lookup** (n=1974): citation-winning correlation survives model + response-length control (+0.055) AND is statistically significant against zero (t=+2.463, p=0.0139) at the corrected sample size - note: p is in the [0.01, 0.05) borderline range given 9 uncorrected tests, so treat this one a little more cautiously than the others in this list.
- **Recommendation** (n=1026): citation-winning correlation survives model + response-length control (+0.118) AND is statistically significant against zero (t=+3.807, p=0.0001) at the corrected sample size.
- **Info Synthesis** (n=1892): citation-winning correlation survives model + response-length control (+0.054) AND is statistically significant against zero (t=+2.350, p=0.0189) at the corrected sample size - note: p is in the [0.01, 0.05) borderline range given 9 uncorrected tests, so treat this one a little more cautiously than the others in this list.
- **Analysis** (n=901): citation-winning correlation survives model + response-length control (+0.069) AND is statistically significant against zero (t=+2.087, p=0.0372) at the corrected sample size - note: p is in the [0.01, 0.05) borderline range given 9 uncorrected tests, so treat this one a little more cautiously than the others in this list.
- **Creative Generation** (n=604): citation-winning correlation survives model + response-length control (+0.091) AND is statistically significant against zero (t=+2.230, p=0.0261) at the corrected sample size - note: p is in the [0.01, 0.05) borderline range given 9 uncorrected tests, so treat this one a little more cautiously than the others in this list.

- conv_id uniqueness and the citation-aggregation fix (established in `intent_complexity_analysis_v2.py`) remain solid - both were verified against real, full-scan data rather than assumed.

## What remains uncertain or exploratory

- **Other**: collapsed to ~0 - mostly verbosity bias.
- **Guidance**: collapsed to ~0 - mostly verbosity bias.
- **Text Processing**: correlation survived the length control (+0.080) but sample size is below the 200-decisive-vote threshold - not yet confirmed.
- **Explanation**: correlation survived the length control (+0.064) but p=0.0543 does not clear alpha=0.05 - not yet confirmed.
- Low-power cells in the 9x2 table: Text Processing/single_turn (n=140), Text Processing/multi_turn (n=44), Other/multi_turn (n=122), Explanation/multi_turn (n=164), Recommendation/multi_turn (n=197), Guidance/multi_turn (n=168), Analysis/multi_turn (n=164), Creative Generation/multi_turn (n=100).
- The single-turn vs multi-turn gap for **Info Synthesis** is not statistically significant (p=0.1092) - do not present this as a confirmed effect.
- The single-turn vs multi-turn gap for **Analysis** is not statistically significant (p=0.8169) - do not present this as a confirmed effect.
- The single-turn vs multi-turn gap for **Creative Generation** is not statistically significant (p=0.8912) - do not present this as a confirmed effect.
- The vote-presence finding (Step 5) is exploratory regardless of control variables: it is fundamentally correlational and the stated friction-vs-satisfaction ambiguity cannot be resolved with this data.
- **Multiple comparisons**: the 9 per-category significance tests in Section 2-3 were run with no correction for multiple comparisons. At alpha=0.05 across 9 tests, roughly 0.4-0.5 false positives are expected by chance alone even if every category's true effect were zero. Treat any single category's significance as suggestive, not proof, until replicated.
- The cost-tier comparison now uses verified real blended pricing (Part B), which is more solid than the earlier generic-pattern approximation, but two caveats remain: `sonar-reasoning`'s price isn't independently published (assumed mid-tier), and the OpenAI search-preview models' per-call search fees aren't folded into the blended figures used for tiering - so treat tier placement as approximately, not exactly, cost-ordered.
- The mid-tier beating the expensive tier in several categories (e.g. Other 60.2%, Explanation 59.3%, Info Synthesis 55.5%, and 54.7% overall - see the three-tier table) is a genuinely surprising result worth flagging on its own: it suggests 'expensive' does not mean 'better' in this data, independent of the cheap-tier routing question. Worth a follow-up look before presenting as a settled finding.
- All partial correlations here use OLS residualization (fixed effects for categorical controls, linear for response length) with Pearson correlation on the residuals - not a formal significance-tested regression model. The Step 6 single-turn/multi-turn gap and the Fix 1 by-intent tests do have real p-values attached (both using n_decisive as the effective sample size), but every other 'controlled' correlation here (Step 4's 9x2 table, Step 5's vote-presence result) is still an untested directional effect-size estimate.
