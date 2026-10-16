# Citation Support Check: Full Results (240-conversation sample + 80-conversation short_light expansion)

> **Update:** the original 240-conversation sample (120 long_heavy + 120 short_light) left the `fetched_url` row below under-sampled for `short_light` (n=95, overlapping CIs). An additional 80 short_light conversations were drawn from the same pool (bottom-quartile length+citations, `winner_citations>=1`, excluding all 240 already-processed conversations) and run through the identical, unmodified pipeline -- see `scripts/citation_support_check_expand.py` and `outputs/citation_support_check_sample_expansion.csv`. This resolved the ambiguity: `fetched_url` CIs now separate cleanly (below). All numbers in this file reflect the combined dataset.

**Verdict, stated plainly:** **Evidence consistent with John's red-herring hypothesis, within the methodologically valid comparison.** Comparing long_heavy vs short_light WITHIN the same judgment method (apples-to-apples, not the pooled number): on `proxy_no_fetch` judgments, long_heavy's supported rate (0.83) is clearly lower than short_light's (0.93), 95% CIs do not overlap. On `fetched_url` judgments the same direction holds (0.22 vs 0.38) but is also statistically clear.

## A measurement-validity issue found while reviewing these results (read this before the numbers below)

The `fetched_url` judge prompt requires the source text to **directly support** the claim. The `proxy_no_fetch` judge prompt (used when a page is blocklisted or fails the content-quality check) asks only whether the claim is **plausible** given the domain -- explicitly NOT confirmed. These are different, non-comparable standards: pooling across them means a group's pooled rate is pulled toward whichever path it happens to rely on more, independent of actual citation quality. Concretely: `real_fetched_article` support rate is 0.226 overall; `blocklisted_or_fallback` is 0.839 -- that gap is mostly the STRICTNESS difference between the two prompts, not a quality difference between fetched and fallback citations. **The pooled long_heavy-vs-short_light number below is reported for completeness but should not be read as the primary result** -- the within-method comparison is.

## Within-method comparison (the methodologically valid test)

| judgment method | long_heavy rate [95% CI] | short_light rate [95% CI] | result |
|---|---|---|---|
| `fetched_url` | 0.221 [0.202,0.240] | 0.377 [0.316,0.443] | CIs do not overlap -- long_heavy lower |
| `proxy_no_fetch` | 0.831 [0.816,0.845] | 0.926 [0.898,0.947] | CIs do not overlap -- long_heavy lower |

## Pooled comparison (reported for completeness -- see validity caveat above)

| group | n judged | n supported | supported rate | 95% Wilson CI |
|---|---|---|---|---|
| long_heavy | 4295 | 2456 | 0.572 | [0.557, 0.587] |
| short_light | 655 | 486 | 0.742 | [0.707, 0.774] |

95% CIs do not overlap. Part of this pooled gap reflects long_heavy and short_light relying on the two judgment paths in different proportions (see below), not purely a within-method quality difference.

## URL-category breakdown (requested in addition to the group comparison)

| category | n | supported rate | note |
|---|---|---|---|
| real_fetched_article | 2045 | 0.238 |  |
| blocklisted_or_fallback | 2905 | 0.845 |  |
| index_page_not_article | 178 | 0.000 | Not an LLM judgment -- a deterministic URL-shape fact. 'Supported' is trivially 0% by construction (this verdict means no support judgment was made at all, not that the citation is unsupported content). |

## Scale-level check: does the idx=18399 pattern hold across many conversations?

Across **212** conversations with enough judged pairs: correlation(fraction of citations that were real fetched articles, supported rate) = **-0.618**.

- Conversations MORE reliant on tag/aggregator/fallback citations (below-median real-article share, n=106): mean supported rate = **0.804**
- Conversations LESS reliant on them (above-median, n=106): mean supported rate = **0.493**

**This check is CONFOUNDED by the same measurement-validity issue above and should NOT be read as a clean test of the original tag/aggregator-page hypothesis.** `fetched_url` judgments are inherently stricter (lower baseline rate ~0.22) than `proxy_no_fetch` judgments (lenient, ~0.83) -- so a conversation's `frac_real_fetched` correlates with its `supported_rate` largely because of WHICH JUDGE PROMPT its citations happen to route through, not necessarily because of citation-source quality. The idx=18399 example that originally motivated this check most likely reflects the strict `fetched_url` standard being hard to satisfy from a 4,000-character excerpt, not a genuine tag-page-citation problem. **The within-method comparison above is the trustworthy result for the original length/citation-count hypothesis; this specific check does not add valid additional evidence either way.**

## Caveats

1. **`index_page_not_article` is excluded from all supported-rate denominators** -- it is a URL-shape fact, not a content judgment, and folding it into "unsupported" would conflate two different problems (citing the wrong kind of page vs. citing a page that doesn't back the claim).
2. **The URL-category comparison is observational, not causal** -- it describes what was found, not why certain conversations lean on weaker citation types.
3. **Known limitation carried over from the pilot:** math/notation-heavy pages can trip the code-symbol-density content-quality check and get routed to the weaker fallback even when genuinely real articles -- this was accepted explicitly rather than chased further, per the pilot writeup.
4. **The two judge prompts (`fetched_url` vs `proxy_no_fetch`) are not on the same scale** -- found while reviewing the full-run results, not anticipated in the pilot. This affects the pooled group comparison, the URL-category table, and invalidates the originally-planned scale-level pattern check (see above). It does NOT affect the within-method comparison, which is the result this summary's verdict is actually based on.
5. **Resolved:** `fetched_url` n for `short_light` was 95 in the original 240-sample, giving a wide, overlapping CI. After the targeted 80-conversation expansion above, n=221 and the CIs now separate cleanly (long_heavy 0.221 [0.202,0.240] vs short_light 0.377 [0.316,0.443]) -- this row is no longer a small-sample caveat.

