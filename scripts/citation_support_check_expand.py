"""
citation_support_check_expand.py

TARGETED EXPANSION of citation_support_check.py's short_light sample, aimed
specifically at the one weak comparison in the full-240 results: the
`fetched_url` within-method row (long_heavy 0.221 [0.202,0.240] vs
short_light 0.326 [0.240,0.426]) has overlapping CIs only because
short_light's `fetched_url` n (96) is small next to long_heavy's (1825),
even though the point estimate already points the same direction as the
`proxy_no_fetch` row (which IS statistically clear). This script does not
re-run the 240-conversation sample or touch the long_heavy side at all --
it pulls MORE short_light winners from the same population, judges them
with the exact same pipeline, and appends to the same results file.

ADDITIVE ONLY, per the task's instruction:
- Does not modify citation_support_check.py (imported unchanged below).
- Does not modify citation_support_check_sample.csv (the original 240-sample
  record) -- the new batch's sample metadata is saved to a SEPARATE file,
  OUT_SAMPLE_EXPANSION_CSV.
- Appends new judgment rows to the EXISTING outputs/citation_support_check.csv
  (same file run_full_sample already appends to across resumed runs), rather
  than creating a parallel results file -- so the final analysis draws from
  one combined, consistent dataset, per Step 3 of the task.

STEP 1 FINDING (see select_additional_sample docstring for the full number):
short_light's low `fetched_url` COUNT is not because its citations are less
likely to be real, fetchable articles than long_heavy's (33.2% vs 41.2% of
cited claims route to a real fetch -- comparable, if anything slightly
LOWER-strict for short_light). It's because short_light conversations have
far fewer citations per conversation BY DESIGN (bottom quartile on citation
count) -- 4.59 cited claims/conversation on average vs long_heavy's 36.91.
Worse: 63/120 (52.5%) of the already-used short_light sample had winner_
citations==0 in the official field and contributed ZERO cited rows at all.
So expanding the sample will produce more fetched_url judgments (the
fetch-eligible rate is fine), but only efficiently if the new batch avoids
re-drawing a large fraction of structurally-zero-citation conversations.
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from citation_support_check import (  # noqa: E402  (after sys.path insert)
    DERIVED_CSV, OUT_CSV, OUT_SAMPLE_CSV, OUT_MD, TRAIN_JSONL, RANDOM_SEED,
    process_one_conversation, load_already_done, is_daily_quota,
    extract_claims, analyze_full_results, write_full_results_summary,
)

OUT_SAMPLE_EXPANSION_CSV = "outputs/citation_support_check_sample_expansion.csv"

# A seed distinct from the original RANDOM_SEED -- not load-bearing for
# avoiding overlap (overlap is prevented explicitly, by excluding already-used
# idx below), just kept distinct so this batch's draw is independently
# reproducible from the original 240-sample's draw.
EXPANSION_SEED = RANDOM_SEED + 1

# STEP 1 EFFICIENCY FILTER: restrict the new batch to short_light pool
# conversations with winner_citations >= 1. Conversations with
# winner_citations == 0 are still legitimately "short_light" by the bucket
# definition (bottom quartile on length AND citation count), but they
# mathematically cannot contribute a single cited claim, let alone a
# fetched_url judgment -- the original 120-sample included 63 of them (52.5%)
# and they returned a mean of 0.22 fetched_url judgments/conversation vs 1.44
# for citations>=1 conversations. Excluding them from THIS expansion batch
# does not bias the support-rate result: a citations==0 conversation
# contributes zero citation-level rows to the analysis either way, so
# skipping it changes which conversations get SAMPLED (for efficiency), not
# which citations get judged or how they're judged.
MIN_WINNER_CITATIONS = 1

# From Step 1's measured rate: citations>=1 short_light conversations
# produced 82 fetched_url judgments across 57 conversations in the original
# sample = 1.439/conversation. Target is to go from 96 to 200+ fetched_url
# judgments (+104), needing ~73 conversations at that rate; a 25% buffer
# brings target_n to ~92. Set here explicitly rather than computed at
# import time, so a re-run with a different target is a one-line change.
TARGET_ADDITIONAL_CONVERSATIONS = 92


def select_additional_sample(target_n=TARGET_ADDITIONAL_CONVERSATIONS,
                              min_citations=MIN_WINNER_CITATIONS,
                              seed=EXPANSION_SEED):
    print("=" * 78)
    print("STEP 2  Select additional short_light winners (new batch, not in the "
          "existing 240-sample)")
    print("=" * 78)
    df = pd.read_csv(DERIVED_CSV)
    dec = df[df["is_decisive"] == True].copy()  # noqa: E712

    is_a = dec["winner"] == "model_a"
    dec["winner_side"] = np.where(is_a, "a", "b")
    dec["winner_model"] = np.where(is_a, dec["base_model_a"], dec["base_model_b"])
    dec["winner_length"] = np.where(is_a, dec["response_length_a"], dec["response_length_b"])
    dec["winner_citations"] = np.where(is_a, dec["citations_a"], dec["citations_b"])

    q75_len = dec["winner_length"].quantile(0.75)
    q25_len = dec["winner_length"].quantile(0.25)
    q75_cit = dec["winner_citations"].quantile(0.75)
    q25_cit = dec["winner_citations"].quantile(0.25)
    # Same stratified population definition as citation_support_check.py's
    # select_sample() -- bottom quartile on both length and citation count
    # among decisive-battle winners. Not re-derived differently.
    short_light = dec[(dec["winner_length"] <= q25_len) & (dec["winner_citations"] <= q25_cit)]
    print(f"short_light pool (full population, same definition as original script): "
          f"{len(short_light)}")

    existing_sample = pd.read_csv(OUT_SAMPLE_CSV, index_col=0)
    already_used = set(existing_sample[existing_sample["group"] == "short_light"].index)
    print(f"already in the existing 240-sample: {len(already_used)}")

    pool = short_light[~short_light.index.isin(already_used)]
    print(f"remaining after excluding already-processed: {len(pool)}")

    pool_eligible = pool[pool["winner_citations"] >= min_citations]
    print(f"remaining with winner_citations >= {min_citations} "
          f"(the efficiency filter -- see module docstring): {len(pool_eligible)}")

    n = min(target_n, len(pool_eligible))
    sample = pool_eligible.sample(n, random_state=seed)
    print(f"selected: {n} (target was {target_n}"
          f"{', fewer than target available!' if n < target_n else ''})")

    sample = sample.assign(group="short_light")
    cols = ["group", "winner_side", "winner_model", "winner_length",
            "winner_citations", "primary_intent", "winner"]
    sample = sample[cols]
    sample.to_csv(OUT_SAMPLE_EXPANSION_CSV)
    print(f"[saved] {OUT_SAMPLE_EXPANSION_CSV} (new file -- original "
          f"{OUT_SAMPLE_CSV} is untouched)")
    return sample


def run_batch(api_key, sample, out_csv=OUT_CSV, stop_after=None):
    """
    Same incremental-append, resumable structure as citation_support_check.py's
    run_full_sample -- reusing process_one_conversation and load_already_done
    UNCHANGED (imported, not reimplemented). `stop_after` lets this be called
    first for a small real-timed slice (the Step 5 cost/time check) without
    committing to the whole batch in one call; calling it again with the same
    `sample` and no stop_after resumes seamlessly, since already-written idx
    are skipped via load_already_done exactly as in the original script.
    """
    already_done = load_already_done(out_csv)
    todo = sample[~sample.index.isin(already_done)]
    todo_idx = set(todo.index)
    if not todo_idx:
        print("Nothing left to do in this batch -- all idx already in", out_csv)
        return dict(n_done=0, elapsed_s=0.0, total_tokens=0, aborted=False,
                    per_conv_times=[], per_conv_cited=[])

    write_header = not os.path.exists(out_csv)
    t0 = time.time()
    total_tokens = 0
    n_done = 0
    aborted = False
    url_cache_errors = {}
    per_conv_times = []
    per_conv_cited = []

    with open(TRAIN_JSONL, encoding="utf-8") as f, \
         open(out_csv, "a", encoding="utf-8", newline="") as out_f:
        for i, line in enumerate(f):
            if i not in todo_idx:
                continue
            r = json.loads(line)
            row = todo.loc[i]
            t_conv0 = time.time()
            pair_rows, tok, err, hit_quota = process_one_conversation(
                api_key, i, row, r, url_cache_errors)
            dt = time.time() - t_conv0
            per_conv_times.append(dt)
            n_cited = sum(1 for p in pair_rows if p.get("marker") is not None)
            per_conv_cited.append(n_cited)
            total_tokens += tok

            if pair_rows:
                pd.DataFrame(pair_rows).to_csv(out_f, header=write_header, index=False)
                write_header = False
                out_f.flush()

            n_done += 1
            print(f"  [{n_done}/{len(todo_idx)}] idx={i:6} cited_pairs={n_cited:3} "
                  f"tokens={tok:5} time={dt:5.1f}s")

            if hit_quota:
                print("DAILY QUOTA WALL hit -- stopping this batch now, "
                      f"{n_done} conversations written to {out_csv} so far.")
                aborted = True
                break
            if stop_after and n_done >= stop_after:
                print(f"\nStopping after {stop_after} conversations (mini-pilot slice) "
                      f"as requested -- {len(todo_idx) - n_done} remain in this batch "
                      "for a later call.")
                break

    elapsed = time.time() - t0
    return dict(n_done=n_done, elapsed_s=elapsed, total_tokens=total_tokens,
                aborted=aborted, per_conv_times=per_conv_times,
                per_conv_cited=per_conv_cited, n_remaining_in_batch=len(todo_idx) - n_done)


def report_cost_time(stats, remaining_target, label):
    print("\n" + "=" * 78)
    print(f"STEP 5  Cost/time report -- {label}")
    print("=" * 78)
    n = stats["n_done"]
    if n == 0:
        print("No conversations processed in this slice -- nothing to extrapolate from.")
        return
    elapsed = stats["elapsed_s"]
    tokens = stats["total_tokens"]
    print(f"processed: {n} conversations, {elapsed:.1f}s total "
          f"({elapsed/n:.1f}s/conversation avg), {tokens:,} tokens "
          f"({tokens/n:.0f} tokens/conversation avg)")
    n_cited_total = sum(stats["per_conv_cited"])
    print(f"cited claim-pairs produced: {n_cited_total} ({n_cited_total/n:.2f}/conversation avg)")

    est_time_s = elapsed / n * remaining_target
    est_tokens = tokens / n * remaining_target
    print(f"\nEXTRAPOLATED to the remaining {remaining_target} conversations in this batch:")
    print(f"  estimated time:   {est_time_s/60:.1f} minutes")
    print(f"  estimated tokens: {est_tokens:,.0f}")
    print("  (Gemini free-tier per-day request-count quotas, not just tokens, are the "
          "more likely constraint -- same caveat as the original script's pilot. If a "
          "DAILY_QUOTA wall is hit mid-batch, this is resumable: re-run with the same "
          "sample and it skips everything already written to outputs/"
          "citation_support_check.csv.)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini-pilot", type=int, default=0,
                    help="Process only this many NEW conversations first, for a "
                         "real cost/time read before committing to the rest.")
    ap.add_argument("--full", action="store_true",
                    help="Process the entire expansion batch (resumable).")
    ap.add_argument("--target-n", type=int, default=TARGET_ADDITIONAL_CONVERSATIONS)
    ap.add_argument("--recompute-only", action="store_true",
                    help="Skip selection/running entirely -- just re-run Step 4's "
                         "analysis + summary over whatever is currently in "
                         "outputs/citation_support_check.csv.")
    args = ap.parse_args()

    if args.recompute_only:
        pairs_df = pd.read_csv(OUT_CSV)
        analysis = analyze_full_results(pairs_df)
        write_full_results_summary(OUT_MD, dict(n_done=pairs_df["idx"].nunique(),
                                                 n_total=pairs_df["idx"].nunique()),
                                   analysis)
        sys.exit(0)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set -- cannot run Step 2/3 (selection needs no "
              "API key, but is skipped here since nothing can follow it). Set the key "
              "and re-run.")
        sys.exit(2)

    sample = select_additional_sample(target_n=args.target_n)

    if args.mini_pilot:
        stats = run_batch(api_key, sample, stop_after=args.mini_pilot)
        remaining = stats.get("n_remaining_in_batch", len(sample) - stats["n_done"])
        report_cost_time(stats, remaining, f"mini-pilot of {args.mini_pilot}")
        if not stats["aborted"]:
            print(f"\nNot running the remaining {remaining} conversations yet -- "
                  "per Step 5, confirm before committing to the rest.")
        sys.exit(0)

    if args.full:
        stats = run_batch(api_key, sample)
        print(f"\nBatch complete: {stats['n_done']} conversations processed, "
              f"{stats['elapsed_s']/60:.1f} min, {stats['total_tokens']:,} tokens, "
              f"aborted={stats['aborted']}")
        if stats["aborted"]:
            sys.exit(3)
        pairs_df = pd.read_csv(OUT_CSV)
        analysis = analyze_full_results(pairs_df)
        write_full_results_summary(OUT_MD, dict(n_done=pairs_df["idx"].nunique(),
                                                 n_total=pairs_df["idx"].nunique()),
                                   analysis)
        print(f"\nRecomputed {OUT_MD} over the combined dataset.")
        sys.exit(0)

    print("Sample selected. Pass --mini-pilot N for a real cost/time read, "
          "--full to run the whole batch, or --recompute-only to just refresh "
          "the analysis over the current outputs/citation_support_check.csv.")
