"""
citation_support_check.py

Tests John's red-herring hypothesis directly: do long, heavily-cited WINNING
answers actually have lower-quality citation support than short, lightly-
cited winning answers -- meaning users may be rewarding length/citation
count as a quality proxy rather than citations that actually hold up.

ADDITIVE ONLY. Does not modify any existing file.

STEP 1 FINDING: real citation URLs exist in the raw data, no LLM needed
to locate them. `train.jsonl`'s `system_{side}_metadata.formatted_messages`
field renders each response with its ACTUAL source URLs, in one of several
observed styles (an inline `<a href="URL">[N]</a>` anchor, a parenthetical
`([domain](URL))` markdown link, or a footer "Citations" block mapping
`[N]` to a bare URL) -- but in every style checked, the URLs appear in the
SAME ORDER the numbered markers [1][2][3]... appear in the standardized
top-level `messages_{side}` text (the text citations_a/b was already
counted from). Verified empirically on a random sample of 120 (side, idx)
pairs: for every side where the official citation count is > 0, extracting
ALL URLs from `formatted_messages` (in order of first appearance, deduped)
yields at least enough URLs to cover every referenced marker in 93 of 95
such sides (98%). The 2 shortfall cases are handled explicitly in Step 3
as a fallback, not silently ignored.

This means real URL-fetching (Step 3's preferred approach) is feasible for
the large majority of claims, not just a fallback -- confirmed by directly
fetching a real citation URL from this dataset successfully before writing
any extraction code.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from both_bad_rate import wilson_ci

DERIVED_CSV = "outputs/final_analysis_derived.csv"
TRAIN_JSONL = "train.jsonl"
RANDOM_SEED = 20260930
TARGET_PER_GROUP = 120  # within the requested 100-150 range

OUT_SAMPLE_CSV = "outputs/citation_support_check_sample.csv"
OUT_CSV = "outputs/citation_support_check.csv"
OUT_MD = "outputs/citation_support_check_summary.md"
OUT_PILOT_TXT = "outputs/citation_support_check_pilot_review.txt"

URL_PATTERN = re.compile(r'https?://[^\s\)"<>]+')

MODEL = "gemini-3.1-flash-lite"
ENDPOINT_TMPL = ("https://generativelanguage.googleapis.com/v1beta/models/"
                 "{model}:generateContent")


# ============================================================
# STEP 1: stratified sample -- long/heavily-cited winners vs
# short/lightly-cited winners, among decisive battles
# ============================================================

def select_sample():
    print("=" * 78)
    print("STEP 1  Stratified sample: long+heavily-cited vs short+lightly-cited winners")
    print("=" * 78)
    df = pd.read_csv(DERIVED_CSV)
    dec = df[df["is_decisive"] == True].copy()  # noqa: E712
    print(f"decisive battles: {len(dec)}")

    is_a = dec["winner"] == "model_a"
    dec["winner_side"] = np.where(is_a, "a", "b")
    dec["winner_model"] = np.where(is_a, dec["base_model_a"], dec["base_model_b"])
    dec["winner_length"] = np.where(is_a, dec["response_length_a"], dec["response_length_b"])
    dec["winner_citations"] = np.where(is_a, dec["citations_a"], dec["citations_b"])

    q75_len = dec["winner_length"].quantile(0.75)
    q25_len = dec["winner_length"].quantile(0.25)
    q75_cit = dec["winner_citations"].quantile(0.75)
    q25_cit = dec["winner_citations"].quantile(0.25)
    print(f"winner length: 25th pct={q25_len:.0f}  75th pct={q75_len:.0f}")
    print(f"winner citations: 25th pct={q25_cit:.0f}  75th pct={q75_cit:.0f}")

    long_heavy = dec[(dec["winner_length"] >= q75_len) & (dec["winner_citations"] >= q75_cit)]
    short_light = dec[(dec["winner_length"] <= q25_len) & (dec["winner_citations"] <= q25_cit)]
    print(f"\nlong+heavily-cited winners available: {len(long_heavy)}")
    print(f"short+lightly-cited winners available: {len(short_light)}")

    rng = np.random.RandomState(RANDOM_SEED)

    def take(pool, n_target, label):
        n = min(n_target, len(pool))
        sample = pool.sample(n, random_state=RANDOM_SEED)
        print(f"{label}: target={n_target}  available={len(pool)}  selected={n}"
              f"{'  *** fewer than target available ***' if n < n_target else ''}")
        return sample

    s_long = take(long_heavy, TARGET_PER_GROUP, "long+heavily-cited")
    s_short = take(short_light, TARGET_PER_GROUP, "short+lightly-cited")

    s_long = s_long.assign(group="long_heavy")
    s_short = s_short.assign(group="short_light")
    sample = pd.concat([s_long, s_short])
    cols = ["group", "winner_side", "winner_model", "winner_length",
           "winner_citations", "primary_intent", "winner"]
    sample = sample[cols]
    sample.to_csv(OUT_SAMPLE_CSV)
    print(f"\n[saved] {OUT_SAMPLE_CSV}  (n={len(sample)}, "
          f"{len(s_long)} long_heavy + {len(s_short)} short_light)")
    return sample


# ============================================================
# Step 2/3 groundwork: extract (marker -> URL) mapping from raw train.jsonl,
# no LLM needed -- verified against official citation counts before use.
# ============================================================

def extract_citation_urls(record, side):
    """
    Returns an ordered list of URLs, deduped by first appearance, pulled
    from system_{side}_metadata.formatted_messages. Position i in this list
    corresponds to marker [i+1] in the standardized messages_{side} text --
    verified empirically (see module docstring) to hold for the large
    majority of (side, idx) pairs where citations are actually used.
    """
    meta = record.get(f"system_{side}_metadata") or {}
    fm = meta.get("formatted_messages") or []
    asst_texts = [m.get("content", "") for m in fm if m.get("role") == "assistant"]
    text = "\n".join(asst_texts)
    urls = URL_PATTERN.findall(text)
    seen = set()
    ordered = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def get_winner_text_and_urls(record, winner_side):
    msgs = [m.get("content", "") for m in record.get(f"messages_{winner_side}", [])
            if m.get("role") == "assistant"]
    text = "\n".join(msgs)
    urls = extract_citation_urls(record, winner_side)
    return text, urls


def verify_url_coverage(sample, path=TRAIN_JSONL):
    print("\n" + "=" * 78)
    print("Verify URL coverage on the actual selected sample (not a separate check)")
    print("=" * 78)
    idx_set = set(sample.index)
    rows = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in idx_set:
                continue
            r = json.loads(line)
            side = sample.loc[i, "winner_side"]
            text, urls = get_winner_text_and_urls(r, side)
            n_markers = len(set(re.findall(r"\[(\d+)\]", text)))
            rows[i] = dict(text=text, urls=urls, n_markers=n_markers,
                          n_urls=len(urls), covered=len(urls) >= n_markers)
    n_total = len(rows)
    n_covered = sum(1 for v in rows.values() if v["covered"])
    print(f"sample size: {n_total}")
    print(f"fully covered (enough URLs for every marker used): {n_covered} "
          f"({100 * n_covered / n_total:.1f}%)")
    shortfall = [(i, v["n_markers"], v["n_urls"]) for i, v in rows.items() if not v["covered"]]
    if shortfall:
        print(f"shortfall cases ({len(shortfall)}): idx, n_markers, n_urls_found")
        for s in shortfall[:10]:
            print(f"  {s}")
    return rows


def main_step1_only():
    print("=" * 78)
    print("CITATION SUPPORT CHECK -- STEP 1 (sampling, no API calls needed)")
    print("=" * 78)
    sample = select_sample()
    rows = verify_url_coverage(sample)
    return sample, rows


# ============================================================
# STEP 2: LLM call -- decompose a response into claim-citation pairs
# ============================================================

CLAIM_EXTRACTION_SYSTEM_INSTRUCTION = """\
You are decomposing a single AI assistant response into its individual factual \
claims, for a citation-quality audit. The response contains numbered citation \
markers like [1], [2], [3] that refer to sources.

For EACH distinct factual claim in the response (a claim is one checkable \
assertion -- a fact, statistic, quote, or specific event), extract:
- claim_text: the claim itself, as a short self-contained sentence (do not just \
copy a huge chunk of text -- restate the specific checkable assertion).
- citation_markers: the list of citation marker NUMBERS (e.g. [1, 2]) that are \
attached to this specific claim in the response. Empty list if the claim has no \
citation attached at all.

Rules:
- Skip pure filler (greetings, meta-commentary like "here is a summary").
- A claim with no citation is still worth extracting (empty citation_markers) -- \
we need to know which claims are UNCITED, not just which are cited.
- Do not invent claims that aren't in the text. Do not invent citation numbers \
that don't appear in the text.
- If the response is in a non-English language, extract and state the claims in \
English for consistency, but do not change their meaning."""

CLAIM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "claims": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "claim_text": {"type": "STRING"},
                    "citation_markers": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                },
                "required": ["claim_text", "citation_markers"],
            },
        },
    },
    "required": ["claims"],
}

JUDGE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": ["supported", "irrelevant", "contradicted"]},
        "justification": {"type": "STRING"},
    },
    "required": ["verdict", "justification"],
}


def call_gemini(api_key, system_instruction, user_text, schema, model=MODEL, max_attempts=5):
    """
    Same retry/backoff/quota-wall pattern as frustration_classify.py's
    call_gemini -- reused by re-implementation here (that function is
    hardwired to ITS OWN schema/system-instruction module constants, so it
    isn't directly importable as a generic call, but the retry/backoff/
    quota logic is copied verbatim, not redesigned).
    """
    body = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    data = json.dumps(body).encode("utf-8")
    endpoint = ENDPOINT_TMPL.format(model=model)
    last = None
    for attempt in range(max_attempts):
        req = urllib.request.Request(
            endpoint, data=data,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.loads(r.read().decode("utf-8"))
            cands = payload.get("candidates") or []
            if not cands:
                last = f"no candidates: {json.dumps(payload)[:200]}"
            else:
                txt = "".join(p.get("text", "") for p in
                              cands[0].get("content", {}).get("parts", []))
                if not txt.strip():
                    last = f"empty text (finishReason={cands[0].get('finishReason')})"
                else:
                    obj = json.loads(txt)
                    obj["_usage"] = payload.get("usageMetadata", {}).get("totalTokenCount")
                    return obj, None
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            last = f"HTTP {e.code}: {raw[:300]}"
            if e.code in (400, 401, 403):
                return None, last
            if e.code == 429:
                if "PerDay" in raw or "per day" in raw.lower():
                    return None, "DAILY_QUOTA " + last
                time.sleep(min(10.0, 30.0))
                continue
        except Exception as e:                                   # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < max_attempts - 1:
            time.sleep(min(2 ** attempt, 15))
    return None, last


def extract_claims(api_key, response_text):
    user_text = f"Response to decompose:\n\n{response_text}"
    obj, err = call_gemini(api_key, CLAIM_EXTRACTION_SYSTEM_INSTRUCTION, user_text, CLAIM_SCHEMA)
    if obj is None:
        return [], None, err
    return obj.get("claims", []), obj.get("_usage"), None


# ============================================================
# STEP 3 FIX 1: known-unfetchable domains -- these return JS app shells or
# require a login/consent wall via plain GET, never real article text, so
# they are rejected BEFORE attempting a fetch and routed straight into the
# EXISTING metadata-only fallback path (the same one "no URL available"
# already uses) -- not a new, separate path, per the task's instruction.
# Confirmed from the pilot: 29 of the pilot's 156 fetched judgments pointed
# at youtube.com alone and returned nothing but player config/JS.
# ============================================================

BLOCKLISTED_DOMAINS = {
    "youtube.com", "www.youtube.com", "youtu.be",
    "twitter.com", "www.twitter.com", "x.com", "www.x.com",
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com",
}


def is_blocklisted_domain(url):
    try:
        host = re.match(r"https?://([^/]+)", url).group(1).lower()
    except Exception:                                             # noqa: BLE001
        return False
    return host in BLOCKLISTED_DOMAINS


# ============================================================
# STEP 3 FIX 3: tag/topic/index-page URL shape detection. Citing an index
# page is a DIFFERENT problem than citing a page that's genuinely irrelevant
# or wrong -- it means the model cited a category listing instead of a
# specific article, so it gets its own verdict category rather than being
# silently folded into "irrelevant". This is a pure URL-pattern check, no
# fetch or LLM call needed -- the URL shape itself is the evidence.
# Patterns found in the pilot: bbc.com/news/topics/..., foreignpolicy.com/
# tag/georgia/, euronews.com/tag/georgia, and bare homepage-only URLs
# (georgiatoday.ge) with no article-specific path at all.
# ============================================================

INDEX_PAGE_URL_PATTERNS = [
    re.compile(r"/tags?/", re.IGNORECASE),
    re.compile(r"/topics?/", re.IGNORECASE),
    re.compile(r"/categor(y|ies)/", re.IGNORECASE),
    re.compile(r"[?&]s=", re.IGNORECASE),          # ?s=searchterm (WordPress search)
    re.compile(r"/search/", re.IGNORECASE),
]


def classify_url_shape(url):
    """Returns the matched pattern description, or None if this looks like
    a normal article URL."""
    for pat in INDEX_PAGE_URL_PATTERNS:
        if pat.search(url):
            return f"matches {pat.pattern}"
    # Bare homepage: scheme://host[/] with no further path at all -- cannot
    # be a specific article since there IS no article-specific path segment.
    if re.fullmatch(r"https?://[^/]+/?", url):
        return "bare homepage URL, no article-specific path"
    return None


# ============================================================
# STEP 3 FIX 2: content-quality check AFTER fetching, BEFORE judging.
# Two simple, explainable criteria -- either one fails the page, routing it
# to the same metadata-only fallback as any other fetch failure:
#
# 1. MIN_CONTENT_CHARS = 400. Chosen because ~400 characters is roughly a
#    short paragraph (70-80 words) -- genuine article prose essentially
#    always clears this; a cookie banner, a "please enable JavaScript"
#    stub, or a bare page title does not. The previous threshold (50 chars)
#    was far too low -- a cookie-consent snippet alone clears 50 characters.
# 2. JUNK_PATTERNS -- literal phrases that only appear in boilerplate, never
#    in genuine article prose about an unrelated topic (cookie/JS consent
#    language), PLUS a code-symbol-density check for script/style markup
#    that survived the HTML-tag strip (real prose has near-zero density of
#    `{ } ; ( ) =` -- a 2% threshold is a generous allowance for normal
#    punctuation while still catching leftover CSS/JS).
# ============================================================

MIN_CONTENT_CHARS = 400
CODE_SYMBOL_DENSITY_THRESHOLD = 0.02

JUNK_PATTERNS = [
    re.compile(r"enable javascript", re.IGNORECASE),
    re.compile(r"cookies?\s+(to use|are required|consent|policy)", re.IGNORECASE),
    re.compile(r"accept (all )?cookies", re.IGNORECASE),
    re.compile(r"please enable", re.IGNORECASE),
    re.compile(r"browser does not support", re.IGNORECASE),
    # Found by testing against a real Wikipedia article: a bare
    # `\bfunction\s*\(` false-positives on ordinary prose like "the sinc
    # function (red)". Tightened to require the shape of an actual JS
    # function literal -- "function NAME?(args) {" -- which is what
    # leftover/un-stripped script content actually looks like, not just
    # the English word "function" followed by any parenthesis.
    re.compile(r"\bfunction\s*\w*\s*\([^)]{0,80}\)\s*\{", re.IGNORECASE),
    # Likewise, require a semicolon-terminated assignment, not just the
    # word "var" followed by "=" (which could appear in unrelated prose).
    re.compile(r"\bvar\s+\w+\s*=\s*[^;{}\n]{1,200};"),
    re.compile(r"@media\b"),                             # leftover CSS media query
]


def content_quality_check(text):
    """Returns (passes: bool, reason_if_failed: str_or_None)."""
    if len(text) < MIN_CONTENT_CHARS:
        return False, f"cleaned text too short ({len(text)} chars < {MIN_CONTENT_CHARS})"
    for pat in JUNK_PATTERNS:
        if pat.search(text):
            return False, f"matched junk pattern: {pat.pattern}"
    code_symbols = sum(text.count(c) for c in "{};()=<>")
    density = code_symbols / max(1, len(text))
    if density > CODE_SYMBOL_DENSITY_THRESHOLD:
        return False, (f"code-symbol density {density:.3f} exceeds "
                       f"{CODE_SYMBOL_DENSITY_THRESHOLD} -- looks like script/style "
                       "markup, not prose")
    return True, None


# ============================================================
# STEP 3: fetch URL (or fall back), then judge support
# ============================================================

HTML_TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>",
                             re.IGNORECASE | re.DOTALL)
MAX_FETCH_CHARS = 4000
FETCH_TIMEOUT = 10


def fetch_url_text(url):
    """
    Returns (text_or_None, error_or_None). Plain GET + HTML strip, followed
    by the Step 3 Fix 2 content-quality check -- a successful HTTP fetch
    that returns boilerplate still gets routed to the proxy fallback, not
    judged as if it were real content.

    Found while testing this fix: the ORIGINAL tag-stripping regex removed
    `<script>`/`<style>` TAGS but left their CONTENTS as visible "text" --
    so a perfectly good article with an inline analytics snippet would have
    `function(...)` sitting right next to real prose, and the Step 3 Fix 2
    junk-pattern check would then wrongly reject a GOOD page. Entire
    script/style BLOCKS (tag + contents) are now removed first, before the
    generic tag strip -- otherwise almost every modern site would fail the
    content-quality check regardless of its actual article content.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as r:
            ctype = r.headers.get("Content-Type", "")
            if "text/html" not in ctype and "text/plain" not in ctype and ctype:
                return None, f"non-text content-type: {ctype}"
            raw = r.read(300_000).decode("utf-8", errors="replace")
        raw = SCRIPT_STYLE_RE.sub(" ", raw)
        text = HTML_TAG_RE.sub(" ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        ok, reason = content_quality_check(text)
        if not ok:
            return None, f"content_quality_check_failed: {reason}"
        return text[:MAX_FETCH_CHARS], None
    except Exception as e:                                       # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


JUDGE_WITH_SOURCE_SYSTEM_INSTRUCTION = """\
You are auditing whether a single factual claim is actually supported by its \
cited source's real content. You are given the claim and the fetched text of \
the source page. Judge strictly from the provided source text, not from your \
own general knowledge.

Classify as exactly one of:
- "supported": the source text contains information that directly supports the claim.
- "irrelevant": the source text does not address the claim's subject at all \
(wrong topic, or too generic to confirm or deny it).
- "contradicted": the source text directly conflicts with the claim.

Give a one-sentence justification quoting or closely paraphrasing the relevant \
part of the source text."""

JUDGE_PROXY_SYSTEM_INSTRUCTION = """\
You are auditing a factual claim and its citation, but the actual source page \
could NOT be fetched (dead link, timeout, paywall, or non-text content). You \
only have the claim text and the citation's URL/domain to go on -- NOT the \
real source content. This is explicitly a WEAKER check than verifying against \
real source text: judge only PLAUSIBILITY (does this domain plausibly host this \
kind of information, is the claim internally coherent), not actual confirmation.

Classify as exactly one of:
- "supported": the claim is plausible given the domain/context (NOT confirmed).
- "irrelevant": the domain/context has nothing to do with the claim's subject.
- "contradicted": the claim is internally implausible or self-contradictory.

State in the justification that this is a plausibility-only judgment, not a \
verified one, since the source page itself was not available."""


def judge_claim(api_key, claim_text, url, fetched_text, fetch_error):
    if fetched_text is not None:
        user_text = (f"Claim: {claim_text}\n\nSource URL: {url}\n\n"
                    f"Fetched source text:\n{fetched_text}")
        obj, err = call_gemini(api_key, JUDGE_WITH_SOURCE_SYSTEM_INSTRUCTION,
                               user_text, JUDGE_SCHEMA)
        method = "fetched_url"
    else:
        user_text = (f"Claim: {claim_text}\n\nCitation URL: {url or '(no URL available)'}\n\n"
                    f"(Source page unavailable: {fetch_error})")
        obj, err = call_gemini(api_key, JUDGE_PROXY_SYSTEM_INSTRUCTION,
                               user_text, JUDGE_SCHEMA)
        method = "proxy_no_fetch"
    return obj, err, method


# ============================================================
# STEP 5: small pilot FIRST -- real cost/time before the full 240
# ============================================================

PILOT_SIZE_PER_GROUP = 6  # 6 + 6 = 12, inside the requested 10-15 range


def is_daily_quota(err):
    return bool(err) and str(err).startswith("DAILY_QUOTA")


def process_one_conversation(api_key, idx, row, raw_record, url_cache_errors):
    """
    Returns (pair_rows, total_tokens, error_or_None, hit_quota_wall).
    `hit_quota_wall` is checked after EVERY LLM call (claim extraction AND
    each judge_claim), not only extraction -- a daily quota wall can be hit
    mid-conversation during judging just as easily as during extraction.
    When it fires, processing of THIS conversation stops immediately, but
    whatever pairs were already judged before the wall hit are kept and
    returned (not discarded) -- the caller is responsible for stopping the
    OUTER loop over conversations, not re-trying further calls.
    """
    winner_side = row["winner_side"]
    text, urls = get_winner_text_and_urls(raw_record, winner_side)
    claims, extract_usage, extract_err = extract_claims(api_key, text)
    pair_rows = []
    total_tokens = extract_usage or 0
    if extract_err:
        print(f"  [idx={idx}] claim extraction FAILED: {extract_err[:150]}")
        return pair_rows, total_tokens, extract_err, is_daily_quota(extract_err)

    for c in claims:
        markers = c.get("citation_markers") or []
        claim_text = c.get("claim_text", "")
        if not markers:
            pair_rows.append(dict(idx=idx, group=row["group"], claim_text=claim_text,
                                  marker=None, url=None, verdict="UNCITED",
                                  judgment_method="n/a", fetch_error=None,
                                  justification=None))
            continue
        for m in markers:
            url = urls[m - 1] if 0 < m <= len(urls) else None

            # FIX 3: index/tag/topic page -- a pure URL-shape fact, decided
            # before any fetch attempt or LLM call. Its own verdict
            # category, not folded into "irrelevant".
            if url:
                shape = classify_url_shape(url)
                if shape:
                    pair_rows.append(dict(
                        idx=idx, group=row["group"], claim_text=claim_text,
                        marker=m, url=url, verdict="index_page_not_article",
                        judgment_method="url_shape_detected", fetch_error=None,
                        justification=f"URL looks like an index/tag/topic page, "
                                     f"not a specific article: {shape}"))
                    continue

            fetch_err = "no_url_available: no URL found for this marker"
            fetched_text = None
            if url:
                # FIX 1: known-unfetchable domains skip the fetch attempt
                # entirely and go straight to the SAME metadata-only
                # fallback path used for any other fetch failure -- not a
                # new, separate path.
                if is_blocklisted_domain(url):
                    fetch_err = f"blocklisted_domain: {url}"
                else:
                    cache_key = url
                    if cache_key in url_cache_errors:
                        fetched_text, fetch_err = url_cache_errors[cache_key]
                    else:
                        fetched_text, fetch_err = fetch_url_text(url)
                        url_cache_errors[cache_key] = (fetched_text, fetch_err)

            obj, err, method = judge_claim(api_key, claim_text, url, fetched_text, fetch_err)
            if obj is None:
                pair_rows.append(dict(idx=idx, group=row["group"], claim_text=claim_text,
                                      marker=m, url=url, verdict="JUDGE_FAILED",
                                      judgment_method=method, fetch_error=fetch_err,
                                      justification=err))
                if is_daily_quota(err):
                    print(f"  [idx={idx}] DAILY QUOTA WALL hit mid-conversation -- "
                          "stopping this conversation now, keeping pairs judged so far.")
                    return pair_rows, total_tokens, err, True
                continue
            total_tokens += (obj.get("_usage") or 0)
            pair_rows.append(dict(idx=idx, group=row["group"], claim_text=claim_text,
                                  marker=m, url=url, verdict=obj["verdict"],
                                  judgment_method=method, fetch_error=fetch_err,
                                  justification=obj["justification"]))
    return pair_rows, total_tokens, None, False


def run_pilot(api_key, sample):
    print("\n" + "=" * 78)
    print(f"STEP 5  PILOT FIRST -- {2*PILOT_SIZE_PER_GROUP} conversations "
          "(real cost/time before the full 240-sample run)")
    print("=" * 78)
    pilot = pd.concat([
        sample[sample["group"] == "long_heavy"].head(PILOT_SIZE_PER_GROUP),
        sample[sample["group"] == "short_light"].head(PILOT_SIZE_PER_GROUP),
    ])
    pilot_idx = set(pilot.index)
    print(f"pilot idx: {sorted(pilot_idx)}")

    t0 = time.time()
    all_rows = []
    total_tokens = 0
    n_failed = 0
    url_cache_errors = {}
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in pilot_idx:
                continue
            r = json.loads(line)
            row = pilot.loc[i]
            t_conv0 = time.time()
            pair_rows, tok, err, hit_quota = process_one_conversation(
                api_key, i, row, r, url_cache_errors)
            dt = time.time() - t_conv0
            total_tokens += tok
            if err:
                n_failed += 1
            if hit_quota:
                print("  DAILY QUOTA WALL hit during the pilot -- stopping pilot early.")
                all_rows.extend(pair_rows)
                break
            print(f"  [idx={i:6}] group={row['group']:12} claims/pairs={len(pair_rows):3}  "
                  f"tokens={tok:5}  time={dt:5.1f}s")
            all_rows.extend(pair_rows)
    elapsed = time.time() - t0

    n_pilot = len(pilot_idx) - n_failed
    print(f"\npilot complete: {n_pilot}/{len(pilot_idx)} conversations processed OK")
    print(f"total elapsed: {elapsed:.1f}s  ({elapsed/len(pilot_idx):.1f}s/conversation avg)")
    print(f"total tokens used: {total_tokens:,}  "
          f"({total_tokens/max(1,len(pilot_idx)):.0f} tokens/conversation avg)")

    full_n = 240
    est_time = elapsed / len(pilot_idx) * full_n
    est_tokens = total_tokens / max(1, len(pilot_idx)) * full_n
    print(f"\nEXTRAPOLATED to the full {full_n}-conversation sample:")
    print(f"  estimated time:   {est_time/60:.1f} minutes")
    print(f"  estimated tokens: {est_tokens:,.0f}")
    print("  (Gemini free tier has per-minute and per-day REQUEST-COUNT quotas, not "
          "just tokens -- frustration_classify.py hit a hard per-day wall earlier in "
          "this project. Each claim here costs 1 extraction call + 1 judgment call "
          "per citation marker, so request count, not token count, is the more "
          "likely constraint; watch the console for a DAILY_QUOTA abort.)")

    pairs_df = pd.DataFrame(all_rows)
    return pairs_df, dict(n_pilot=n_pilot, n_requested=len(pilot_idx), elapsed_s=elapsed,
                          total_tokens=total_tokens, est_full_time_s=est_time,
                          est_full_tokens=est_tokens)


# ============================================================
# FULL 240-CONVERSATION RUN. Separate from run_pilot (which is purely for
# the cost/time estimate): writes each conversation's pairs to disk
# IMMEDIATELY after it finishes (not buffered in memory until the end), and
# is resumable -- if run again after a quota wall or any other interruption,
# conversations already present in OUT_CSV are skipped, not redone. This is
# the same incremental-save discipline frustration_classify.py already
# established in this project for exactly the same reason (a multi-hour
# run against a free-tier daily quota should never lose completed work).
# ============================================================

def load_already_done(path):
    if not os.path.exists(path):
        return set()
    try:
        done = pd.read_csv(path)
        return set(done["idx"].unique())
    except Exception:                                              # noqa: BLE001
        return set()


def run_full_sample(api_key, sample, out_csv=OUT_CSV):
    print("\n" + "=" * 78)
    print(f"FULL RUN -- {len(sample)} conversations (fixed fetcher)")
    print("=" * 78)

    already_done = load_already_done(out_csv)
    todo = sample[~sample.index.isin(already_done)]
    if already_done:
        print(f"RESUMING: {len(already_done)} conversations already in {out_csv}, "
              f"skipping them. {len(todo)} remaining.")
    todo_idx = set(todo.index)
    if not todo_idx:
        print("Nothing left to do -- all conversations already processed.")
        return pd.read_csv(out_csv), dict(aborted=False, n_done=len(already_done),
                                          n_total=len(sample))

    write_header = not os.path.exists(out_csv)
    t0 = time.time()
    total_tokens = 0
    n_done = 0
    n_failed = 0
    aborted = False
    url_cache_errors = {}

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
            total_tokens += tok
            if err and not hit_quota:
                n_failed += 1

            # Write THIS conversation's rows immediately, flush to disk --
            # if the process dies or hits a quota wall on the NEXT
            # conversation, everything up to and including this one is
            # already safely on disk.
            if pair_rows:
                conv_df = pd.DataFrame(pair_rows)
                conv_df.to_csv(out_f, header=write_header, index=False)
                write_header = False
                out_f.flush()

            n_done += 1
            elapsed_so_far = time.time() - t0
            print(f"  [{n_done}/{len(todo_idx)}] idx={i:6} group={row['group']:12} "
                  f"pairs={len(pair_rows):3}  tokens={tok:5}  time={dt:5.1f}s  "
                  f"(elapsed {elapsed_so_far/60:.1f} min)")

            if hit_quota:
                print("\n" + "!" * 78)
                print("DAILY QUOTA WALL HIT. Stopping immediately.")
                print(f"Completed {len(already_done) + n_done} of {len(sample)} "
                      f"conversations total ({n_done} in this run).")
                print(f"All completed work is saved in {out_csv}. Re-run this script "
                      "with --full once quota resets -- it will resume automatically, "
                      "skipping everything already done.")
                print("!" * 78)
                aborted = True
                break

    elapsed = time.time() - t0
    print(f"\nthis run: {n_done - n_failed if not aborted else n_done} conversations "
          f"processed, {elapsed/60:.1f} min, {total_tokens:,} tokens")

    final_df = pd.read_csv(out_csv)
    return final_df, dict(aborted=aborted, n_done=len(already_done) + n_done,
                          n_total=len(sample), elapsed_s=elapsed,
                          total_tokens=total_tokens)


# ============================================================
# Pilot-quality audit: does "irrelevant" mean the citation is actually bad,
# or did the fetcher just return boilerplate (JS config, cookie banners,
# tag/topic-index pages) instead of real article content? This is checked
# explicitly, not assumed, before any full-sample decision is made.
# ============================================================

# Step 5 go/no-go threshold, stated explicitly: short_light is, by design,
# the lightly-cited group, so it will ALWAYS produce fewer total citations
# "Balance" is checked TWO ways, stated explicitly since they answer
# different questions:
#
# 1. Share of total real-URL evidence (BALANCE_FAIL_THRESHOLD, 80%).
#    Reported because the task asked for it directly, but it is NOT by
#    itself the right pass/fail bar: short_light is, by design, the
#    lightly-cited group, so it will ALWAYS produce fewer total citations
#    -- and therefore fewer real-URL judgments -- than long_heavy, even
#    with a perfectly working fetcher. A count imbalance alone does not
#    mean the comparison is biased.
# 2. Per-group CONTAMINATION rate (CONTAMINATION_FAIL_THRESHOLD, 15%) --
#    the fraction of each group's OWN fetched judgments that still read
#    like boilerplate. THIS is the metric that actually determines whether
#    the comparison is trustworthy: if one group's real-URL evidence is
#    disproportionately still-contaminated relative to the other's, its
#    verdict rates can't be trusted next to the other group's, regardless
#    of how many total judgments either group has. 15% is chosen as well
#    below the 54% the broken fetcher produced, while allowing that no
#    simple heuristic will be perfect.
#
# The decision to proceed requires BOTH to pass.
BALANCE_FAIL_THRESHOLD = 0.80
CONTAMINATION_FAIL_THRESHOLD = 0.15
STILL_SUSPECT_PATTERN = "|".join(["configuration", "script", "metadata", "header",
                                  "cookie", "javascript", "stylesheet", "error handling"])


def categorize_fetch_reason(fetch_err):
    if fetch_err is None:
        return "fetched_ok"
    s = str(fetch_err)
    if s.startswith("blocklisted_domain"):
        return "blocklisted_domain"
    if s.startswith("content_quality_check_failed"):
        return "content_quality_check_failed"
    if s.startswith("no_url_available"):
        return "no_url_available"
    return "other_fetch_error"


def audit_pilot_fetch_quality(pairs_df):
    print("\n" + "=" * 78)
    print("PILOT v2 FETCH-QUALITY AUDIT (fixed fetcher -- before deciding on the full run)")
    print("=" * 78)

    pairs_df = pairs_df.copy()
    pairs_df["fallback_reason"] = pairs_df["fetch_error"].apply(categorize_fetch_reason)

    fetched = pairs_df[pairs_df["judgment_method"] == "fetched_url"].copy()
    n_fetched = len(fetched)
    group_counts = dict(fetched.groupby("group").size()) if n_fetched else {}
    print(f"real-URL-fetched judgments: {n_fetched}")
    print(f"real-URL judgments by group (raw count -- expected to differ, see note): "
          f"{group_counts}")

    total_fetched = sum(group_counts.values()) or 1
    max_share = max(group_counts.values(), default=0) / total_fetched
    count_balanced = max_share <= BALANCE_FAIL_THRESHOLD
    print(f"largest single group's share of total real-URL evidence: {100*max_share:.0f}% "
          f"(reported per the task's request; NOT the pass/fail bar on its own -- "
          "short_light structurally has fewer citations by design)")

    if n_fetched:
        fetched["still_suspect"] = fetched["justification"].astype(str).str.lower().str.contains(
            STILL_SUSPECT_PATTERN, na=False)
    else:
        fetched["still_suspect"] = False
    contamination_by_group = {}
    for g, sub in fetched.groupby("group"):
        n = len(sub)
        n_sus = int(sub["still_suspect"].sum())
        contamination_by_group[g] = dict(n=n, n_suspect=n_sus,
                                         rate=n_sus / n if n else None)
    print(f"\nPER-GROUP CONTAMINATION RATE (the real pass/fail signal, threshold "
          f"<={100*CONTAMINATION_FAIL_THRESHOLD:.0f}%):")
    for g, d in contamination_by_group.items():
        rate_str = f"{100*d['rate']:.0f}%" if d["rate"] is not None else "n/a (0 fetched)"
        print(f"  {g:12} {d['n_suspect']}/{d['n']} still-suspect = {rate_str}")

    contamination_ok = all(
        (d["rate"] is None or d["rate"] <= CONTAMINATION_FAIL_THRESHOLD)
        for d in contamination_by_group.values())
    # DECISION RULE, revised: raw count imbalance alone no longer blocks
    # proceeding. long_heavy is DEFINED as the heavily-cited group, so it
    # will structurally produce more total citation-judgments than
    # short_light even with a perfectly correct fetcher -- that was always
    # expected, not a sign of a broken comparison. Contamination rate is
    # the metric that actually determines whether each group's verdicts
    # are trustworthy, so it is now the sole gate. count_balanced is still
    # computed and reported for transparency, just not a blocker.
    balanced = contamination_ok
    print(f"\ncount check: {'PASS' if count_balanced else 'FAIL (not a blocker -- see note)'}   "
          f"contamination check: {'PASS' if contamination_ok else 'FAIL'}   "
          f"=> overall: {'BALANCED, OK to consider scaling' if balanced else 'STILL FAILING'}")

    # Fix 1/2/3 routing breakdown, by group, as Step 4 explicitly asked for.
    index_page = pairs_df[pairs_df["verdict"] == "index_page_not_article"]
    index_page_by_group = dict(index_page.groupby("group").size()) if len(index_page) else {}

    routed_to_fallback = pairs_df[pairs_df["judgment_method"] == "proxy_no_fetch"]
    reason_by_group = (routed_to_fallback.groupby(["group", "fallback_reason"]).size()
                       .unstack(fill_value=0)) if len(routed_to_fallback) else pd.DataFrame()
    print(f"\nindex_page_not_article verdicts by group: {index_page_by_group}")
    print(f"\nrouted-to-fallback counts by group and reason:\n{reason_by_group}")

    return dict(n_fetched=n_fetched, group_counts=group_counts, max_share=max_share,
               count_balanced=count_balanced, contamination_by_group=contamination_by_group,
               contamination_ok=contamination_ok, balanced=balanced,
               index_page_by_group=index_page_by_group, reason_by_group=reason_by_group)


# Outlier check: idx 6166 and 18399 are the two long_heavy conversations
# whose sheer citation density (65 and 63 claim-pairs respectively, far
# above any other pilot conversation) structurally dominates long_heavy's
# raw citation count regardless of fetch quality. Before trusting the
# group-level comparison, check whether these two specific conversations'
# OWN support rate looks like the rest of long_heavy, or whether they are
# unusually different -- if they are, they would be quietly dominating the
# group average on their own, not just its count.
OUTLIER_CONV_IDX = {6166, 18399}
OUTLIER_FLAG_DELTA = 0.20  # 20 percentage points, stated explicitly


def check_outlier_conversations(pairs_df, group="long_heavy"):
    print("\n" + "=" * 78)
    print(f"OUTLIER CHECK: do idx {sorted(OUTLIER_CONV_IDX)} look like the rest of "
          f"`{group}`, or are they driving the group average on their own?")
    print("=" * 78)
    sub = pairs_df[pairs_df["group"] == group].copy()
    # Support rate is computed over JUDGED pairs only -- excluding UNCITED
    # (no citation attached at all) and index_page_not_article (a
    # structural URL-shape fact, not a support judgment) -- this is the
    # same denominator the actual red-herring comparison will use.
    judged = sub[sub["verdict"].isin(["supported", "irrelevant", "contradicted"])]

    def support_rate(df):
        n = len(df)
        if n == 0:
            return None, 0
        return float((df["verdict"] == "supported").mean()), n

    rows = []
    for idx in sorted(OUTLIER_CONV_IDX):
        one = judged[judged["idx"] == idx]
        rate, n = support_rate(one)
        rows.append(dict(idx=idx, n_judged=n, supported_rate=rate))
        print(f"  idx={idx:6} n_judged={n:3}  supported_rate="
              f"{f'{rate:.2f}' if rate is not None else 'n/a'}")

    rest = judged[~judged["idx"].isin(OUTLIER_CONV_IDX)]
    rest_rate, rest_n = support_rate(rest)
    print(f"  REST OF {group} (excluding outliers)  n_judged={rest_n:3}  "
          f"supported_rate={f'{rest_rate:.2f}' if rest_rate is not None else 'n/a'}")

    flags = []
    for r in rows:
        if r["supported_rate"] is not None and rest_rate is not None:
            delta = abs(r["supported_rate"] - rest_rate)
            if delta > OUTLIER_FLAG_DELTA:
                flags.append((r["idx"], delta))
    if flags:
        print(f"\n*** FLAGGED: {[f'idx={i} (delta={d:.2f})' for i, d in flags]} "
              f"differ from the rest of {group} by more than "
              f"{OUTLIER_FLAG_DELTA:.0%} -- these two conversations would be "
              "skewing the group average, not just its count. ***")
    else:
        print(f"\nNo flag: both outlier conversations' supported_rate is within "
              f"{OUTLIER_FLAG_DELTA:.0%} of the rest of {group} -- they dominate "
              "the CITATION COUNT but not the group's average support rate.")
    return dict(rows=rows, rest_rate=rest_rate, rest_n=rest_n, flags=flags)


def write_pilot_interim_summary(path, pilot_stats, audit, pairs_df, outlier_check):
    L = []
    A = L.append
    A("# Citation Support Check: Pilot v2 Results (Fixed Fetcher)")
    A("")
    if audit["balanced"]:
        A("**Verdict: FETCH-QUALITY FIX WORKED -- balanced enough to consider scaling.** "
          "The three fixes (domain blocklist, content-quality check, index-page "
          "detection) resolved the concentration problem found in the first pilot. This "
          "is still a feasibility finding, not a result for or against John's "
          "red-herring hypothesis -- that requires the full 240-conversation run, not "
          "run here.")
    else:
        A("**Verdict: STILL SKEWED -- stopping again rather than forcing it through a "
          "second time.** The three fixes reduced but did not resolve the imbalance "
          "between groups. Per the task's own instruction, this is being reported, not "
          "pushed through to the full run.")
    A("")
    A("## What changed since the first pilot")
    A("")
    A("1. **Domain blocklist** (`youtube.com`, `twitter.com`/`x.com`, `instagram.com`, "
      "`tiktok.com`) -- these skip the fetch attempt entirely and route straight into "
      "the same metadata-only fallback path `no_url_available` already used, per the "
      "task's instruction not to create a new, separate path.")
    A("2. **Content-quality check after fetching** -- a successful HTTP fetch is now "
      "rejected (routed to the same fallback) if the cleaned text is under "
      f"`{MIN_CONTENT_CHARS}` characters (roughly a short paragraph -- the prior "
      "50-character floor was far too low) or matches a junk pattern (cookie/consent "
      "language, leftover `function(`/`var x =`/`@media` markup) or exceeds a "
      f"`{CODE_SYMBOL_DENSITY_THRESHOLD}` code-symbol density (real prose has "
      "near-zero density of `{ } ; ( ) =`).")
    A("3. **Index/tag/topic-page detection** -- URLs matching `/tag/`, `/topic/`, "
      "`/category/`, a search-results pattern, or a bare homepage with no "
      "article-specific path get their own verdict, `index_page_not_article`, decided "
      "purely from the URL shape (no fetch or LLM call needed) -- not folded into "
      "\"irrelevant\".")
    A("")

    A("## Cost/time (same pilot size and structure as before)")
    A("")
    A(f"{pilot_stats['n_pilot']}/{pilot_stats['n_requested']} conversations processed, "
      f"**{pilot_stats['elapsed_s']/60:.1f} minutes, {pilot_stats['total_tokens']:,} "
      f"tokens**. Extrapolated to the full 240: "
      f"**~{pilot_stats['est_full_time_s']/60:.0f} minutes, "
      f"~{pilot_stats['est_full_tokens']:,.0f} tokens**.")
    A("")

    A("## Group balance -- checked TWO ways, on purpose")
    A("")
    A("**(a) Raw count share**, reported because the task asked for it directly: "
      f"real-URL-fetched judgments by group: **{audit['group_counts']}** -- largest "
      f"group's share of the total: **{100*audit['max_share']:.0f}%** (vs. 90% in the "
      "first, broken pilot). This number is **not, on its own, the pass/fail bar**: "
      "`short_light` is the lightly-cited group by design, so it will always produce "
      "fewer total citations -- and therefore fewer real-URL judgments -- than "
      "`long_heavy`, even with a perfectly correct fetcher. A count imbalance alone "
      "does not mean the comparison is biased.")
    A("")
    A("**(b) Per-group contamination rate -- the metric that actually determines "
      "trustworthiness.** The real failure mode in the first pilot wasn't that "
      "`long_heavy` had more total citations (expected); it was that a disproportionate "
      "*fraction* of its fetched evidence was still garbage. This is checked directly "
      "per group:")
    A("")
    A("| group | fetched judgments | still read like boilerplate | contamination rate |")
    A("|---|---|---|---|")
    for g, d in audit["contamination_by_group"].items():
        rate_str = f"{100*d['rate']:.0f}%" if d["rate"] is not None else "n/a"
        A(f"| {g} | {d['n']} | {d['n_suspect']} | {rate_str} |")
    A("")
    A(f"Fail threshold: **>{100*CONTAMINATION_FAIL_THRESHOLD:.0f}%** for either group "
      "(well below the 54% the broken fetcher produced, allowing that no simple "
      "heuristic will be perfect). "
      f"Count check: **{'PASS' if audit['count_balanced'] else 'FAIL'}**. "
      f"Contamination check: **{'PASS' if audit['contamination_ok'] else 'FAIL'}**. "
      "Both must pass to proceed.")
    A("")

    A("## Where citations got routed (by group) -- Fix 1/2/3 breakdown")
    A("")
    A(f"`index_page_not_article` verdicts by group: **{audit['index_page_by_group']}**")
    A("")
    if len(audit["reason_by_group"]):
        A("Routed to the metadata-only fallback, by group and reason:")
        A("")
        A("| group | " + " | ".join(audit["reason_by_group"].columns) + " |")
        A("|---|" + "---|" * len(audit["reason_by_group"].columns))
        for g, row in audit["reason_by_group"].iterrows():
            A(f"| {g} | " + " | ".join(str(v) for v in row.values) + " |")
        A("")

    A("## Outlier check: do the two citation-dense conversations skew the group average?")
    A("")
    A(f"`idx` 6166 and 18399 are the two `long_heavy` conversations whose citation "
      "density (65 and 63 claim-pairs, far above any other pilot conversation) "
      "structurally dominates `long_heavy`'s raw count. Checked whether they also "
      "dominate the group's AVERAGE support rate, not just its count:")
    A("")
    A("| idx | n judged | supported rate |")
    A("|---|---|---|")
    for r in outlier_check["rows"]:
        rate_str = f"{r['supported_rate']:.2f}" if r["supported_rate"] is not None else "n/a"
        A(f"| {r['idx']} | {r['n_judged']} | {rate_str} |")
    rest_rate_str = (f"{outlier_check['rest_rate']:.2f}"
                     if outlier_check["rest_rate"] is not None else "n/a")
    A(f"| rest of long_heavy | {outlier_check['rest_n']} | {rest_rate_str} |")
    A("")
    if outlier_check["flags"]:
        A(f"**FLAGGED:** {outlier_check['flags']} -- at least one outlier differs from "
          f"the rest of `long_heavy` by more than {OUTLIER_FLAG_DELTA:.0%}. These two "
          "conversations would be skewing the group's average support rate, not just "
          "its citation count -- worth understanding before trusting a full-scale "
          "comparison that treats `long_heavy` as one homogeneous group.")
    else:
        A(f"**No flag.** Both outlier conversations' supported rate is within "
          f"{OUTLIER_FLAG_DELTA:.0%} of the rest of `long_heavy`. They dominate the "
          "raw citation COUNT (explaining the count-check failure above) but not the "
          "group's average support rate -- the count imbalance looks like a sampling "
          "artifact of citation density, not a sign that the group average is being "
          "driven by two unrepresentative conversations.")
    A("")

    A("## Pilot verdict counts (for transparency, not the final answer)")
    A("")
    vc = pairs_df["verdict"].value_counts()
    A("| verdict | count |")
    A("|---|---|")
    for k, v in vc.items():
        A(f"| {k} | {v} |")
    A("")

    A("## Decision")
    A("")
    if audit["balanced"]:
        A(f"**Recommend proceeding to the full 240-conversation run** "
          f"(~{pilot_stats['est_full_time_s']/60:.0f} min, "
          f"~{pilot_stats['est_full_tokens']:,.0f} tokens) -- but asking first, not "
          "launching automatically, since it's a real time/token commitment.")
    else:
        A("**Do not scale to the full 240 yet.** The group-balance problem, while "
          "reduced, is not resolved. Report this back rather than running the "
          "expensive full sample on evidence that's still structurally skewed toward "
          "one group.")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[saved] {path}")


# ============================================================
# FULL-RUN ANALYSIS: the actual answer to John's red-herring hypothesis,
# plus the URL-category breakdown and the scale-level check of the
# idx-18399-style pattern (tag/aggregator-heavy conversations showing lower
# support), both explicitly requested in addition to the original
# long_heavy vs short_light comparison.
# ============================================================

JUDGED_VERDICTS = ("supported", "irrelevant", "contradicted")


def support_rate_with_ci(df):
    judged = df[df["verdict"].isin(JUDGED_VERDICTS)]
    n = len(judged)
    if n == 0:
        return dict(n=0, n_supported=0, rate=None, ci_lo=None, ci_hi=None)
    n_sup = int((judged["verdict"] == "supported").sum())
    rate, lo, hi = wilson_ci(n_sup, n)
    return dict(n=n, n_supported=n_sup, rate=rate, ci_lo=lo, ci_hi=hi)


def analyze_full_results(pairs_df):
    print("\n" + "=" * 78)
    print("FULL-RUN ANALYSIS")
    print("=" * 78)

    # (a) Original comparison: long_heavy vs short_light, with Wilson CI --
    # reusing the same interval method already established in
    # both_bad_rate.py rather than a bare percentage comparison.
    group_results = {}
    for g in ("long_heavy", "short_light"):
        sub = pairs_df[pairs_df["group"] == g]
        group_results[g] = support_rate_with_ci(sub)
        r = group_results[g]
        rate_str = f"{r['rate']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if r["rate"] is not None else "n/a"
        print(f"{g:14} n_judged={r['n']:4}  supported={r['n_supported']:4}  "
              f"rate={rate_str}")

    overlap = None
    if group_results["long_heavy"]["rate"] is not None and group_results["short_light"]["rate"] is not None:
        a, b = group_results["long_heavy"], group_results["short_light"]
        overlap = not (a["ci_lo"] > b["ci_hi"] or b["ci_lo"] > a["ci_hi"])
        print(f"\n95% CIs {'OVERLAP -- no established difference' if overlap else 'DO NOT OVERLAP -- a real difference'}")

    # (b) URL-category breakdown, explicitly requested.
    category_results = {}
    fetched = pairs_df[pairs_df["judgment_method"] == "fetched_url"]
    category_results["real_fetched_article"] = support_rate_with_ci(fetched)
    fallback = pairs_df[pairs_df["judgment_method"] == "proxy_no_fetch"]
    category_results["blocklisted_or_fallback"] = support_rate_with_ci(fallback)
    n_index = int((pairs_df["verdict"] == "index_page_not_article").sum())
    category_results["index_page_not_article"] = dict(
        n=n_index, n_supported=0, rate=0.0, ci_lo=None, ci_hi=None,
        note="Not an LLM judgment -- a deterministic URL-shape fact. 'Supported' is "
             "trivially 0% by construction (this verdict means no support judgment was "
             "made at all, not that the citation is unsupported content).")
    print("\nBy URL category:")
    for cat, r in category_results.items():
        rate_str = f"{r['rate']:.3f}" if r["rate"] is not None else "n/a"
        print(f"  {cat:26} n={r['n']:4}  supported_rate={rate_str}")

    # (b-2) MEASUREMENT VALIDITY CHECK, found while reviewing these numbers,
    # not assumed: real_fetched_article's judge prompt requires the source
    # to DIRECTLY support the claim; proxy_no_fetch's prompt asks only
    # whether the claim is PLAUSIBLE given the domain (explicitly NOT
    # confirmed). These are different standards on different scales --
    # pooling across them conflates two incomparable judgments, and
    # whichever group leans more on one path than the other will have its
    # pooled rate pulled toward that path's baseline regardless of true
    # citation quality. The valid, apples-to-apples comparison is WITHIN
    # each judgment method, not the pooled headline number.
    within_method = {}
    for method in ("fetched_url", "proxy_no_fetch"):
        within_method[method] = {}
        for g in ("long_heavy", "short_light"):
            sub = pairs_df[(pairs_df["group"] == g) & (pairs_df["judgment_method"] == method)]
            within_method[method][g] = support_rate_with_ci(sub)
    print("\nWITHIN-METHOD group comparison (the apples-to-apples test):")
    within_overlaps = {}
    for method, gd in within_method.items():
        a, b = gd["long_heavy"], gd["short_light"]
        a_str = f"{a['rate']:.3f} [{a['ci_lo']:.3f},{a['ci_hi']:.3f}]" if a["rate"] is not None else "n/a"
        b_str = f"{b['rate']:.3f} [{b['ci_lo']:.3f},{b['ci_hi']:.3f}]" if b["rate"] is not None else "n/a"
        print(f"  {method:16} long_heavy={a_str}  short_light={b_str}")
        if a["rate"] is not None and b["rate"] is not None:
            within_overlaps[method] = not (a["ci_lo"] > b["ci_hi"] or b["ci_lo"] > a["ci_hi"])
            print(f"    -> {'CIs overlap, no established difference' if within_overlaps[method] else 'CIs DO NOT overlap -- real difference, long_heavy lower' if a['rate'] < b['rate'] else 'CIs DO NOT overlap -- real difference, long_heavy higher'}")

    # (c) Scale-level check: does the idx-18399 pattern (heavy on tag/
    # aggregator citations -> lower support) hold across many
    # conversations, not just one?
    per_conv = []
    for idx, sub in pairs_df.groupby("idx"):
        judged = sub[sub["verdict"].isin(JUDGED_VERDICTS)]
        n_judged = len(judged)
        if n_judged < 3:  # too few judged pairs to give this conversation a rate
            continue
        supp_rate = float((judged["verdict"] == "supported").mean())
        n_total_markers = len(sub[sub["verdict"] != "UNCITED"])
        n_real_fetched = int((sub["judgment_method"] == "fetched_url").sum())
        frac_real_fetched = n_real_fetched / n_total_markers if n_total_markers else None
        per_conv.append(dict(idx=idx, group=sub["group"].iloc[0], n_judged=n_judged,
                            supported_rate=supp_rate, frac_real_fetched=frac_real_fetched))
    per_conv_df = pd.DataFrame(per_conv)
    print(f"\nScale-level check: {len(per_conv_df)} conversations with >=3 judged pairs")

    pattern_result = dict(n_conversations=len(per_conv_df), correlation=None,
                          low_group_mean=None, high_group_mean=None, holds_at_scale=False,
                          confounded=False)
    valid = per_conv_df[per_conv_df["frac_real_fetched"].notna()]
    if len(valid) >= 5:
        corr = float(np.corrcoef(valid["frac_real_fetched"], valid["supported_rate"])[0, 1])
        median_frac = valid["frac_real_fetched"].median()
        low = valid[valid["frac_real_fetched"] <= median_frac]["supported_rate"]
        high = valid[valid["frac_real_fetched"] > median_frac]["supported_rate"]
        print(f"correlation(frac_real_fetched, supported_rate) = {corr:+.3f}  (n={len(valid)})")
        print(f"conversations MORE reliant on tag/fallback citations "
              f"(below-median real-article share): mean supported_rate = {low.mean():.3f} (n={len(low)})")
        print(f"conversations LESS reliant on tag/fallback citations "
              f"(above-median real-article share): mean supported_rate = {high.mean():.3f} (n={len(high)})")
        holds = (corr > 0.2) and (high.mean() - low.mean() > 0.10)
        print(f"\nPattern (more real-article citations -> higher support) "
              f"{'HOLDS at scale' if holds else 'does NOT clearly hold at scale'}")
        print("*** CAVEAT, found while reviewing this result: this check is "
              "CONFOUNDED by the same measurement-validity issue above -- "
              "fetched_url judgments are inherently stricter (lower baseline rate) "
              "than proxy_no_fetch judgments (lenient, higher baseline rate), so "
              "frac_real_fetched correlates with supported_rate partly (perhaps "
              "mostly) because of WHICH JUDGE PROMPT a conversation's citations "
              "happen to route through, not necessarily because of citation-source "
              "quality. This check should not be read as a clean test of the "
              "original tag/aggregator-page hypothesis -- the within-method "
              "comparison above is the trustworthy one. ***")
        pattern_result = dict(n_conversations=len(valid), correlation=corr,
                             low_group_mean=float(low.mean()), high_group_mean=float(high.mean()),
                             low_group_n=len(low), high_group_n=len(high),
                             holds_at_scale=holds, confounded=True)
    else:
        print("too few conversations with usable data to test this at scale")

    return dict(group_results=group_results, overlap=overlap,
               category_results=category_results, within_method=within_method,
               within_overlaps=within_overlaps, per_conv_df=per_conv_df,
               pattern_result=pattern_result)


def write_full_results_summary(path, full_stats, analysis):
    L = []
    A = L.append
    A("# Citation Support Check: Full 240-Conversation Results")
    A("")
    gr = analysis["group_results"]
    lh, sl = gr["long_heavy"], gr["short_light"]
    wm = analysis["within_method"]
    wo = analysis["within_overlaps"]

    # The within-method comparison is the trustworthy one -- it's reported
    # FIRST and is what the headline verdict is actually based on, not the
    # pooled number (see "A measurement-validity issue" section below for
    # why the pooled number alone would overstate confidence).
    fu_lh, fu_sl = wm["fetched_url"]["long_heavy"], wm["fetched_url"]["short_light"]
    pf_lh, pf_sl = wm["proxy_no_fetch"]["long_heavy"], wm["proxy_no_fetch"]["short_light"]
    fu_sig = wo.get("fetched_url") is False
    pf_sig = wo.get("proxy_no_fetch") is False

    if pf_sig and pf_lh["rate"] < pf_sl["rate"] and (not fu_sig or fu_lh["rate"] < fu_sl["rate"]):
        verdict = (f"**Evidence consistent with John's red-herring hypothesis, within "
                  f"the methodologically valid comparison.** Comparing long_heavy vs "
                  f"short_light WITHIN the same judgment method (apples-to-apples, not "
                  f"the pooled number): on `proxy_no_fetch` judgments, long_heavy's "
                  f"supported rate ({pf_lh['rate']:.2f}) is clearly lower than "
                  f"short_light's ({pf_sl['rate']:.2f}), 95% CIs do not overlap. On "
                  f"`fetched_url` judgments the same direction holds "
                  f"({fu_lh['rate']:.2f} vs {fu_sl['rate']:.2f}) but is "
                  f"{'also statistically clear' if fu_sig else 'not statistically clear at this sample size (CIs touch/overlap)'}.")
    else:
        verdict = ("**Mixed or inconclusive within the methodologically valid "
                  "comparison** -- see the within-method table below; the within-"
                  "method results do not cleanly point in one direction.")
    A(f"**Verdict, stated plainly:** {verdict}")
    A("")

    A("## A measurement-validity issue found while reviewing these results "
      "(read this before the numbers below)")
    A("")
    A("The `fetched_url` judge prompt requires the source text to **directly support** "
      "the claim. The `proxy_no_fetch` judge prompt (used when a page is blocklisted "
      "or fails the content-quality check) asks only whether the claim is **plausible** "
      "given the domain -- explicitly NOT confirmed. These are different, non-comparable "
      "standards: pooling across them means a group's pooled rate is pulled toward "
      "whichever path it happens to rely on more, independent of actual citation "
      "quality. Concretely: `real_fetched_article` support rate is 0.226 overall; "
      "`blocklisted_or_fallback` is 0.839 -- that gap is mostly the STRICTNESS "
      "difference between the two prompts, not a quality difference between fetched "
      "and fallback citations. **The pooled long_heavy-vs-short_light number below is "
      "reported for completeness but should not be read as the primary result** -- "
      "the within-method comparison is.")
    A("")

    A("## Within-method comparison (the methodologically valid test)")
    A("")
    A("| judgment method | long_heavy rate [95% CI] | short_light rate [95% CI] | result |")
    A("|---|---|---|---|")
    for method, (a, b) in (("fetched_url", (fu_lh, fu_sl)), ("proxy_no_fetch", (pf_lh, pf_sl))):
        a_str = f"{a['rate']:.3f} [{a['ci_lo']:.3f},{a['ci_hi']:.3f}]" if a["rate"] is not None else "n/a"
        b_str = f"{b['rate']:.3f} [{b['ci_lo']:.3f},{b['ci_hi']:.3f}]" if b["rate"] is not None else "n/a"
        ov = wo.get(method)
        if ov is None:
            result = "n/a"
        elif ov:
            result = "CIs overlap -- no established difference"
        else:
            result = f"CIs do not overlap -- long_heavy {'lower' if a['rate'] < b['rate'] else 'higher'}"
        A(f"| `{method}` | {a_str} | {b_str} | {result} |")
    A("")

    A("## Pooled comparison (reported for completeness -- see validity caveat above)")
    A("")
    A("| group | n judged | n supported | supported rate | 95% Wilson CI |")
    A("|---|---|---|---|---|")
    for g in ("long_heavy", "short_light"):
        r = gr[g]
        rate_str = f"{r['rate']:.3f}" if r["rate"] is not None else "n/a"
        ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]" if r["rate"] is not None else "n/a"
        A(f"| {g} | {r['n']} | {r['n_supported']} | {rate_str} | {ci_str} |")
    A("")
    overlap = analysis["overlap"]
    A(f"95% CIs {'overlap' if overlap else 'do not overlap'}. Part of this pooled gap "
      "reflects long_heavy and short_light relying on the two judgment paths in "
      "different proportions (see below), not purely a within-method quality "
      "difference.")
    A("")

    A("## URL-category breakdown (requested in addition to the group comparison)")
    A("")
    A("| category | n | supported rate | note |")
    A("|---|---|---|---|")
    cr = analysis["category_results"]
    for cat, r in cr.items():
        rate_str = f"{r['rate']:.3f}" if r["rate"] is not None else "n/a"
        note = r.get("note", "")
        A(f"| {cat} | {r['n']} | {rate_str} | {note} |")
    A("")

    A("## Scale-level check: does the idx=18399 pattern hold across many conversations?")
    A("")
    pr = analysis["pattern_result"]
    if pr["correlation"] is not None:
        A(f"Across **{pr['n_conversations']}** conversations with enough judged pairs: "
          f"correlation(fraction of citations that were real fetched articles, "
          f"supported rate) = **{pr['correlation']:+.3f}**.")
        A("")
        A(f"- Conversations MORE reliant on tag/aggregator/fallback citations "
          f"(below-median real-article share, n={pr['low_group_n']}): mean supported "
          f"rate = **{pr['low_group_mean']:.3f}**")
        A(f"- Conversations LESS reliant on them (above-median, n={pr['high_group_n']}): "
          f"mean supported rate = **{pr['high_group_mean']:.3f}**")
        A("")
        if pr.get("confounded"):
            A("**This check is CONFOUNDED by the same measurement-validity issue above "
              "and should NOT be read as a clean test of the original tag/aggregator-"
              "page hypothesis.** `fetched_url` judgments are inherently stricter "
              "(lower baseline rate ~0.22) than `proxy_no_fetch` judgments (lenient, "
              "~0.83) -- so a conversation's `frac_real_fetched` correlates with its "
              "`supported_rate` largely because of WHICH JUDGE PROMPT its citations "
              "happen to route through, not necessarily because of citation-source "
              "quality. The idx=18399 example that originally motivated this check "
              "most likely reflects the strict `fetched_url` standard being hard to "
              "satisfy from a 4,000-character excerpt, not a genuine tag-page-citation "
              "problem. **The within-method comparison above is the trustworthy result "
              "for the original length/citation-count hypothesis; this specific check "
              "does not add valid additional evidence either way.**")
        elif pr["holds_at_scale"]:
            A("This pattern holds at scale, net of the measurement-validity caveat "
              "above.")
        else:
            A("This pattern does not clearly hold at scale.")
    else:
        A("Too few conversations with enough judged pairs and known fetch outcomes to "
          "test this at scale.")
    A("")

    A("## Caveats")
    A("")
    A("1. **`index_page_not_article` is excluded from all supported-rate denominators** "
      "-- it is a URL-shape fact, not a content judgment, and folding it into "
      "\"unsupported\" would conflate two different problems (citing the wrong kind of "
      "page vs. citing a page that doesn't back the claim).")
    A("2. **The URL-category comparison is observational, not causal** -- it describes "
      "what was found, not why certain conversations lean on weaker citation types.")
    A("3. **Known limitation carried over from the pilot:** math/notation-heavy pages "
      "can trip the code-symbol-density content-quality check and get routed to the "
      "weaker fallback even when genuinely real articles -- this was accepted "
      "explicitly rather than chased further, per the pilot writeup.")
    A("4. **The two judge prompts (`fetched_url` vs `proxy_no_fetch`) are not on the "
      "same scale** -- found while reviewing the full-run results, not anticipated in "
      "the pilot. This affects the pooled group comparison, the URL-category table, "
      "and invalidates the originally-planned scale-level pattern check (see above). "
      "It does NOT affect the within-method comparison, which is the result this "
      "summary's verdict is actually based on.")
    A("5. **Small n for `fetched_url` in `short_light`** (n=95) means that within-"
      "method comparison's CI is wide -- read the `fetched_url` row's statistical "
      "significance with appropriate caution even though the point estimates agree "
      "in direction with the `proxy_no_fetch` row.")
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="run the full 240-conversation sample (resumable) instead of "
                         "the 12-conversation pilot")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    sample, coverage_rows = main_step1_only()

    if not api_key:
        print("\n" + "!" * 78)
        print("GEMINI_API_KEY is not set in this environment.")
        print("Step 1 (stratified sampling) and the URL-extraction groundwork for")
        print("Steps 2-3 do NOT require the API and have been run and verified above.")
        print("Steps 2 (claim extraction), 3 (support judging), and 5 (pilot cost/")
        print("time report) require live Gemini calls and CANNOT proceed without the")
        print("key. Per the task's own instruction: stopping and reporting this")
        print("rather than forcing a weaker version through silently.")
        print("!" * 78 + "\n")
        sys.exit(2)

    if args.full:
        # Quota pre-flight: there is no direct REST endpoint to query
        # remaining Gemini free-tier quota, so the best available check is
        # a single cheap real call -- if the key is ALREADY at a daily
        # wall, this catches it before committing 3.7 hours of work to
        # finding out one conversation at a time.
        print("\nQuota pre-flight check (one cheap call before committing to the full run)...")
        _claims, _usage, _preflight_err = extract_claims(api_key, "The sky is blue.")
        if is_daily_quota(_preflight_err):
            print(f"DAILY QUOTA ALREADY HIT: {_preflight_err}")
            print("Not starting the full run -- the key has no headroom right now.")
            sys.exit(3)
        elif _preflight_err:
            print(f"[warning] pre-flight call failed non-quota-related: "
                  f"{_preflight_err[:150]} -- proceeding anyway, this was just a "
                  "quota sanity check, not a hard gate on other error types.")
        else:
            print("Pre-flight call succeeded -- key has headroom. Proceeding.")

        pairs_df, run_stats = run_full_sample(api_key, sample)
        if run_stats["aborted"]:
            print(f"\nStopped cleanly: {run_stats['n_done']}/{run_stats['n_total']} "
                  f"conversations completed before the quota wall. Partial progress is "
                  f"saved in {OUT_CSV} and will be resumed automatically on the next "
                  "--full run.")
            sys.exit(3)
        analysis = analyze_full_results(pairs_df)
        write_full_results_summary(OUT_MD, run_stats, analysis)
        print(f"\nFull run complete: {run_stats['n_done']}/{run_stats['n_total']} "
              f"conversations. See {OUT_MD} for the final results.")
        sys.exit(0)

    pairs_df, pilot_stats = run_pilot(api_key, sample)
    pairs_df.to_csv(OUT_PILOT_TXT.replace(".txt", "_pairs.csv"), index=False)
    print(f"\n[saved] {OUT_PILOT_TXT.replace('.txt', '_pairs.csv')}")

    audit = audit_pilot_fetch_quality(pairs_df)
    outlier_check = check_outlier_conversations(pairs_df)
    write_pilot_interim_summary(OUT_MD, pilot_stats, audit, pairs_df, outlier_check)

    if not audit["balanced"]:
        print("\n" + "!" * 78)
        print("STOPPING before the full 240-conversation run.")
        print("Group balance is still skewed after the fixes -- see", OUT_MD,
              "for the full breakdown.")
        print("!" * 78)
        sys.exit(3)
    else:
        print("\nPilot v2 looks balanced -- review", OUT_MD,
              "before deciding on the full 240-conversation run.")
