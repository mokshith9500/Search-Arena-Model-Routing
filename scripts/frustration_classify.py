"""
frustration_classify.py

Classifies Search Arena conversations as frustrated / engaged / neutral from the
USER's turns, using Gemini via the Generative Language REST API.

UNIT OF ANALYSIS: one conversation, NOT one side. Verified empirically: the
user's turns are byte-identical between model_a's side and model_b's side in all
1,714 decisive multi-turn conversations, so frustration is a conversation-level
property and a per-side difference would be identically zero.

SCOPE: only conversations with >= 2 stored user turns. The `turn` metadata field
exceeds the number of stored rounds for some records, so 724 of the 1,714
"multi_turn" decisive battles actually contain a single user turn and cannot
express frustration at all.

Reads the API key from the GEMINI_API_KEY environment variable. Never hardcode it.

Usage:
    GEMINI_API_KEY=... python frustration_classify.py --input pilot20.pkl \
        --out-txt frustration_pilot_review.txt --out-jsonl frustration_pilot.jsonl
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
import urllib.error
import urllib.request

# Default model. Free-tier daily request quotas are per-model, so this is
# overridable with --model; see the quota note in the pilot report.
MODEL = "gemini-3.1-flash-lite"
ENDPOINT_TMPL = ("https://generativelanguage.googleapis.com/v1beta/models/"
                 "{model}:generateContent")
MAX_TURN_CHARS = 1500
LABELS = ("frustrated", "engaged", "neutral")

SYSTEM_INSTRUCTION = """\
You are annotating real multi-turn conversations between a user and an AI search \
assistant. Judge ONLY the user's own messages: what the user does across their \
turns. Do not judge whether the assistant's answers were factually good.

Classify the conversation into exactly one of three labels:

frustrated - The user corrects the assistant, repeats or re-states a request the \
assistant already failed to satisfy, pushes back, expresses dissatisfaction or \
impatience, or restates the same need because the previous answer missed it.

engaged - The user builds on the answer: asks a genuine follow-up, goes deeper, \
explores an adjacent question, or extends the task. The user is moving forward, \
not re-litigating.

neutral - Neither is clearly present. Includes unrelated topic switches, purely \
mechanical continuations, short acknowledgements, or genuinely ambiguous intent.

Rules:
- The conversation may be in any language. Judge it in its own language; do not \
penalise or change your labelling because it is not English.
- Terse or blunt phrasing is NOT automatically frustration. Many languages and \
many users are simply direct. Require actual evidence of correction, repetition, \
or dissatisfaction.
- A follow-up asked because the answer was INCOMPLETE and the user says so is \
"frustrated". A follow-up asked out of curiosity is "engaged".
- If you are genuinely unsure, answer "neutral". Do not guess.

Give a one-sentence justification in English, max ~30 words. Quote the decisive \
phrase from the conversation when there is one."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING", "enum": list(LABELS)},
        "justification": {"type": "STRING"},
    },
    "required": ["label", "justification"],
}


def build_user_block(conv):
    langs = conv.get("languages")
    langs = ", ".join(langs) if isinstance(langs, (list, tuple)) else str(langs)
    parts = [f"Conversation language(s): {langs}",
             f"Number of user turns: {len(conv['user_turns'])}",
             "", "The user's turns, in order:"]
    for i, t in enumerate(conv["user_turns"], 1):
        t = t or ""
        if len(t) > MAX_TURN_CHARS:
            t = t[:MAX_TURN_CHARS] + f"\n...[truncated, {len(t)} chars total]"
        parts.append(f"\n[USER TURN {i}]\n{t}")
    return "\n".join(parts)


def call_gemini(api_key, user_block, model=MODEL, max_attempts=6):
    """
    POST with retries. The API intermittently returns 404/503 even for a valid
    model, so transient failures are retried with exponential backoff + jitter.
    Returns (result_dict, error_string).
    """
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": user_block}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    data = json.dumps(body).encode("utf-8")
    endpoint = ENDPOINT_TMPL.format(model=model)
    last = None
    for attempt in range(max_attempts):
        req = urllib.request.Request(
            endpoint, data=data,
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": api_key},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read().decode("utf-8"))
            cands = payload.get("candidates") or []
            if not cands:
                last = f"no candidates: {json.dumps(payload)[:200]}"
            else:
                cand = cands[0]
                txt = "".join(p.get("text", "")
                              for p in cand.get("content", {}).get("parts", []))
                if not txt.strip():
                    last = f"empty text (finishReason={cand.get('finishReason')})"
                else:
                    try:
                        obj = json.loads(txt)
                    except json.JSONDecodeError:
                        last = f"unparseable JSON: {txt[:200]}"
                    else:
                        if obj.get("label") not in LABELS:
                            last = f"bad label: {obj.get('label')!r}"
                        else:
                            obj["_usage"] = payload.get(
                                "usageMetadata", {}).get("totalTokenCount")
                            return obj, None
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            last = f"HTTP {e.code}: {raw[:300]}"
            if e.code in (400, 401, 403):   # not transient -- stop early
                return None, last
            if e.code == 429:
                # Each attempt spends quota. If this is the per-DAY quota there
                # is nothing to wait for, so give up immediately rather than
                # burning the remaining allowance on retries. The DAILY_QUOTA
                # marker tells the caller to abort the whole run, not just this
                # item -- continuing would mark every remaining item failed.
                if "PerDay" in raw or "per day" in raw.lower():
                    return None, "DAILY_QUOTA " + last
                delay = None
                try:
                    for d in json.loads(raw)["error"].get("details", []):
                        if d.get("@type", "").endswith("RetryInfo"):
                            delay = float(str(d["retryDelay"]).rstrip("s"))
                except Exception:                                # noqa: BLE001
                    pass
                time.sleep(min((delay or 10.0) + 1.0, 65.0))
                continue
        except Exception as e:                                   # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < max_attempts - 1:
            time.sleep(min(2 ** attempt + random.uniform(0, 1.0), 30))
    return None, last


def write_review_txt(path, results, title, model=MODEL):
    """Human-reviewable dump, in the style of zero_citation_manual_review.txt."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(title + "\n")
        f.write("=" * 78 + "\n")
        f.write(f"model: {model}   temperature: 0   items: {len(results)}\n")
        f.write("Unit of analysis: ONE CONVERSATION (user turns are identical on\n"
                "both sides, so frustration is not a per-side property).\n")
        f.write("Labels: frustrated / engaged / neutral, each with a one-sentence\n"
                "justification so every call can be checked by hand.\n")
        f.write("=" * 78 + "\n\n")
        for n, r in enumerate(results, 1):
            f.write("-" * 78 + "\n")
            f.write(f"[{n}/{len(results)}]  record_idx={r['idx']}  "
                    f"intent={r['intent']}  languages={r['languages']}\n")
            f.write(f"user_turns={r['n_user_turns']}  "
                    f"models: {r['model_a']} vs {r['model_b']}  winner={r['winner']}\n")
            f.write(f"citations a/b={r['cit_a']}/{r['cit_b']}  "
                    f"resp_len a/b={r['len_a']}/{r['len_b']}\n")
            f.write("-" * 78 + "\n")
            if r.get("error"):
                f.write(f"*** CLASSIFICATION FAILED: {r['error']}\n\n")
            else:
                f.write(f">>> LABEL        : {r['label'].upper()}\n")
                f.write(f">>> JUSTIFICATION: {r['justification']}\n\n")
            f.write("USER TURNS (verbatim, as sent to the classifier):\n")
            for i, t in enumerate(r["user_turns"], 1):
                shown = t if len(t) <= MAX_TURN_CHARS else (
                    t[:MAX_TURN_CHARS] + f"\n...[truncated, {len(t)} chars total]")
                f.write(f"\n  [TURN {i}]\n")
                for line in shown.splitlines() or [""]:
                    f.write(f"    {line}\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="pickle of conversation dicts")
    ap.add_argument("--out-txt", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--title", default="FRUSTRATION CLASSIFIER OUTPUT")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--resume", action="store_true",
                    help="skip conversations already present in --out-jsonl "
                         "(also lets an earlier pilot's results be reused)")
    ap.add_argument("--resume-from", default=None,
                    help="additional jsonl of already-classified records to seed from")
    ap.add_argument("--sleep", type=float, default=0.6,
                    help="pause between calls, to stay inside free-tier limits")
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Refusing to run.")

    convs = pickle.load(open(args.input, "rb"))
    print(f"loaded {len(convs)} conversations from {args.input}")
    print(f"model={args.model}  endpoint=v1beta  temperature=0")

    # ---- resume: reuse anything already classified with this same model+prompt
    done = {}
    seeds = []
    if args.resume_from and os.path.exists(args.resume_from):
        seeds.append(args.resume_from)
    if args.resume and os.path.exists(args.out_jsonl):
        seeds.append(args.out_jsonl)
    for sp in seeds:
        with open(sp, encoding="utf-8") as sf:
            for line in sf:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("label"):
                    done[r["idx"]] = r
    if done:
        print(f"resume: {len(done)} already-classified records available "
              f"({len(set(c['idx'] for c in convs) & set(done))} of them in this input)")

    results = []
    n_fail = 0
    aborted = None
    with open(args.out_jsonl, "w", encoding="utf-8") as jf:
        for i, c in enumerate(convs, 1):
            if c["idx"] in done:
                rec = done[c["idx"]]
                rec["user_turns"] = c["user_turns"]       # keep text for review
                results.append(rec)
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jf.flush()
                print(f"  [{i}/{len(convs)}] idx={c['idx']:6} "
                      f"{str(c.get('languages')):26} -> {rec['label']} (reused)")
                continue
            obj, err = call_gemini(api_key, build_user_block(c),
                                   model=args.model)
            rec = dict(idx=c["idx"], intent=c["intent"],
                       languages=str(c.get("languages")),
                       n_user_turns=len(c["user_turns"]),
                       model_a=c["model_a"], model_b=c["model_b"],
                       winner=c["winner"], cit_a=c["cit_a"], cit_b=c["cit_b"],
                       len_a=c["len_a"], len_b=c["len_b"],
                       user_turns=c["user_turns"])
            if obj:
                rec.update(label=obj["label"],
                           justification=obj["justification"],
                           error=None, usage=obj.get("_usage"))
            else:
                rec.update(label=None, justification=None, error=err, usage=None)
                n_fail += 1
            results.append(rec)
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()
            lab = rec["label"] or f"FAIL({err[:40]})"
            print(f"  [{i}/{len(convs)}] idx={c['idx']:6} "
                  f"{str(c.get('languages')):26} -> {lab}", flush=True)

            if err and err.startswith("DAILY_QUOTA"):
                # Hard daily wall: stop the run NOW. Everything classified so far
                # is already on disk; re-run with --resume tomorrow to continue.
                aborted = err
                print(f"\n!! HARD DAILY QUOTA WALL at item {i}/{len(convs)}. "
                      "Aborting the run immediately.", flush=True)
                break

            if i < len(convs):
                time.sleep(args.sleep)

    write_review_txt(args.out_txt, results, args.title, model=args.model)
    ok = sum(1 for r in results if r.get("label"))
    print(f"\nclassified OK : {ok}")
    print(f"failures      : {n_fail}/{len(results)}")
    print(f"[saved] {args.out_jsonl}")
    print(f"[saved] {args.out_txt}")
    if aborted:
        print("\nRUN INCOMPLETE - hard daily quota reached.")
        print(f"  completed {ok} of {len(convs)} requested")
        print(f"  detail: {aborted[:300]}")
        print("  resume with the same command plus --resume once quota resets.")
        sys.exit(3)


if __name__ == "__main__":
    main()
