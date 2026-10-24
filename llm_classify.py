#!/usr/bin/env python3
"""Classify emails with a local model through Ollama (standard library only).

The model only does the fuzzy part: pick a label, a deadline date and the
sentence it relied on. Code does the risky part: every date and every quote the
model reports is checked against the email text before it is trusted.
This script only READS the input file. It never touches a mailbox and never
deletes anything.

Usage:
  ollama pull gemma3:4b
  python3 llm_classify.py data/test.jsonl --model gemma3:4b --out preds_gemma.jsonl
  python3 llm_classify.py data/test.jsonl --limit 3        # quick speed test

Safe failure: if the model's answer cannot be parsed or has an invalid label,
the email is labeled needs_reply (so it is shown, not hidden) and flagged.
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import date

from baseline import find_dates, strip_quotes

LABELS = ["deadline", "needs_reply", "might_interest", "skippable"]

SYSTEM = """You sort one email for its recipient, Sam. Reply with JSON only.

Labels (pick exactly one):
- deadline: Sam must DO something by a specific date, or must attend a mandatory event on a fixed date. A reply request with a date is a deadline.
- needs_reply: someone is waiting for Sam's answer or action, and no date is attached.
- might_interest: optional but possibly interesting to a CS student (events, contests, newsletters, opportunities). An optional event with a date is might_interest, not deadline.
- skippable: ads, OTPs, receipts, notifications, FYIs, auto-replies, and anything where someone else has the deadline.

Rules:
- Ignore quoted old text (lines starting with >). Judge the newest message only.
- Fake urgency in ads ("sale ends tonight") is skippable.
- If the label is deadline, give deadline_date as YYYY-MM-DD. Resolve words like "Friday" or "tomorrow" using the SENT date given. If the email has two dates, use the first one Sam must act on.
- evidence: copy ONE short sentence or phrase from the email exactly as written that supports your label.
- Otherwise deadline_date must be null.

JSON format: {"label": "...", "deadline_date": "YYYY-MM-DD or null", "evidence": "..."}"""


def build_user_message(email):
    return (
        f"SENT: {email['sent']} ({date.fromisoformat(email['sent']).strftime('%A')})\n"
        f"FROM: {email['from']}\n"
        f"SUBJECT: {email['subject']}\n\n"
        f"{email['body']}"
    )


def call_ollama(base_url, model, email, timeout):
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 150},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_message(email)},
        ],
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)["message"]["content"]
    except urllib.error.HTTPError as e:
        # Ollama explains the failure in the response body (e.g. not enough memory).
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {e.code} from Ollama: {detail}") from None


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_answer(raw):
    """Return (dict or None). Tolerates extra text around the JSON object."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", raw or "", re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def verify(email, obj):
    """Check the model's claims against the email. Returns (result, flags)."""
    flags = []
    label = str(obj.get("label", "")).strip().lower().replace(" ", "_")
    if label not in LABELS:
        return None, ["invalid_label"]

    sent = date.fromisoformat(email["sent"])
    text = f"{email['subject']}\n{strip_quotes(email['body'])}"

    evidence = str(obj.get("evidence") or "").strip()
    if evidence and norm(evidence) not in norm(email["subject"] + " " + email["body"]):
        flags.append("evidence_not_in_email")

    deadline_date = None
    if label == "deadline":
        raw_date = obj.get("deadline_date")
        try:
            claimed = date.fromisoformat(str(raw_date))
        except (ValueError, TypeError):
            claimed = None
        if claimed is None:
            flags.append("deadline_without_valid_date")
        else:
            supported = {d for _, d in find_dates(text, sent)}
            if claimed in supported and claimed >= sent:
                deadline_date = claimed.isoformat()
            else:
                flags.append("date_not_found_in_email")
    return {"label": label, "deadline_date": deadline_date, "evidence": evidence}, flags


def classify(email, args):
    raw, error = None, None
    for attempt in range(2):
        try:
            raw = call_ollama(args.base_url, args.model, email, args.timeout)
        except Exception as e:  # connection refused, timeout, bad response
            error = f"{type(e).__name__}: {e}"
            continue
        obj = parse_answer(raw)
        if obj is not None:
            result, flags = verify(email, obj)
            if result is not None:
                return {**result, "flags": flags, "raw": raw}
        error = "unparseable_or_invalid_answer"
    # Safe failure: show the email instead of hiding it.
    return {"label": "needs_reply", "deadline_date": None, "evidence": "",
            "flags": ["model_failed", error or ""], "raw": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emails")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--out", default="preds_gemma.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="only the first N emails")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    emails = [json.loads(l) for l in open(args.emails, encoding="utf-8") if l.strip()]
    if args.limit:
        emails = emails[: args.limit]

    done = {}
    try:  # resume: keep earlier results so a crash or sleep doesn't cost the run
        for l in open(args.out, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if "model_failed" not in r.get("flags", []):  # retry failures
                    done[r["id"]] = r
    except FileNotFoundError:
        pass

    start = time.time()
    with open(args.out, "a", encoding="utf-8") as out:
        for i, email in enumerate(emails, 1):
            if email["id"] in done:
                continue
            t0 = time.time()
            res = classify(email, args)
            res["id"] = email["id"]
            res["seconds"] = round(time.time() - t0, 1)
            out.write(json.dumps(res, ensure_ascii=False) + "\n")
            out.flush()
            flag = f"  flags={res['flags']}" if res["flags"] else ""
            print(f"[{i}/{len(emails)}] {email['id']} -> {res['label']} "
                  f"{res['deadline_date'] or ''} ({res['seconds']}s){flag}")
    print(f"Done in {time.time() - start:.0f}s. Predictions in {args.out}")


if __name__ == "__main__":
    main()
