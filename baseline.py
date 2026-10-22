#!/usr/bin/env python3
"""Keyword/regex baseline for email triage (standard library only).

This is what the language model has to beat. It is meant to be reasonable,
not a strawman: skip patterns, deadline phrases, date parsing (including
"by Friday" resolved against the sent date) and interest keywords.

Order of rules: skippable signals, then deadline, then needs_reply, then
might_interest, then fall back to skippable.

Usage:
  python3 baseline.py data/dev.jsonl --out preds_baseline.jsonl
"""
import argparse
import json
import re
from datetime import date, timedelta

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTH_RE = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")

P_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
P_SLASH = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")  # assumes day/month/year
P_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?{MONTH_RE}\b", re.I)
P_MONTH_DAY = re.compile(rf"\b{MONTH_RE}\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", re.I)
P_WEEKDAY = re.compile(r"\b(" + "|".join(WEEKDAYS) + r")\b", re.I)
P_REL = re.compile(r"\b(today|tonight|tomorrow)\b", re.I)

SKIP_RE = re.compile(
    r"unsubscribe|no-?reply|\botp\b|one-time password|\d+% off|flash sale|\binvoice\b|"
    r"payment received|\breceipt\b|notifications?@|viewed your profile|view in browser|"
    r"manage notification", re.I)
DEADLINE_RE = re.compile(
    r"\bdeadline\b|\bdue (?:on|by|date|in|at)\b|\b(?:is|are) due\b|\bsubmit\w*|\bsubmission\b|"
    r"\blast date\b|\bno later than\b|"
    r"\bby (?:this |next )?(?:" + "|".join(WEEKDAYS) + r")\b|"
    r"\bby (?:tomorrow|tonight|today|eod|end of day)\b|"
    rf"\bby \d{{1,2}}(?:st|nd|rd|th)? {MONTH_RE}\b|\bby {MONTH_RE} \d{{1,2}}\b", re.I)
REPLY_RE = re.compile(r"\?|please (?:reply|respond|confirm)|let me know|get back to me|reply to this", re.I)


def month_number(token):
    return MONTHS.index(token[:3].lower()) + 1


def make_date(day, month, sent):
    try:
        d = date(sent.year, month, day)
        if d < sent:
            d = date(sent.year + 1, month, day)
        return d
    except ValueError:
        return None


def find_dates(text, sent):
    """Return [(position, date)] in text order, without overlapping matches."""
    found = []
    for m in P_ISO.finditer(text):
        try:
            found.append((m.start(), m.end(), date(int(m[1]), int(m[2]), int(m[3]))))
        except ValueError:
            pass
    for m in P_SLASH.finditer(text):
        year = int(m[3]) + (2000 if int(m[3]) < 100 else 0)
        try:
            found.append((m.start(), m.end(), date(year, int(m[2]), int(m[1]))))
        except ValueError:
            pass
    for m in P_DAY_MONTH.finditer(text):
        d = make_date(int(m[1]), month_number(m[2]), sent)
        if d:
            found.append((m.start(), m.end(), d))
    for m in P_MONTH_DAY.finditer(text):
        d = make_date(int(m[2]), month_number(m[1]), sent)
        if d:
            found.append((m.start(), m.end(), d))
    for m in P_WEEKDAY.finditer(text):
        delta = (WEEKDAYS.index(m[1].lower()) - sent.weekday()) % 7 or 7
        found.append((m.start(), m.end(), sent + timedelta(days=delta)))
    for m in P_REL.finditer(text):
        offset = 1 if m[1].lower() == "tomorrow" else 0
        found.append((m.start(), m.end(), sent + timedelta(days=offset)))
    found.sort()
    kept, last_end = [], -1
    for start, end, d in found:
        if start >= last_end:
            kept.append((start, d))
            last_end = end
    return kept


def strip_quotes(body):
    return "\n".join(l for l in body.splitlines() if not l.lstrip().startswith(">"))


def load_interests(path):
    words = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line and not line.startswith("#"):
                words.append(line)
    return words


def classify(email, interests):
    sent = date.fromisoformat(email["sent"])
    body = strip_quotes(email["body"])
    text = f"{email['subject']}\n{body}"
    everything = f"{email['from']}\n{text}"

    if SKIP_RE.search(everything):
        return "skippable", None
    if DEADLINE_RE.search(text):
        dates = [d for _, d in find_dates(text, sent) if d >= sent]
        return "deadline", (dates[0].isoformat() if dates else None)
    if REPLY_RE.search(text):
        return "needs_reply", None
    lowered = text.lower()
    if any(re.search(r"\b" + re.escape(w) + r"\b", lowered) for w in interests):
        return "might_interest", None
    return "skippable", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emails", help="JSONL file of emails")
    ap.add_argument("--out", default="preds_baseline.jsonl")
    ap.add_argument("--interests", default="interests.txt")
    args = ap.parse_args()
    interests = load_interests(args.interests)
    n = 0
    with open(args.emails, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            email = json.loads(line)
            label, dd = classify(email, interests)
            out.write(json.dumps({"id": email["id"], "label": label, "deadline_date": dd}) + "\n")
            n += 1
    print(f"Wrote {n} predictions to {args.out}")


if __name__ == "__main__":
    main()
