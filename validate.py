#!/usr/bin/env python3
"""Check a JSONL email set for labeling mistakes.

Usage: python3 validate.py data/test.jsonl
Exit code is 1 if there are errors (warnings do not fail).
"""
import json
import re
import sys
from collections import Counter
from datetime import date

LABELS = ["deadline", "needs_reply", "might_interest", "skippable"]
REQUIRED = ["id", "from", "to", "sent", "subject", "body", "label"]


def parse_date(s):
    try:
        y, m, d = (int(x) for x in s.split("-"))
        return date(y, m, d)
    except Exception:
        return None


def shingles(text, n=4):
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rows = []
    errors, warnings = [], []
    with open(sys.argv[1], encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                errors.append(f"line {n}: invalid JSON ({e})")

    seen = set()
    for r in rows:
        rid = r.get("id", "?")
        for k in REQUIRED:
            if not r.get(k):
                errors.append(f"{rid}: missing {k}")
        if rid in seen:
            errors.append(f"{rid}: duplicate id")
        seen.add(rid)
        label = r.get("label")
        if label not in LABELS:
            errors.append(f"{rid}: label {label!r} is not one of {LABELS}")
        sent = parse_date(r.get("sent") or "")
        if sent is None:
            errors.append(f"{rid}: sent must be YYYY-MM-DD")
        dd = r.get("deadline_date")
        if label == "deadline":
            d = parse_date(dd or "")
            if d is None:
                errors.append(f"{rid}: label deadline needs deadline_date as YYYY-MM-DD")
            elif sent and d < sent:
                errors.append(f"{rid}: deadline_date {dd} is before sent date {r['sent']}")
        elif dd:
            errors.append(f"{rid}: deadline_date is only for label deadline")
        words = len((r.get("body") or "").split())
        if words < 12:
            warnings.append(f"{rid}: body is very short ({words} words)")
        if words > 300:
            warnings.append(f"{rid}: body is long ({words} words)")

    sets = [(r.get("id", "?"), shingles(r.get("body", ""))) for r in rows]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i][1], sets[j][1]
            if a and b and len(a & b) / len(a | b) > 0.6:
                warnings.append(f"{sets[i][0]} and {sets[j][0]} look like near-duplicates")

    counts = Counter(r.get("label") for r in rows)
    print(f"{len(rows)} emails")
    for lab in LABELS:
        share = counts[lab] / len(rows) if rows else 0
        print(f"  {lab:15s} {counts[lab]:4d}  ({share:.0%})")
        if len(rows) >= 50 and share < 0.10:
            warnings.append(f"label {lab} is under 10% of the set")
    for w in warnings:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    if errors:
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
