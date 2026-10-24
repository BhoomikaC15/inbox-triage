#!/usr/bin/env python3
"""Let code resolve a missing deadline date (standard library only).

Small models often get the label right but the weekday arithmetic wrong
("by Friday", "within 3 days"). The checker in llm_classify.py drops such dates.
This step then asks plain code to fill the gap, but only when the answer is
unambiguous:
  - exactly one distinct date in the email that is on or after the sent date, or
  - the phrase "within N days" (sent date + N).
If the email holds several candidate dates we leave the date empty rather than guess.

Usage:
  python3 postprocess.py emails.jsonl preds.jsonl --out preds_fixed.jsonl
"""
import argparse
import json
import re
from datetime import date, timedelta

from baseline import find_dates, strip_quotes

WITHIN = re.compile(r"\bwithin (\d{1,2}) days?\b", re.I)


def resolve_date(email):
    """Return (iso_date, how) or (None, None)."""
    sent = date.fromisoformat(email["sent"])
    text = f"{email['subject']}\n{strip_quotes(email['body'])}"
    m = WITHIN.search(text)
    if m:
        return (sent + timedelta(days=int(m.group(1)))).isoformat(), "within_n_days"
    dates = {d for _, d in find_dates(text, sent) if d >= sent}
    if len(dates) == 1:
        return next(iter(dates)).isoformat(), "single_date_in_email"
    return None, None


def apply(email, pred):
    """Return a copy of pred with a code-resolved date when the model's was dropped."""
    if pred.get("label") == "deadline" and not pred.get("deadline_date"):
        d, how = resolve_date(email)
        if d:
            fixed = dict(pred)
            fixed["deadline_date"] = d
            fixed["flags"] = list(pred.get("flags", [])) + ["date_resolved_by_code"]
            return fixed
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emails")
    ap.add_argument("preds")
    ap.add_argument("--out", default="preds_fixed.jsonl")
    args = ap.parse_args()
    emails = {}
    for l in open(args.emails, encoding="utf-8"):
        if l.strip():
            e = json.loads(l)
            emails[e["id"]] = e
    n = fixed = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for l in open(args.preds, encoding="utf-8"):
            if not l.strip():
                continue
            p = json.loads(l)
            q = apply(emails[p["id"]], p) if p["id"] in emails else p
            fixed += q is not p
            n += 1
            out.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"{n} predictions, {fixed} deadline dates filled in by code -> {args.out}")


if __name__ == "__main__":
    main()
