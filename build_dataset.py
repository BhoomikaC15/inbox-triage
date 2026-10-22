#!/usr/bin/env python3
"""Convert the plain-text email authoring format into JSONL.

Usage: python3 build_dataset.py data/dev.txt data/dev.jsonl

Authoring format (see data/dev.txt): header lines, one blank line, then the
body. Emails are separated by a line containing only =====
"""
import json
import re
import sys

REQUIRED = ["id", "from", "to", "sent", "subject", "label"]
KNOWN = set(REQUIRED) | {"deadline_date", "notes"}


def parse_block(block, number):
    lines = block.splitlines()
    # Drop leading blank lines and leading comment lines (# ...).
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("#")):
        lines.pop(0)
    if not lines:
        return None
    text = "\n".join(lines)
    header, sep, body = text.partition("\n\n")
    if not sep:
        sys.exit(f"Email #{number}: no blank line between header and body.")
    record = {}
    for line in header.splitlines():
        if not line.strip():
            continue
        key, colon, value = line.partition(":")
        if not colon:
            sys.exit(f"Email #{number}: header line without a colon: {line!r}")
        key = key.strip().lower()
        if key not in KNOWN:
            sys.exit(f"Email #{number}: unknown header {key!r}. Known: {sorted(KNOWN)}")
        record[key] = value.strip()
    missing = [k for k in REQUIRED if not record.get(k)]
    if missing:
        sys.exit(f"Email #{number} ({record.get('id', '?')}): missing {missing}")
    record["body"] = body.strip()
    record["deadline_date"] = record.get("deadline_date") or None
    if "notes" not in record:
        record["notes"] = ""
    return record


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        raw = f.read()
    blocks = re.split(r"^=====\s*$", raw, flags=re.M)
    records = []
    for i, block in enumerate(blocks, 1):
        rec = parse_block(block, i)
        if rec:
            records.append(rec)
    with open(dst, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} emails to {dst}")


if __name__ == "__main__":
    main()
