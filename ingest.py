#!/usr/bin/env python3
"""Turn an .mbox file, a folder of .eml files, or a single .eml into emails JSONL.

Read-only: it never writes to or deletes from the mailbox.

Usage:
  python3 ingest.py inbox.mbox --out emails.jsonl
  python3 ingest.py path/to/folder_of_eml --out emails.jsonl
Export an mbox from Gmail with Google Takeout (Mail only), or save messages as .eml.
"""
import argparse
import email
import email.policy
import email.utils
import html
import json
import mailbox
import os
import re
from datetime import date


def html_to_text(s):
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", s)
    s = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", s)).strip()


def body_of(msg):
    plain = msg.get_body(preferencelist=("plain",))
    if plain is not None:
        return plain.get_content().strip()
    rich = msg.get_body(preferencelist=("html",))
    if rich is not None:
        return html_to_text(rich.get_content())
    return ""


def convert(msg, idx):
    try:
        dt = email.utils.parsedate_to_datetime(msg["Date"])
        sent = dt.date().isoformat()
    except Exception:
        sent = date.today().isoformat()
    return {
        "id": f"m{idx:05d}",
        "from": str(msg["From"] or "unknown"),
        "to": str(msg["To"] or ""),
        "sent": sent,
        "subject": str(msg["Subject"] or "(no subject)"),
        "body": body_of(msg)[:6000],
    }


def messages(path):
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.lower().endswith(".eml"):
                with open(os.path.join(path, name), "rb") as f:
                    yield email.message_from_binary_file(f, policy=email.policy.default)
    elif path.lower().endswith(".mbox"):
        for m in mailbox.mbox(path, factory=lambda f: email.message_from_binary_file(f, policy=email.policy.default)):
            yield m
    else:
        with open(path, "rb") as f:
            yield email.message_from_binary_file(f, policy=email.policy.default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="emails.jsonl")
    args = ap.parse_args()
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, msg in enumerate(messages(args.path), 1):
            out.write(json.dumps(convert(msg, i), ensure_ascii=False) + "\n")
            n += 1
    print(f"Read {n} emails from {args.path} -> {args.out}")


if __name__ == "__main__":
    main()
