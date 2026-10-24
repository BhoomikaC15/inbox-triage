#!/usr/bin/env python3
"""One command: mailbox export -> labels -> HTML report. Everything runs locally.

  python triage.py inbox.mbox --model gemma3:4b --out report.html
  python triage.py folder_of_eml_files --limit 20          # try a few first

Steps: ingest.py (read-only) -> llm_classify.py (local Ollama model, every date and
quote checked against the email) -> postprocess.py (code fills in dropped dates) ->
make_report.py. It never deletes, moves or sends anything.

To use predictions from another model (for example the Tinker fine-tune), skip this
script and run make_report.py directly on the emails and predictions files.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*cmd):
    print("->", " ".join(cmd))
    subprocess.run([sys.executable] + [os.path.join(HERE, cmd[0])] + list(cmd[1:]), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mailbox", help=".mbox file, .eml file, or folder of .eml files")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--workdir", default="triage_work")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    emails = os.path.join(args.workdir, "emails.jsonl")
    preds = os.path.join(args.workdir, "preds.jsonl")
    fixed = os.path.join(args.workdir, "preds_fixed.jsonl")

    run("ingest.py", args.mailbox, "--out", emails)
    cmd = ["llm_classify.py", emails, "--model", args.model, "--out", preds, "--base-url", args.base_url]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    run(*cmd)
    run("postprocess.py", emails, preds, "--out", fixed)
    run("make_report.py", emails, fixed, "--out", args.out, "--model-name", args.model)
    print(f"\nOpen {args.out} in your browser.")


if __name__ == "__main__":
    main()
