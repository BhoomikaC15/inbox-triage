#!/usr/bin/env python3
"""Score predictions against gold labels (standard library only).

Usage:
  python3 evaluate.py data/test.jsonl preds.jsonl [--json metrics.json]

Predictions are JSONL with: id, label, deadline_date (null unless label is deadline).
"""
import argparse
import json
import random
import sys
from collections import Counter

LABELS = ["deadline", "needs_reply", "might_interest", "skippable"]
IMPORTANT = {"deadline", "needs_reply"}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}:{n}: invalid JSON ({e})")
    return rows


def bootstrap_ci(correct, n_boot=10000, seed=0):
    rng = random.Random(seed)
    n = len(correct)
    accs = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("preds")
    ap.add_argument("--json", help="also save the metrics to this file")
    args = ap.parse_args()

    gold = {r["id"]: r for r in load_jsonl(args.gold)}
    pred = {r["id"]: r for r in load_jsonl(args.preds)}
    missing = [i for i in gold if i not in pred]
    if missing:
        sys.exit(f"Predictions are missing {len(missing)} ids, e.g. {missing[:5]}")
    ids = list(gold)
    n = len(ids)
    y_true = [gold[i]["label"] for i in ids]
    y_pred = [pred[i].get("label") for i in ids]

    correct = [int(t == p) for t, p in zip(y_true, y_pred)]
    acc = sum(correct) / n
    lo, hi = bootstrap_ci(correct)
    print(f"Emails: {n}")
    print(f"Accuracy: {acc:.3f}  (95% bootstrap CI {lo:.3f} to {hi:.3f})")

    print(f"\n{'label':15s} {'prec':>6s} {'recall':>6s} {'f1':>6s} {'support':>8s}")
    f1s, per_label = [], {}
    for lab in LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = tp + fn
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        if support:
            f1s.append(f1)
        print(f"{lab:15s} {prec:6.2f} {rec:6.2f} {f1:6.2f} {support:8d}")
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    print(f"Macro-F1: {macro:.3f}")

    print("\nConfusion matrix (rows = gold, columns = predicted)")
    print(f"{'':15s}" + "".join(f"{l[:10]:>12s}" for l in LABELS))
    cm = Counter(zip(y_true, y_pred))
    for t in LABELS:
        print(f"{t:15s}" + "".join(f"{cm[(t, p)]:12d}" for p in LABELS))

    important = [i for i in ids if gold[i]["label"] in IMPORTANT]
    dangerous = [i for i in important if pred[i].get("label") == "skippable"]
    gold_dl = [i for i in ids if gold[i]["label"] == "deadline"]
    dl_found = [i for i in gold_dl if pred[i].get("label") == "deadline"]
    dl_date_ok = [i for i in dl_found if pred[i].get("deadline_date") == gold[i]["deadline_date"]]
    invented = [i for i in ids if pred[i].get("label") == "deadline" and gold[i]["label"] != "deadline"]

    print("\nThe numbers that matter for an inbox cleaner")
    print(f"Important emails marked skippable (dangerous): {len(dangerous)} of {len(important)}")
    print(f"Deadlines found: {len(dl_found)} of {len(gold_dl)}")
    print(f"Deadlines found with the exact right date: {len(dl_date_ok)} of {len(gold_dl)}")
    print(f"Deadlines invented (predicted deadline, gold says otherwise): {len(invented)}")

    wrong = [i for i in ids if gold[i]["label"] != pred[i].get("label")]
    if wrong:
        print("\nMistakes")
        for i in wrong:
            extra = ""
            if gold[i]["label"] == "deadline" and pred[i].get("label") == "deadline":
                extra = f"  (date: gold {gold[i]['deadline_date']}, pred {pred[i].get('deadline_date')})"
            print(f"  {i}: gold={gold[i]['label']}, pred={pred[i].get('label')}  | {gold[i]['subject']}{extra}")
    date_wrong = [i for i in dl_found if i not in dl_date_ok]
    for i in date_wrong:
        print(f"  {i}: label right, date wrong: gold {gold[i]['deadline_date']}, pred {pred[i].get('deadline_date')}")

    if args.json:
        metrics = {
            "n": n, "accuracy": acc, "accuracy_ci95": [lo, hi], "macro_f1": macro,
            "per_label": per_label,
            "dangerous_skips": [len(dangerous), len(important)],
            "deadlines_found": [len(dl_found), len(gold_dl)],
            "deadline_dates_exact": [len(dl_date_ok), len(gold_dl)],
            "deadlines_invented": len(invented),
            "mistake_ids": wrong,
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {args.json}")


if __name__ == "__main__":
    main()
