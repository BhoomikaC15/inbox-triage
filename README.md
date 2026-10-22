# Email triage (work in progress)

Sorts an inbox into: **deadline** (with date), **needs reply**, **might interest you**, **skippable**.
It never deletes anything. It only labels.

## Files

| File | Purpose |
|---|---|
| `LABELING_GUIDE.md` | Label rules and examples for writing the dataset |
| `data/dev.txt` | 12 hand-written dev emails (for building, not for reporting) |
| `build_dataset.py` | Converts `.txt` authoring format to `.jsonl` |
| `validate.py` | Checks labels, dates, class balance, near-duplicates |
| `baseline.py` | Regex and keyword baseline (the number to beat) |
| `evaluate.py` | Accuracy with CI, per-label F1, confusion matrix, dangerous skips, deadline-date accuracy |
| `interests.txt` | Your interests, used by the baseline for "might interest you" |

## Run

```
python3 build_dataset.py data/dev.txt data/dev.jsonl
python3 validate.py data/dev.jsonl
python3 baseline.py data/dev.jsonl --out preds_baseline.jsonl
python3 evaluate.py data/dev.jsonl preds_baseline.jsonl
```

Standard library only, no installs.

## Dev-set baseline result (12 emails, so only a smoke test)

Accuracy 10/12. Both mistakes are the designed traps: an optional event mistaken for needing a reply, and someone else's deadline mistaken for yours.

## Next

- Gemma classifier via Ollama (same output format as the baseline)
- Code check that every date the model reports appears in the email
- 100-email test set, then the same scorer on baseline vs Gemma vs (optional) Tinker fine-tune
