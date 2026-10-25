# Inbox Triage

*My entry for the Hacktoberfest Open-Source AI Challenge, Week 3: Slow and Steady.*

I hate going through emails, and I believe we all do. I don't want to know that someone I barely know commented on a LinkedIn post. I want to know whether my pull request was merged. So I built a tool that reads a mailbox export and sorts every email into a folder, then shows the few messages that matter first.

It only **reads**. It never deletes, moves, sends, or edits anything. The final decision stays in your hands.

## The folders

Every email gets exactly one label.

| Label | What it means |
| --- | --- |
| **Deadline** | You must do something by a date. The report shows the date and the weekday. |
| **Needs reply** | Someone is waiting for your answer, and no date is attached. |
| **Might interest you** | Optional, but possibly worth a look: events, contests, newsletters, opportunities. |
| **Skippable** | Ads, receipts, notifications, FYIs, auto-replies, and other people's deadlines. Spam and promotions land here too. |
| **OTP codes** | Filled by a plain rule, not by the model (see below). The code is shown on the row. |

The report is a single HTML file styled like Gmail for familiarity. The first screen shows only the counts. You open one folder at a time, and each email shows the exact line on which its label was based.

## Results

I tested on 96 held-out synthetic emails (25 deadline, 25 needs reply, 19 might interest, 27 skippable), about a third of them hard on purpose.

| Method | Accuracy (95% interval) | Important emails marked skippable | Exact deadline dates |
| --- | --- | --- | --- |
| Regex baseline | 68.8% (59-78%) | 11 of 50 | 14 of 25 |
| Gemma 3 4B, local | 76.0% (68-84%) | 0 of 50 | 21 of 25 |
| Qwen3.5-4B, untuned | 94.8% (90-99%) | 1 of 50 | 20 of 25 |
| Qwen3.5-4B, fine-tuned on Tinker | 97.9% (95-100%) | 1 of 50 | 23 of 25 |

Honest reading:

- The fine-tuned model is clearly better than the baseline and Gemma.
- Against untuned Qwen the gap is within noise (it fixed 5 emails and broke 2, p about 0.45). I do not claim the fine-tune is more accurate.
- What the fine-tune changes is the prompt: 133 input tokens per email instead of 368. That is roughly $0.09 instead of $0.16 per 1,000 emails at Tinker's list prices, and no saving if the long prompt's prefix is cached. These are estimates from the price table, not from a bill.
- The 24-of-25 date figure with the code fallback was added after I saw the test errors, so it is post hoc.
- **The OTP rule is outside these numbers.** I added it late and checked it on only a few OTP emails.
- Everything is synthetic. I also spot-checked a small number of my own real emails and disagreed with a few labels. That is an anecdote, not a benchmark.

## Why I trust the labels: code checks the model

The model only does the fuzzy part: one label, a deadline date, and one quoted line. Plain Python then checks it:

- A deadline date is trusted only if it really appears in the email (words like "Friday" are resolved against the sent date) and is not before it.
- The quoted line must really be in the email.
- If the model's answer can't be read, the email is shown as **Needs reply** with a "Check me" note, so it is never hidden.
- If the model drops a date, code fills it in only when the email has exactly one candidate date or says "within N days".
- OTPs: if an email names a one-time code and contains a 4-8 digit code, it goes to the OTP folder by rule.

## Two ways to run it

**1. Local with Ollama (private).** Nothing leaves your machine. This is the path I use for real mail. Its speed depends on your RAM and GPU.

**2. Fine-tuned Qwen on Tinker (cloud).** This was my best result. It sends email text to Tinker's servers, so use it on the synthetic data, not on a real inbox, unless you are fine with that.

## Quick start (local)

You need Python 3.9 or newer and [Ollama](https://ollama.com/). The local pipeline uses only the standard library, so there is no `pip install`.

```text
ollama pull gemma3:4b
python triage.py path\to\inbox.mbox --model gemma3:4b --out report.html
```

Try a few emails first:

```text
python triage.py path\to\folder_of_eml_files --limit 20
```

To get a mailbox: use Google Takeout and pick **Mail** only (you get an `.mbox`), or save emails as `.eml` files. Open `report.html` in a browser. It makes no network requests.

**Edit `interests.txt` first.** It contains example interests. Rewrite it according to your interests or whatever you would like to look for. Put one interest per line (music, hackathons, open source, or anything else you care about); the model uses these interests to decide what might interest you.

Intermediate files go to `triage_work\`. The classifier can resume from earlier results in this folder. Delete the folder first when you want to start a completely new run.

## Reproduce the results

```text
python evaluate.py test.jsonl preds_baseline.jsonl
python evaluate.py test.jsonl preds_gemma.jsonl
python evaluate.py test.jsonl preds_qwen_base.jsonl
python evaluate.py test.jsonl preds_tinker.jsonl
```

To redo the fine-tune you need `pip install tinker` and a Tinker login (`tinker auth login`). The key is never in this repo. Run `python tinker_finetune.py --help` for the `train` and `predict` options. I trained on 320 synthetic emails made by `generate_train.py`, with any email too similar to a test email dropped.

To measure the prompt size and estimated cost:

```text
python measure_tokens.py test.jsonl preds_qwen_base.jsonl preds_tinker.jsonl
```

## Labeling rules

If more than one label fits, this is the priority: deadline, then needs reply, then might interest, then skippable. A date alone does not make a deadline. Optional events stay in Might interest, and someone else's deadline is Skippable. Old quoted text is ignored where possible. Details and edge cases are in [`LABELING_GUIDE.md`](LABELING_GUIDE.md).

## Files

| File | What it does |
| --- | --- |
| `triage.py` | Runs the whole mailbox-to-report pipeline |
| `ingest.py` | Reads `.mbox` / `.eml` without modifying them |
| `llm_classify.py` | Calls the local Ollama model and verifies its claims |
| `postprocess.py` | Fills in a missing date only when it is unambiguous |
| `otp.py` | The rule that recognizes OTP emails |
| `make_report.py` | Builds the Gmail-style HTML report |
| `tinker_finetune.py` | Fine-tunes and runs Qwen3.5-4B on Tinker |
| `generate_train.py` | Generates the synthetic training emails |
| `measure_tokens.py` | Measures prompt size and estimated cost |
| `baseline.py` | The keyword/regex baseline and date finder |
| `evaluate.py` | Accuracy, per-label scores, dangerous skips, date accuracy |
| `build_dataset.py`, `validate.py` | Turn authored emails into JSONL and check them |
| `interests.txt` | My interests, used by the model |
| `LABELING_GUIDE.md` | The labeling rules with examples |

## Limits

- All test and training emails are synthetic. I used Claude's help to draft the hundreds of test and training emails (I reviewed every label), which could make them easier for a model than real mail.
- My training emails are templated and the fine-tune's loss fell to almost zero. The held-out test set is the real check, but real mail is messier.
- Skippable means "you can ignore it", not "it is safe". The tool does not detect phishing.
- There is no label for important notifications like "your pull request was merged". That is the first thing I would add.
- I did not test a closed model, so I can't say how much better one would do.
