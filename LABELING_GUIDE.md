# Labeling guide

Every email gets exactly ONE label. If two seem to fit, use the priority order:

1. `deadline`: something YOU must do by a specific date or day. Fill `deadline_date` (YYYY-MM-DD). Resolve "this Friday" against the `sent` date.
2. `needs_reply`: someone is waiting for YOUR answer, and no date is attached.
3. `might_interest`: not required, but you might want it (events, contests, newsletters you like).
4. `skippable`: sales, OTPs, receipts, notifications, FYIs, someone else's deadline.

Rules of thumb:
- A date in the email is not enough. Ask: "Do I have to do something by then?"
- Deadline beats needs_reply (d09 in the dev set).
- Fake urgency ("sale ends tonight") is skippable (d04).
- An optional event with a date is might_interest, not deadline (d06).
- Quoted old text (lines starting with >) does not count.

## Two examples

**Example A: deadline**

```
id: t001
from: Prof. R. Mehta <rmehta@college.example>
to: Sam <sam@college.example>
sent: 2026-10-19
subject: DBMS Assignment 3 - submission
label: deadline
deadline_date: 2026-10-28
notes: Two dates. Deadline is the first, the second is a hard cutoff.

Dear students,

Assignment 3 is open on the portal. Submit your PDF by 28 October, 11:59 PM. Late submissions lose 20% and nothing is accepted after 30 October.

Regards,
Prof. Mehta
```

**Example B: needs_reply**

```
id: t002
from: Ananya <ananya@mail.example>
to: Sam <sam@mail.example>
sent: 2026-10-21
subject: slides?
label: needs_reply

Hey Sam, did you finish your part of the slides? Can you send it when you get a chance? Also, which day works for the team call, Saturday or Sunday?

Ananya
```

## How to write the test set quickly

1. Copy the format from data/dev.txt into data/test.txt. One email per block, separated by a line of `=====`.
2. Aim for about 100 emails: roughly 25 deadline, 25 needs_reply, 20 might_interest, 30 skippable.
3. Write them yourself in your own words. Fictional names and fake domains only. Do not paste real emails with other people's details.
4. Include hard cases on purpose, because that is where the model and baseline differ: relative dates ("by Friday"), two dates in one email, someone else's deadline, fake urgency, quoted replies, and a "deadline" that is really an optional event.
5. Run `python3 build_dataset.py data/test.txt data/test.jsonl` then `python3 validate.py data/test.jsonl`.
6. NEVER use the test set to train or to tune prompts. Build a separate train set for the Tinker fine-tune.
