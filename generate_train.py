#!/usr/bin/env python3
"""Generate the synthetic TRAINING set for the fine-tune (standard library only).

Gold labels and deadline dates are computed by code, not guessed, so every
label here is consistent with the labeling guide. The set is kept separate from
data/test.jsonl: any email whose text is too similar to a test email is dropped.

Usage:
  python3 generate_train.py --test test.jsonl --out train.jsonl --n 320 --seed 11
"""
import argparse
import json
import random
import re
from datetime import date, timedelta

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]

PROFS = [("Prof. Nair", "nair@college.example"), ("Dr. Banerjee", "banerjee@college.example"),
         ("Prof. Kulkarni", "kulkarni@college.example"), ("Dr. Sethi", "sethi@college.example"),
         ("Prof. Ghosh", "ghosh@college.example"), ("Dr. Pillai", "pillai@college.example")]
OFFICES = [("Registrar Office", "registrar@college.example"), ("Exam Section", "exams@college.example"),
           ("Training and Placement", "tnp@college.example"), ("Accounts Office", "accounts@college.example"),
           ("Student Affairs", "affairs@college.example"), ("Hostel Warden", "warden@college.example")]
PEERS = [("Vikas", "vikas@mail.example"), ("Ishita", "ishita@mail.example"), ("Tanvi", "tanvi@mail.example"),
         ("Arjun", "arjun@mail.example"), ("Sana", "sana@mail.example"), ("Dev", "dev@mail.example"),
         ("Mitali", "mitali@mail.example"), ("Yash", "yash@mail.example")]
COMPANIES = [("HireLoop", "talent@hireloop.example"), ("CodeNest", "recruit@codenest.example"),
             ("DataForge", "hr@dataforge.example"), ("PixelWorks", "careers@pixelworks.example")]
BRANDS = [("MegaMart", "deals@megamart.example"), ("TrendyWear", "offers@trendywear.example"),
          ("GadgetHub", "promo@gadgethub.example"), ("TravelNow", "sale@travelnow.example"),
          ("SnackBox", "hello@snackbox.example")]
ORGS = [("Tech Society", "tech@college.example"), ("Dev Circle", "hello@devcircle.example"),
        ("Linux Users Group", "lug@college.example"), ("Data Science Club", "dsc@college.example"),
        ("Cyber Cell", "cyber@college.example"), ("Hack Club", "hack@college.example")]

THINGS = ["lab record", "project report", "scholarship form", "hostel clearance form", "internship letter",
          "exam registration form", "fee receipt copy", "seminar topic", "mini-project synopsis",
          "course feedback form", "no-dues certificate", "attendance sheet", "workshop report"]
TOPICS = ["web security", "Rust for beginners", "machine learning basics", "Docker and containers",
          "data structures", "open-source contributions", "SQL performance", "Linux internals",
          "prompt engineering", "competitive programming", "cloud deployment", "Git workflows"]
TIMES = ["10 AM", "11:30 AM", "2 PM", "3:30 PM", "4 PM", "9 AM"]
OPENERS = ["Hi Sam,", "Hey Sam,", "Hello Sam,", "Sam,", ""]


def fmt_date(d, rng):
    style = rng.choice(["dm", "md", "ord", "iso", "slash", "short"])
    if style == "dm":
        return f"{d.day} {MONTHS[d.month - 1]}"
    if style == "md":
        return f"{MONTHS[d.month - 1]} {d.day}"
    if style == "ord":
        suf = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
        return f"{d.day}{suf} {MONTHS[d.month - 1]}"
    if style == "iso":
        return d.isoformat()
    if style == "slash":
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    return f"{d.day} {MONTHS[d.month - 1][:3]}"


def pick(rng, seq):
    return rng.choice(seq)


def person(rng, pool):
    n, a = pick(rng, pool)
    return n, a


def mk(rng, sender, subject, sentences, label, ev_idx=0, dd=None, sent=None, joiner=" ", sign=None, extra=""):
    opener = pick(rng, OPENERS)
    body_sents = list(sentences)
    body = joiner.join(body_sents)
    evidence = body_sents[ev_idx] if body_sents else ""
    text = (opener + "\n\n" if opener else "") + body
    if sign:
        text += f"\n\n{sign}"
    if extra:
        text += f"\n\n{extra}"
    return {"from": f"{sender[0]} <{sender[1]}>", "to": "Sam <sam@mail.example>",
            "sent": sent.isoformat(), "subject": subject, "body": text, "label": label,
            "deadline_date": dd.isoformat() if dd else None, "notes": "", "evidence": evidence}


# ---------------------------------------------------------------- deadline
def gen_deadline(rng):
    kind = rng.choice(["abs", "abs", "abs", "weekday", "tomorrow", "next", "event", "subject",
                       "two", "reply", "within", "money", "peer"])
    sent = date(2026, 9, 1) + timedelta(days=rng.randrange(0, 44))
    thing = pick(rng, THINGS)
    sender = pick(rng, [pick(rng, PROFS), pick(rng, OFFICES)])
    sign = sender[0]

    if kind == "abs":
        d = sent + timedelta(days=rng.randrange(3, 25))
        ds = fmt_date(d, rng)
        s = pick(rng, [f"Please submit your {thing} by {ds}.", f"The last date for the {thing} is {ds}.",
                       f"Your {thing} is due on {ds}. Late entries will not be accepted.",
                       f"Kindly complete your {thing} before {ds}.",
                       f"We need your {thing} no later than {ds}.", f"Final deadline for the {thing}: {ds}."])
        return mk(rng, sender, f"{thing.capitalize()}", [s, "Incomplete entries will be rejected."], "deadline", 0, d, sent, sign=sign)
    if kind == "weekday":
        ahead = rng.randrange(1, 5)
        d = sent + timedelta(days=ahead)
        wd = WEEKDAYS[d.weekday()]
        s = pick(rng, [f"Please send your {thing} by {wd}.", f"Your {thing} needs to reach us by this {wd}.",
                       f"Submit the {thing} by {wd} evening without fail."])
        return mk(rng, sender, f"{thing.capitalize()} reminder", [s, "No extensions will be given."], "deadline", 0, d, sent, sign=sign)
    if kind == "tomorrow":
        d = sent + timedelta(days=1)
        s = pick(rng, [f"Reminder: the {thing} is due tomorrow at {pick(rng, TIMES)}.",
                       f"Your {thing} must be handed in tomorrow before noon.",
                       f"Just a reminder that I need the {thing} tomorrow."])
        return mk(rng, sender, f"Due tomorrow: {thing}", [s], "deadline", 0, d, sent, sign=sign)
    if kind == "next":
        for _ in range(20):
            d = sent + timedelta(days=rng.randrange(5, 12))
            if (d - sent).days >= 5:
                break
        wd = WEEKDAYS[d.weekday()]
        # "next <weekday>" is only unambiguous when the first such weekday is >= 5 days away
        first = sent + timedelta(days=(d.weekday() - sent.weekday()) % 7 or 7)
        if first != d:
            d = first
        if (d - sent).days < 4:
            d = d + timedelta(days=7)
        s = f"Please share your {thing} by next {WEEKDAYS[d.weekday()]}."
        return mk(rng, sender, f"{thing.capitalize()}", [s, "Students who miss this will be marked absent."], "deadline", 0, d, sent, sign=sign)
    if kind == "event":
        d = sent + timedelta(days=rng.randrange(2, 14))
        ev = pick(rng, ["Unit test 1", "The lab exam", "Mock group discussion", "The orientation session",
                        "Your project viva", "The aptitude test"])
        s = f"{ev} will be held on {WEEKDAYS[d.weekday()]} {fmt_date(d, rng)} at {pick(rng, TIMES)}."
        return mk(rng, sender, f"{ev}: schedule", [s, "Attendance is mandatory. Bring your ID card."], "deadline", 0, d, sent, sign=sign)
    if kind == "subject":
        d = sent + timedelta(days=rng.randrange(4, 20))
        ds = fmt_date(d, rng)
        subj = f"Action needed by {ds}: {thing}"
        return mk(rng, sender, subj, [f"Please complete the {thing} on the portal.", "Accounts with pending items will be restricted."],
                  "deadline", 0, d, sent, sign=sign)
    if kind == "two":
        d1 = sent + timedelta(days=rng.randrange(4, 12))
        d2 = d1 + timedelta(days=rng.randrange(7, 15))
        s = f"The first draft of the {thing} is due {fmt_date(d1, rng)}. The final version is due {fmt_date(d2, rng)}."
        return mk(rng, sender, f"{thing.capitalize()} timeline", [s, "Feedback will be given on the draft."], "deadline", 0, d1, sent, sign=sign)
    if kind == "reply":
        d = sent + timedelta(days=rng.randrange(2, 6))
        wd = WEEKDAYS[d.weekday()]
        s = f"Please reply with your preferred slot for the {pick(rng, ['viva', 'meeting', 'counselling session'])} by {wd}."
        return mk(rng, sender, "Slot confirmation", [s, "Unconfirmed students will get the last slot."], "deadline", 0, d, sent, sign=sign)
    if kind == "within":
        n = rng.randrange(2, 6)
        d = sent + timedelta(days=n)
        s = f"Please upload the signed {thing} within {n} days."
        return mk(rng, sender, f"Upload {thing}", [s, "Late uploads are not processed."], "deadline", 0, d, sent, sign=sign)
    if kind == "money":
        d = sent + timedelta(days=rng.randrange(5, 20))
        amt = rng.choice([450, 1200, 2750, 3900, 860])
        s = f"Amount due: Rs. {amt}. Pay by {fmt_date(d, rng)} to avoid a late fee."
        spender = pick(rng, [("Northline Bank", "bills@northline.example"), ("PowerGrid Utility", "billing@powergrid.example"),
                             ("NetBroad", "accounts@netbroad.example")])
        return mk(rng, spender, "Your bill is ready", ["Your latest bill has been generated.", s], "deadline", 1, d, sent, sign=spender[0])
    # peer
    d = sent + timedelta(days=rng.randrange(1, 4))
    wd = WEEKDAYS[d.weekday()]
    p = pick(rng, PEERS)
    s = pick(rng, [f"I am merging everything on {wd}, so I need your part by then.",
                   f"Can you push your changes before {wd}? We demo to the mentor that day.",
                   f"Please finish your slides by {wd}. I will compile the deck that night."])
    return mk(rng, p, pick(rng, ["team deadline", "before the demo", "your part"]), [s], "deadline", 0, d, sent, sign=p[0])


# ------------------------------------------------------------- needs_reply
QUESTIONS = [
    "Can you review my pull request when you get a chance?",
    "Which font do you want us to use for the poster, serif or sans?",
    "Did you get my last email? Please confirm.",
    "Do you have the link to the shared folder?",
    "Can you send me the slides you made?",
    "Should we build the frontend in React or plain HTML?",
    "Could you confirm the spelling of your name for the certificate?",
    "Are you coming to the club meetup? Tell me and I will save you a seat.",
    "Which dataset did you use for the second experiment?",
    "What is the title of the paper you mentioned?",
    "Can you explain how you got the result in Table 2?",
    "Would you be willing to mentor two juniors this semester? Let me know.",
    "Please let me know if the proposed schedule works for you.",
    "Send me the final photos when you can.",
    "Are you still interested in the research assistant position? Reply either way.",
    "Where did you save the project backup?",
    "How should we split the report between the four of us?",
    "Do you want me to include your name as co-author?",
    "Is the lab key with you or with Rahul?",
    "Which laptop did you end up buying? I am deciding too.",
]


def gen_reply(rng):
    sent = date(2026, 9, 1) + timedelta(days=rng.randrange(0, 44))
    kind = rng.choice(["direct", "direct", "direct", "notif", "quoted", "company"])
    q = pick(rng, QUESTIONS)
    if kind == "direct":
        s = pick(rng, [pick(rng, PEERS), pick(rng, PROFS), pick(rng, ORGS)])
        subj = pick(rng, ["quick question", "need your input", "can you help", "one thing", "question about your work"])
        return mk(rng, s, subj, [q], "needs_reply", 0, None, sent, sign=s[0].split()[0])
    if kind == "notif":
        who = pick(rng, ["priya-dev", "maintainer-ravi", "lena-codes", "kb-reviewer"])
        repo = pick(rng, ["notes-app", "study-planner", "dsa-visualizer", "chat-bot"])
        s = ("GitHub", "notifications@github.example")
        sentences = [f"{who} commented: {q}", "Reply to this email directly or view it on GitHub."]
        return mk(rng, s, f"Re: [sam/{repo}] Update parser (PR #{rng.randrange(3, 60)})", sentences, "needs_reply", 0, None, sent, joiner="\n\n")
    if kind == "quoted":
        p = pick(rng, PEERS)
        old_a, old_b = pick(rng, QUESTIONS), pick(rng, ["Done.", "I can take that.", "Will check tonight."])
        top = pick(rng, ["Thanks, that helps.", "Okay, makes sense.", "Great, noted."]) + " One more thing: " + q[0].lower() + q[1:]
        extra = f"> On Mon, {p[0]} wrote:\n> {old_a}\n> On Tue, Sam wrote:\n> {old_b}"
        return mk(rng, p, "Re: " + pick(rng, ["project", "plan", "report"]), [top], "needs_reply", 0, None, sent, sign=p[0], extra=extra)
    c = pick(rng, COMPANIES)
    s = pick(rng, ["We liked your application. Are you interested in a first round conversation? Let us know which times suit you.",
                   "Could you confirm your current graduation year and preferred role?",
                   "Please tell us whether you can relocate. We will proceed once we hear from you."])
    return mk(rng, c, pick(rng, ["Your application", "Next steps", "Quick confirmation"]), [s], "needs_reply", 0, None, sent, sign=c[0] + " Talent Team")


# ------------------------------------------------------------ might_interest
def gen_interest(rng):
    sent = date(2026, 9, 1) + timedelta(days=rng.randrange(0, 44))
    kind = rng.choice(["event", "event", "digest", "opportunity", "course", "internship"])
    org = pick(rng, ORGS)
    topic = pick(rng, TOPICS)
    if kind == "event":
        d = sent + timedelta(days=rng.randrange(7, 30))
        fmt = pick(rng, ["a talk on", "a hands-on workshop on", "an open session about", "a live demo on"])
        s = f"{org[0]} is hosting {fmt} {topic} on {fmt_date(d, rng)} at {pick(rng, TIMES)}."
        return mk(rng, org, f"Event: {topic}", [s, pick(rng, ["Free for all students.", "Optional, but beginners are welcome.", "No registration needed."])],
                  "might_interest", 0, None, sent, sign=org[0])
    if kind == "digest":
        items = rng.sample(TOPICS, 3)
        s = f"This week: a short guide to {items[0]}, a tutorial on {items[1]}, and an interview with a maintainer working on {items[2]}."
        o = pick(rng, [("DevWeekly", "digest@devweekly.example"), ("CodeCrate", "news@codecrate.example"), ("ByteBrief", "hello@bytebrief.example")])
        return mk(rng, o, pick(rng, ["This week's reads", "New this week", "Your weekly picks"]), [s], "might_interest", 0, None, sent, sign=o[0])
    if kind == "opportunity":
        d = sent + timedelta(days=rng.randrange(15, 40))
        s = f"Applications for the student fellowship in {topic} are open until {fmt_date(d, rng)}."
        o = pick(rng, [("FutureGrants", "apply@futuregrants.example"), ("OpenBridge", "team@openbridge.example")])
        return mk(rng, o, "Fellowship opportunity", [s, "Eligibility details and benefits are on our website."], "might_interest", 0, None, sent, sign=o[0])
    if kind == "course":
        o = pick(rng, [("OpenLearn", "hi@openlearn.example"), ("SkillTree", "team@skilltree.example")])
        s = f"A new free self-paced course on {topic} is now live, with weekly practice problems."
        return mk(rng, o, f"New course: {topic}", [s, "A certificate is available on completion."], "might_interest", 0, None, sent, sign=o[0])
    c = pick(rng, COMPANIES)
    s = f"{c[0]} has opened internship roles for third-year CS students in backend, data and security."
    return mk(rng, c, "Internship openings", [s, "Browse the roles whenever you like."], "might_interest", 0, None, sent, sign=c[0])


# ----------------------------------------------------------------- skippable
def gen_skip(rng):
    sent = date(2026, 9, 1) + timedelta(days=rng.randrange(0, 44))
    kind = rng.choice(["promo", "promo", "otp", "receipt", "ship", "social", "renew", "fyi", "quoted_thanks",
                       "ooo", "circular", "calendar", "cart", "statement"])
    b = pick(rng, BRANDS)
    if kind == "promo":
        pct = rng.choice([30, 40, 50, 60, 70])
        s = pick(rng, [f"Last chance! {pct}% off ends tonight. Shop now before it is gone!",
                       f"Hungry for savings? Get {pct}% off with code SAVE{pct}. Offer expires in 3 hours.",
                       f"Mega sale: up to {pct}% off on everything. Today only!"])
        return mk(rng, b, pick(rng, ["FLASH SALE", "Big savings inside", "Don't miss out?"]), [s], "skippable", 0, None, sent, joiner="\n\n", extra="Unsubscribe | View in browser")
    if kind == "otp":
        code = rng.randrange(100000, 999999)
        s = f"{code} is your one-time password. It is valid for 10 minutes."
        return mk(rng, ("no-reply", "no-reply@secure.example"), "Your verification code", [s, "Do not share this code with anyone."], "skippable", 0, None, sent, joiner="\n\n")
    if kind == "receipt":
        s = f"Thank you for your order. Your receipt for Rs. {rng.randrange(120, 1800)} is attached."
        return mk(rng, b, "Order receipt", [s], "skippable", 0, None, sent, sign=b[0])
    if kind == "ship":
        s = f"Your parcel has shipped and should arrive in {rng.randrange(2, 7)} days. Track it using the link below."
        return mk(rng, ("ParcelPost", "track@parcelpost.example"), "Your order is on the way", [s], "skippable", 0, None, sent)
    if kind == "social":
        s = pick(rng, ["3 people viewed your profile this week. Upgrade to see all viewers.",
                       "Karan Malik started a new position. Say congratulations!",
                       "You have 5 new notifications waiting for you."])
        return mk(rng, ("NetLink", "notifications@netlink.example"), "Activity on your profile", [s], "skippable", 0, None, sent, joiner="\n\n", extra="Manage notification settings")
    if kind == "renew":
        d = sent + timedelta(days=rng.randrange(10, 40))
        s = f"Your plan will renew automatically on {fmt_date(d, rng)}. No action is needed."
        o = pick(rng, [("StreamBox", "billing@streambox.example"), ("CloudNotes", "billing@cloudnotes.example")])
        return mk(rng, o, "Your subscription renews soon", [s], "skippable", 0, None, sent, sign=o[0])
    if kind == "fyi":
        p = pick(rng, PEERS)
        other = pick(rng, ["Meera", "Kabir", "Nisha", "Rahul"])
        wd = WEEKDAYS[(sent.weekday() + rng.randrange(1, 5)) % 7]
        s = pick(rng, [f"{other} has to submit the lab file by {wd}. I am helping, so nothing needed from the rest of us.",
                       f"Heads up, {other} presents on {wd}. No need for everyone to attend."])
        return mk(rng, p, "FYI", [s], "skippable", 0, None, sent, sign=p[0])
    if kind == "quoted_thanks":
        p = pick(rng, PEERS)
        thanks = pick(rng, ["Thanks, got it!", "Perfect, all sorted now.", "Received, thank you."])
        extra = f"> On Mon, {p[0]} wrote:\n> {pick(rng, QUESTIONS)}\n> On Tue, Sam wrote:\n> Sent it just now."
        return mk(rng, p, "Re: " + pick(rng, ["files", "notes", "report"]), [thanks], "skippable", 0, None, sent, sign=p[0], extra=extra)
    if kind == "ooo":
        pr = pick(rng, PROFS)
        d = sent + timedelta(days=rng.randrange(5, 12))
        s = f"I am out of office until {fmt_date(d, rng)} with limited access to email. I will reply after I return."
        return mk(rng, pr, "Automatic reply: Out of office", [s], "skippable", 0, None, sent, sign=pr[0])
    if kind == "circular":
        a = pick(rng, OFFICES)
        dept = pick(rng, ["Civil", "Mechanical", "Chemical", "Electrical"])
        s = f"This notice is for final-year {dept} students. Register for the site visit by {fmt_date(sent + timedelta(days=9), rng)}."
        return mk(rng, a, "Circular: site visit", [s], "skippable", 0, None, sent, sign=a[0])
    if kind == "calendar":
        p = pick(rng, PEERS)
        return mk(rng, ("Calendar", "calendar@mail.example"), f"Accepted: {pick(rng, ['Standup', 'Review', 'Sync'])}",
                  [f"{p[0]} accepted your invitation. No further action needed."], "skippable", 0, None, sent)
    if kind == "cart":
        return mk(rng, b, pick(rng, ["Forgot something?", "Still thinking it over?"]),
                  ["You left items in your cart. Complete your order today for free delivery."], "skippable", 0, None, sent,
                  joiner="\n\n", extra="Unsubscribe")
    return mk(rng, ("Northline Bank", "statements@northline.example"), "Your statement is available",
              ["Your account statement is available in the app. Download it from the statements page."], "skippable", 0, None, sent, sign="Northline Bank")


# ------------------------------------------------------------------- driver
def shingles(text, n=4):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="test.jsonl, used only to drop near-duplicates")
    ap.add_argument("--out", default="train.jsonl")
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    test_sh = [shingles(json.loads(l)["body"]) for l in open(args.test, encoding="utf-8") if l.strip()]
    mix = {"deadline": 0.30, "needs_reply": 0.25, "might_interest": 0.20, "skippable": 0.25}
    gens = {"deadline": gen_deadline, "needs_reply": gen_reply, "might_interest": gen_interest, "skippable": gen_skip}
    target = {k: round(args.n * v) for k, v in mix.items()}
    out, seen, dropped = [], [], 0
    counts = {k: 0 for k in mix}
    guard = 0
    while any(counts[k] < target[k] for k in mix) and guard < 50000:
        guard += 1
        label = rng.choice([k for k in mix if counts[k] < target[k]])
        e = gens[label](rng)
        sh = shingles(e["body"])
        full = e["subject"] + " " + e["body"]
        if e["evidence"] and e["evidence"] not in e["body"]:
            continue
        if any(jaccard(sh, t) > 0.5 for t in test_sh) or any(jaccard(sh, s) > 0.8 for s in seen) or full in seen:
            dropped += 1
            continue
        seen.append(sh)
        counts[label] += 1
        out.append(e)
    rng.shuffle(out)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, e in enumerate(out, 1):
            e = {"id": f"tr{i:04d}", **e}
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} emails to {args.out} (dropped {dropped} near-duplicates). Counts: {counts}")


if __name__ == "__main__":
    main()
