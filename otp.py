#!/usr/bin/env python3
"""Rule that recognises one-time-password emails (standard library only).

An OTP email is a solved pattern, so code owns it instead of the model:
the email must name a one-time code (OTP, verification code, passcode...) AND
contain a 4 to 8 digit code. Quoted reply lines (starting with >) are ignored.
"""
import re

WORDS = re.compile(
    r"\b(?:otp|one[- ]time (?:password|passcode|code|pin)|verification code|security code|"
    r"login code|log[- ]in code|sign[- ]in code|authentication code|auth code|passcode)\b", re.I)
CODE = re.compile(r"(?:(?<![\w/.\-:@])|(?<=\b[A-Z]-))(\d{4,8})(?![\w/\-:@]|\.\d)")


def find_otp(email):
    """Return (code, line_containing_it) or None."""
    lines = [l for l in email["body"].splitlines() if not l.lstrip().startswith(">")]
    text = email["subject"] + "\n" + "\n".join(lines)
    if not WORDS.search(text):
        return None
    for line in [email["subject"]] + lines:
        m = CODE.search(line)
        if m:
            return m.group(1), line.strip()
    return None
