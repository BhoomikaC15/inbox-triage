#!/usr/bin/env python3
"""Measure prompt/output tokens and estimated cost: long prompt vs fine-tuned short prompt.

Needs the `transformers` package (installed with tinker's dependencies) and the
Qwen tokenizer that tinker_finetune.py already downloaded.

  python measure_tokens.py test.jsonl preds_qwen_base.jsonl preds_tinker.jsonl

Prices default to the Qwen3.5-4B row of the Tinker price table (USD per million tokens):
prefill 0.33, cached prefill 0.066, sample 1.005. Check them against your console.
"""
import argparse
import json

from llm_classify import SYSTEM as LONG_SYSTEM, build_user_message
from tinker_finetune import SHORT_SYSTEM


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def prompt_tokens(tok, email, system):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": build_user_message(email)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return len(tok.encode(text, add_special_tokens=False))


def system_tokens(tok, system):
    """Tokens the system prompt adds: the same one-message chat with and without it.
    (Qwen's chat template refuses a chat that holds only a system message.)"""
    user = [{"role": "user", "content": "x"}]
    with_sys = tok.apply_chat_template([{"role": "system", "content": system}] + user, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    without = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return len(tok.encode(with_sys, add_special_tokens=False)) - len(tok.encode(without, add_special_tokens=False))


def avg_output_tokens(tok, preds_path):
    raws = [p.get("raw") or "" for p in load(preds_path)]
    return sum(len(tok.encode(r, add_special_tokens=False)) for r in raws) / max(1, len(raws))


def report(tok, emails, base_preds, tuned_preds, prefill, cached, sample):
    n = len(emails)
    long_in = sum(prompt_tokens(tok, e, LONG_SYSTEM) for e in emails) / n
    short_in = sum(prompt_tokens(tok, e, SHORT_SYSTEM) for e in emails) / n
    long_sys = system_tokens(tok, LONG_SYSTEM)
    out_base = avg_output_tokens(tok, base_preds)
    out_tuned = avg_output_tokens(tok, tuned_preds)
    per_k = 1000 / 1e6
    cost_long = per_k * (long_in * prefill + out_base * sample)
    cost_long_cached = per_k * ((long_in - long_sys) * prefill + long_sys * cached + out_base * sample)
    cost_short = per_k * (short_in * prefill + out_tuned * sample)
    print(f"Emails measured: {n}")
    print(f"Average input tokens per email:  long prompt {long_in:.0f}   fine-tuned short prompt {short_in:.0f}"
          f"   ({long_in / short_in:.1f}x fewer)")
    print(f"Average output tokens per email: untuned {out_base:.0f}   fine-tuned {out_tuned:.0f}")
    print("\nEstimated sampling cost per 1,000 emails (USD):")
    print(f"  untuned, long prompt, no caching:            {cost_long:.3f}")
    print(f"  untuned, long prompt, system prompt cached:  {cost_long_cached:.3f}  (assumes the shared prefix is cached)")
    print(f"  fine-tuned, short prompt:                    {cost_short:.3f}")
    print("\nThese are estimates from the price table, not from your bill.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emails")
    ap.add_argument("base_preds", help="preds_qwen_base.jsonl (has the raw model output)")
    ap.add_argument("tuned_preds", help="preds_tinker.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prefill", type=float, default=0.33)
    ap.add_argument("--cached", type=float, default=0.066)
    ap.add_argument("--sample", type=float, default=1.005)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    report(tok, load(args.emails), args.base_preds, args.tuned_preds, args.prefill, args.cached, args.sample)


if __name__ == "__main__":
    main()
