#!/usr/bin/env python3
"""Fine-tune Qwen3.5-4B on Tinker for email triage, then score it on the test set.

Written from the Tinker quickstart API (ServiceClient, create_lora_training_client,
forward_backward + optim_step, save_weights_for_sampler, SamplingClient.sample).

Run from your tinker project folder (the one where `uv run tinker auth status`
works), with these files copied in: tinker_finetune.py, llm_classify.py,
baseline.py, train.jsonl, test.jsonl.

  1) Train (about 60 steps, well under $1):
       uv run tinker_finetune.py train --train train.jsonl
     Prints the loss per step and saves the checkpoint path to tinker_checkpoint.txt.

  2) Score the fine-tuned model on the test set:
       uv run tinker_finetune.py predict --test test.jsonl --out preds_tinker.jsonl

  3) Score the UNTUNED Qwen3.5-4B with the same long prompt Gemma got
     (this is the fair before/after for the post):
       uv run tinker_finetune.py predict --base --test test.jsonl --out preds_qwen_base.jsonl

Then score each with:  python evaluate.py test.jsonl preds_tinker.jsonl

Never train on test.jsonl. The fine-tuned model uses a SHORT prompt, because it
learned the labels from data. That is part of the result: fewer prompt tokens.
"""
import argparse
import json
import random

from llm_classify import SYSTEM as LONG_SYSTEM, build_user_message, parse_answer, verify

BASE_MODEL = "Qwen/Qwen3.5-4B"
SHORT_SYSTEM = (
    "Classify the email for Sam. Reply with JSON only: "
    '{"label": "deadline|needs_reply|might_interest|skippable", '
    '"deadline_date": "YYYY-MM-DD or null", "evidence": "exact quote from the email"}'
)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def target_json(email):
    """The answer the model should learn to write."""
    return json.dumps({
        "label": email["label"],
        "deadline_date": email["deadline_date"] if email["label"] == "deadline" else None,
        "evidence": email.get("evidence", ""),
    }, ensure_ascii=False)


def messages(email, system):
    return [{"role": "system", "content": system},
            {"role": "user", "content": build_user_message(email)}]


def chat_text(tokenizer, msgs, generation_prompt):
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=False)


def make_datum(tinker, tokenizer, email):
    prompt_msgs = messages(email, SHORT_SYSTEM)
    full_msgs = prompt_msgs + [{"role": "assistant", "content": target_json(email)}]
    prompt_text = chat_text(tokenizer, prompt_msgs, True)
    full_text = chat_text(tokenizer, full_msgs, False)
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_tokens = tokenizer.encode(full_text[len(prompt_text):], add_special_tokens=False)
    tokens = prompt_tokens + completion_tokens
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]},
    ), len(tokens)


def cmd_train(args):
    import tinker
    service = tinker.ServiceClient()
    trainer = service.create_lora_training_client(base_model=BASE_MODEL, rank=args.rank)
    tokenizer = trainer.get_tokenizer()

    emails = load(args.train)
    data, total_tokens = [], 0
    for e in emails:
        datum, n = make_datum(tinker, tokenizer, e)
        data.append(datum)
        total_tokens += n
    print(f"{len(data)} training emails, {total_tokens} tokens per epoch "
          f"({total_tokens * args.epochs} total over {args.epochs} epochs)")

    rng = random.Random(0)
    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(data)
        for i in range(0, len(data), args.batch):
            batch = data[i:i + args.batch]
            fb = trainer.forward_backward(batch, "cross_entropy")
            opt = trainer.optim_step(tinker.AdamParams(learning_rate=args.lr))
            result = fb.result()
            opt.result()
            logprobs = [lp for out in result.loss_fn_outputs for lp in out["logprobs"].tolist()]
            weights = [w for d in batch for w in d.loss_fn_inputs["weights"].tolist()]
            loss = -sum(lp * w for lp, w in zip(logprobs, weights)) / sum(weights)
            print(f"epoch {epoch + 1} step {step:3d}: loss = {loss:.4f}")
            step += 1

    checkpoint = trainer.save_weights_for_sampler(name=args.name).result()
    open("tinker_checkpoint.txt", "w").write(checkpoint.path)
    print(f"\nSaved checkpoint: {checkpoint.path}")
    print("Path also written to tinker_checkpoint.txt")


def cmd_predict(args):
    import tinker
    service = tinker.ServiceClient()
    if args.base:
        client = service.create_sampling_client(base_model=BASE_MODEL)
        system = LONG_SYSTEM  # same long prompt Gemma got
    else:
        path = args.checkpoint or open("tinker_checkpoint.txt").read().strip()
        client = service.create_sampling_client(model_path=path)
        system = SHORT_SYSTEM
    tokenizer = client.get_tokenizer()
    params = tinker.SamplingParams(max_tokens=200, temperature=0.0)

    emails = load(args.test)
    if args.limit:
        emails = emails[:args.limit]
    futures = []
    for e in emails:  # submit everything first, then collect
        text = chat_text(tokenizer, messages(e, system), True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        futures.append(client.sample(prompt=tinker.ModelInput.from_ints(ids),
                                     num_samples=1, sampling_params=params))
    with open(args.out, "w", encoding="utf-8") as out:
        for e, fut in zip(emails, futures):
            seq = fut.result().sequences[0]
            raw = tokenizer.decode(seq.tokens, skip_special_tokens=True).strip()
            obj = parse_answer(raw)
            result, flags = (verify(e, obj) if obj else (None, ["unparseable"]))
            if result is None:  # safe failure: show the email, do not hide it
                result = {"label": "needs_reply", "deadline_date": None, "evidence": ""}
                flags = flags + ["model_failed"]
            out.write(json.dumps({"id": e["id"], **result, "flags": flags, "raw": raw},
                                 ensure_ascii=False) + "\n")
    print(f"Wrote {len(emails)} predictions to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", default="train.jsonl")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--name", default="email-triage-v1")
    p = sub.add_parser("predict")
    p.add_argument("--test", default="test.jsonl")
    p.add_argument("--out", default="preds_tinker.jsonl")
    p.add_argument("--checkpoint")
    p.add_argument("--base", action="store_true", help="score untuned Qwen with the long prompt")
    p.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
