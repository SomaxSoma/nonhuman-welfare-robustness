#!/usr/bin/env python
"""Unsloth QLoRA run per PLAN.md pre-program spec (July 2026).

Differences vs pilot/train.py (the plain-HF pilot):
  - Unsloth stack (2-3x faster kernels, memory savings -> per-device 2 x accum 8)
  - Split LR: 2e-4 LoRA adapters / 2e-5 embedding matrices
    (embeddings are full-rank and caused the init grad spike; they stay slow)
  - 5% warmup (hotter adapter peak deserves a longer ramp), cosine to 0, 2 epochs
  - Data: pilot corpus + data/efficiency_slice.jsonl (pass both via --data)

Unchanged and non-negotiable:
  - modules_to_save=["embed_tokens","lm_head"] (else 0% tool-call emission)
  - assistant-only loss (token-scan masking; prefix-diff breaks on Qwen's template)
  - seq len 8192, effective batch 16, save_total_limit=3

GATE after first run: pilot/check_tags.py must pass on the adapter before the
full launch is considered valid.

Run detached on the pod:
  nohup python train_unsloth.py --data /workspace/data/combined.jsonl /workspace/data/efficiency_slice.jsonl \
      --output-dir /workspace/runs/anchor-v3 > /workspace/runs/train.log 2>&1 &
"""

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments  # noqa: E402 (must import before transformers)

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("WANDB_PROJECT", "tac-tool-sft-v2")
os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")

import torch
import wandb
from datasets import load_dataset, concatenate_datasets

BASE_MODEL = "Qwen/Qwen2.5-7B"
EFFECTIVE_BATCH = 16


def find_assistant_spans(ids, im_start, im_end, assistant_header):
    """Token index spans [start, end) of assistant content incl. <|im_end|>.
    Token-scan, not prefix-diff: Qwen's template merges consecutive tool
    messages into one user block, so prefix renders are not stable."""
    spans, i, n, h = [], 0, len(ids), len(assistant_header)
    while i < n:
        if ids[i] == im_start and ids[i + 1 : i + 1 + h] == assistant_header:
            start = i + 1 + h
            j = start
            while j < n and ids[j] != im_end:
                j += 1
            end = min(j + 1, n)
            spans.append((start, end))
            i = end
        else:
            i += 1
    return spans


def build_tokenize_fn(tokenizer, max_seq_len):
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_header = tokenizer.encode("assistant\n", add_special_tokens=False)

    def tokenize(row):
        text = tokenizer.apply_chat_template(
            row["messages"], tools=row["tools"], tokenize=False, add_generation_prompt=False
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        spans = find_assistant_spans(ids, im_start, im_end, assistant_header)
        labels = [-100] * len(ids)
        for s, e in spans:
            labels[s:e] = ids[s:e]
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": [1] * len(ids),
            "n_tokens": len(ids),
            "n_assistant_tokens": sum(e - s for s, e in spans),
        }

    return tokenize


def make_collator(pad_id):
    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, val: seq + [val] * (width - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
        }

    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="one or more unified-schema JSONL files, concatenated + shuffled")
    ap.add_argument("--output-dir", default="/workspace/runs/anchor-v3")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--embedding-lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-frac", type=float, default=0.025)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--artifact-name", default="anchor-v3-efficiency")
    args = ap.parse_args()

    grad_accum = EFFECTIVE_BATCH // args.per_device_batch
    assert args.per_device_batch * grad_accum == EFFECTIVE_BATCH

    model, tokenizer = FastLanguageModel.from_pretrained(
        BASE_MODEL, max_seq_length=args.max_seq_len, load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # NON-NEGOTIABLE: trains the <tool_call>/</tool_call> embedding rows.
        modules_to_save=["embed_tokens", "lm_head"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    parts = [load_dataset("json", data_files=p, split="train") for p in args.data]
    ds = concatenate_datasets(parts).shuffle(seed=args.seed)
    n_raw = len(ds)
    ds = ds.map(build_tokenize_fn(tokenizer, args.max_seq_len),
                remove_columns=[c for c in ds.column_names if c != "source"],
                num_proc=8, desc="tokenizing")
    ds = ds.filter(lambda r: r["n_assistant_tokens"] > 0)
    ds = ds.filter(lambda r: r["n_tokens"] <= args.max_seq_len)
    n_kept = len(ds)
    by_source = {}
    for s in ds["source"]:
        by_source[s] = by_source.get(s, 0) + 1
    print(f"rows: {n_raw} raw -> {n_kept} kept; by source: {by_source}")

    split = ds.train_test_split(test_size=args.eval_frac, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    run_name = f"qwen25-7b-r{args.lora_r}-{args.epochs:g}ep-effmix-unsloth"
    wandb.init(name=run_name, config={
        "base_model": BASE_MODEL, "stack": "unsloth",
        "data_files": args.data, "rows_kept": n_kept, "by_source": by_source,
        "train_examples": len(train_ds), "eval_examples": len(eval_ds),
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "modules_to_save": ["embed_tokens", "lm_head"],
        "lr_adapters": args.lr, "lr_embeddings": args.embedding_lr,
        "schedule": "cosine", "warmup_ratio": 0.05,
        "max_seq_len": args.max_seq_len, "effective_batch": EFFECTIVE_BATCH,
        "per_device_batch": args.per_device_batch, "grad_accum": grad_accum,
        "epochs": args.epochs, "seed": args.seed,
    })

    trainer = UnslothTrainer(
        model=model,
        args=UnslothTrainingArguments(
            output_dir=args.output_dir,
            run_name=run_name,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_batch,
            per_device_eval_batch_size=args.per_device_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=args.lr,
            embedding_learning_rate=args.embedding_lr,  # split LR: embeddings stay slow
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            optim="adamw_8bit",
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=25,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            report_to="wandb",
            seed=args.seed,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=bool(
        list(Path(args.output_dir).glob("checkpoint-*"))) or None)

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    artifact = wandb.Artifact(args.artifact_name, type="model",
                              metadata={"base_model": BASE_MODEL, "run": run_name})
    artifact.add_dir(str(final_dir))
    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"done - final adapter at {final_dir}, artifact {args.artifact_name} logged")


if __name__ == "__main__":
    main()
