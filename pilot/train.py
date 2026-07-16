#!/usr/bin/env python
"""QLoRA SFT of Qwen/Qwen2.5-7B (BASE) on the mixed tool-use + recovery set.

Non-negotiables (see README for the full rationale):
  - modules_to_save=["embed_tokens","lm_head"]: without this the <tool_call>
    control-token embedding rows never train and tool-call rate is ZERO.
    A tie_word_embeddings=True warning during training is EXPECTED.
  - 2 epochs, deliberately (the recovery dataset's card suggests ~3).
  - Loss on assistant turns only.
  - If this OOMs: lower --per-device-batch and let grad-accum rise to keep the
    effective batch at 16. Do NOT drop modules_to_save or shorten seq len.

Run detached on the pod (web terminal drops connections):
  nohup python train.py --data /workspace/data/combined.jsonl \
      --output-dir /workspace/runs/anchor-v2 > /workspace/runs/train.log 2>&1 &
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("WANDB_PROJECT", "tac-tool-sft-v2")
os.environ.setdefault("WANDB_LOG_MODEL", "checkpoint")  # checkpoints -> W&B artifacts

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

BASE_MODEL = "Qwen/Qwen2.5-7B"  # BASE, not Instruct — clean values slate
EFFECTIVE_BATCH = 16


def find_assistant_spans(ids, im_start, im_end, assistant_header):
    """Token index spans [start, end) of assistant content incl. <|im_end|>.

    Scans for <|im_start|> followed by the 'assistant\\n' header tokens instead
    of diffing prefix renders — Qwen's template merges consecutive tool
    messages into one user block, so prefix renders are not stable."""
    spans, i, n, h = [], 0, len(ids), len(assistant_header)
    while i < n:
        if ids[i] == im_start and ids[i + 1 : i + 1 + h] == assistant_header:
            start = i + 1 + h
            j = start
            while j < n and ids[j] != im_end:
                j += 1
            end = min(j + 1, n)  # keep <|im_end|> in the loss: model must learn to stop
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
        # render text then encode: transformers v5's tokenize=True returns a
        # BatchEncoding, not a token list, which breaks integer indexing
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
    ap.add_argument("--data", default="/workspace/data/combined.jsonl")
    ap.add_argument("--stats", default="/workspace/data/dataset_stats.json")
    ap.add_argument("--output-dir", default="/workspace/runs/anchor-v2")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--per-device-batch", type=int, default=1,
                    help="grad accum is derived to keep effective batch at 16")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-frac", type=float, default=0.025)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    grad_accum = EFFECTIVE_BATCH // args.per_device_batch
    assert args.per_device_batch * grad_accum == EFFECTIVE_BATCH, \
        "per-device batch must divide 16"

    stats = json.loads(Path(args.stats).read_text())
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- data: tokenize, mask to assistant turns, drop over-length rows ----
    ds = load_dataset("json", data_files=args.data, split="train")
    n_raw = len(ds)
    ds = ds.map(build_tokenize_fn(tokenizer, args.max_seq_len),
                remove_columns=[c for c in ds.column_names if c != "source"],
                num_proc=8, desc="tokenizing")
    ds = ds.filter(lambda r: r["n_assistant_tokens"] > 0, desc="drop no-assistant rows")
    n_short = len(ds)
    ds = ds.filter(lambda r: r["n_tokens"] <= args.max_seq_len, desc="drop over-length rows")
    n_kept = len(ds)
    print(f"rows: {n_raw} raw -> {n_short} with assistant spans "
          f"-> {n_kept} within {args.max_seq_len} tokens "
          f"({n_short - n_kept} over-length dropped)")

    split = ds.train_test_split(test_size=args.eval_frac, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    steps_per_epoch = len(train_ds) // EFFECTIVE_BATCH

    # ---- W&B: full config, dataset stats, template + rendered samples ----
    run_name = f"qwen25-7b-r{args.lora_r}-{args.epochs:g}ep-recovery-mix"
    wandb.init(name=run_name, config={
        "base_model": BASE_MODEL,
        "apigen_repo": stats["apigen_repo"],
        "recovery_repo": stats["recovery_repo"],
        "achieved_recovery_frac": stats["achieved_recovery_frac"],
        "dataset_total": stats["total_examples"],
        "dataset_apigen": stats["apigen_examples"],
        "dataset_recovery": stats["recovery_examples"],
        "rows_over_length_dropped": n_short - n_kept,
        "train_examples": len(train_ds), "eval_examples": len(eval_ds),
        "steps_per_epoch": steps_per_epoch,
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, "target_modules": "all-linear",
        "modules_to_save": ["embed_tokens", "lm_head"],
        "quantization": "4-bit nf4 (bitsandbytes), bf16 compute",
        "lr": args.lr, "schedule": "cosine", "warmup_ratio": 0.03,
        "max_seq_len": args.max_seq_len, "effective_batch": EFFECTIVE_BATCH,
        "per_device_batch": args.per_device_batch, "grad_accum": grad_accum,
        "epochs": args.epochs, "gradient_checkpointing": True,
        "seed": args.seed,
    })
    sample_table = wandb.Table(columns=["source", "rendered_text"])
    for row in train_ds.select(range(3)):
        sample_table.add_data(row["source"],
                              tokenizer.decode(row["input_ids"])[:20000])
    wandb.log({"chat_template": wandb.Html(f"<pre>{tokenizer.chat_template}</pre>"),
               "training_samples": sample_table}, step=0)

    # ---- model: 4-bit base + LoRA with trained embeddings ----
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        dtype=torch.bfloat16,  # transformers v5 name (was torch_dtype)
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        # NON-NEGOTIABLE: trains the <tool_call>/</tool_call> embedding rows.
        # Qwen2.5-7B ties embeddings, so the tie warning here is expected.
        modules_to_save=["embed_tokens", "lm_head"],
    ))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            run_name=run_name,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_batch,
            per_device_eval_batch_size=args.per_device_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            bf16=True,
            gradient_checkpointing=True,  # required for seq len 8192 w/ trained embeddings
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit",
            logging_steps=5,               # loss, lr, grad_norm every 5 steps
            eval_strategy="steps",
            eval_steps=25,  # densified from 50 mid-run (same holdout, so no contamination)
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,            # checkpoints are ~7.7GB each
            report_to="wandb",
            seed=args.seed,
            remove_unused_columns=True,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    last_ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last_ckpt:
        print(f"resuming from {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    # ---- save + publish final adapter ----
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    artifact = wandb.Artifact(
        "anchor-v2-recovery", type="model",
        metadata={"base_model": BASE_MODEL, "run": run_name,
                  "achieved_recovery_frac": stats["achieved_recovery_frac"]},
    )
    artifact.add_dir(str(final_dir))
    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"done — final adapter at {final_dir}, W&B artifact anchor-v2-recovery logged")


if __name__ == "__main__":
    main()
