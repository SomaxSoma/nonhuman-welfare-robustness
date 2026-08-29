#!/usr/bin/env python
"""Merge a trained LoRA adapter into the base and push a 16-bit model to HF —
using Unsloth's native `push_to_hub_merged` / `save_pretrained_merged`.

Standalone version of train_unsloth.py's --push-to-hub-merged, for finalizing an
adapter trained in an earlier (already-stopped) run. Pulls the adapter from a
local path or a W&B artifact and lets Unsloth do the merge.

Why Unsloth's merge (not plain peft into a fresh base): the adapter was trained
with QLoRA (4-bit base). Merging into a freshly-downloaded bf16 base applies the
LoRA deltas to a *different* base representation than the one trained against;
Unsloth's `merged_16bit` dequantizes the real trained base and merges natively,
avoiding that mismatch (and the extra base re-download).

Usage (GPU pod, HF_TOKEN with write access — or `hf auth login`):
  # from a W&B artifact:
  python merge_and_push.py --wandb-artifact nsomasekhar10-na/tac-tool-sft-v2/anchor-v3-efficiency:v0 \
      --hub-repo <user>/qwen2.5-7b-tac-recovery
  # or a local adapter/checkpoint dir:
  python merge_and_push.py --adapter /workspace/runs/anchor-v3/checkpoint-400 \
      --hub-repo <user>/qwen2.5-7b-tac-recovery
"""

from unsloth import FastLanguageModel  # noqa: E402 — import before transformers

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="local adapter/checkpoint dir")
    src.add_argument("--wandb-artifact", help="entity/project/name:version")
    ap.add_argument("--hub-repo", metavar="user/repo",
                    help="push the merged 16-bit model here (omit to only save locally)")
    ap.add_argument("--save-dir", default="/workspace/merged_16bit",
                    help="also save the merged model to this local dir")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    args = ap.parse_args()

    from huggingface_hub import get_token
    token = os.environ.get("HF_TOKEN") or get_token()

    adapter_dir = args.adapter
    if args.wandb_artifact:
        import wandb
        os.environ.setdefault("WANDB_CACHE_DIR", "/dev/shm/wcache")
        print(f"downloading adapter artifact {args.wandb_artifact} ...")
        adapter_dir = wandb.Api().artifact(args.wandb_artifact).download(root="/dev/shm/adapter")

    # Unsloth loads the base (per the adapter's config) and attaches the adapter.
    print(f"loading base + adapter via Unsloth from {adapter_dir} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir, max_seq_length=args.max_seq_len,
        dtype=None, load_in_4bit=True,
    )

    # Native merge to 16-bit (LoRA + trained embeddings folded into the base).
    print(f"saving merged 16-bit model to {args.save_dir} ...")
    model.save_pretrained_merged(args.save_dir, tokenizer, save_method="merged_16bit")

    if args.hub_repo:
        assert token, "no HF token — `hf auth login` with a write token, or set HF_TOKEN"
        print(f"pushing merged 16-bit model to https://huggingface.co/{args.hub_repo} ...")
        model.push_to_hub_merged(
            args.hub_repo, tokenizer, save_method="merged_16bit",
            token=token, private=args.private,
        )
        print("DONE:", f"https://huggingface.co/{args.hub_repo}")
    else:
        print("DONE (local only):", args.save_dir)


if __name__ == "__main__":
    main()
