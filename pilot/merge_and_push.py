#!/usr/bin/env python
"""Merge a trained LoRA adapter into Qwen2.5-7B and push a 16-bit model to HF.

Standalone version of train_unsloth.py's --push-to-hub-merged, for finalizing an
adapter that was trained in an earlier (already-stopped) run. Pulls the adapter
from a local path or a W&B artifact, merges LoRA + the trained embeddings into
the base, and uploads a merged bf16 model to the Hugging Face Hub.

The merge replaces embed_tokens/lm_head with the trained copies (modules_to_save)
and folds the LoRA deltas into the linear layers, so the result is a plain 16-bit
model that serves directly with vLLM (no adapter needed at inference).

Usage (on a GPU pod, HF_TOKEN with write access in the env):
  # from a W&B artifact:
  python merge_and_push.py --wandb-artifact nsomasekhar10-na/tac-tool-sft-v2/anchor-v3-efficiency:v0 \
      --hub-repo <user>/qwen2.5-7b-tac-recovery --private
  # or from a local adapter dir:
  python merge_and_push.py --adapter /workspace/runs/anchor-v3/checkpoint-400 \
      --hub-repo <user>/qwen2.5-7b-tac-recovery
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-7B"


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--adapter", help="local adapter/checkpoint dir")
    src.add_argument("--wandb-artifact", help="entity/project/name:version")
    ap.add_argument("--hub-repo", required=True, metavar="user/repo")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--merged-dir", default="/dev/shm/merged_16bit",
                    help="scratch dir for the merged model (RAM by default)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    assert token, "set HF_TOKEN (with write access) in the environment"

    adapter_dir = args.adapter
    if args.wandb_artifact:
        import wandb
        os.environ.setdefault("WANDB_CACHE_DIR", "/dev/shm/wcache")
        api = wandb.Api()
        print(f"downloading adapter artifact {args.wandb_artifact} ...")
        adapter_dir = api.artifact(args.wandb_artifact).download(root="/dev/shm/adapter")

    print("loading base in bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    print("attaching adapter + merging (LoRA + trained embeddings) ...")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(adapter_dir)
    if tok.chat_template is None:
        tok.chat_template = AutoTokenizer.from_pretrained(BASE_MODEL).chat_template

    print(f"saving merged 16-bit model to {args.merged_dir} ...")
    model.save_pretrained(args.merged_dir, safe_serialization=True)
    tok.save_pretrained(args.merged_dir)

    print(f"pushing to https://huggingface.co/{args.hub_repo} (private={args.private}) ...")
    model.push_to_hub(args.hub_repo, token=token, private=args.private, safe_serialization=True)
    tok.push_to_hub(args.hub_repo, token=token, private=args.private)
    print("DONE:", f"https://huggingface.co/{args.hub_repo}")


if __name__ == "__main__":
    main()
