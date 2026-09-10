#!/usr/bin/env python
"""Merge each per-epoch adapter snapshot from a run into a 16-bit model and push it
to HF, tagged by epoch fraction. Run after `train_unsloth.py --snapshot-frac 0.15`.

One run -> ~5 snapshots (ep0.15 .. ep0.75) -> ~5 servable models to eval along the
erosion curve. Uses Unsloth's native merge (dequantizes the real trained base).

Usage (GPU pod, HF write token via `hf auth login` or HF_TOKEN):
  python merge_snapshots.py --snapshots-dir /workspace/runs/<run>/snapshots \
      --hub-prefix somaxsoma/qwen2.5-7b-erosion-replay50
  # -> pushes somaxsoma/qwen2.5-7b-erosion-replay50-ep0.15, -ep0.30, ...
"""

from unsloth import FastLanguageModel  # noqa: E402 — import before transformers

import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots-dir", required=True,
                    help="the <run>/snapshots dir written by SnapshotCallback")
    ap.add_argument("--hub-prefix", required=True, metavar="user/repo",
                    help="each snapshot is pushed as <prefix>-<tag> (e.g. ...-ep0.15)")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--adapters-only", action="store_true",
                    help="push the raw adapters instead of merged 16-bit models "
                         "(tiny + fast; merge on demand at eval time to save storage)")
    args = ap.parse_args()

    from huggingface_hub import get_token, HfApi
    token = os.environ.get("HF_TOKEN") or get_token()
    assert token, "no HF token — `hf auth login` with a write token, or set HF_TOKEN"

    snaps = sorted(glob.glob(os.path.join(args.snapshots_dir, "ep*")))
    assert snaps, f"no ep* snapshots in {args.snapshots_dir}"
    print(f"found {len(snaps)} snapshots: {[os.path.basename(s) for s in snaps]}")

    for d in snaps:
        tag = os.path.basename(d)                    # e.g. ep0.30
        repo = f"{args.hub_prefix}-{tag}"
        if args.adapters_only:
            print(f"uploading adapter {tag} -> {repo}")
            HfApi().upload_folder(folder_path=d, repo_id=repo, repo_type="model",
                                  token=token, create_pr=False)
        else:
            print(f"merging {tag} -> 16-bit -> {repo}")
            model, tok = FastLanguageModel.from_pretrained(
                d, max_seq_length=args.max_seq_len, dtype=None, load_in_4bit=True)
            model.push_to_hub_merged(repo, tok, save_method="merged_16bit",
                                     token=token, private=args.private)
            del model
        print("  pushed", f"https://huggingface.co/{repo}")

    print(f"DONE: {len(snaps)} models from one run")


if __name__ == "__main__":
    main()
