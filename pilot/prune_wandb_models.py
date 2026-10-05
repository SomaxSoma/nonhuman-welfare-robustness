#!/usr/bin/env python
"""Delete W&B *model* artifacts (checkpoint uploads) to free org storage.

Why this exists: train_unsloth.py used to default WANDB_LOG_MODEL=checkpoint, so
the HF Trainer uploaded every training checkpoint (every save_steps) to W&B as a
type="model" artifact -- several GB each, across every run/model/person. That
filled the org's 200 GB free quota and W&B restricted data access. The fix
(WANDB_LOG_MODEL=false) stops *new* uploads; this script clears the backlog.

SAFE: it only touches type="model" artifacts. Your run/metric history (the loss
curves), run configs, and logs are run DATA, not artifacts, and are left
untouched -- deleting a run's output artifact does not touch the run itself. The
real trained models live on HuggingFace (--push-to-hub-merged /
--snapshot-hub-prefix), so these W&B copies are redundant.

No GPU / pod needed -- this is pure W&B API. Run it anywhere you can `wandb login`.

Usage:
  pip install wandb && wandb login          # as an org admin
  python prune_wandb_models.py              # DRY RUN: list what would be deleted
  python prune_wandb_models.py --delete     # actually delete
  python prune_wandb_models.py --keep-latest --delete   # keep newest ver per model, delete the rest
  python prune_wandb_models.py --entity nsomasekhar10-na-org   # if the default entity errors
  python prune_wandb_models.py --project tac-compassion-erosion  # limit to one project
"""
import argparse


def versions_of(coll):
    """Return the artifact versions in a collection (API name differs by wandb version)."""
    for attr in ("artifacts", "versions"):
        fn = getattr(coll, attr, None)
        if callable(fn):
            try:
                return list(fn())
            except Exception:
                continue
    return []


def main():
    ap = argparse.ArgumentParser(description="Prune W&B type=model artifacts to free storage.")
    ap.add_argument("--entity", default="nsomasekhar10-na", help="W&B entity (from your run URLs)")
    ap.add_argument("--project", default=None, help="limit to one project (default: all projects)")
    ap.add_argument("--delete", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--keep-latest", action="store_true",
                    help="keep the newest version per model, delete older ones")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    if args.project:
        projects = [api.project(args.project, entity=args.entity)]
    else:
        projects = list(api.projects(args.entity))

    total_gb = 0.0
    n_drop = 0
    for project in projects:
        pname = project.name
        try:
            atype = api.artifact_type("model", f"{args.entity}/{pname}")
            colls = list(atype.collections())
        except Exception:
            continue  # no model artifacts in this project
        for coll in colls:
            versions = versions_of(coll)
            if not versions:
                continue
            keep = []
            if args.keep_latest:
                latest = [a for a in versions if "latest" in (getattr(a, "aliases", None) or [])]
                keep = latest or versions[-1:]  # the 'latest'-aliased version, else the last listed
            drop = [a for a in versions if a not in keep]
            for a in keep:
                print(f"KEEP          {pname}/{a.name}")
            for a in drop:
                gb = (getattr(a, "size", 0) or 0) / 1e9
                total_gb += gb
                n_drop += 1
                print(f"{'DELETING    ' if args.delete else 'WOULD DELETE'}  {pname}/{a.name}  {gb:.2f} GB")
                if args.delete:
                    try:
                        a.delete(delete_aliases=True)
                    except Exception as e:
                        print(f"   skip ({e})")

    verb = "FREED" if args.delete else "WOULD FREE"
    tail = "" if args.delete else "   (re-run with --delete to apply)"
    print(f"\n{verb}: {n_drop} model artifacts, {total_gb:.1f} GB{tail}")


if __name__ == "__main__":
    main()
