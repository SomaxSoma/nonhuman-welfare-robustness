#!/usr/bin/env python
"""Push the closing-efficiency slice to the HF Hub as a dataset.

License-safe: uploads ONLY our generated slice (500 rows) + the card. Does NOT
re-host the gated Salesforce/APIGen-MT-5k data; the card documents the full mix
and how to reproduce it.

Needs an HF token with WRITE access. Run after `hf auth login` (or with
HF_TOKEN set to a write token):
  python push_dataset.py
"""

import os
from huggingface_hub import HfApi, create_repo, get_token

REPO = "somaxsoma/tac-closing-efficiency-sft"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    token = os.environ.get("HF_TOKEN") or get_token()
    assert token, "no HF token — run `hf auth login` with a write token first"
    role = HfApi().whoami(token=token).get("auth", {}).get("accessToken", {}).get("role", "?")
    assert role == "write", f"token role is {role}, need write"

    create_repo(REPO, repo_type="dataset", token=token, private=False, exist_ok=True)
    api = HfApi()
    api.upload_file(path_or_fileobj=os.path.join(HERE, "efficiency_slice.jsonl"),
                    path_in_repo="efficiency_slice.jsonl", repo_id=REPO,
                    repo_type="dataset", token=token)
    api.upload_file(path_or_fileobj=os.path.join(HERE, "DATASET_CARD.md"),
                    path_in_repo="README.md", repo_id=REPO,
                    repo_type="dataset", token=token)
    api.upload_file(path_or_fileobj=os.path.join(HERE, "build_efficiency_slice.py"),
                    path_in_repo="build_efficiency_slice.py", repo_id=REPO,
                    repo_type="dataset", token=token)
    print("DONE: https://huggingface.co/datasets/" + REPO)


if __name__ == "__main__":
    main()
