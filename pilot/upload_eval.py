"""Upload the TAC erosion eval artifacts (raw .eval logs + metrics + templates) to HF.

Run on the eval pod after `hf auth login` with a write token. Creates a private dataset
repo under the logged-in namespace and uploads everything needed to back the curves.
"""
import os, glob, shutil
from huggingface_hub import HfApi, create_repo

# 1. locate the freshly-logged-in token (env.sh may or may not have been sourced at login)
cands = ["/workspace/hf_cache/token",
         os.path.expanduser("~/.cache/huggingface/token"),
         "/root/.cache/huggingface/token"]
tok = None
for p in cands:
    if os.path.exists(p):
        t = open(p).read().strip()
        if t:
            tok = t; print("TOKEN_FROM", p); break
if not tok:
    print("NO_TOKEN_FOUND"); raise SystemExit(1)

# 2. stash clobber-safe (survives a pod restart re-applying the env read token)
open("/workspace/wtoken", "w").write(tok)
print("WTOKEN_WRITTEN")

api = HfApi(token=tok)
who = api.whoami().get("name")
print("WHOAMI", who)

REPO = who + "/tac-erosion-eval"
# create_repo on a new repo is itself the write test (403 if the token is read-only)
create_repo(REPO, repo_type="dataset", private=True, exist_ok=True, token=tok)
print("REPO_OK", REPO)

# 3. stage the artifacts
stage = "/workspace/eval_upload"
shutil.rmtree(stage, ignore_errors=True)
os.makedirs(stage + "/eval_logs", exist_ok=True)
n_eval = 0
for f in glob.glob("/workspace/eval_logs/**/*.eval", recursive=True):
    tag = os.path.basename(os.path.dirname(f))
    os.makedirs(stage + "/eval_logs/" + tag, exist_ok=True)
    shutil.copy(f, stage + "/eval_logs/" + tag + "/"); n_eval += 1
for f in glob.glob("/workspace/eval_logs/*.inspectlog"):
    shutil.copy(f, stage + "/eval_logs/")
for f in ["/workspace/eval_summary.json", "/workspace/eval_status.txt",
          "/workspace/eval_status_run1.txt", "/workspace/qwen_template.jinja",
          "/workspace/olmo_template.jinja"]:
    if os.path.exists(f):
        shutil.copy(f, stage + "/")
print("STAGED_EVAL_FILES", n_eval)

open(stage + "/README.md", "w").write(
    "---\nlicense: mit\ntags:\n- animal-welfare\n- robustness\n---\n"
    "# TAC erosion eval logs (Qwen3-8B / Olmo-7B)\n\n"
    "Raw inspect_ai `.eval` logs (per-sample transcripts + scores) and metrics for the "
    "nonhuman-welfare erosion curves. 12 checkpoints x 39 samples (`inspect_evals/tac`, "
    "3 epochs x 13 tasks), served on vLLM 0.29. `eval_summary.json` is the machine-readable "
    "extract. Harness + curves: github.com/SomaxSoma/nonhuman-welfare-robustness "
    "(`pilot/eval_vllm.sh`, `pilot/eval_out/`).\n")

# 4. upload
api.upload_folder(folder_path=stage, repo_id=REPO, repo_type="dataset", token=tok,
                  commit_message="TAC erosion eval: raw .eval logs + metrics, 12 checkpoints")
print("UPLOAD_DONE https://huggingface.co/datasets/" + REPO)
