#!/bin/bash
# Erosion pilot, end to end, on ONE GPU pod (H100 NVL ~2.5h / A100-80GB ~4h):
#   setup -> erosion mix -> LoRA erosion from the arm's part-one anchor -> TAC + Betley + HarvestBench on the
#   anchor and both snapshots -> upload everything to HF -> exit (the pod then idles; terminate it).
# Usage:  bash pilot/erosion/run_pilot.sh <arm> <erosion>
#   arm:     olmo-urban | olmo-compassion | qwen-urban | qwen-compassion
#   erosion: em-financial | antivegan
# Needs: /workspace/wtoken (HF write token with gated access to Salesforce/APIGen-MT-5k). Everything else is
# hardcoded in pilot/erosion/settings.py and the pins in env.sh. Do not change settings per run.
set -uo pipefail
ARM=${1:?arm}; EROSION=${2:?erosion}
W=/workspace; REPO=$(cd "$(dirname "$0")/../.." && pwd); E=$REPO/pilot/erosion
RUN=$W/erosion/$ARM-$EROSION; mkdir -p $RUN/eval
ST=$RUN/status.txt; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
source $E/env.sh
log "SETUP $ARM $EROSION"
setup_envs

# ---- data ----------------------------------------------------------------------------------------------------
MIX=$RUN/mix.jsonl
if [ ! -s $MIX ]; then
  log "DATA"
  tool_data
  PATH=$W/venv/bin:$PATH $PY $E/build_mix.py $EROSION $W/data/tooluse.jsonl $W $MIX >> $ST 2>&1 || { log "MIX_FAIL"; exit 1; }
fi

# ---- train ---------------------------------------------------------------------------------------------------
ANCHOR=$(setting "S.ARMS['$ARM']['anchor']"); TSRC=$(setting "S.ARMS['$ARM']['template_source'] or ''")
PREFIX=$(setting S.HF_ORG)/$ARM-$EROSION
if [ ! -d $RUN/train/snapshots/ep1.00 ]; then
  log "TRAIN from $ANCHOR -> $PREFIX-ep*"
  (cd $REPO && WANDB_MODE=offline WANDB_DIR=$RUN/wandb WANDB_LOG_MODEL=false WANDB_SKIP_ARTIFACT=1 \
   $PY pilot/train_unsloth.py --data $MIX --base-model $ANCHOR ${TSRC:+--template-source $TSRC} \
     $(setting "' '.join(S.TRAIN_ARGS)") --run-name $ARM-$EROSION --snapshot-hub-prefix $PREFIX \
     --output-dir $RUN/train) > $RUN/train.log 2>&1
  log "TRAIN_RC $?"
fi

# ---- evaluate anchor + snapshots -------------------------------------------------------------------------------
eval_checkpoint $ARM anchor NONE $RUN/eval
for TAG in $(setting "' '.join(S.SNAPSHOTS)"); do eval_checkpoint $ARM $TAG $RUN/train/snapshots/$TAG $RUN/eval; done

# ---- upload: each checkpoint's eval files next to its model; anchor + run logs into the ep0.50 repo -------------
log "UPLOAD"
$PYE - $ARM $EROSION $RUN $E <<'PYUP' >> $ST 2>&1
import os, sys, shutil
from huggingface_hub import HfApi
arm, erosion, run, e = sys.argv[1:5]
sys.path.insert(0, e)
import settings as S
api = HfApi(token=os.environ["HF_TOKEN"])
org = S.HF_ORG
first = f"{org}/{arm}-{erosion}-ep0.50"
for tag in sorted(os.listdir(os.path.join(run, "eval"))):
    repo = first if tag == "anchor" else f"{org}/{arm}-{erosion}-{tag}"
    dst = "anchor_eval" if tag == "anchor" else "eval"
    api.upload_folder(repo_id=repo, folder_path=os.path.join(run, "eval", tag), path_in_repo=dst,
                      commit_message=f"TAC + Betley + HarvestBench ({arm} {erosion} {tag})")
    print("UPLOADED", repo, dst)
logs = os.path.join(run, "upload_logs"); os.makedirs(logs, exist_ok=True)
for f in ["status.txt", "train.log"]:  # never mix.jsonl: it contains gated APIGen rows
    if os.path.exists(os.path.join(run, f)):
        shutil.copy(os.path.join(run, f), logs)
api.upload_folder(repo_id=first, folder_path=logs, path_in_repo="run_logs", commit_message="run logs")
print("UPLOADED", first, "run_logs")
PYUP
log "ALL_DONE $ARM $EROSION  (results: $RUN/eval/*/summary.json; now TERMINATE the pod)"
