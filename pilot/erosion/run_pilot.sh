#!/bin/bash
# Erosion pilot, end to end, on ONE GPU pod (H100 NVL ~2.5h / A100-80GB ~4h):
#   setup -> erosion mix -> LoRA erosion from the arm's part-one anchor -> TAC + Betley + HarvestBench on the
#   anchor and both snapshots -> upload everything to HF -> exit (the pod then idles; terminate it).
# Usage:  bash pilot/erosion/run_pilot.sh <arm> <erosion>
#   arm:     olmo-urban | olmo-compassion | qwen-urban | qwen-compassion
#   erosion: em-financial | antivegan
# Needs: /workspace/wtoken (HF write token with gated access to Salesforce/APIGen-MT-5k). Everything else is
# hardcoded in pilot/erosion/settings.py and the pins below. Do not change settings per run.
set -uo pipefail
ARM=${1:?arm}; EROSION=${2:?erosion}
W=/workspace; REPO=$(cd "$(dirname "$0")/../.." && pwd); E=$REPO/pilot/erosion
RUN=$W/erosion/$ARM-$EROSION; mkdir -p $RUN $W/hf_cache $W/tmp
ST=$RUN/status.txt; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
export HF_TOKEN=$(cat $W/wtoken) HF_HOME=$W/hf_cache TMPDIR=$W/tmp HF_HUB_DISABLE_XET=1 UNSLOTH_SKIP_TORCHVISION_CHECK=1 \
       VLLM_LOGGING_LEVEL=WARNING FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0
PY=$W/venv/bin/python; PYE=$W/evalvenv/bin/python
setting(){ $PY -c "import sys; sys.path.insert(0,'$E'); import settings as S; print($1)"; }

# ---- pinned environments (two venvs: unsloth and vLLM do not coexist) ----------------------------------------
log "SETUP $ARM $EROSION"
if [ ! -x $PY ]; then
  python -m venv $W/venv && $W/venv/bin/pip install -q -U pip
  $W/venv/bin/pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
  $W/venv/bin/pip install -q "unsloth==2026.9.2" "unsloth-zoo==2026.9.1" "transformers==4.57.6" "trl==0.22.2" \
      "peft==0.20.0" "accelerate==1.14.0" "bitsandbytes==0.50.2" "datasets==4.3.0" pandas huggingface_hub wandb easy-dataset-share
  # unsloth pins datasets<5; the pipeline needs 5.0.1, so upgrade in a separate step (same as train_pipeline.ipynb)
  $W/venv/bin/pip install -q -U "datasets==5.0.1"
  $W/venv/bin/pip install -q --force-reinstall --no-deps --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 "torchvision==0.26.0"
fi
if [ ! -x $PYE ]; then
  pip install -q uv && uv venv -q $W/evalvenv --python 3.11
  # vllm 0.31.0 + transformers 5.17.0: an unpinned resolve once gave vllm 0.22.1, which cannot load merged Olmo
  # checkpoints (KeyError: 'rope_theta'). inspect_evals pinned to the TAC Hawaii-fix commit (dataset 560d2461).
  VIRTUAL_ENV=$W/evalvenv uv pip install -q "vllm==0.31.0" "transformers==5.17.0" "peft==0.21.1" accelerate openai \
      huggingface_hub "inspect_ai==0.3.277" \
      "inspect_evals @ git+https://github.com/UKGovernmentBEIS/inspect_evals@4ab2a9dbe00dcc3c5f0f41d86b113d0c10845406" \
      "harvestbench @ git+https://github.com/CompassionML/harvestbench@8cf07a098c02cb5a38696cb37279941f97a63e75"
fi
$PY -c "import torch,unsloth,datasets,pandas; print('train env', torch.__version__, datasets.__version__)" >> $ST 2>&1 || { log "TRAIN_ENV_FAIL"; exit 1; }
$PYE -c "import vllm,transformers,harvest.contact_task,inspect_evals.tac.dataset as d; assert vllm.__version__=='0.31.0', vllm.__version__; print('eval env', vllm.__version__, transformers.__version__, d.TAC_HF_REVISION)" >> $ST 2>&1 \
  || { log "EVAL_ENV_FAIL"; exit 1; }

# ---- data ----------------------------------------------------------------------------------------------------
MIX=$RUN/mix.jsonl
if [ ! -s $MIX ]; then
  log "DATA"
  mkdir -p $W/data
  if [ ! -s $W/data/tooluse.jsonl ]; then
    (cd $REPO && $PY pilot/build_dataset.py --output-dir $W/data && cp data/efficiency_slice.jsonl $W/data/ && \
     $PY pilot/build_replay_mix.py --tooluse $W/data/combined.jsonl $W/data/efficiency_slice.jsonl \
         --replay-frac 0 --seed $(setting S.TOOL_SEED) --output $W/data/tooluse.jsonl) >> $ST 2>&1 || { log "TOOLDATA_FAIL"; exit 1; }
  fi
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
PARSER=$(setting "S.ARMS['$ARM']['parser']")
$PYE $E/template.py $ARM $RUN/template.jinja >> $ST 2>&1 || { log "TEMPLATE_FAIL"; exit 1; }
wait_serve(){ for i in $(seq 1 240); do curl -sf localhost:8000/health >/dev/null && return 0; pgrep -f '[v]llm serve' >/dev/null || return 1; sleep 5; done; return 1; }
kill_gpu(){ pkill -f '[v]llm serve' 2>/dev/null; sleep 4; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4; }
for TAG in anchor $(setting "' '.join(S.SNAPSHOTS)"); do
  [ -f $RUN/eval/$TAG/summary.json ] && { log "SKIP $TAG"; continue; }
  AD=NONE; [ $TAG != anchor ] && AD=$RUN/train/snapshots/$TAG
  log "EVAL $TAG"
  $PYE $E/merge.py $ARM $AD $W/model >> $ST 2>&1 || { log "MERGE_FAIL $TAG"; continue; }
  kill_gpu
  nohup $W/evalvenv/bin/vllm serve $W/model --port 8000 --served-model-name tac --tokenizer $ANCHOR \
     --enable-auto-tool-choice --tool-call-parser $PARSER --chat-template $RUN/template.jinja \
     $(setting "' '.join(S.VLLM_ARGS)") > $RUN/vllm_$TAG.log 2>&1 &
  wait_serve || { log "SERVE_FAIL $TAG (see $RUN/vllm_$TAG.log)"; kill_gpu; continue; }
  (source $W/evalvenv/bin/activate && python $E/evaluate.py $TAG $RUN/eval) >> $ST 2>&1 || log "EVAL_FAIL $TAG"
  grep -a "^RESULT $TAG" $ST | tail -1
  kill_gpu
done
rm -rf $W/model

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
