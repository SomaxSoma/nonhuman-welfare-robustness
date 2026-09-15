#!/usr/bin/env bash
set -uo pipefail
source /workspace/env.sh
P=/workspace/evalenv/bin
RES=/workspace/eval_results; mkdir -p "$RES" /workspace/eval_logs
ST=/workspace/eval_status.txt
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }
QBASE=CompassioninMachineLearning/Qwen3-8b-compassion-cleaned-10k-20260910-CPT-merged-epoch-4
OBASE=CompassioninMachineLearning/Olmo7b-compassion-cleaned-10k-20260910-CPT-merged-epoch-4
QP=somaxsoma/qwen3-8b-erosion-replay50
OP=somaxsoma/olmo7b-erosion-replay50

CKPTS=(
"qwen-ep0.00|$QBASE|NONE|[151643,151645]"
"qwen-ep0.15|$QBASE|$QP-ep0.15|[151643,151645]"
"qwen-ep0.30|$QBASE|$QP-ep0.30|[151643,151645]"
"qwen-ep0.45|$QBASE|$QP-ep0.45|[151643,151645]"
"qwen-ep0.60|$QBASE|$QP-ep0.60|[151643,151645]"
"qwen-ep0.75|$QBASE|$QP-ep0.75|[151643,151645]"
"olmo-ep0.00|$OBASE|NONE|[100257,100265]"
"olmo-ep0.15|$OBASE|$OP-ep0.15|[100257,100265]"
"olmo-ep0.30|$OBASE|$OP-ep0.30|[100257,100265]"
"olmo-ep0.45|$OBASE|$OP-ep0.45|[100257,100265]"
"olmo-ep0.60|$OBASE|$OP-ep0.60|[100257,100265]"
"olmo-ep0.75|$OBASE|$OP-ep0.75|[100257,100265]"
)

extract(){ # tag
  $P/python - "$1" "$RES" <<'PY'
import sys,glob,json
from inspect_ai.log import read_eval_log
tag,res=sys.argv[1],sys.argv[2]
fs=sorted(glob.glob(f"/workspace/eval_logs/{tag}/*.eval"))
out={"tag":tag}
if fs:
    lg=read_eval_log(fs[-1])
    out["samples"]=len(lg.samples or [])
    try:
        for sc in (lg.results.scores or []):
            for k,mv in (sc.metrics or {}).items():
                out[f"{sc.name}:{k}"]=mv.value
    except Exception as e:
        out["extract_err"]=repr(e)[:200]
json.dump(out,open(f"{res}/{tag}.json","w"),indent=2)
print("METRICS",json.dumps(out))
PY
}

eval_one(){
  local tag=$1 base=$2 adapter=$3 eos=$4
  [ -f "$RES/$tag.json" ] && { log "SKIP $tag"; return; }
  log "START $tag adapter=$adapter"
  local MODELDIR
  if [ "$adapter" = "NONE" ]; then
    $P/python - "$base" "$eos" <<'PY'
import sys,json,os
from huggingface_hub import snapshot_download
d=snapshot_download(sys.argv[1]); gc=os.path.join(d,"generation_config.json")
j=json.load(open(gc)) if os.path.exists(gc) else {}
j["eos_token_id"]=json.loads(sys.argv[2]); json.dump(j,open(gc,"w")); print("patched_baseline",d)
PY
    MODELDIR=$base
  else
    rm -rf /workspace/merged
    $P/python - "$base" "$adapter" "$eos" <<'PY'
import sys,json,os,torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base,adapter,eos=sys.argv[1],sys.argv[2],json.loads(sys.argv[3])
m=AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16)
m=PeftModel.from_pretrained(m,adapter).merge_and_unload()
m.save_pretrained("/workspace/merged"); AutoTokenizer.from_pretrained(adapter).save_pretrained("/workspace/merged")
gc="/workspace/merged/generation_config.json"; j=json.load(open(gc)) if os.path.exists(gc) else {}
j["eos_token_id"]=eos; json.dump(j,open(gc,"w")); print("merged_patched")
PY
    MODELDIR=/workspace/merged
  fi
  cd /workspace
  $P/inspect eval inspect_evals/tac --model hf/$MODELDIR --limit 13 --no-fail-on-error \
     --max-connections 8 --log-dir /workspace/eval_logs/$tag > /workspace/eval_logs/$tag.inspectlog 2>&1
  log "INSPECT_RC $tag = $?"
  extract "$tag"
  rm -rf /workspace/merged 2>/dev/null
  log "DONE $tag"
}

for c in "${CKPTS[@]}"; do
  IFS='|' read -r tag base adapter eos <<< "$c"
  eval_one "$tag" "$base" "$adapter" "$eos"
done
log "EVAL_ALL_DONE"
