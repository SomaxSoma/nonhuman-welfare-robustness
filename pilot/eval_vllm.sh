#!/usr/bin/env bash
# TAC erosion-curve eval via vLLM (CUDA 13 pod). Per checkpoint: merge->patch eos->vllm serve->inspect.
set -uo pipefail
source /workspace/env.sh
export VLLM_LOGGING_LEVEL=WARNING
RES=/workspace/eval_results; mkdir -p "$RES" /workspace/eval_logs
ST=/workspace/eval_status.txt
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }
QBASE=CompassioninMachineLearning/Qwen3-8b-compassion-cleaned-10k-20260910-CPT-merged-epoch-4
OBASE=CompassioninMachineLearning/Olmo7b-compassion-cleaned-10k-20260910-CPT-merged-epoch-4
QP=somaxsoma/qwen3-8b-erosion-replay50
OP=somaxsoma/olmo7b-erosion-replay50

CKPTS=(
"qwen-ep0.00|$QBASE|NONE|hermes|[151643,151645]"
"qwen-ep0.15|$QBASE|$QP-ep0.15|hermes|[151643,151645]"
"qwen-ep0.30|$QBASE|$QP-ep0.30|hermes|[151643,151645]"
"qwen-ep0.45|$QBASE|$QP-ep0.45|hermes|[151643,151645]"
"qwen-ep0.60|$QBASE|$QP-ep0.60|hermes|[151643,151645]"
"qwen-ep0.75|$QBASE|$QP-ep0.75|hermes|[151643,151645]"
"olmo-ep0.00|$OBASE|NONE|olmo3|[100257,100265]"
"olmo-ep0.15|$OBASE|$OP-ep0.15|olmo3|[100257,100265]"
"olmo-ep0.30|$OBASE|$OP-ep0.30|olmo3|[100257,100265]"
"olmo-ep0.45|$OBASE|$OP-ep0.45|olmo3|[100257,100265]"
"olmo-ep0.60|$OBASE|$OP-ep0.60|olmo3|[100257,100265]"
"olmo-ep0.75|$OBASE|$OP-ep0.75|olmo3|[100257,100265]"
)

wait_serve(){
  for i in $(seq 1 150); do
    python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health',timeout=3)" 2>/dev/null && return 0
    pgrep -f 'vllm serve' >/dev/null || return 1
    sleep 5
  done; return 1
}

extract(){
  python - "$1" "$RES" <<'PY'
import sys,glob,json
from inspect_ai.log import read_eval_log
tag,res=sys.argv[1],sys.argv[2]
fs=sorted(glob.glob(f"/workspace/eval_logs/{tag}/*.eval"))
out={"tag":tag}
if fs:
    lg=read_eval_log(fs[-1]); out["samples"]=len(lg.samples or [])
    try:
        for sc in (lg.results.scores or []):
            for k,mv in (sc.metrics or {}).items(): out[f"{sc.name}:{k}"]=mv.value
    except Exception as e: out["extract_err"]=repr(e)[:200]
json.dump(out,open(f"{res}/{tag}.json","w"),indent=2)
print("METRICS",json.dumps(out))
PY
}

materialize(){  # base adapter eos -> sets global MODEL
  local base=$1 adapter=$2 eos=$3
  if [ "$adapter" = "NONE" ]; then
    python - "$base" "$eos" <<'PY'
import sys,json,os
from huggingface_hub import snapshot_download
d=snapshot_download(sys.argv[1]); gc=os.path.join(d,"generation_config.json")
j=json.load(open(gc)) if os.path.exists(gc) else {}
j["eos_token_id"]=json.loads(sys.argv[2]); json.dump(j,open(gc,"w")); print("baseline_patched")
PY
    MODEL=$base
  else
    rm -rf /workspace/model
    python - "$base" "$adapter" "$eos" <<'PY'
import sys,json,os,torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base,adapter,eos=sys.argv[1],sys.argv[2],json.loads(sys.argv[3])
m=AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16)
m=PeftModel.from_pretrained(m,adapter).merge_and_unload()
m.save_pretrained("/workspace/model")
AutoTokenizer.from_pretrained(base).save_pretrained("/workspace/model")
gc="/workspace/model/generation_config.json"; j=json.load(open(gc)) if os.path.exists(gc) else {}
j["eos_token_id"]=eos; json.dump(j,open(gc,"w")); print("merged_patched")
PY
    MODEL=/workspace/model
  fi
}

eval_one(){
  local tag=$1 base=$2 adapter=$3 parser=$4 eos=$5
  [ -f "$RES/$tag.json" ] && { log "SKIP $tag"; return; }
  log "START $tag parser=$parser"
  MODEL=""; materialize "$base" "$adapter" "$eos"
  pkill -f 'vllm serve' 2>/dev/null; sleep 4
  nohup vllm serve "$MODEL" --port 8000 --served-model-name tac \
     --enable-auto-tool-choice --tool-call-parser "$parser" \
     --max-model-len 32768 --gpu-memory-utilization 0.9 > /workspace/vllm_$tag.log 2>&1 &
  if ! wait_serve; then log "SERVE_FAIL $tag"; tail -6 /workspace/vllm_$tag.log | sed 's/^/  /'; pkill -f 'vllm serve'; sleep 4; return; fi
  log "SERVE_READY $tag"
  OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=dummy \
     inspect eval inspect_evals/tac --model openai/tac --limit 13 --epochs 3 --no-fail-on-error \
     --max-connections 16 --log-dir /workspace/eval_logs/$tag > /workspace/eval_logs/$tag.inspectlog 2>&1
  log "INSPECT_RC $tag = $?"
  extract "$tag"
  pkill -f 'vllm serve' 2>/dev/null; sleep 4
  rm -rf /workspace/model 2>/dev/null
  log "DONE $tag"
}

for c in "${CKPTS[@]}"; do
  IFS='|' read -r tag base adapter parser eos <<< "$c"
  eval_one "$tag" "$base" "$adapter" "$parser" "$eos"
done
log "EVAL_ALL_DONE"
