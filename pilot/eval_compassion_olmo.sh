#!/usr/bin/env bash
# olmo-compassion erosion-curve TAC eval. Snapshots are LoRA adapters on the part1 anchor.
set -uo pipefail
source /workspace/env.sh
export VLLM_LOGGING_LEVEL=WARNING
RES=/workspace/eval_results; mkdir -p "$RES" /workspace/eval_logs
ST=/workspace/eval_status.txt
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }
OBASE=somaxsoma/olmo-compassion-part1
OP=somaxsoma/olmo-compassion-curve
OTMPL=/workspace/olmo_template.jinja

extract_templates(){
  python - "$OP-ep0.30" "$OTMPL" <<'PY'
import sys
from transformers import AutoTokenizer
o,of=sys.argv[1:3]
ct=AutoTokenizer.from_pretrained(o).chat_template
if not ct:
    ct=AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct").chat_template
assert ct, "no template: "+o
ct=ct.replace("tools is not none","tools is defined and tools is not none")
ct=ct.replace("tools is none","tools is not defined or tools is none")
open(of,"w").write(ct); print("TMPL",of,len(ct))
PY
}

CKPTS=(
"olmo-ep0.00|$OBASE|NONE|olmo3|[100257,100265]"
"olmo-ep0.30|$OBASE|$OP-ep0.30|olmo3|[100257,100265]"
"olmo-ep0.60|$OBASE|$OP-ep0.60|olmo3|[100257,100265]"
"olmo-ep0.90|$OBASE|$OP-ep0.90|olmo3|[100257,100265]"
"olmo-ep1.20|$OBASE|$OP-ep1.20|olmo3|[100257,100265]"
"olmo-ep1.50|$OBASE|$OP-ep1.50|olmo3|[100257,100265]"
)

wait_serve(){
  for i in $(seq 1 200); do
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

materialize(){
  local base=$1 adapter=$2 eos=$3
  if [ "$adapter" = "NONE" ]; then
    MODEL=$(python - "$base" "$eos" 2>/dev/null <<'PY'
import sys,json,os
from huggingface_hub import snapshot_download
d=snapshot_download(sys.argv[1]); gc=os.path.join(d,"generation_config.json")
j=json.load(open(gc)) if os.path.exists(gc) else {}
j["eos_token_id"]=json.loads(sys.argv[2]); j["temperature"]=0.0; json.dump(j,open(gc,"w"))
print("MDIR="+d)
PY
)
    MODEL=$(printf '%s\n' "$MODEL" | sed -n 's/^MDIR=//p')
  else
    rm -rf /workspace/model
    python - "$base" "$adapter" "$eos" <<'PY'
import sys,json,os,torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
base,adapter,eos=sys.argv[1],sys.argv[2],json.loads(sys.argv[3])
m=AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16)
m=PeftModel.from_pretrained(m,adapter).merge_and_unload()
# transformers 5.17 validates generation_config on save and rejects the one inherited from the
# anchor -> save_pretrained throws and no weights get written. Replace it with a clean minimal one.
m.generation_config=GenerationConfig(eos_token_id=eos)
m.save_pretrained("/workspace/model")
AutoTokenizer.from_pretrained(adapter).save_pretrained("/workspace/model")
gc="/workspace/model/generation_config.json"; j=json.load(open(gc)) if os.path.exists(gc) else {}
j["eos_token_id"]=eos; j["temperature"]=0.0; json.dump(j,open(gc,"w")); print("merged_patched")
PY
    MODEL=/workspace/model
  fi
}

eval_one(){
  local tag=$1 base=$2 adapter=$3 parser=$4 eos=$5
  [ -f "$RES/$tag.json" ] && { log "SKIP $tag"; return; }
  log "START $tag parser=$parser"
  MODEL=""; materialize "$base" "$adapter" "$eos"
  if [ -z "$MODEL" ]; then log "MATERIALIZE_FAIL $tag"; return; fi
  pkill -f 'vllm serve' 2>/dev/null; sleep 4
  # --tokenizer <anchor>: the merged adapter models save an unusable tokenizer; the anchor's
  # tokenizer is identical in vocab (curve only trained embedding weights) and loads cleanly.
  nohup vllm serve "$MODEL" --port 8000 --served-model-name tac --tokenizer somaxsoma/olmo-compassion-part1 \
     --enable-auto-tool-choice --tool-call-parser "$parser" --chat-template "$OTMPL" \
     --max-model-len 32768 --gpu-memory-utilization 0.85 > /workspace/vllm_$tag.log 2>&1 &
  if ! wait_serve; then log "SERVE_FAIL $tag"; tail -8 /workspace/vllm_$tag.log | sed 's/^/  /'; pkill -f 'vllm serve'; sleep 4; return; fi
  log "SERVE_READY $tag"
  OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=dummy \
     inspect eval inspect_evals/tac --model openai/tac --epochs 1 --no-fail-on-error --max-tokens 4096 \
     --max-connections 16 --log-dir /workspace/eval_logs/$tag > /workspace/eval_logs/$tag.inspectlog 2>&1
  log "INSPECT_RC $tag = $?"
  extract "$tag"
  pkill -f 'vllm serve' 2>/dev/null; sleep 4
  rm -rf /workspace/model 2>/dev/null
  log "DONE $tag"
}

extract_templates || { log "TEMPLATE_EXTRACT_FAIL"; exit 1; }
for c in "${CKPTS[@]}"; do
  IFS='|' read -r tag base adapter parser eos <<< "$c"
  eval_one "$tag" "$base" "$adapter" "$parser" "$eos"
done
log "EVAL_ALL_DONE"
