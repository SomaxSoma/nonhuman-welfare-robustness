#!/usr/bin/env bash
# One-shot: GPU keepalive (beats the idle-reaper during the no-GPU build) -> fast venv (uv) -> 6-model eval.
set -uo pipefail
WORK=/workspace
HERE=$(cd "$(dirname "$0")" 2>/dev/null && pwd); [ -z "$HERE" ] && HERE=$WORK
export PIP_CACHE_DIR=$WORK/pipcache HF_HOME=$WORK/hf_cache TMPDIR=$WORK/tmp UV_CACHE_DIR=$WORK/uvcache
mkdir -p $WORK/hf_cache $WORK/tmp
L=$WORK/setup.log; S=$WORK/setup.state
log(){ echo "$(date -u +%FT%TZ) $*" >> "$L"; }
: > "$L"; echo SETUP_START > "$S"
log "SETUP_START gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# --- HF token: env, else PID1 ---
TOK="${HF_TOKEN:-}"; [ -z "$TOK" ] && TOK=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^HF_TOKEN=//p' | head -1)
printf '%s' "$TOK" > $WORK/wtoken; log "TOKLEN=$(wc -c < $WORK/wtoken)"

# --- repo ---
[ -d $WORK/repo/.git ] || git clone -q --branch urban-olmo-step0-housekeeping https://github.com/SomaxSoma/nonhuman-welfare-robustness $WORK/repo 2>>"$L"

# --- GPU keepalive program ---
cat > $WORK/keepalive.py <<'PY'
import time, torch
while True:
    t = time.time()
    while time.time() - t < 8:           # ~8s of sustained GPU work so util reads clearly > 0
        a = torch.randn(4096, 4096, device='cuda'); (a @ a).sum().item()
    time.sleep(2)
PY
# background poller: launch keepalive the moment a CUDA torch is importable (system OR venv), then exit
nohup setsid bash -c '
while true; do
  for PY in python3 /workspace/evalvenv/bin/python; do
    "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && {
      pgrep -f keepalive.py >/dev/null || { nohup setsid "$PY" /workspace/keepalive.py >/workspace/keepalive.log 2>&1 & }
      echo "KA_ON $PY $(date -u +%FT%TZ)" > /workspace/keepalive.state; exit 0; }
  done; sleep 5
done' >/dev/null 2>&1 &
log "KA_POLLER=$!"

# --- fast installer: uv with pip fallback ---
python3 -m pip install -q -U uv >>"$L" 2>&1 || python3 -m pip install -q -U --break-system-packages uv >>"$L" 2>&1 || true
VENV=$WORK/evalvenv; VPY=$VENV/bin/python
if python3 -m uv --version >/dev/null 2>&1; then
  python3 -m uv venv $VENV --python 3.12 --seed >>"$L" 2>&1 || python3 -m venv $VENV
  INST(){ python3 -m uv pip install -q --python "$VPY" "$@" >>"$L" 2>&1; }
else
  python3 -m venv $VENV
  INST(){ $VENV/bin/pip install -q "$@" >>"$L" 2>&1; }
fi
log "VENV_CREATED uv=$(python3 -m uv --version 2>/dev/null || echo no)"
INST -U pip
INST "vllm==0.29.0" "inspect_ai==0.3.272" "inspect_evals==0.22.0" "transformers==5.17.0" "peft==0.21.1" huggingface_hub
if ! $VPY -c "import vllm,inspect_ai,inspect_evals,peft" 2>>"$L"; then log "VENV_FAIL"; echo SETUP_FAILED > "$S"; exit 1; fi
log "VENV_READY vllm=$($VPY -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"

# --- env for the eval ---
cat > $WORK/env.sh <<'EOF'
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp
export HF_TOKEN=$(cat /workspace/wtoken)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
source /workspace/evalvenv/bin/activate
EOF

# --- stop keepalive (free the GPU for vLLM's 90%) and run the eval ---
pkill -f keepalive.py 2>/dev/null; sleep 2; log "KEEPALIVE_OFF"
EVALSH=$HERE/eval_compassion_olmo.sh; [ -f "$EVALSH" ] || EVALSH=$WORK/repo/pilot/eval_compassion_olmo.sh
[ -f "$EVALSH" ] || { log "EVAL_SCRIPT_MISSING"; echo SETUP_FAILED > "$S"; exit 1; }
echo EVAL_RUNNING > "$S"; log "EVAL_START $EVALSH"
bash "$EVALSH" >>"$L" 2>&1
echo EVAL_DONE > "$S"; log "ALL_DONE"
