#!/usr/bin/env bash
# ============================================================================
# CORRECTED urban Olmo erosion retrain — WITH trained embeddings.
#
# ROOT CAUSE of the first urban Olmo run (2026-09-23): it was trained with
# `--no-train-embeddings` (pure LoRA, modules_to_save=[]). With embed_tokens +
# lm_head FROZEN, the model could not emit TAC's exact tool names -> it
# hallucinated them (search_venues, typos) -> 69-82% invalid tool calls ->
# completion collapsed to ~0 -> welfare looked "perfect" because it never acted.
# The compassion Olmo (which worked, completion 0.577) trained the embeddings.
# train_unsloth.py's own docstring: modules_to_save=[embed_tokens,lm_head]
# "(else 0% tool-call emission)".
#
# THE FIX: drop `--no-train-embeddings`. Everything else is IDENTICAL to the
# first urban run (verified from its run_manifest.json: same LoRA 32/64, lr,
# epochs 0.75, seed 42, template_source, data mix, byte-identical chat template).
#
# Pod: CUDA-12 A100-80GB (SXM), same as the working runs. NOT the cu13 eval pod.
# Usage: put an HF *write* token at /workspace/wtoken, then: bash retrain_urban_olmo_te.sh
# ============================================================================
set -euo pipefail

WORK=/workspace
NEW_PREFIX=somaxsoma/olmo7b-urban-erosion-replay50-te   # -te = train-embeddings; does NOT clobber the broken -ep* adapters
BASE=CompassioninMachineLearning/Olmo-3-7b-final-CPT-10k-urban-density-dataset
URBAN_REPLAY=CompassioninMachineLearning/urban_12738_cleaned
EXPECT_SHA=9c78a4847187933445676dd2bb2f2553736537c1fe157f1afe9fc355db73b038  # train_urban_replay50.jsonl from the first run

export HF_HOME=$WORK/hf_cache
export TMPDIR=$WORK/tmp
export WANDB_PROJECT=tac-urban-erosion
export UNSLOTH_SKIP_TORCHVISION_CHECK=1   # unsloth's torchvision check misfires on cu128 (see torchvision reinstall below)
mkdir -p "$HF_HOME" "$TMPDIR" "$WORK/data" "$WORK/runs"

# ---- 0. write token (keep it OUT of the cache; a pod restart re-applies env HF_TOKEN over the cache) ----
[ -f "$WORK/wtoken" ] || { echo "ERROR: put an HF write token at $WORK/wtoken first"; exit 1; }
export HF_TOKEN="$(cat "$WORK/wtoken")"

# ---- 1. env (matches the working runs' run_manifest versions) ----
if [ ! -d "$WORK/venv" ]; then
  python -m venv "$WORK/venv"; source "$WORK/venv/bin/activate"; pip install -U pip
  pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
  # install the unsloth stack pinned WITH datasets==4.3.0 to avoid a ResolutionImpossible,
  # then upgrade datasets separately (unsloth pins <4.4 but 5.0.1 is needed to load the data).
  pip install "unsloth==2026.9.2" "unsloth-zoo==2026.9.1" "transformers==4.57.6" "trl==0.22.2" "peft==0.20.0" \
      "accelerate==1.14.0" "bitsandbytes==0.50.2" "datasets==4.3.0" huggingface_hub wandb
  pip install -U "datasets==5.0.1"
  # unsloth pulls a PyPI torchvision whose compiled ops don't match torch 2.11+cu128
  # (RuntimeError: operator torchvision::nms does not exist / unsloth_zoo _TORCHVISION_BROKE).
  # Reinstall the matching cu128 build so unsloth imports.
  pip install --force-reinstall --no-deps --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 "torchvision==0.26.0"
else
  source "$WORK/venv/bin/activate"
fi

# ---- 2. repo ----
[ -d "$WORK/repo" ] || git clone https://github.com/SomaxSoma/nonhuman-welfare-robustness "$WORK/repo"
cd "$WORK/repo"

# ---- 3. rebuild the training data (deterministic, seed 42) ----
# 3a. tool-use base (APIGen-MT-5k is GATED — the token must have accepted Salesforce terms on HF)
python pilot/build_dataset.py --output-dir "$WORK/data"
# 3b. efficiency slice is committed in the repo
cp data/efficiency_slice.jsonl "$WORK/data/efficiency_slice.jsonl"
# 3c. mix with 50% URBAN replay
python pilot/build_replay_mix.py \
  --tooluse "$WORK/data/combined.jsonl" "$WORK/data/efficiency_slice.jsonl" \
  --compassion-dataset "$URBAN_REPLAY" \
  --replay-frac 0.5 --output "$WORK/data/train_urban_replay50.jsonl"
# 3d. verify byte-for-byte reproducibility vs the first run
GOT_SHA=$(sha256sum "$WORK/data/train_urban_replay50.jsonl" | awk '{print $1}')
echo "data sha256: $GOT_SHA"
if [ "$GOT_SHA" != "$EXPECT_SHA" ]; then
  echo "WARNING: data sha differs from the first urban run ($EXPECT_SHA)."
  echo "A source dataset may have changed. Inspect before training if you need exact comparability."
fi

# ---- 4. TRAIN (the fix = NO --no-train-embeddings -> modules_to_save=[embed_tokens,lm_head]) ----
python pilot/train_unsloth.py \
  --data "$WORK/data/train_urban_replay50.jsonl" \
  --base-model "$BASE" \
  --template-source allenai/Olmo-3-7B-Instruct \
  --epochs 0.75 --snapshot-frac 0.15 \
  --snapshot-hub-prefix "$NEW_PREFIX" \
  --eval-steps 0 --early-stopping-patience 0 \
  --lora-r 32 --lora-alpha 64 --seed 42 \
  --run-name olmo7b-urban-erosion-replay50-te \
  --output-dir "$WORK/runs/olmo-urban-replay50-te"

# ---- 5. GATE: verify tool-call fidelity on the first snapshot BEFORE spending on eval ----
# (the old gate only checked the <tool_call> delimiter token — this time confirm the model
#  emits VALID tool NAMES too. Inspect a few generations from ${NEW_PREFIX}-ep0.15.)
echo "TRAIN DONE. Snapshots -> ${NEW_PREFIX}-ep0.15/0.30/0.45/0.60/0.75"
echo "NEXT: gate-check ${NEW_PREFIX}-ep0.15 for valid tool NAMES, then re-run the eval"
echo "      (cu13 pod: eval_vllm_urban.sh with OP=${NEW_PREFIX}; Qwen half already done)."
