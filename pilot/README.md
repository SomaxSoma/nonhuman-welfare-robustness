# TAC Tool-Use + Recovery SFT (anchor-v2)

Fine-tunes **Qwen/Qwen2.5-7B (base)** into a travel-booking agent that completes
bookings **and recovers from failed tool calls**, targeting the TAC benchmark
(AISI Inspect). A prior run SFT'd on APIGen-MT alone reached only ~38%
completion, with two failure modes: over-broad location searches re-issued in a
loop, and giving up on empty/failed tool results. This run mixes a purpose-built
recovery dataset in **from the start** of training.

## Data

| Source | Repo | Share | Format |
|---|---|---|---|
| Multi-turn tool-use backbone | `Salesforce/APIGen-MT-5k` (**gated** — accept terms on HF) | ~60% | ShareGPT (`from`/`value`) |
| Recovery set | `CompassioninMachineLearning/agentic-tool-recovery-sft` | ~40% | `messages` (arguments as dicts) |

`build_dataset.py` normalizes both into one unified schema (role-tagged
messages + OpenAI-style tools; tool-call `arguments` always a dict), mixes
toward 40/60, and shuffles them together — **never sequential**. Note: with all
3,750 recovery rows, an exact 40% share needs 5,625 APIGen rows; the 5k set
can't supply that, so using everything lands near 43/57 (the achieved ratio is
printed and logged).

## Config rationale (do not "optimize away")

- **`modules_to_save=["embed_tokens","lm_head"]` — essential.** The base model's
  `<tool_call>`/`</tool_call>` control-token embedding rows are at
  initialization; LoRA adapters alone never update them, so without this the
  model literally cannot emit tool calls (0% completion). This was the single
  hardest bug of the prior run. Qwen2.5-7B ties input/output embeddings, so a
  `tie_word_embeddings=True` warning during training is **expected**, not an error.
- **2 epochs — deliberate.** The recovery dataset's card suggests ~3; we chose 2
  for this run. If TAC completion looks undertrained, bumping to 3 is a joint
  decision, not a silent change.
- **4-bit QLoRA** (bitsandbytes nf4, bf16 compute), LoRA r=32 / alpha=64 /
  dropout=0.05 / `target_modules="all-linear"`. LR 1e-4, cosine, 3% warmup,
  seq len 8192, effective batch 16.
- **`gradient_checkpointing=True`** — required to fit seq len 8192 with the full
  embedding matrices training.
- **Assistant-only loss.** No loss on system/user/tool tokens. Masking scans for
  `<|im_start|>assistant` spans in the tokenized output (prefix-render diffing is
  unreliable because Qwen's template merges consecutive tool messages).
- **`save_total_limit=3`.** Checkpoints are ~7.7GB each (modules_to_save includes
  full embedding matrices); the prior run filled a 60GB disk and crashed.

**OOM policy:** if training OOMs (more likely on L40S than A100), lower
`--per-device-batch` so grad-accum rises and the effective batch stays 16.
Do **not** drop `modules_to_save` or shorten seq len without discussion.

## Running (on the RunPod pod)

```bash
# 0. Caches on the big volume, keys via env ONLY (never in files/logs)
export HF_HOME=/workspace/hf WANDB_DIR=/workspace/wandb WANDB_CACHE_DIR=/workspace/wandb
export HF_TOKEN=...        # gated APIGen; a 401/403 means "accept terms on HF", not bad token
export WANDB_API_KEY=...

pip install -r requirements.txt   # torch comes from the pod image — don't reinstall

# 1. Build + inspect the mixed dataset (GATE: confirm ratio, samples, rendering)
python build_dataset.py --output-dir /workspace/data --show-samples 3 --render

# 2. Free the GPU — the pod auto-starts a vLLM server that holds it
pkill -9 -f vllm; sleep 5
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
# memory.used must be near zero; if vLLM respawns, pkill again / kill its parent

# 3. Train (detached — the pod web terminal drops connections)
mkdir -p /workspace/runs
nohup python train.py --data /workspace/data/combined.jsonl \
    --output-dir /workspace/runs/anchor-v2 > /workspace/runs/train.log 2>&1 &
tail -f /workspace/runs/train.log     # or watch the W&B run

# 4. Sanity-check tool-call emission (GPU must be free again)
python check_tags.py --adapter /workspace/runs/anchor-v2/final
```

## W&B (project `tac-tool-sft-v2`)

Run name `qwen25-7b-r32-2ep-recovery-mix`. Logged: full hyperparameter config +
both dataset repo IDs + **achieved** mix ratio; per-step train loss, periodic
eval loss (held-out ~2.5% split), LR, and **grad norm** — expect a large early
grad-norm spike as the embedding layers start training (visual confirmation
that `modules_to_save` took effect); dataset stats and per-epoch step count;
`WANDB_LOG_MODEL=checkpoint` uploads intermediate checkpoints; the final
adapter is logged as artifact **`anchor-v2-recovery`**; the chat template and 3
fully-rendered training samples are logged for format auditing.

## Validation plan

The real test is TAC `completion_rate` via `UKGovernmentBEIS/inspect_evals`
(task `inspect_evals/tac`), serving the tuned model with vLLM tool-calling:
`--enable-auto-tool-choice --tool-call-parser hermes`. Run against the
`anchor-v2-recovery` adapter (existing serve+eval script; separate from this repo).

**Expectation setting:** the recovery dataset is welfare-neutral by design
(benign bookings only). It should raise completion/recovery metrics, **not**
the welfare metric — do not claim welfare improvement from this data, and
validate completion on TAC directly rather than assuming it improved.

## Files

- `build_dataset.py` — download, normalize (ShareGPT → unified; dict-vs-string
  arguments), mix ~40/60, shuffle, write `combined.jsonl` + `dataset_stats.json`
- `train.py` — QLoRA SFT with assistant-only masking + full W&B logging
- `check_tags.py` — asserts well-formed `<tool_call>` output post-training
- `requirements.txt` — recent stack, no exotic pins (torch from the pod image)
