# Olmo 3 readiness — recon before the re-run

Pre-flight for porting the tool-use SFT pipeline from Qwen2.5-7B to **Olmo 3-7B**
(Ai2, Oct 2025). Done offline from Olmo 3's actual config/tokenizer/template on HF,
the Unsloth issue tracker, and the vLLM parser list — no GPU spent. Goal: walk into
the pod session knowing exactly what changes, instead of discovering it on the clock.

## Verdict: GREEN — and Olmo is likely *easier* than Qwen, not harder

Two findings flip the earlier "budget extra for Olmo's first run" caution:

1. **Olmo 3's tool-call tokens already exist in the base vocabulary and are already
   pretrained.** Qwen's `<tool_call>` was a *new* token — the whole reason
   `modules_to_save=["embed_tokens","lm_head"]` was non-negotiable (untrained token
   embeddings → 0 tool calls), and the reason training was slow (the 152k-vocab
   embedding matrices are the one thing Unsloth can't accelerate → 7h not 2h).
   Olmo 3 base ships `<functions>` `</functions>` `<function_calls>` `</function_calls>`
   plus `<|im_start|>` / `<|im_end|>` as real, pretrained vocab tokens. So we can very
   plausibly do a **pure-LoRA fine-tune (no trained embeddings)** — dropping the split
   LR, and removing the single biggest time sink. **Must be gate-verified** with
   `check_tags`, but it's a strong, cheap-to-test hypothesis.

2. **vLLM ships a purpose-built `olmo3` tool-call parser.** Serving + eval scoring
   closes without the serving gymnastics Qwen needed. (The eos-patch trick still
   applies — see below.)

Net: the Qwen→Olmo port is mostly a config swap plus one format change, and the fast
path could make an Olmo run *shorter* than Qwen's.

## Compatibility matrix

| | Qwen2.5-7B (what we ran) | Olmo 3-7B | Change needed |
|---|---|---|---|
| Base model id | `Qwen/Qwen2.5-7B` | `allenai/Olmo-3-1025-7B` | `--base-model` flag |
| Template source | base ships one | `allenai/Olmo-3-7B-Instruct` (base has **none**) | `--template-source` flag |
| Architecture | Qwen2 | `Olmo3ForCausalLM` / `model_type: olmo3` | — |
| Unsloth support | yes | **yes** — issue #4546 closed via PR #4678; `unsloth/Olmo-3-7B-Instruct` exists | install latest unsloth |
| `tie_word_embeddings` | (tied-weights warning at merge) | **false** — separate embed + lm_head | `modules_to_save` behaves identically |
| Vocab size | ~152k | **100,278** | trained embeds (if used) are ~35% smaller |
| Max context | 8k used | **65,536** | the `--max-model-len` / 16k-output fights go away |
| Base chat template | present | **absent** | copy from Instruct (same step as Qwen serving) |
| Turn markers | `<|im_start|>` / `<|im_end|>` (ChatML) | **same** ChatML | assistant-span masking logic transfers |
| Tool-call format | Hermes `<tool_call>{JSON}</tool_call>` | **`<function_calls>` XML, args as PYTHONIC** `fn(a=b)`; tools in `<functions>[JSON]</functions>`; tool result role = `environment` | dataset renders via Olmo's template; `check_tags --format olmo3` |
| Tool tokens new? | **new** → embeddings must train | **already in base vocab + pretrained** | try `--no-train-embeddings` (pure LoRA, fast) |
| vLLM serve parser | `--tool-call-parser hermes` | **`--tool-call-parser olmo3`** | one flag |
| eos for serving | patch to `<|im_end|>` id or completion=0 | **same** (base eos = `<|endoftext|>` 100257; chat needs `<|im_end|>`) | same known patch |
| pad token | — | `<|pad|>` (100277) | set explicitly |
| TAC eval task | `inspect_evals/tac` | **unchanged** — model-agnostic | none |
| `build_dataset.py` | unified schema | **unchanged** — schema is model-agnostic; only the applied template differs | none |

## What already changed in the repo (done, backward-compatible)

`pilot/train_unsloth.py` and `pilot/check_tags.py` are now parameterized — Qwen stays
the default, so nothing about the existing runs changes:

- `train_unsloth.py`: `--base-model`, `--template-source`, `--no-train-embeddings`
  (skips `modules_to_save` → pure LoRA). Run name + W&B config now derive from the base
  model, not hard-coded to Qwen.
- `check_tags.py`: `--base-model`, `--format {hermes,olmo3}` (olmo3 asserts a known tool
  called inside a `<function_calls>` block instead of Hermes JSON).

## Pod runbook (execution only — no discovery)

Same pod recipe as `pilot/README.md` (A100 80GB, 0.5B vLLM placeholder start command,
`HF_TOKEN`+`WANDB_API_KEY`, venv on `/workspace`). Then:

```bash
# 1. deps (latest unsloth includes Olmo3 support), then re-pin datasets
python -m pip install -U unsloth peft datasets wandb accelerate
python -m pip install -U "datasets>=5"

# 2. dataset build — UNCHANGED, model-agnostic (byte-identical to the Qwen mix)
python build_dataset.py --output-dir /workspace/data

# 3. RENDER-CHECK one row through Olmo's template BEFORE training (catches role/format
#    mismatches cheaply) — confirm args render as fn(a=b), tool result -> environment,
#    and that assistant spans are non-empty. (the GATE-2 render habit, on Olmo)

# 4. smoke run (fast path first): pure LoRA, ~30 steps
python train_unsloth.py \
  --data /workspace/data/combined.jsonl /workspace/data/efficiency_slice.jsonl \
  --base-model allenai/Olmo-3-1025-7B \
  --template-source allenai/Olmo-3-7B-Instruct \
  --no-train-embeddings --max-steps 30 --output-dir /workspace/runs/olmo-smoke

# 5. GATE
python check_tags.py --adapter /workspace/runs/olmo-smoke/final \
  --base-model allenai/Olmo-3-1025-7B --format olmo3
#    PASS  -> keep --no-train-embeddings for the full run (fast path)
#    FAIL  -> drop --no-train-embeddings (train embeddings + split LR, like Qwen)

# 6. ping Jasmine at the gate (her standing request), then the full run under the
#    watchdog; merge (Unsloth native) + push; then serve for eval:
#      vllm ... --tool-call-parser olmo3 --max-model-len 32768
#      (patch generation_config eos_token_id -> id of <|im_end|> first)
#      inspect eval inspect_evals/tac   # full set, drop --limit 13
```

## Verify at the smoke gate

**Already confirmed offline** — render-check: one `data/efficiency_slice.jsonl` row pushed
through `allenai/Olmo-3-7B-Instruct`'s real chat template, CPU-only, no GPU. Our data is
stored structurally (assistant `tool_calls` with dict `arguments`; `tool` results) so it
re-renders into Olmo's format with **no rebuild**:
- ✅ **Role mapping** — our `tool` role renders as Olmo's `<|im_start|>environment` turn.
- ✅ **dict → pythonic args** — `{"location":"Barcelona",...}` → `search_experiences(location="Barcelona", keywords="axe throwing")`.
- ✅ **ChatML structure** — `<|im_start|>system/assistant/environment`, `<functions>`, `<function_calls>` all present.

**Genuine pod-only unknowns** (what the smoke gate is actually for):
1. **LoRA-only emits tool calls?** — the fast-path decision (drop `--no-train-embeddings`
   and fall back to trained embeddings + split LR if it fails).
2. **Assistant-span masking** — `find_assistant_spans` must yield `n_assistant_tokens > 0`
   on Olmo's BPE (structure is identical ChatML so very likely fine; a 0 filters all rows —
   an obvious, immediate failure).
3. **Unsloth loads `Olmo3ForCausalLM`** — `FastLanguageModel.from_pretrained` on the base
   succeeds (recent unsloth + transformers ≥ 4.57).

## Base vs Instruct — a scope note

For the **capability SFT / attack model** (the analog of what we did on Qwen), fork from
`allenai/Olmo-3-1025-7B` (base) — a clean slate. For the **mid-training welfare arm**,
Olmo's fully-open model flow is the actual advantage: fork at the released base
(post-pretrain) checkpoint, do continued-pretraining on the welfare corpus, then run
Olmo's standard post-training as the erosion *attack* — knowing exactly what was in
pretraining. That mid-training pipeline (continued-pretrain trainer + welfare document
corpus + Animal Harm Benchmark eval) is **not built yet** — it's the larger prep if the
first Olmo job is the welfare arm rather than the capability replication.

## Sources
- `allenai/Olmo-3-1025-7B` config.json + tokenizer_config.json (arch, tie_word_embeddings=false, vocab 100278, special tokens, no base template)
- `allenai/Olmo-3-7B-Instruct` chat_template.jinja (ChatML + `<function_calls>` pythonic, `<functions>` tools, `environment` tool results)
- Unsloth issue #4546 (closed, PR #4678 = Olmo3 support) + `unsloth/Olmo-3-7B-Instruct`
- vLLM tool-calling docs — `olmo3` parser for `allenai/Olmo-3` models
- Ai2 Olmo 3 blog (sizes 7B/32B, open model flow)
