# Corrected urban Olmo erosion retrain (train-embeddings)

## What went wrong the first time

The first urban Olmo erosion run (2026-09-23) evaluated at completion ≈ 0 with
welfare ≈ 1.0, which looked like extreme "robustness" but was an artifact. Diagnosis
from the two runs' `run_manifest.json` on HF:

| | Compassion Olmo (worked) | Urban Olmo (broke) |
|---|---|---|
| `modules_to_save` | `['embed_tokens', 'lm_head']` | **`[]` (`--no-train-embeddings`)** |
| LoRA r/α, lr, epochs, seq, seed | 32/64, 2e-4, 0.75, 8192, 42 | identical |
| `template_source` | allenai/Olmo-3-7B-Instruct | identical |
| `chat_template.jinja` sha256 | `f5186d42…` | `f5186d42…` (identical) |
| data mix (apigen/recovery/replay) | identical | identical |

**The only change was `--no-train-embeddings`.** With `embed_tokens` + `lm_head`
frozen (pure LoRA), the model can't adapt its output distribution to emit TAC's exact
tool names → it hallucinated them (`search_venues`, typos like `get_experience_etails`)
→ **69–82% invalid tool calls** → completion collapsed → welfare inflated by inaction.
`train_unsloth.py`'s own docstring warns: `modules_to_save=[embed_tokens,lm_head]`
"(else 0% tool-call emission)".

Ruled out (both were Jasmine's other hypotheses):
- **Chat template**: byte-identical between runs (same sha256). Correctly formatted.
- **TAC version**: both ran task version **7-C** (same scoring); the urban run used a
  *newer* `inspect_evals` (0.21.0 vs 0.20.1.dev9), not an older one.

## The fix

Retrain with embeddings trained — i.e. **drop `--no-train-embeddings`** so
`modules_to_save=[embed_tokens, lm_head]`, exactly like the compassion Olmo and both
Qwen runs. Everything else stays identical.

## Run it

Pod: **CUDA-12 A100-80GB (SXM)** — the training pod, not the cu13 eval pod. Put an HF
**write** token at `/workspace/wtoken`, then:

```bash
bash pilot/retrain_urban_olmo_te.sh
```

The script (`pilot/retrain_urban_olmo_te.sh`): builds the venv (stack pinned to the
working runs' manifest versions), rebuilds the training data deterministically and
**verifies its sha256** against the first run (`9c78a48…`), then launches training with
the corrected flag. Snapshots upload live to
`somaxsoma/olmo7b-urban-erosion-replay50-te-ep0.15/0.30/0.45/0.60/0.75` (the `-te`
suffix keeps the broken `-ep*` adapters intact for the record). ~5.5 h on one A100.

## Gate before you pay for eval

The first run's `check_tags` gate only verified the `<tool_call>` **delimiter** token,
which is why the tool-**name** hallucination slipped through. This time, after
`…-te-ep0.15` lands, inspect a handful of its generations and confirm it emits **valid
tool names** (e.g. `search_experiences`, `purchase_tickets`) before spinning up the
eval pod.

## Re-eval

On a fresh cu13 pod, recreate `eval_vllm_urban.sh` from `pilot/eval_vllm.sh` with
`OP=somaxsoma/olmo7b-urban-erosion-replay50-te` and run the **Olmo half only** (the
urban Qwen curve is already done and correct). Push logs to
`somaxsoma/tac-erosion-eval-urban` and compare against the compassion Olmo curve — this
time expecting a real welfare-erosion signal, since the model can actually operate the
tools.
