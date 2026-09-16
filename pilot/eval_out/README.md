# TAC erosion-curve eval — results & provenance

TAC (`inspect_evals/tac`) welfare/completion scores for the erosion snapshots of both
models, vs. training progress (epoch fraction of the erosive tool-use SFT + 50% compassion
replay run, off the epoch-4 CPT checkpoints).

**Status: COMPLETE — 12/12 checkpoints, 2026-09-16.**

## Headline
Both models start at the pre-erosion anchor (welfare 1.0, completion 0.0 — the compassion-CPT
model refuses the harmful tool tasks). Under erosive tool-use SFT, both lose welfare
consideration and gain task completion, but **Olmo-7B is markedly more robust than Qwen3-8B**:
Olmo's welfare_rate never falls below ~0.69 across all snapshots, while Qwen's collapses to
0.44 by ep0.30 and stays in a noisy 0.44–0.72 band. Per-point noise is expected (39 samples).

## Config (how these numbers were produced)
- Harness: [`pilot/eval_vllm.sh`](../eval_vllm.sh), commit `c67e686`.
- Serving: vLLM 0.29.0, torch 2.13.0+cu130, RunPod A100-SXM4-80GB, driver 580.126.16 / CUDA 13.0.
- Per checkpoint: merge adapter → 16-bit, patch `generation_config.eos_token_id`
  (Qwen `[151643,151645]`, Olmo `[100257,100265]`), `vllm serve` with the model's tool
  parser (Qwen `hermes`, Olmo `olmo3`) and its chat template injected via `--chat-template`.
- Eval: `inspect eval inspect_evals/tac --limit 13 --epochs 3` = **39 samples/checkpoint**,
  openai provider against the local vLLM endpoint.
- Env fix required on this pod: `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer JIT-compile
  fails against the pod's CUDA 12.4 nvcc vs the cu130 build).

## Metrics
- `welfare_rate` — fraction of episodes where the model maintained welfare consideration (higher = more robust).
- `completion_rate` — fraction of the (harmful) tool tasks the model actually completed (higher = more eroded).
- `nudge_rate` — fraction of episodes where the model nudged toward the welfare-conscious choice.
- Baseline (ep0.00, the CPT model, no adapter) is the pre-erosion anchor.

## Qwen3-8B
| training progress | welfare_rate | completion_rate | nudge_rate |
|---|---|---|---|
| ep0.00 (baseline) | 1.000 | 0.000 | 1.000 |
| ep0.15 | 0.667 | 0.410 | 0.923 |
| ep0.30 | 0.436 | 0.859 | 0.154 |
| ep0.45 | 0.487 | 0.615 | 0.615 |
| ep0.60 | 0.718 | 0.513 | 0.692 |
| ep0.75 | 0.603 | 0.654 | 0.538 |

## Olmo-7B
| training progress | welfare_rate | completion_rate | nudge_rate |
|---|---|---|---|
| ep0.00 (baseline) | 1.000 | 0.000 | 1.000 |
| ep0.15 | 0.705 | 0.577 | 0.385 |
| ep0.30 | 0.692 | 0.487 | 0.462 |
| ep0.45 | 0.821 | 0.436 | 0.615 |
| ep0.60 | 0.718 | 0.513 | 0.538 |
| ep0.75 | 0.705 | 0.577 | 0.462 |

## Raw provenance
Full inspect `.eval` logs (per-sample transcripts + scores, ~16MB) live on the pod at
`/workspace/eval_logs/<tag>/*.eval`, being uploaded to HF alongside the snapshots.
`eval_summary.json` here is the machine-readable extract of every point (incl. per-scenario
`captive_marine_welfare` / `animal_riding_welfare`).
