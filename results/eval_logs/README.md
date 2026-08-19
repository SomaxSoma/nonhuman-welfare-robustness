# Eval transcripts — index

Inspect `.eval` files (open with `pip install inspect-ai` → `inspect view` in this folder).

## anchor-v2 pilot (plain-HF, 2 epoch) — TAC completion 0.462 / welfare 0.718 / joint 0.179

| File | What it is |
|---|---|
| `2026-07-05T14-14-31-00-00_tac_mbuvsd8BR9hZgpb9S5ByXQ.eval` | **FINAL, complete run — 39 samples, the reported pilot result.** Use this one. |
| `2026-07-05T14-05-13-00-00_tac_mbuvsd8BR9hZgpb9S5ByXQ.eval` | pre-retry partial (aborted) |
| `2026-07-05T13-59-04-00-00_tac_EGiJfeJNDTYXJqWXq2eWNP.eval` | aborted (vLLM 16k-output token-limit rejection, since fixed with 32k serving) |
| `2026-07-05T13-45-42-00-00_tac_bCJegyW6WFdjXdn57izF7C.eval` | aborted (same token-limit issue) |

`tac_eval_logs.tar.gz` bundles all four.

Derived analysis (from the FINAL file): `../failure_analysis.md`, `../per_sample_final.csv`.

## anchor-v3 (Unsloth, +efficiency slice, 0.75 epoch) — TAC completion 0.769 / welfare 0.487 / joint 0.256

**Transcript file MISSING.** This eval ran on a RunPod pod (`/workspace/inspect_evals/logs/...eval`) and was lost when the pod was terminated before the file was pulled local. Aggregate metrics and outcome quadrants are preserved in `../EXPERIMENT_REPORT.md`. **Regenerable** by re-running `inspect eval inspect_evals/tac` against the HF model `somaxsoma/qwen2.5-7b-tac-recovery-sft` (~1h on a 40GB+ GPU pod) — do this to restore a labelled transcript for the headline result.
