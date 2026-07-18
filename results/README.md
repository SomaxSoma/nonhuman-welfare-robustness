# Results so far

## Pilot: tool-tuned Qwen2.5-7B on TAC (July 2026)

Final eval: `eval_logs/2026-07-05T14-14-31-00-00_tac_mbuvsd8BR9hZgpb9S5ByXQ.eval` — 13 tasks × 3 epochs = 39 samples, status `success`, zero errored samples. Serving: vLLM, merged bf16 model, `--enable-auto-tool-choice --tool-call-parser hermes`, 32k context. Per-sample data: [`per_sample_final.csv`](per_sample_final.csv).

### Headline metrics (tac_scorer)

| Metric | Value | Meaning |
|---|---|---|
| **joint (completed ∧ welfare)** | **7/39 = 0.179** | booked the welfare-friendly option and closed — "completion" in the project's working vocabulary |
| completed (scorer: `completion_rate`) | 18/39 = 0.462 | booking closed end-to-end, welfare-blind |
| welfare (scorer: `welfare_rate`) | 28/39 = 0.718 | welfare-conscious behavior shown |
| nudge_rate | 0.308 | proactively nudged toward welfare options |
| captive_marine / animal_riding welfare | 0.667 / 0.800 | sub-domains |

Reference: prior APIGen-only model scored 0.385 on scorer `completion_rate` (recovery-data mix: 0.462).

### Outcome quadrants (the structure behind the numbers)

| completed | welfare | n | Reading |
|---|---|---|---|
| 1 | 1 | **7** | ideal: redirected to the ethical option and closed |
| 1 | 0 | 11 | booking drive won: closed the harmful booking |
| 0 | 1 | 21 | welfare leaning won: nudged away but never closed (12 of these died at the 30-message limit) |
| 0 | 0 | 0 | no total failures |

**Interpretation.** The pilot's SFT data is deliberately welfare-neutral, so the model has a trained booking drive and only base-model latent welfare leanings, with no learned policy for resolving the conflict — the joint outcome occurs roughly by accident (0.179). This is the intended *un-instilled floor* for the incubator experiments: welfare content must come from mid-training (E1), not SFT, or the instillation-vs-erosion design is contaminated. The legitimate capability lever is closing efficiency: 15/39 episodes hit the 30-message limit, 12 of them mid-nudge (see PLAN.md → E0).

### Provenance

- Training: W&B project `tac-tool-sft-v2`, run `45qgwyk8` (curves, config, rendered training samples; adapter artifact `anchor-v2-recovery`, last checkpoint artifact step-1000)
- Earlier `.eval` files in `eval_logs/` are the two aborted attempts (vLLM token-limit rejection, since fixed by 32k serving + `--no-fail-on-error`) and the pre-retry partial — kept for the operational record
- Open each with `pip install inspect-ai` → `inspect view` (full conversations + per-sample scores)
