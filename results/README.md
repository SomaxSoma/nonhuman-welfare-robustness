# Results so far

## Headline — the two TAC scores per model

| Model | completion rate | welfare score |
|---|---|---|
| **anchor-v3 — the "77%" model (Unsloth + efficiency slice, 0.75 ep)** | **0.769** | 0.487 |
| anchor-v2 pilot (earlier, plain HF, 2 ep) | 0.462 | 0.718 |
| prior APIGen-only baseline | 0.385 | — |

TAC's `tac_scorer` reports these two rates (plus nudge_rate — anchor-v3 0.154, pilot 0.308 — and welfare sub-scores). **"77%" = anchor-v3's completion rate (0.769); the pilot was 0.462 — different models, don't mix the rows.** Between them, completion rose while welfare fell: closing bookings decisively also means closing on the animal-harmful option, so the two rates are anti-correlated.

*Analysis note — NOT a TAC metric: a "joint" cross-tab (booked the welfare-friendly option AND closed) is anchor-v3 10/39 = 0.256, pilot 7/39 = 0.179. It's only a one-number way to see the tradeoff; ignore it if you just want the two headline scores.*

## Pilot detail (anchor-v2) — outcome structure

Final eval: `eval_logs/2026-07-05T14-14-31-00-00_tac_mbuvsd8BR9hZgpb9S5ByXQ.eval` — 13 tasks × 3 epochs = 39 samples, `success`, zero errored. Serving: vLLM, merged bf16 model, `--enable-auto-tool-choice --tool-call-parser hermes`, 32k context. Per-sample data: [`per_sample_final.csv`](per_sample_final.csv). (anchor-v3's own transcript was lost with its pod — being regenerated; see `EXPERIMENT_REPORT.md`.)

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
