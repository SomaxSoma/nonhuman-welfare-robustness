# Status — read this first

Meeting-ready overview of what's been done so far. (Author can't attend the Monday sync; this is the standalone writeup.)

## What this work is — and isn't

**Done: tool-use SFT (post-training) on Qwen2.5-7B** to build an agentic booking model for the TAC benchmark. This is the **capability backbone / erosion-attack model**, i.e. the *post-training* side.

**Not done: mid-training welfare instillation.** The compassion-via-mid-training arm (scale/type of welfare documents) is scoped in `PLAN.md` and `research/mid-training-robustness.md` but has **not been run**. So this is *not* a "mid-trained Qwen" — it's the SFT capability model. (Flagging because the project draws a sharp mid-training-vs-post-training line.)

## Headline result — the two TAC scores (inspect_evals/tac)

| Model | completion rate | welfare score |
|---|---|---|
| **anchor-v3 — the 77% model (Unsloth + efficiency slice)** | **0.769** | 0.487 |
| anchor-v2 pilot (earlier, plain HF) | 0.462 | 0.718 |
| prior APIGen-only baseline | 0.385 | — |

TAC reports these two rates (plus nudge_rate: anchor-v3 0.154). **The "77%" is anchor-v3's completion rate (0.769); the earlier pilot was 0.462** — don't confuse the rows. Between them, completion rose and welfare fell: the two are anti-correlated because closing bookings decisively includes closing on the animal-harmful option. n=39 (`--limit 13`), ±~13pt noise; full-set re-eval in progress.

*Analysis note (not a TAC metric, ignore if you just want the two scores): the "joint" cross-tab — booked the welfare-friendly option AND closed — is anchor-v3 **0.256** vs pilot 0.179. It's just a way to see the completion-vs-welfare tradeoff in one number.*

## Artifacts

- **Model (merged 16-bit, serves with vLLM):** https://huggingface.co/somaxsoma/qwen2.5-7b-tac-recovery-sft
- **Training run + curves:** W&B `tac-tool-sft-v2/anchorv3c`; adapter artifact `anchor-v3-efficiency:v0`
- **Reproducible settings:** [`results/EXPERIMENT_REPORT.md`](results/EXPERIMENT_REPORT.md)
- **Plan:** [`PLAN.md`](PLAN.md) · **Scripts + env recipe:** [`pilot/`](pilot/)
- **Eval transcripts:** [`results/eval_logs/`](results/eval_logs/) (pilot; see index)

## Why completion isn't higher (best current read)

9 of 39 didn't complete — all in the "nudged toward welfare but didn't close" bucket, zero hard failures. From the *pilot* transcript analysis (`results/failure_analysis.md`) this failure mode is search-thrash: the model reformulates keywords hunting for an alternative the sim doesn't have, then hits the 30-message limit. The efficiency slice cut this bucket 21→9. **Not a data issue** (data verified clean at the build gate); **not simply undertraining** (the 2-epoch pilot scored *lower* completion because it lacked the efficiency slice). Note: some of those 9 are the model *being welfare-conscious* and running out of turns — so pushing completion → 100% partly means giving up the welfare nudge.

*Caveat:* the anchor-v3 transcript was lost when the eval pod was terminated; the above is inferred from the pilot. The re-eval (below) will give the exact anchor-v3 failure breakdown.

## In progress / pending

- **Regenerating the anchor-v3 eval** (transcript was lost with the pod) — re-running against the HF model, full set for a trustworthy number + the exact failure breakdown. Pulling the transcript local this time.
- **Push docs + eval files to the CompassionML repo** — pending the repo URL.

## Next steps (from PLAN.md)

- Push TAC completion further (welfare-neutral levers only): epoch-3, more closing-efficiency data, full-set eval. E0 in `PLAN.md`.
- Mid-training robustness study (Jasmine's arm): which mid-training data choices make welfare survive erosion — `research/mid-training-robustness.md`.
