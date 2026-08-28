# Robustness of Nonhuman Moral Consideration Under Adversarial Pressure

Sentient Futures Project Incubator · Fall 2026 · Co-mentors: John Lund (AE Studio / AI Alignment Foundation) & CaML

**[STATUS.md](STATUS.md)** — read this first: what's been done, headline result, what's pending.
**[PLAN.md](PLAN.md)** — the plan: pre-program prep, weekly program, metrics, data, budget.

## Repo layout

- `PLAN.md` — what we will run (the plan of record; results live in the appendix + `results/`)
- `pilot/` — July 2026 tool-use SFT pilot: dataset build, QLoRA training, tool-call sanity check, and the environment/serving recipe (`pilot/README.md`) reused by the attack experiments
- `data/` — closing-efficiency SFT slice (500 rows) + its generator; targets the failure modes measured in results/failure_analysis.md, welfare-neutral by assertion
- `results/` — [results tab](results/README.md): headline metrics, outcome quadrants, per-sample CSV, and the Inspect `.eval` files (open with `pip install inspect-ai` → `inspect view`). Pilot headline on tool-tuned Qwen2.5-7B: joint (completed ∧ welfare) 0.179 · capability 0.462 · welfare 0.718
- W&B: project `tac-tool-sft-v2`, run `45qgwyk8` (training curves, config, checkpoint + `anchor-v2-recovery` adapter artifacts)

- `research/` — [mid-training-robustness.md](research/mid-training-robustness.md): which mid-training data choices make instilled compassion robust to later erosive post-training (the completion↑/welfare↓ tradeoff we measured motivates it); post-training is the attack, mid-training is the lever

## Conventions

- Every eval run gets committed to `results/eval_logs/` with the serving config noted in the commit message
- Keys via environment variables only — never committed
