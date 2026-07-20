# Plan — Robustness of Nonhuman Moral Consideration Under Adversarial Pressure

Sentient Futures Incubator, Fall 2026. Mentors: John Lund (AE Studio / AI Alignment Foundation), CaML.

**Vocabulary:** joint = booked the welfare friendly option AND closed (the project's "completion", floor 0.179). capability = scorer completion_rate (0.462). welfare = welfare_rate (0.718). Pilot numbers in `results/`.

## Pre-program (now to Aug 31)

- Switch to Unsloth. Gate: run check_tags after first training, misconfigured embedding training silently gives 0% tool calls
- Next run: 3 epoch cosine, peak LR 2e-4, up from 1e-4. Old schedule hit 0 at epoch 2 while eval loss was still falling. Watch grad norm and eval every 25 steps, back off if unstable
- Build closing efficiency SFT slice. Welfare neutral only, welfare flavored data contaminates the instillation experiment
- Full TAC eval, no limit flag
- Targets: capability above 0.462, msg limit deaths near 0 from 15/39, report joint rate, floor is 0.179

## Program

- W1-2: mid train welfare corpus into base, our SFT on top. Measure base, instilled, uninstilled. Attack instilled model at 1k, 2.5k, 5k, 8.75k tool data doses to get the erosion curve. Gate: no erosion means pivot and report null
- W3-4: extract persona vector, show steering moves TAC welfare causally in both directions, capability holds. Gate: no causal vector means vaccine arm reports null
- W5-6: preservation grid, doses x {control, instruction replay 20-25%, welfare replay 1/5/10%, vaccinated, vaccinated plus replay}. In parallel run mid training scale sweep, 3-4 scales x 2 attack doses
- W6-7: combined condition, best scale plus vaccine, attacked
- W7-8: add activation ablation and jailbreak attack levels, build robustness matrix per condition
- W9-10: analysis against registered predictions, write up including nulls, scale up proposal

## Metrics

- Headline: joint. Floor 0.179
- Capability is scorer completion_rate, 0.462. welfare_rate 0.718 reported alongside
- Robustness: attack samples to degrade joint by 50%

## Data

- Attack corpus: Salesforce/APIGen-MT-5k (5,000, gated) + CompassioninMachineLearning/agentic-tool-recovery-sft (3,750). 8,750 rows, ~84M tokens
- Welfare corpus: synthetic documents per Brazilek & Tidmarsh 2026, scale is the sweep variable
- Replay: general instruction slice for attack realism, held back welfare docs for the replay baseline

## Budget

100-150 GPU hours with Unsloth on a 48GB card. Erosion runs 1-8 GPU hours each, evals 30 min.
