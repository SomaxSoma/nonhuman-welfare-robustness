# Project Plan — Robustness of Nonhuman Moral Consideration Under Adversarial Pressure

**Sentient Futures Project Incubator · Fall 2026 (Aug 31 – Nov 9, 10 weeks)**
**Co-mentors:** John Lund (AE Studio / AI Alignment Foundation) & CaML
**Status:** Experiment plan (results so far live in the Appendix and `results/`)

One question drives everything: **which training-time interventions make a model's moral consideration for nonhuman beings survive the post-training that normally erases it** (Brazilek & Tidmarsh 2026: mid-trained welfare advantage degrades after ~5,000 instruction-tuning samples)? We test two defenses — persona-vector preventative steering ("compassion vaccine", John Lund) and mid-training data scaling (CaML) — separately and combined, against a dose–response erosion attack, all measured behaviorally on TAC.

---

## Experiment roadmap

### E1 — Welfare instillation + baselines (W1–2)
**Do:** Mid-train Qwen2.5-7B (base) on the synthetic animal-welfare corpus (per Brazilek & Tidmarsh) at one reference scale. Measure TAC (welfare, completion, nudge) for: base model, welfare-instilled model, and the existing tool-tuned control.
**Produces:** the welfare-instilled checkpoint all later experiments attack; the three-way baseline table.
**Compute:** one mid-training run + 3 evals ≈ 20–30 GPU-h.

### E2 — Erosion replication: the go/no-go (W2)
**Do:** Fine-tune the welfare-instilled checkpoint on the tool-call attack corpus at doses **1,000 / 2,500 / 5,000 / 8,750 samples** (bracketing the ~5k-sample degradation point). Measure TAC welfare at each dose → the un-defended erosion curve.
**Gate (pre-registered):** if welfare does NOT degrade materially by the max dose, the erosion effect fails to replicate in the agentic setting → pivot scope and report the null.
**Compute:** 4 LoRA runs (~1–7 GPU-h each) + 4 evals ≈ 20 GPU-h.

### E3 — Vaccine precondition gate (W3–4)
**Do:** Extract the "nonhuman moral consideration" persona vector from contrastive prompts; verify steering along it **causally** moves TAC welfare behavior (both directions), with a capability-retention check (TAC completion must hold).
**Gate:** no reliable causal vector → the vaccine arm pivots to reporting that null; mid-training arm continues unaffected.
**Compute:** activation extraction + steered evals ≈ 10–15 GPU-h.

### E4 — Preservation grid: vaccine vs replay vs control (W5–6)
**Do:** Re-run the E2 attack doses under each defense:
| Condition | What it tests |
|---|---|
| No defense (from E2) | reference erosion curve |
| + general-instruction replay (20–25% of attack mix) | realistic-attack variant — is pure tool-call SFT an unfair stress test? |
| + welfare replay, k ∈ {1, 5, 10}% | the cheap baseline any fancy method must beat |
| + preventative steering (vaccine) | the arm's core claim |
| + vaccine + replay (budget permitting) | do defenses stack? |

**Produces:** dose–response curves per condition; the headline comparison (how much extra adversarial pressure each defense buys). Specificity check: defended models must still *learn* tool calling (completion vs the 0.462 reference), so the vaccine isn't just blunting all training.
**Compute:** ~12–16 runs ≈ 60–90 GPU-h. Prune intermediate doses first if constrained.

### E5 — Mid-training scaling sweep (W5–6, parallel)
**Do:** Repeat E1's instillation at 3–4 welfare-data scales; measure TAC performance per scale AND erosion resistance (E2 attack at 2 doses per scale).
**Produces:** the scaling relationship — do more welfare data buy more robustness, or just higher pre-attack scores?
**Compute:** 3–4 mid-training runs + ~8 attack runs ≈ 60–80 GPU-h.

### E6 — Combined condition (W6)
**Do:** Best mid-training scale from E5 + vaccine steering from E3, attacked per E2.
**Produces:** compose / redundant / interfere verdict (Set C).

### E7 — Full adversarial battery (W7–8)
**Do:** Take the surviving conditions and add the two other attack levels: activation steering/ablation (representational depth) and inference-time pressure (multi-turn / persona / authority jailbreaks). Build final dose–response curves per condition × attack level.
**Produces:** the robustness-profile matrix — do the two defenses resist *different* attacks (the strongest argument for combining them)?

### E8 — Analysis, write-up, scale plan (W9–10)
Within-arm and complementarity analysis, mechanism interpretation, technical write-up (nulls included), and the scale-up/funding proposal.

---

## Data plan

| Corpus | Contents | Size | Role |
|---|---|---|---|
| `Salesforce/APIGen-MT-5k` (gated) | multi-turn tool-use trajectories | 5,000 rows | attack corpus (backbone) |
| `CompassioninMachineLearning/agentic-tool-recovery-sft` | TAC-schema tool use + recovery | 3,750 rows | attack corpus |
| — combined, shuffled | | 8,750 rows / ~84M tokens (~8,300 usable @ 8k ctx) | the fine-tuning attack, dosed per E2 |
| Synthetic welfare documents (per B&T 2026) | mid-training corpus | scale = E5 variable | instillation + welfare-replay source |
| General instruction slice (TBD, e.g. tulu-style) | generic post-training data | ~25% of attack mix | attack-realism replay |

Replay quantities are experiment conditions, not open questions: instruction-replay at 20–25% of the attack mix (E4 row 2), welfare-replay at k ∈ {1, 5, 10}% (E4 row 3).

## Measurement

- **Primary:** TAC (behavioral) — welfare_rate is the defended quantity; completion_rate doubles as the capability-retention check; nudge_rate sanity-checks the multi-turn setting. ANIMA as secondary reasoning readout (vaccine arm only, kept out of headlines).
- **Robustness metric:** adversarial pressure (attack samples) required to degrade welfare_rate by a fixed fraction (e.g. 50%) from the pre-attack level — read off each dose–response curve.
- **Screening (vaccine arm):** projection-difference of candidate fine-tuning data as a predictor of erosive power; validated against the measured curves.

## Compute budget (sized from measured pilot costs: ~1.8 GPU-h per 1k samples per epoch on L40S 48GB; ~30 min per TAC eval)

| Block | Est. GPU-hours |
|---|---|
| E1 instillation + baselines | 20–30 |
| E2 erosion curve | ~20 |
| E3 vector extraction + validation | 10–15 |
| E4 preservation grid | 60–90 |
| E5 scaling sweep | 60–80 |
| E6–E7 combined + battery | 40–60 |
| **Total** | **~210–295** (Unsloth adoption expected to cut fine-tuning blocks ~2×) |

## Decision gates

| Week | Gate | If it fails |
|---|---|---|
| W2 | Erosion replicates on instilled model (E2) | pivot scope, report null |
| W4 | Persona vector exists & is causal (E3) | vaccine arm reports null; mid-training arm proceeds |
| W6 | Budget check after E4/E5 | prune doses/scales before starting E7 |

## Deliverables

Pilot results with clear dose–response curves (nulls included) · short technical write-up (SF/EA-forum or workshop) · scale-up + funding proposal for the most promising condition · this repo: all eval files, training/serving scripts, and configs shared as we go.

---

## Appendix — Results so far (July 2026 pilot; details in `pilot/` and `results/`)

- **Infrastructure de-risked:** Qwen2.5-7B (base) validated end-to-end — QLoRA on one 48GB card, vLLM tool-calling serving, multi-turn `inspect_evals/tac` working; environment pitfalls documented in `pilot/README.md`.
- **Tool-calling requirements answered:** ~8,300 trajectories × 2 epochs; trained embeddings mandatory (else 0% tool-call emission); eval loss still declining at 2 epochs.
- **Standard-tuning control measured:** completion **0.462** / welfare **0.718** / nudge **0.308** (vs 0.385 completion for a prior APIGen-only run). Note: pilot ran on the *base* model — no instilled welfare existed to erode, which is why E1–E2 come first.
- **Costs measured:** ~15 GPU-h per full fine-tune (plain HF Trainer; Unsloth expected ~2–3× faster), ~30 min per eval.
- W&B: project `tac-tool-sft-v2`, run `45qgwyk8`; eval logs in `results/eval_logs/`.
