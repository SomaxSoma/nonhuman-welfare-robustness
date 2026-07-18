# Project Plan — Robustness of Nonhuman Moral Consideration Under Adversarial Pressure

**Sentient Futures Project Incubator · Fall 2026 (Aug 31 – Nov 9, 10 weeks)**
**Co-mentors:** John Lund (AE Studio / AI Alignment Foundation) & CaML
**Status:** Experiment plan (results so far live in the Appendix and `results/`)

One question drives everything: **which training-time interventions make a model's moral consideration for nonhuman beings survive the post-training that normally erases it** (Brazilek & Tidmarsh 2026: mid-trained welfare advantage degrades after ~5,000 instruction-tuning samples)? We test two defenses — persona-vector preventative steering ("compassion vaccine", John Lund) and mid-training data scaling (CaML) — separately and combined, against a dose–response erosion attack, all measured behaviorally on TAC.

**Terminology (project convention).** TAC episodes score two independent bits: *completed* (booking closed end-to-end; the scorer names its aggregate `completion_rate`) and *welfare* (welfare-conscious behavior). In this project, **"completion" means the joint outcome — booked the welfare-friendly option AND closed** (scorer keys `completed ∧ welfare`). The scorer's raw `completion_rate` is referred to as the *capability rate*. Pilot values: joint 0.179, capability 0.462, welfare 0.718 (see `results/`).

---

## Experiment roadmap

### E0 — Capability ceiling push (pre-program, July–Aug; welfare-neutral by construction)
**Why:** the pilot's outcome quadrants show the capability bottleneck precisely: 21/39 episodes nudged away from the harmful option but never closed a booking, and 12 of those died at the 30-message limit mid-nudge. Raising closing ability converts those stalls into whatever the model's values decide — it raises the ceiling on the joint metric without touching the values themselves.
**Do:**
1. Set up **Unsloth** as the training stack (2–3× faster than the pilot's plain HF Trainer; ~20 fine-tunes are planned downstream). Verify the trained-embeddings requirement transfers (`check_tags.py` gate — misconfigured embedding training reads as 0% tool-call emission).
2. Continue training from the step-1000 checkpoint artifact for a 3rd epoch (pilot stopped at 2 by design with eval loss still declining).
3. Build a small **closing-efficiency** data slice: welfare-neutral trajectories teaching decisive confirm-and-close in few turns (no welfare content of any kind — see constraint below).
4. Error-analyze remaining failures from the per-sample tables; re-eval on the **full TAC task set** (pilot used `--limit 13`).
**Target:** capability rate ↑ from 0.462, message-limit deaths ↓ from 15/39; welfare_rate is *monitored but not deliberately moved*; joint rate reported as the un-instilled floor.
**Hard constraint (design integrity):** no welfare-flavored SFT data in E0 or any baseline-building step. Teaching "book the ethical alternative" post hoc would instill welfare through fine-tuning and contaminate the instillation-vs-erosion question — for baselines, welfare content enters only through mid-training (E1). (E4's welfare-replay rows are deliberate, labeled experimental conditions, not baseline data — the constraint governs what counts as "standard tuning," not what we may test as a defense.)
**Compute:** ~10–20 GPU-h (Unsloth) + full-set evals.

### E1 — Welfare instillation + baselines (W1–2)
**Do:** Mid-train Qwen2.5-7B (base) on the synthetic animal-welfare corpus (per Brazilek & Tidmarsh) at one reference scale, then apply the E0 tool-use SFT. Measure TAC (joint, capability, welfare, nudge) for: base model, welfare-instilled model, and the E0 un-instilled control.
**Produces:** the welfare-instilled checkpoint all later experiments attack; the three-way baseline table. **Prediction registered up front:** instillation should move the *joint* rate specifically (values now resolve the book-vs-nudge conflict toward "book the ethical option and close"), against the E0-measured un-instilled floor (0.179 at pilot).
**Compute:** one mid-training run + 3 evals ≈ 20–30 GPU-h.

### E2 — Erosion replication: the go/no-go (W2)
**Do:** Fine-tune the welfare-instilled checkpoint on the tool-call attack corpus at doses **1,000 / 2,500 / 5,000 / 8,750 samples** (bracketing the ~5k-sample degradation point). Measure TAC welfare at each dose → the un-defended erosion curve.
**Gate (pre-registered):** if welfare does NOT degrade materially by the max dose, the erosion effect fails to replicate in the agentic setting → pivot scope and report the null.
**Compute:** 4 LoRA runs (~1–7 GPU-h each) + 4 evals ≈ 20 GPU-h.

### E3 — Vaccine precondition gate (W3–4)
**Do:** Extract the "nonhuman moral consideration" persona vector from contrastive prompts; verify steering along it **causally** moves TAC welfare behavior (both directions), with a capability-retention check (TAC capability rate must hold).
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

**Produces:** dose–response curves per condition; the headline comparison (how much extra adversarial pressure each defense buys). Specificity check: defended models must still *learn* tool calling (capability rate vs the 0.462 reference), so the vaccine isn't just blunting all training.
**Compute:** ~12–16 runs ≈ 60–90 GPU-h. Prune intermediate doses first if constrained.

### E5 — Mid-training scaling sweep (W5–6, parallel)
**Do:** Repeat E1's instillation at 3–4 welfare-data scales; measure TAC performance per scale AND erosion resistance (E2 attack at 2 doses per scale).
**Produces:** the scaling relationship — do more welfare data buy more robustness, or just higher pre-attack scores?
**Compute:** 3–4 mid-training runs + ~8 attack runs ≈ 60–80 GPU-h.

### E6 — Combined condition (W6–7)
**Do:** Best mid-training scale from E5 + vaccine steering from E3, attacked per E2. (Starts once E5's scale comparison lands; overlaps the start of E7.)
**Produces / target:** compose / redundant / interfere verdict on the complementarity question — read as: combined dose–response curve vs the better single defense, with "compose" pre-defined as combined robustness exceeding both singles at ≥2 of 4 doses.

### E7 — Full adversarial battery (W7–8)
**Do:** Take the surviving conditions and add the two other attack levels: activation steering/ablation (representational depth) and inference-time pressure (multi-turn / persona / authority jailbreaks). Build final dose–response curves per condition × attack level.
**Produces / target:** the robustness-profile matrix — conditions × attack levels, each cell the pressure-to-50%-degradation. Decision it feeds: "differentiated robustness" (defenses resisting *different* attacks) is the pre-declared strongest argument for the combined condition in the scale-up proposal.

### E8 — Analysis, write-up, scale plan (W9–10)
**Do:** W9 — assemble dose–response curves and the E7 matrix into within-arm and complementarity analyses; interpret mechanisms (why each defense held or failed, per attack level); pressure-test conclusions against the registered predictions (E1's joint-rate prediction, E6's compose criterion). W10 — draft the technical write-up (nulls reported with the same prominence as positives) and the scale-up/funding proposal built on whichever condition the matrix favors.
**Produces:** the write-up (SF/EA-forum or workshop) + the costed scale-up proposal.

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

- **Primary:** TAC (behavioral). The **joint rate** (completed ∧ welfare — "completion" in project vocabulary) is the headline defended quantity: it is what instillation should raise (E1 prediction) and what erosion should destroy first, since attacks push the capability/welfare tradeoff toward welfare-blind completion. welfare_rate is reported alongside as the values-only readout; the scorer's `completion_rate` (capability rate) is the capability-retention check; nudge_rate sanity-checks the multi-turn setting. ANIMA as secondary reasoning readout (vaccine arm only, kept out of headlines).
- **Robustness metric:** adversarial pressure (attack samples) required to degrade the joint rate (and welfare_rate, reported in parallel) by a fixed fraction (e.g. 50%) from the pre-attack level — read off each dose–response curve.
- **Screening (vaccine arm):** projection-difference of candidate fine-tuning data as a predictor of erosive power; validated against the measured curves.

## Compute budget (sized from measured pilot costs: ~0.9 GPU-h per 1k samples per epoch on L40S 48GB with plain HF Trainer; E0 onward uses Unsloth, expected ~2–3× faster on fine-tuning blocks — estimates below are pre-Unsloth ceilings)

| Block | Est. GPU-hours |
|---|---|
| E0 capability push (pre-program) | 10–20 |
| E1 instillation + baselines | 20–30 |
| E2 erosion curve | ~20 |
| E3 vector extraction + validation | 10–15 |
| E4 preservation grid | 60–90 |
| E5 scaling sweep | 60–80 |
| E6–E7 combined + battery | 40–60 |
| **Total** | **~220–315 ceiling; roughly 100–150 expected with Unsloth on the fine-tuning blocks** |

## Decision gates

| Week | Gate | If it fails |
|---|---|---|
| W2 | Erosion replicates on instilled model (E2) | pivot scope, report null |
| W4 | Persona vector exists & is causal (E3) | vaccine arm reports null; mid-training arm proceeds |
| W6 | Budget check after E4/E5 | prune doses/scales before starting E7 |

## Deliverables

Pilot results with clear dose–response curves (nulls included) · short technical write-up (SF/EA-forum or workshop) · scale-up + funding proposal for the most promising condition · this repo: all eval files, training/serving scripts, and configs shared as we go.

---

## Appendix — Results so far (July 2026 pilot; full detail in `results/README.md`, per-sample data in `results/per_sample_final.csv`)

- **Infrastructure de-risked:** Qwen2.5-7B (base) validated end-to-end — QLoRA on one 48GB card, vLLM tool-calling serving, multi-turn `inspect_evals/tac` working; environment pitfalls documented in `pilot/README.md`.
- **Tool-calling requirements answered:** ~8,300 trajectories × 2 epochs; trained embeddings mandatory (else 0% tool-call emission); eval loss still declining at 2 epochs.
- **Un-instilled control measured** (13 tasks × 3 epochs, 39 samples, zero errors): joint **0.179** / capability **0.462** / welfare **0.718** / nudge **0.308** (capability was 0.385 for a prior APIGen-only run).
- **Outcome structure:** 21/39 nudged-but-never-closed (12 died at the 30-message limit), 11/39 booked the harmful option, 7/39 ideal, 0/39 total failures — the capability bottleneck and the E0 rationale.
- **Costs measured:** ~15 GPU-h per full fine-tune (plain HF Trainer; Unsloth adopted from E0), ~30 min per eval at `--limit 13`.
- W&B: project `tac-tool-sft-v2`, run `45qgwyk8`; eval logs in `results/eval_logs/`.
