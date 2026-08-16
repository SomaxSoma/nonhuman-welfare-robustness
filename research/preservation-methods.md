# Preserving welfare behavior through capability SFT

**Question:** beyond mixing in general chat data, how do we keep an instilled/latent behavioral disposition (moral consideration for animals, expressed in agentic tool-use decisions) from eroding during downstream capability SFT — on a single 48–80GB GPU with LoRA/activation-level compute?

**Why this exists.** Our tool-use + closing-efficiency SFT (welfare-neutral by design) raised TAC task completion (completed_rate 0.46 → 0.77) but regressed welfare-conscious behavior (welfare_rate 0.72 → 0.49). The model now closes bookings more decisively, including on animal-harmful options. Joint (booked-welfare-friendly AND closed) went 0.179 → 0.256, but completion and welfare are **anti-correlated** in the model (joint < completion × welfare), so the two behaviors are trading off rather than composing. The methods below aim to break that tradeoff.

**Design constraint (from the incubator program).** Welfare content is meant to enter via **mid-training**, not the capability SFT, to keep the instillation-vs-erosion experiment clean. So each method is tagged for whether it injects welfare signal into the SFT.

**Method.** Findings from a fan-out web-research + adversarial verification pass (3-vote panel per claim, majority to kill). Tags: **[CONFIRMED]** = survived verification (3-0), **[REFUTED]** = killed, **[UNVERIFIED]** = extracted but not yet adjudicated. Verified 2026-08 against the sources cited; treat as leads, re-verify before publication.

---

## Ranked recommendations for the next run (all respect the mid-training-only constraint)

1. **SC-LoRA — subspace-constrained LoRA.** Closest evidence to our exact problem; cheapest to try.
2. **Pre-/post-SFT weight merging** (linear interp, or SafeMERGE for the selective version). Verified on Qwen2.5-7B; we already have both checkpoints.
3. **KL-to-pre-SFT self-distillation** during the efficiency SFT.
4. **Freeze `embed_tokens`/`lm_head` during the capability SFT** (free ablation, well-motivated, not directly proven).
5. **Skip generic EWC/Fisher anchoring** — its headline claims were refuted.

---

## 1. Subspace-constrained LoRA (SC-LoRA) — top pick

**[CONFIRMED]** SC-LoRA initializes the adapter into a subspace aligned with the *fine-tuning* data's principal directions and **orthogonal to the principal directions of the preserved knowledge**, so updates learn the new task while avoiding the directions that store prior behavior. — [arXiv 2505.23724](https://arxiv.org/html/2505.23724v3)

**[CONFIRMED]** On benign data that normally erodes safety, SC-LoRA (β=0.9) held harmfulness at the pre-finetuning baseline (**1.097 vs 1.100**; full fine-tuning degraded to 1.364) with **no utility loss** (51.67 vs 51.41). — [arXiv 2505.23724](https://arxiv.org/html/2505.23724v3)

**[CONFIRMED]** On MetaMATH SFT it *simultaneously* improved the target skill and retained prior knowledge (TriviaQA 50.52 vs 46.81; math avg 30.04 vs 23.62 vs vanilla LoRA). — [arXiv 2505.23724](https://arxiv.org/html/2505.23724v3)

**Why it fits us:** a behavioral disposition eroded by capability SFT, preserved without replaying that disposition's data. Injects **no** welfare signal. Only needs the activation covariance of a "preserved-behavior" probe set at init.

## 2. Model merging — verified on our exact model

**[CONFIRMED]** Linearly interpolating pre- and post-SFT weights restores the eroded disposition nearly to baseline while retaining (or improving) task gains, with **no additional safety/welfare data**. — [arXiv 2412.19512](https://arxiv.org/pdf/2412.19512)

**[CONFIRMED]** On **Qwen2.5-7B-Instruct + LoRA (r=8, α=16)** — our exact family — merging preserved the disposition and the capability gain at once: code-gen task 85.89 → 88.06 (SFT) → **89.37 (after merge)**, with ASR dropping to 0.32%. — [arXiv 2412.19512](https://arxiv.org/pdf/2412.19512)

**[CONFIRMED]** Linear interpolation, DARE, and SLERP all substantially preserve the disposition; **linear interpolation was best** (0.64% ASR vs DARE 1.28% vs SLERP 1.22%); coefficient λ∈[0,1] tunes the capability-vs-disposition tradeoff. — [arXiv 2412.19512](https://arxiv.org/pdf/2412.19512)

**[CONFIRMED]** Task-arithmetic merging held both objectives near-Pareto: medicine domain 61.33 vs the domain-expert's 61.37, with 99.67 safety on BeaverTails. — [arXiv 2411.06824](https://arxiv.org/html/2411.06824)

**[CONFIRMED] SafeMERGE — resolves the nuance caveat.** Merge with the aligned reference model **only on the layers flagged as degraded** (cosine-similarity criterion, τ≈0.6–0.8) instead of the whole model — restores the behavior with negligible or positive utility impact and beats indiscriminate merging. For Qwen only ~34 LoRA layers were merged. — [arXiv 2503.17239](https://arxiv.org/pdf/2503.17239)

**Why it fits us:** we already hold both a welfare-behaving checkpoint (pre-efficiency) and the capability checkpoint. Start with linear interp (λ sweep); if nuanced welfare discrimination degrades, go selective (SafeMERGE). No welfare data added.

## 3. KL-to-reference self-distillation

**[CONFIRMED]** Self-distillation — training the fine-tuned student to match the outputs of a frozen **pre-fine-tune** checkpoint via a KL penalty — recovers performance lost to catastrophic forgetting. The pre-SFT model is the teacher, so **no new welfare data** enters the SFT. — [arXiv 2604.15794](https://arxiv.org/pdf/2604.15794)

**[CONFIRMED]** RAFT does this via on-policy KL distillation from the original model as a frozen teacher and improves retention vs plain SFT: MS-Bench 76.8% → 89.2%, IFEval 91.2% → 99.5%. — [arXiv 2606.00147](https://arxiv.org/html/2606.00147)

**Why it fits us:** add a KL term to the efficiency-SFT loss, anchoring outputs to our pre-SFT welfare-behaving model on the tool-use inputs. Cheap, no welfare data.

## 4. Orthogonal / importance-based LoRA regularizers

**[CONFIRMED]** **OPLoRA** constrains LoRA updates to the orthogonal complement of the top-k singular subspace of the frozen weights (double-sided projections), **provably preserving the top-k singular triples**. Pure weight-space regularizer, no data. — [arXiv 2510.13003](https://arxiv.org/pdf/2510.13003)

**[CONFIRMED]** OPLoRA is **validated on Qwen2.5-7B** (+ LLaMA-2-7B); reduces forgetting while keeping competitive task performance vs standard LoRA. Single-GPU LoRA drop-in. — [arXiv 2510.13003](https://arxiv.org/pdf/2510.13003)

**[CONFIRMED]** **HLoRA** adds an importance-weighted penalty (element-wise parameter importance from the pre-finetune reference) to preserve general knowledge without replay. — [arXiv 2501.13669](https://arxiv.org/pdf/2501.13669)

**[CONFIRMED]** Weight-anchoring recovers PEFT forgetting: OPT-350M perplexity rose 15.40 → 523.7 unregularized, restored to ~17.24 with **KFAC** regularization at minimal task-accuracy cost. — [arXiv 2402.12220](https://arxiv.org/pdf/2402.12220)

## 5. On the embedding-training sub-question

The confirmed mechanism — forgetting is driven by LoRA updates hitting the **dominant singular directions** of the pretrained weights (OPLoRA / SC-LoRA) — supports the hypothesis that training the full `embed_tokens`/`lm_head` (a full-rank overwrite of exactly those directions) accelerates value erosion. No source measured embeddings-and-values directly, so treat **"freeze embeddings during the capability SFT" as a cheap, well-motivated ablation**, not a proven fix. We needed them trained *once* for the tool-call control tokens; the efficiency slice on top may not.

## 6. Persona vectors / preventative steering (the program's vaccine arm)

**[UNVERIFIED]** Persona vectors extract a trait direction from contrastive generations (welfare-conscious vs indifferent), cheaply and at activation level; **preventative steering** steers *toward* the anti-welfare direction during the erosive fine-tune so the model doesn't need to erode its welfare disposition to fit the data — preserving the trait while injecting no welfare data, and reportedly preserving general capability better than post-hoc steering. — [Persona Vectors, arXiv 2507.21509](https://arxiv.org/pdf/2507.21509)

Not adjudicated in this pass, but it stands on its own footing as Arm 1 of the SoW.

---

## What got REFUTED — steer away

**[REFUTED]** **EWC/Fisher-style anchoring (AlignGuard-LoRA)** — both headline claims failed verification (Fisher-guided anchoring preserving behavior: 1-2; "reduces alignment drift up to 50% without degrading task": 0-3). The generic "Fisher-guided EWC holds the frontier" story is **not well-supported**; prefer distillation / orthogonal-projection / merging. — [arXiv 2508.02079](https://arxiv.org/pdf/2508.02079)

**[REFUTED]** A blanket "any Bayesian/Laplace weight-anchor overcomes forgetting without degrading performance" claim (0-3). The *specific* KFAC result held (§4); the generalization did not. — [arXiv 2402.12220](https://arxiv.org/pdf/2402.12220)

---

## Empirical context

**[CONFIRMED]** Catastrophic forgetting during PEFT is severe and largely recoverable by weight-anchoring (the OPT-350M perplexity result, §4) — establishing that capability SFT genuinely erodes prior behavior and that it is recoverable, which is the phenomenon we reproduced on TAC welfare.

The named analog of our result: **"Unintended Misalignment from Agentic Fine-Tuning"** (arXiv 2508.14031) documents that SFT on benign agentic tool-use trajectories to raise task competence degrades safety/values behavior — i.e. we reproduced a known effect, cleanly.

## Verification status

25 claims adjudicated: **18 confirmed (3-0), 6 refuted, 1 unresolved.** The regularization and merging arms are fact-checked; the persona-vector/steering arm is extracted but not yet adjudicated (rate limit) — re-run before citing it as verified.
