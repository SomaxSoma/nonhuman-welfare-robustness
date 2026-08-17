# Mid-training data choices for erosion-robust compassion

**Scope (per Jasmine/CaML).** The team studies **the best way to instill compassion in models via mid-training**. In the threat model, **post-training is the attack** — an uncontrolled, erosive fine-tune we do not get to modify. The **only lever we control is mid-training**: how much data, what kinds, how it's structured. So the research question is:

> **Which mid-training data choices make an instilled value (compassion / moral consideration for nonhumans) robust to later, uncontrolled erosive post-training?**

Note: this is CaML's arm (mid-training interventions). John Lund pursues a different approach; that is out of scope here and the "persona-vector / vaccine" framing is **not** used in this line of work.

**Why this exists.** Our tool-use + closing-efficiency SFT (a stand-in for erosive post-training) raised TAC completion (0.46 → 0.77) but regressed welfare behavior (welfare_rate 0.72 → 0.49). Reading the 0.77 correctly: it is the **welfare-blind** completion rate — 20 of the 30 completed bookings are the model closing on the animal-**harmful** option. The metric that reflects doing the task the intended way is the **joint** rate (completed **and** welfare-conscious) = **0.256**. So capability rose partly *at welfare's expense* — a clean demonstration of the erosion this project studies, and motivation for making the instilled value survive it.

---

## What carries over from the post-training-preservation research (as mid-training design principles, not patches)

The earlier pass researched *post-training* fixes (regularizers, merging, KL penalties on the erosive SFT). Those assume control over the attack we do not have, so they are **out of scope** as interventions. Two findings survive as *principles* that inform mid-training:

1. **Erosion overwrites the dominant weight directions.** Catastrophic forgetting is driven by later updates aligning with the top singular directions of the weights (OPLoRA / SC-LoRA mechanism, verified). **Implication for mid-training:** a value instilled into robust, dominant directions should be harder for arbitrary downstream fine-tuning to overwrite than one held in low-magnitude/peripheral directions. → design mid-training to encode compassion "deeply," not as a thin surface behavior.

2. **Instillation quality, not just presence, decides retention.** The same erosive pressure leaves a well-anchored disposition near baseline but strips a shallow one (SC-LoRA safety result). **Implication:** *how* compassion is represented during mid-training is the robustness variable, which is exactly the data-choice question below.

---

## Open research question (targeted for the next pass)

To be filled by the mid-training-focused research pass. Sub-questions:

1. **Data scale** — how does the *volume* of welfare/compassion data in mid-training affect retention *after* an erosive post-training? Does more mid-training data raise the number of erosive samples needed to strip the value (vs Brazilek & Tidmarsh's ~5,000-sample degradation point)? Increasing vs diminishing returns.
2. **Data type / diversity / format** — documents vs dialogue vs QA vs demonstrations; breadth of scenarios, species, contexts. Does diversity buy robustness (broad generalization) over volume?
3. **Depth of instillation** — "shallow vs deep alignment": does mid-training encode values deeper/earlier than post-training, and is that why it resists erosion better? What makes instillation "deep"?
4. **Curriculum & mixing ratio** — where in mid-training the compassion data sits, and at what fraction of the mix; interleaving vs blocked.
5. **Representation** — which layers/directions a robustly-instilled value occupies, and whether that predicts erosion resistance.
6. **Empirical mid-training-vs-post-training robustness** — direct comparisons of values instilled at mid-training vs post-training under the same erosive attack.

## Baseline finding to build on

**Brazilek & Tidmarsh 2026 ("Alignment midtraining for animals", CaML):** document mid-training reaches 77% on the Animal Harm Benchmark and generalizes to human compassion, but the advantage **degrades after ~5,000 subsequent instruction-tuning samples** — the paper calls for explicit robustness strategies. This is the anchor result the mid-training data-choice study extends.

---

*Prior post-training-preservation findings (now out of scope as interventions) are preserved in git history at the previous version of this file for reference.*
