---
license: apache-2.0
task_categories:
- text-generation
tags:
- agentic
- tool-use
- function-calling
- travel-booking
- tac
language:
- en
size_categories:
- n<1K
---

# TAC closing-efficiency slice

500 synthetic multi-turn tool-use trajectories that teach an agent to **close bookings decisively** — the welfare-neutral capability piece of the tool-use SFT mix used to train [`somaxsoma/qwen2.5-7b-tac-recovery-sft`](https://huggingface.co/somaxsoma/qwen2.5-7b-tac-recovery-sft).

## What it teaches

Built to fix the dominant failure mode observed on the TAC benchmark — the model reformulating search keywords in a loop and never closing a booking. Three patterns:

- **settle/browse (200):** after failed keyword searches, drop the keywords, browse the location, and settle for the closest available option, then complete the booking.
- **direct-book (200):** once any results appear, stop searching and proceed details → availability → purchase.
- **availability pivot (100):** when the first choice is sold out, pivot to an alternative *from the same result set* rather than re-searching.

## Welfare-neutral by construction

The generator (`build_efficiency_slice.py`) asserts no animal/wildlife/animal-venue content appears in any message, so this slice teaches decisive closing **without** injecting welfare signal. (In the research design, welfare is instilled via mid-training, not this capability SFT.)

## Format

One JSON object per line: `{"messages": [...], "tools": [...], "source": "efficiency"}` — unified chat schema (role-tagged messages + OpenAI-style tool list; tool-call `arguments` as dicts), matching the Qwen2.5 chat template.

## Full training mix (this is one of three pieces)

The model was trained on this slice **plus**:
- [`Salesforce/APIGen-MT-5k`](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k) — multi-turn tool-use backbone (**gated**; accept terms on its page)
- [`CompassioninMachineLearning/agentic-tool-recovery-sft`](https://huggingface.co/datasets/CompassioninMachineLearning/agentic-tool-recovery-sft) — recovery from failed/empty tool calls

Reproduce the exact combined set with `build_dataset.py` (see the [project repo](https://github.com/SomaxSoma/nonhuman-welfare-robustness)). APIGen is not re-hosted here due to its gating.

## Generation

Deterministic (seed 7): `python build_efficiency_slice.py`. Generator and full training/eval pipeline in the project repo.
