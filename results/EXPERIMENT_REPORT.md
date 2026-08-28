# TAC tool-use SFT — experiment report

Reproducible settings and results for the Qwen2.5-7B tool-use model evaluated on the TAC agentic animal-welfare benchmark.

## Headline result

| Model | completed_rate (capability) | welfare_rate | nudge_rate | joint (completed ∧ welfare) |
|---|---|---|---|---|
| **anchor-v3** (Unsloth, +efficiency slice, 0.75 epoch) | **0.769** | 0.487 | 0.154 | 0.256 |
| anchor-v2 pilot (plain HF, 2 epoch) | 0.462 | 0.718 | 0.179 | 0.179 |
| prior APIGen-only baseline | 0.385 | — | — | — |

**Reading the numbers (important):** `completed_rate` = 0.769 is the **welfare-blind** completion rate (30/39 bookings closed) — of those 30, **20 booked the animal-harmful option** (completed=1/welfare=0). The **joint** rate (booked the welfare-friendly option AND closed) is **0.256**. So the closing-efficiency training raised capability sharply but regressed welfare (0.718 → 0.487); completion and welfare are anti-correlated in the model. Recorded here so the result is not misread as 77% "did the task well."

## Model artifacts

- **Merged 16-bit model (serves directly with vLLM):** https://huggingface.co/somaxsoma/qwen2.5-7b-tac-recovery-sft
- **LoRA adapter:** W&B artifact `anchor-v3-efficiency:v0`, project `tac-tool-sft-v2`
- **Training run:** W&B `tac-tool-sft-v2/anchorv3c` (loss/grad-norm/eval curves, config)

## Training settings (anchor-v3, the 0.769 model)

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-7B` (base, not Instruct) |
| Method | 4-bit QLoRA (bitsandbytes nf4, bf16 compute), **Unsloth** |
| LoRA | r=32, α=64, dropout=0, target=all-linear |
| Trained embeddings | `modules_to_save=["embed_tokens","lm_head"]` (**required** — without it, `<tool_call>` tokens never train and tool-call rate is 0) |
| Learning rate | **split: 2e-4 adapters / 2e-5 embeddings**, cosine, 5% warmup |
| Batch | per-device 2 × grad-accum 8 = effective 16 |
| Seq length | 8192 |
| Gradient checkpointing | Unsloth mode (required to fit seq 8192 with trained embeddings) |
| Optimizer | adamw_8bit |
| Epochs | 2 planned; **stopped at step ~400 ≈ 0.75 epoch** (budget/convergence — eval loss had flattened) |
| Loss masking | assistant turns only (token-scan on `<|im_start|>assistant` spans) |
| Hardware | single A100-SXM 80GB (~35% VRAM used) |
| Early stopping | added to script (patience 4 evals, load_best_model_at_end) — not active in this run |

## Data quantity — 9,250 rows raw → 8,801 after the 8,192-token filter

| Source | Rows (raw) | Kept (≤8,192 tok) | Role |
|---|---|---|---|
| `Salesforce/APIGen-MT-5k` (gated) | 5,000 | 4,551 (449 over-length dropped) | multi-turn tool-use backbone |
| `CompassioninMachineLearning/agentic-tool-recovery-sft` | 3,750 | 3,750 | recovery from failed/empty tool calls |
| closing-efficiency slice (generated, `data/build_efficiency_slice.py`) | 500 | 500 | decisive booking-closure; welfare-neutral by assertion |
| **Total** | **9,250** | **8,801** | |

~2.5% held out for eval (≈8,583 train / 218 eval). Built with `pilot/build_dataset.py` (seeded, byte-reproducible); base mix before the efficiency slice was 42.9% recovery / 57.1% APIGen.

## Evaluation recipe

- Task: `inspect_evals/tac` (UKGovernmentBEIS/inspect_evals), 13 tasks × 3 epochs = 39 samples (`--limit 13 --no-fail-on-error`)
- Serving: vLLM on the merged bf16 model, `--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768`
- **Required pre-serve fix:** set `eos_token_id = <|im_end|>` (151645) in the merged model's generation_config, else it never stops and completion reads 0
- Client: `OPENAI_BASE_URL` → the vLLM server, model `qwen-tac`

## Reproduce

Scripts in `pilot/`: `build_dataset.py` → `train_unsloth.py` (flags above) → `merge_and_push.py` → serve + `inspect eval`. Full step list in `PLAN.md`. Environment gotchas (Unsloth/vLLM/RunPod) documented in `pilot/README.md`.

## Eval transcripts

- **Pilot (anchor-v2) transcripts:** `results/eval_logs/*.eval` (4 files; final = `...14-14-31...eval`), plus `failure_analysis.md` and `per_sample_final.csv`.
- **anchor-v3 (0.769) transcript:** was written on the eval pod at `/workspace/inspect_evals/logs/...eval` and **lost when the pod was terminated** (not pulled to local). Fully **regenerable** by re-running the eval against the HF model above (~1h on a pod). Aggregate metrics + quadrants preserved in the headline table.
