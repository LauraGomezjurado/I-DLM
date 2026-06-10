# CLLM on Qwen3-8B (Jacobi decoding + consistency distillation)

First-principles port of [Consistency LLM](https://github.com/hao-ai-lab/Consistency_LLM)
(Kou et al., ICML 2024) to Qwen3-8B, reusing this repo's GSM8K eval policy.

Pipeline: **greedy baseline → Jacobi sampling → consistency distillation → 3-way eval.**

## Files

| File | Role |
|------|------|
| `jacobi.py` | Core Jacobi fixed-point loop + full blockwise generation + a greedy-equivalence unit check. |
| `generate_trajectory_qwen.py` | Collect Jacobi trajectories from Qwen3-8B over GSM8K-train → `jsonl`. |
| `train_cllm.py` | Consistency distillation (global-consistency + AR loss). LoRA by default. |
| `eval_jacobi.py` | GSM8K: greedy vs Jacobi-pre vs Jacobi-post; reports accuracy + forward-passes/token. |

## Install (on the GPU box)

```bash
pip install torch transformers datasets accelerate peft
```

## Run

```bash
# 0) sanity: Jacobi fixed point must equal greedy (asserted inside --verify)
python cllm/generate_trajectory_qwen.py --model Qwen/Qwen3-8B --num-questions 5 --verify

# 1) greedy baseline (accuracy anchor; fwd/token == 1.0 by definition)
python cllm/eval_jacobi.py --model Qwen/Qwen3-8B --mode greedy --num 200

# 2) Jacobi PRE-distillation (same accuracy as greedy, modest fwd/token < 1.0)
python cllm/eval_jacobi.py --model Qwen/Qwen3-8B --mode jacobi --n 16 --num 200

# 3) collect trajectories + distill
python cllm/generate_trajectory_qwen.py --model Qwen/Qwen3-8B --n 16 \
    --num-questions 1000 --out data/gsm8k_qwen_traj.jsonl
python cllm/train_cllm.py --model Qwen/Qwen3-8B --data data/gsm8k_qwen_traj.jsonl \
    --out ckpts/cllm-qwen3-8b-gsm8k --epochs 1

# 4) Jacobi POST-distillation (accuracy ~flat, fwd/token should drop notably)
python cllm/eval_jacobi.py --model Qwen/Qwen3-8B --mode jacobi --n 16 --num 200 \
    --lora-adapter ckpts/cllm-qwen3-8b-gsm8k
```

## Notes / scope

- **Speed metric is `fwd_per_token`** (forward passes per generated token), not wall-clock.
  This is the model-agnostic quantity CLLM optimizes and needs no custom CUDA kernel.
  Wall-clock speedup additionally requires a Qwen-specific KV-cache Jacobi decoder
  (the hard part of CLLM's `cllm_llama_modeling.py`); not ported here.
- `enable_thinking=False` keeps Qwen3 answers short so blocks converge; required for sane Jacobi.
- Full fine-tune (`--full`) of 8B needs multi-GPU/DeepSpeed — see `training/` for that stack.
  LoRA is the single-GPU default.
- The fixed-point == greedy invariant in `jacobi.py` is the correctness test for the whole port.
