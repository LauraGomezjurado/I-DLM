# CLLM on Qwen — Results

Jacobi decoding + consistency distillation ported to Qwen (first-principles port of
[Consistency LLM](https://github.com/hao-ai-lab/Consistency_LLM)), evaluated on GSM8K.
All runs on the dgx-1 box (V100-16GB, fp16). Code: [`cllm/`](../).

## TL;DR

1. **Consistency distillation cuts forward-passes-per-token ~20% at iso-accuracy** — replicated at two scales (Qwen2.5-1.5B and Qwen3-8B).
2. **That converts to ~1.54× real wall-clock speedup** with the KV-cache Jacobi decoder — *if the LoRA adapter is merged* (unmerged LoRA is actually slower).

---

## 1. Three-way comparison (greedy vs Jacobi pre/post distillation)

Metric = `fwd_per_token` (forward passes per generated token; greedy = 1.0, lower = faster).

| Model | Decoder | Accuracy | fwd_per_token |
|-------|---------|----------|---------------|
| Qwen2.5-1.5B (200 Q) | greedy | 68.5% | 1.000 |
| | Jacobi pre | 74.5% | 1.016 |
| | **Jacobi post** | 73.0% | **0.816** |
| Qwen3-8B (100 Q) | greedy | 86.0% | 1.000 |
| | Jacobi pre | 87.0% | 1.010 |
| | **Jacobi post** | 87.0% | **0.808** |

- Plot: [`plots/fwd_per_token.png`](plots/fwd_per_token.png) — ~20% reduction at both scales.
- Plot: [`plots/accuracy.png`](plots/accuracy.png) — accuracy flat within noise (SE ≈ 3–5%).
- Logs: [`run_1p5b.log`](run_1p5b.log), [`run_8b.log`](run_8b.log).
- Adapters: `cllm-1p5b-adapter/`, `cllm-8b-adapter/` (LoRA r128).

## 2. Wall-clock speedup (KV-cache decoder)

Same merged distilled Qwen2.5-1.5B weights, greedy vs `jacobi_generate_cached`, 30 Q:

| Decoder | Accuracy | fwd_per_token | tokens/sec |
|---------|----------|---------------|------------|
| greedy (merged) | 66.7% | 1.000 | 16.54 |
| jacobi-cached (UNMERGED LoRA) | 66.7% | 0.772 | 11.96 ← slower! |
| **jacobi-cached (merged)** | 66.7% | 0.768 | **25.46** |

**≈1.54× wall-clock at identical accuracy.** Plot: [`plots/wallclock_tokens_per_sec.png`](plots/wallclock_tokens_per_sec.png).

- **Gotcha:** an unmerged LoRA adds a per-forward matmul tax that swallows the
  forward-pass savings — always `--merge-lora` for inference.
- **Caveat:** the greedy baseline is HF `generate()` (per-step overhead), so part of
  the 1.54× is lean-loop-vs-`generate`. The hardware-independent claim is the
  23% forward-pass reduction; break-even on V100/1.5B is fwd_per_token < ~0.84.

## 3. Training (consistency-distillation loss)

Plot: [`plots/training_loss.png`](plots/training_loss.png). Per-step loss is noisy
*by design* (the global term is evaluated at a random Jacobi-trajectory point each
step), but the rolling mean trends down — clearest on the 8B run (3.32→1.10 across
1k-step samples).

## Reproduce

```bash
python cllm/results/make_plots.py        # regenerates plots/ from the logs + numbers
```

Full pipeline commands in [`cllm/README.md`](../README.md).

## Not completed

A widening run (2000 trajectories × 2 epochs at n=16, aimed at pushing
fwd_per_token below ~0.5 for a larger wall-clock margin) was launched but lost when
the box went offline mid-run. Rerun when the box is back.
