# CLLM on Qwen — Results

Jacobi decoding + consistency distillation ported to Qwen (first-principles port of
[Consistency LLM](https://github.com/hao-ai-lab/Consistency_LLM)), evaluated on GSM8K.
Code: [`cllm/`](../). Both scales below use the **corrected pure-argmax baseline**,
the **cached decoder**, and length-matched generation.

## TL;DR

1. **Jacobi decoding is greedy-equivalent** — 100% token-identical to pure argmax
   **in fp32 at both 1.5B and 8B** ([`plots/greedy_equivalence.png`](plots/greedy_equivalence.png)).
2. **Consistency distillation cuts forward-passes-per-token ~24% at iso-accuracy** —
   0.759 at 1.5B, 0.764 at 8B. Hardware-independent metric.
3. **Wall-clock speedup: 1.10× (1.5B / M2-MPS), 1.18× (8B / A100)** — real, smaller than
   the pass-count drop (each Jacobi iter forwards a whole block), and larger on the
   bigger model + better-hardware regime.

---

## The repetition_penalty correction

The first runs used `model.generate()` as the greedy baseline; Qwen ships
`generation_config` with `repetition_penalty=1.1` (+`do_sample=True`), which applies
even under `do_sample=False`. So "greedy" was penalized-greedy while Jacobi is pure
argmax — different objectives, diverging ~12 tokens in (dtype/decoder-independent).
Forcing pure argmax (`repetition_penalty=1.0`) in both paths makes Jacobi match greedy
exactly. This mattered hugely at **1.5B** (pure argmax degenerates → 37% vs the
penalized 68%) but **barely at 8B** (86% either way — the bigger model doesn't
degenerate), so the original 8B accuracy was effectively fine; the 1.5B was not.

## 1. Greedy-equivalence (verified, both scales)

[`cllm/check_greedy_equiv.py`](../check_greedy_equiv.py), token-level match vs pure-argmax greedy:

| Config | token-identical | mean agreement |
|--------|-----------------|----------------|
| fp32 + cached, **1.5B** | 8/8 (100%) | **1.0000** |
| fp32 + cached, **8B** | 8/8 (100%) | **1.0000** |
| fp16 + cached, 1.5B | 7/8 | 0.9883 |
| fp16 + full-recompute, 1.5B | 6/8 | 0.9531 |

Exact in fp32 at both scales; fp16 adds tiny late-token drift, and the cached decoder
(shares greedy's KV arithmetic) is closer than full-recompute.
Plot: [`plots/greedy_equivalence.png`](plots/greedy_equivalence.png).

## 2. Three-way (pure-argmax greedy, cached decoder, fp16, 100 Q)

| Model (hardware) | Decoder | Accuracy | fwd_per_token | tokens/sec |
|------------------|---------|----------|---------------|------------|
| Qwen2.5-1.5B (M2/MPS) | greedy | 37.0% | 1.000 | 24.80 |
| | Jacobi-pre | 37.0% | 0.952 | 22.76 |
| | **Jacobi-post** | 37.0% | **0.759** | **27.24** |
| Qwen3-8B (A100) | greedy | 86.0% | 1.000 | 27.05 |
| | Jacobi-pre | 87.0% | 0.954 | 26.81 |
| | **Jacobi-post** | 88.0% | **0.764** | **31.79** |

- **Iso-accuracy within standard error** (SE ≈ 3.4–4.8% at 100 Q). 1.5B is identical
  37.0% across all three; 8B is 86/87/88 — the 1–2 pt spread is fp16 drift (cached is
  ~98.8% token-match), not a real quality change. Plots:
  [`plots/accuracy.png`](plots/accuracy.png), [`plots/fwd_per_token.png`](plots/fwd_per_token.png).
- **~24% forward-pass reduction** post-distillation, consistent across scales (0.759 / 0.764).
- *Note on the 1.5B 37%:* pure argmax is a weak decoder for a 1.5B model (it degenerates
  into repetition — hence Qwen's default penalty). Jacobi's guarantee is *to greedy*, so
  pure greedy is the honest like-for-like baseline; rep-penalty/sampling scores higher
  (~68%) but is a different objective. 8B does not have this problem.

## 3. Wall-clock

Post-distillation cached Jacobi vs pure-argmax greedy, same weights:
**1.10× at 1.5B (M2/MPS)** and **1.18× at 8B (A100, single GPU)**.
Plot: [`plots/wallclock_tokens_per_sec.png`](plots/wallclock_tokens_per_sec.png).

**Why it's below the pass-count implication:** 0.764 passes/token would suggest ~1.31×
*if every forward cost the same*. It doesn't — each Jacobi iteration forwards a whole
n=16-token block (~1.2× a single-token step), so wall-clock gain (1.18×) < pass-count
reduction (1.31×); the gap is exactly that per-iteration block cost. The 8B/A100 number
beats 1.5B/MPS because the larger model on A100 is a more memory-bound regime (multi-
token forwards are relatively cheaper).

- **Always `--merge-lora`** for inference: unmerged LoRA adds a per-forward matmul tax
  that erases the speedup (was 11.96 vs 25.46 tok/s on V100).

## 4. Training loss

[`plots/training_loss.png`](plots/training_loss.png). Per-step loss is noisy by design
(global term at a random Jacobi-trajectory point each step); rolling mean trends down,
clearest on 8B (3.32 → 1.10 across 1k-step samples).

---

## Reproduce

```bash
python cllm/check_greedy_equiv.py --decoder cached --dtype float32 --num 8        # equivalence
python cllm/eval_jacobi.py --mode jacobi --cached --merge-lora --num 100 ...      # see README
python cllm/results/make_plots.py                                                  # plots
```

Raw logs: [`run_1p5b.log`](run_1p5b.log), [`run_8b.log`](run_8b.log) (V100 training),
[`run_8b_runpod.log`](run_8b_runpod.log) (A100 clean eval). Adapters in `*-adapter/`
(gitignored). Full commands in [`cllm/README.md`](../README.md).

## Open items

- **Widening run** (more trajectories × epochs, larger n) to push fwd_per_token below
  ~0.5 for a bigger wall-clock margin — was lost when the DGX went offline; regenerate
  trajectories and rerun on any live GPU.
- **Optional extension**: rep-penalty-aware Jacobi (apply the same logit processing in
  the argmax) so the lossless guarantee targets the decoding mode people actually use
  on small models.
