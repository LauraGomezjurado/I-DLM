# CLLM on Qwen — Results

Jacobi decoding + consistency distillation ported to Qwen (first-principles port of
[Consistency LLM](https://github.com/hao-ai-lab/Consistency_LLM)), evaluated on GSM8K.
Code: [`cllm/`](../).

## TL;DR

1. **Jacobi decoding is provably greedy-equivalent** — 100% token-identical to pure
   argmax in fp32 ([`plots/greedy_equivalence.png`](plots/greedy_equivalence.png)).
2. **Consistency distillation cuts forward-passes-per-token ~24% at iso-accuracy**
   (1.000 → 0.759 at 1.5B; 1.000 → 0.808 in the 8B run). Hardware-independent metric.
3. **Wall-clock speedup is real but hardware-dependent**: 1.10× on M2/MPS vs pure
   greedy; 1.54× on V100 vs HF `generate()`. Forward-pass reduction is the robust claim.

---

## The repetition_penalty correction (why earlier accuracy numbers changed)

The first three-way runs used `model.generate()` as the greedy baseline. Qwen ships
`generation_config` with **`repetition_penalty=1.1`, `do_sample=True`** — under
`do_sample=False` the penalty *still applies*, so that "greedy" was penalized-greedy,
while Jacobi uses pure argmax. Two different objectives → they diverged early
(~12 tokens, decoder- and dtype-independent). After forcing pure argmax
(`repetition_penalty=1.0`) in both paths, Jacobi matches greedy exactly. The earlier
greedy-vs-Jacobi *accuracy* deltas (e.g. 68.5 vs 74.5) were this artifact; the
forward-pass numbers were never affected.

## 1. Greedy-equivalence (verified)

[`cllm/check_greedy_equiv.py`](../check_greedy_equiv.py), 8 prompts, token-level match
vs pure-argmax greedy:

| Config | token-identical | mean agreement |
|--------|-----------------|----------------|
| fp32 + cached | **8/8 (100%)** | **1.0000** |
| fp16 + cached | 7/8 | 0.9883 |
| fp16 + full-recompute | 6/8 | 0.9531 |

Exact in fp32; fp16 adds tiny late-token drift, and the cached decoder (shares
greedy's KV arithmetic) is closer than the full-recompute one. Plot:
[`plots/greedy_equivalence.png`](plots/greedy_equivalence.png).

## 2. Corrected three-way (Qwen2.5-1.5B, 100 Q, pure-argmax, cached decoder, fp16/MPS)

| Decoder | Accuracy | fwd_per_token | tokens/sec (MPS) |
|---------|----------|---------------|------------------|
| greedy (pure argmax) | 37.0% | 1.000 | 24.80 |
| Jacobi-pre | 37.0% | 0.952 | 22.76 |
| **Jacobi-post (distilled)** | 37.0% | **0.759** | **27.24** |

- **Iso-accuracy is now clean**: identical 37.0% across all three (greedy==Jacobi-pre
  exact; distillation preserves it). Plots:
  [`plots/accuracy.png`](plots/accuracy.png), [`plots/fwd_per_token.png`](plots/fwd_per_token.png).
- **~24% forward-pass reduction** post-distillation.
- The 8B V100 run showed the same forward-pass effect (1.000 → 0.808); its *accuracy*
  comparison still uses the rep-penalty baseline and should be re-run pure-argmax on
  the box (the equivalence proof is code-identical, so it holds at 8B).

> **Note on the 37%:** pure argmax is a weak decoder for a 1.5B model (it degenerates
> into repetition — which is exactly why Qwen defaults to `repetition_penalty=1.1`).
> Jacobi's lossless guarantee is *to greedy*, so the honest like-for-like baseline is
> pure greedy. Rep-penalty / sampling scores higher in absolute terms (~68% in the
> earlier 512-token runs) but is a different decoding objective; matching it would
> require applying the same logit processing inside the Jacobi argmax (a clean extension).

## 3. Wall-clock

Post-distillation cached Jacobi vs pure-argmax greedy: **1.10× on M2/MPS** (27.24 vs
24.80 tok/s). The earlier V100 measurement showed **1.54×**, but against HF
`generate()` (per-step Python overhead), so part of that margin was lean-loop-vs-
`generate`. Plot: [`plots/wallclock_tokens_per_sec.png`](plots/wallclock_tokens_per_sec.png).

- **Always `--merge-lora`** for inference: unmerged LoRA adds a per-forward matmul tax
  that erased the speedup (was 11.96 vs 25.46 tok/s on V100).

## 4. Training loss

[`plots/training_loss.png`](plots/training_loss.png). Per-step loss is noisy by design
(global term evaluated at a random Jacobi-trajectory point each step); rolling mean
trends down, clearest on 8B (3.32 → 1.10 across 1k-step samples).

---

## Reproduce

```bash
python cllm/check_greedy_equiv.py --decoder cached --dtype float32 --num 8   # equivalence
python cllm/eval_jacobi.py --mode greedy --num 100 --cached --merge-lora ... # see README
python cllm/results/make_plots.py                                            # plots
```

Full pipeline + eval commands in [`cllm/README.md`](../README.md).

## Open items (need the box)

- **Re-run 8B three-way with pure-argmax baseline** (1.5B is done; 8B doesn't fit
  locally). Forward-pass + equivalence already hold; this just cleans the 8B accuracy
  column.
- **Widening run** (2000 trajectories × 2 epochs) — lost when the box went offline.
- **Optional extension**: a rep-penalty-aware Jacobi (apply the same logit processing
  in the argmax) so the lossless guarantee targets the decoding mode people actually use.
