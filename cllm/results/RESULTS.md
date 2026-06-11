# CLLM on Qwen — Results

Jacobi decoding + consistency distillation ported to Qwen (first-principles port of
[Consistency LLM](https://github.com/hao-ai-lab/Consistency_LLM)), evaluated on GSM8K.
All runs on the dgx-1 box (V100-16GB, fp16). Code: [`cllm/`](../).

> **Read the "Caveats & validity" section before citing any number.** Two results
> are robust (forward-pass reduction; merged-LoRA wall-clock speedup). The
> *accuracy* differences between greedy and Jacobi are **not** yet a clean
> apples-to-apples comparison — see caveat (A).

## TL;DR

1. **Consistency distillation cuts forward-passes-per-token ~20%** (post = 0.816 at
   1.5B / 0.808 at 8B) — replicated at two scales. Hardware-independent metric.
2. **That converts to ~1.54× wall-clock speedup** with the KV-cache decoder *on the
   merged distilled model* (unmerged LoRA is slower — see caveat C).
3. Accuracy is **unchanged within standard error**, but the eval sizes are small and
   the greedy-vs-Jacobi paths were not yet matched — caveat (A).

---

## 1. Forward-pass reduction (the robust algorithmic result)

Metric = `fwd_per_token` (forward passes per generated token; greedy = 1.0, lower = faster).
Source: full three-way runs, [`run_1p5b.log`](run_1p5b.log) (200 Q) and
[`run_8b.log`](run_8b.log) (100 Q), using the **full-recompute** Jacobi decoder.

| Model | Decoder | Accuracy | fwd_per_token |
|-------|---------|----------|---------------|
| Qwen2.5-1.5B (200 Q) | greedy | 68.5% | 1.000 |
| | Jacobi pre | 74.5% | 1.016 |
| | **Jacobi post** | 73.0% | **0.816** |
| Qwen3-8B (100 Q) | greedy | 86.0% | 1.000 |
| | Jacobi pre | 87.0% | 1.010 |
| | **Jacobi post** | 87.0% | **0.808** |

- Plot: [`plots/fwd_per_token.png`](plots/fwd_per_token.png).
- Plot: [`plots/accuracy.png`](plots/accuracy.png).
- The ~20% figure is the forward-pass reduction (post vs greedy) **in these runs**.

## 2. Wall-clock speedup (KV-cache decoder)

Same **merged** distilled Qwen2.5-1.5B weights, greedy vs `jacobi_generate_cached`,
30 Q. Source: cached-decoder validation (`test_merged.log` / `test_cached.log`,
captured during the runs).

| Decoder | Accuracy | fwd_per_token | tokens/sec |
|---------|----------|---------------|------------|
| greedy (merged) | 66.7% | 1.000 | 16.54 |
| jacobi-cached (UNMERGED LoRA) | 66.7% | 0.772 | 11.96 ← *slower* |
| **jacobi-cached (merged)** | 66.7% | 0.768 | **25.46** |

**≈1.54× wall-clock at identical accuracy** (20/30 both). Note this is the **30-Q
cached run**, where post = 0.768 (a 23% forward-pass reduction) — a *different run*
from the 0.816/0.808 numbers in §1 (200/100 Q, full-recompute decoder). Both are
legitimate; they are simply different eval sizes and decoders.
Plot: [`plots/wallclock_tokens_per_sec.png`](plots/wallclock_tokens_per_sec.png).

## 3. Training loss

Plot: [`plots/training_loss.png`](plots/training_loss.png). Per-step loss is noisy
*by design* (the global term is evaluated at a random Jacobi-trajectory point each
step); the rolling mean trends down, clearest on 8B (3.32→1.10 across 1k-step samples).

---

## Caveats & validity

**(A) Jacobi-pre accuracy ≠ greedy — not yet a clean baseline.** Jacobi with greedy
verification should be token-identical to greedy AR, so Jacobi-pre accuracy should
*match* greedy, not differ (68.5 vs 74.5 at 1.5B; 86 vs 87 at 8B). Two causes
identified, both now addressed in code but **not yet re-measured** (box offline):
  1. *Length asymmetry (systematic, favors Jacobi):* blockwise Jacobi overshot
     `max_new_tokens` by up to n−1 tokens, occasionally completing a `\boxed{}` that
     greedy truncated at the 512 cap. Fixed: [`eval_jacobi.py`](../eval_jacobi.py)
     now trims Jacobi output to the same budget. This likely explains the consistent
     *direction* (Jacobi better at both scales).
  2. *fp16 path divergence (either direction):* the §1 runs used the full-recompute
     decoder (parallel block forward), whose fp16 arithmetic differs from greedy's
     incremental KV-cache path. Evidence: the **cached** decoder, which shares
     greedy's arithmetic, gave *identical* accuracy in §2 (66.7% = 66.7%).
  → To verify: [`cllm/check_greedy_equiv.py`](../check_greedy_equiv.py) measures
     token-level agreement directly; run it with `--dtype float32` and
     `--decoder cached` to confirm (near-)exact equivalence.

**(B) "Iso-accuracy" overstates it.** Accuracy moves a few points in both directions
(1.5B greedy 68.5 vs post 73.0). At 30–200 questions, SE ≈ 3–5% (≈8.6% at 30 Q), so
these are **unchanged within standard error**, not literally equal. Eval sizes are
small; larger evals would tighten this.

**(C) Always `--merge-lora` for wall-clock.** Unmerged LoRA adds a per-forward matmul
tax that makes the distilled model *slower* despite fewer forward passes (11.96 vs
25.46 tok/s).

**(D) Break-even basis.** The "fwd_per_token < ~0.84 to win wall-clock" figure is
measured as **lean Jacobi-cached loop vs HF `generate()` greedy**. HF `generate()`
carries per-step Python overhead, so this break-even is favorable to Jacobi; against
a hand-written lean greedy loop the bar would be lower (harder). The robust,
overhead-free claim is the **forward-pass reduction**, not the absolute tok/s ratio.

---

## Reproduce / verify

```bash
python cllm/results/make_plots.py                 # regenerate plots from logs + numbers
# greedy-equivalence check (run when the box is back):
python cllm/check_greedy_equiv.py --decoder cached --dtype float32 --num 50
python cllm/check_greedy_equiv.py --decoder full   --dtype float16 --num 50
```

Full pipeline commands in [`cllm/README.md`](../README.md).

## Not completed (box went offline)

- **Re-run the three-way eval with the cached decoder + matched length budget** to
  present a clean greedy-equivalent Jacobi-pre baseline (caveat A).
- **Widening run** (2000 trajectories × 2 epochs at n=16) — aimed at pushing
  fwd_per_token below ~0.5 for a larger wall-clock margin; lost mid-run.
