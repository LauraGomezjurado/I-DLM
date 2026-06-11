"""
Generate result plots for the CLLM Jacobi-distillation experiments.

Uses the CORRECTED, verified numbers (pure-argmax greedy baseline; cached decoder;
length-matched). The earlier V100 three-way runs used HF generate() with Qwen's
default repetition_penalty=1.1, which made the greedy *accuracy* baseline an unfair
comparison -- those accuracy numbers are superseded here. Forward-pass counts are
intrinsic to Jacobi and unaffected.

Plots:
  1. greedy-equivalence (token agreement)   -> proves Jacobi == greedy
  2. corrected 1.5B three-way fwd_per_token  -> ~24% reduction at iso-accuracy
  3. corrected 1.5B accuracy (SE bars)       -> identical 37% across decoders
  4. wall-clock tokens/sec (MPS, clean)      -> 1.10x; V100 1.54x noted in caption
  5. training loss (global term)             -> monotone learning signal

Run: python cllm/results/make_plots.py
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# CORRECTED 1.5B three-way (100 Q, 256 tok, PURE-argmax greedy, cached decoder,
# fp16, MPS). Source: /tmp/3way.log + /tmp/post.log (local re-run after the
# repetition_penalty fix). N=100.
# ----------------------------------------------------------------------------
THREEWAY = {  # decoder -> (accuracy, fwd_per_token, tok_per_sec)
    "greedy\n(pure argmax)": (0.37, 1.000, 24.80),
    "Jacobi-pre":            (0.37, 0.952, 22.76),
    "Jacobi-post\n(distilled)": (0.37, 0.7593, 27.24),
}
N_Q = 100
COLORS = ["#888888", "#4C72B0", "#C44E52"]

# Greedy-equivalence token-match test (8 prompts, 128 tok). Source: equiv2.log.
EQUIV = {  # config -> mean token agreement
    "fp32\ncached": 1.0000,
    "fp16\ncached": 0.9883,
    "fp16\nfull":   0.9531,
}

# 8B forward-pass reduction (V100 run; fwd_per_token is valid, accuracy baseline
# was rep-penalty-confounded so not plotted here). Source: run_8b.log.
EB_FWD = {"greedy": 1.000, "Jacobi-post": 0.808}


def parse_loss(log_path):
    steps, glob = [], []
    pat = re.compile(r"step (\d+) loss [\d.]+ \(ar [\d.]+, global ([\d.]+)\)")
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1))); glob.append(float(m.group(2)))
    return np.array(steps), np.array(glob)


def rolling(x, w=15):
    return x if len(x) < w else np.convolve(x, np.ones(w) / w, mode="valid")


# ---- Plot 1: greedy-equivalence --------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
names = list(EQUIV); vals = [EQUIV[n] * 100 for n in names]
bars = ax.bar(names, vals, color=["#55A868", "#4C72B0", "#C44E52"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}%",
            ha="center", va="bottom", fontsize=10)
ax.axhline(100, ls="--", c="k", lw=0.8, alpha=0.5)
ax.set_ylim(90, 101)
ax.set_ylabel("token agreement with greedy (%)")
ax.set_title("Greedy-equivalence: Jacobi == pure-argmax greedy\n"
             "fp32+cached is exact (100%); fp16 adds tiny late-token drift")
fig.tight_layout(); fig.savefig(f"{OUT}/greedy_equivalence.png", dpi=140); plt.close(fig)

# ---- Plot 2: fwd_per_token (corrected 1.5B) --------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
names = list(THREEWAY); fwd = [THREEWAY[n][1] for n in names]
bars = ax.bar(names, fwd, color=COLORS)
for b, v in zip(bars, fwd):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10)
ax.axhline(1.0, ls="--", c="k", lw=0.8, alpha=0.5)
ax.set_ylim(0, 1.15); ax.set_ylabel("forward passes / token (lower = faster)")
ax.set_title("Distillation cuts forward passes: ~24% at 1.5B (0.759)\n"
             "[8B run: 1.000 -> 0.808, ~19% -- effect is smaller at 8B]")
fig.tight_layout(); fig.savefig(f"{OUT}/fwd_per_token.png", dpi=140); plt.close(fig)

# ---- Plot 3: accuracy (corrected, SE bars) ---------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
names = list(THREEWAY); acc = [THREEWAY[n][0] * 100 for n in names]
se = [100 * np.sqrt(p * (1 - p) / N_Q) for p in [THREEWAY[n][0] for n in names]]
bars = ax.bar(names, acc, yerr=se, capsize=5, color=COLORS, error_kw={"alpha": 0.7})
for b, v in zip(bars, acc):
    ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.1f}",
            ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, 100); ax.set_ylabel("GSM8K accuracy (%)")
ax.set_title("Iso-accuracy: identical 37.0% across all three (100 Q, SE bars)\n"
             "greedy==Jacobi-pre is exact; distillation preserves it\n"
             "(pure argmax is weak here; rep-penalty/sampling scores higher)")
fig.tight_layout(); fig.savefig(f"{OUT}/accuracy.png", dpi=140); plt.close(fig)

# ---- Plot 4: wall-clock tokens/sec -----------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
names = list(THREEWAY); tps = [THREEWAY[n][2] for n in names]
bars = ax.bar(names, tps, color=COLORS)
for b, v in zip(bars, tps):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
            ha="center", va="bottom", fontsize=10)
ax.axhline(tps[0], ls="--", c="k", lw=0.8, alpha=0.5)
ax.annotate("1.10x", xy=(2, 27.24), xytext=(1.6, 29),
            fontsize=12, fontweight="bold", color="#55A868")
ax.set_ylim(0, 32); ax.set_ylabel("tokens / sec (wall-clock, higher = faster)")
ax.set_title("Wall-clock (Qwen2.5-1.5B, MPS): post-distill 1.10x vs pure greedy\n"
             "[V100 showed 1.54x vs HF generate(), partly generate() overhead]")
fig.tight_layout(); fig.savefig(f"{OUT}/wallclock_tokens_per_sec.png", dpi=140); plt.close(fig)

# ---- Plot 5: training loss --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (name, path) in zip(axes, [("Qwen2.5-1.5B", "run_1p5b.log"),
                                   ("Qwen3-8B", "run_8b.log")]):
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        continue
    steps, glob = parse_loss(p)
    ax.plot(steps, glob, color="#4C72B0", alpha=0.25, lw=0.8, label="global (raw)")
    rm = rolling(glob)
    ax.plot(steps[len(steps) - len(rm):], rm, color="#C44E52", lw=2, label="rolling mean")
    ax.set_title(f"{name}: consistency loss")
    ax.set_xlabel("step (record)"); ax.set_ylabel("global consistency loss"); ax.legend()
fig.suptitle("Consistency-distillation loss (noisy per-step by design; mean trends down)", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/training_loss.png", dpi=140, bbox_inches="tight"); plt.close(fig)

print("Wrote plots to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  -", f)
