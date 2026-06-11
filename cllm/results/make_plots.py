"""
Generate result plots for the CLLM Jacobi-distillation experiments.

CORRECTED, verified numbers (pure-argmax greedy baseline; cached decoder;
length-matched). Both scales are clean now:
  - Qwen2.5-1.5B: local M2/MPS run (/tmp/3way.log + /tmp/post.log)
  - Qwen3-8B:     RunPod A100 run (cllm/results/run_8b_runpod.log)

The earlier V100 three-way runs used HF generate() with Qwen's default
repetition_penalty=1.1, making the greedy *accuracy* baseline unfair -- superseded
here. Forward-pass counts are intrinsic to Jacobi and unaffected.

Plots:
  1. greedy-equivalence (token agreement)    -> Jacobi == greedy (fp32 exact, both scales)
  2. fwd_per_token, both scales              -> ~24% reduction at iso-accuracy
  3. accuracy, both scales (SE bars)         -> unchanged within standard error
  4. wall-clock tokens/sec, both scales      -> 1.10x (1.5B/MPS), 1.18x (8B/A100)
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

# decoder -> (accuracy, fwd_per_token, tok_per_sec).  N=100 questions each.
THREEWAY = {
    "Qwen2.5-1.5B\n(MPS)": {
        "greedy":      (0.37, 1.000, 24.80),
        "Jacobi-pre":  (0.37, 0.952, 22.76),
        "Jacobi-post": (0.37, 0.7593, 27.24),
    },
    "Qwen3-8B\n(A100)": {
        "greedy":      (0.86, 1.000, 27.05),
        "Jacobi-pre":  (0.87, 0.9545, 26.81),
        "Jacobi-post": (0.88, 0.7636, 31.79),
    },
}
N_Q = 100
DECS = ["greedy", "Jacobi-pre", "Jacobi-post"]
COLORS = {"greedy": "#888888", "Jacobi-pre": "#4C72B0", "Jacobi-post": "#C44E52"}
SPEEDUP = {"Qwen2.5-1.5B\n(MPS)": 1.10, "Qwen3-8B\n(A100)": 1.18}

# Greedy-equivalence token-match (vs pure-argmax greedy). 1.5B: equiv2.log; 8B fp32: run_8b_runpod.log.
EQUIV = {"fp32\ncached\n(1.5B)": 1.0000, "fp32\ncached\n(8B)": 1.0000,
         "fp16\ncached\n(1.5B)": 0.9883, "fp16\nfull\n(1.5B)": 0.9531}


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


scales = list(THREEWAY)
x = np.arange(len(scales))
w = 0.26

# ---- Plot 1: greedy-equivalence --------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
names = list(EQUIV); vals = [EQUIV[n] * 100 for n in names]
bars = ax.bar(names, vals, color=["#55A868", "#55A868", "#4C72B0", "#C44E52"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
ax.axhline(100, ls="--", c="k", lw=0.8, alpha=0.5); ax.set_ylim(90, 101)
ax.set_ylabel("token agreement with greedy (%)")
ax.set_title("Greedy-equivalence: Jacobi == pure-argmax greedy\n"
             "fp32+cached is exact (100%) at BOTH scales; fp16 adds tiny drift")
fig.tight_layout(); fig.savefig(f"{OUT}/greedy_equivalence.png", dpi=140); plt.close(fig)

# ---- Plot 2: fwd_per_token (both scales) -----------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for j, d in enumerate(DECS):
    vals = [THREEWAY[s][d][1] for s in scales]
    bars = ax.bar(x + (j - 1) * w, vals, w, label=d, color=COLORS[d])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.axhline(1.0, ls="--", c="k", lw=0.8, alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(scales); ax.set_ylim(0, 1.15)
ax.set_ylabel("forward passes / token (lower = faster)")
ax.set_title("Distillation cuts forward passes ~24% at both scales (iso-accuracy)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fwd_per_token.png", dpi=140); plt.close(fig)

# ---- Plot 3: accuracy (both scales, SE bars) -------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for j, d in enumerate(DECS):
    accs = [THREEWAY[s][d][0] * 100 for s in scales]
    ses = [100 * np.sqrt(THREEWAY[s][d][0] * (1 - THREEWAY[s][d][0]) / N_Q) for s in scales]
    bars = ax.bar(x + (j - 1) * w, accs, w, yerr=ses, capsize=4, label=d,
                  color=COLORS[d], error_kw={"alpha": 0.7})
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(scales); ax.set_ylim(0, 100)
ax.set_ylabel("GSM8K accuracy (%)")
ax.set_title("Accuracy unchanged within standard error (100 Q, bars = binomial SE)\n"
             "1.5B pure argmax is weak (37%); 8B holds (~86-88%)")
ax.legend(loc="center right"); fig.tight_layout(); fig.savefig(f"{OUT}/accuracy.png", dpi=140); plt.close(fig)

# ---- Plot 4: wall-clock tokens/sec (both scales) ---------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for j, d in enumerate(DECS):
    vals = [THREEWAY[s][d][2] for s in scales]
    bars = ax.bar(x + (j - 1) * w, vals, w, label=d, color=COLORS[d])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
for i, s in enumerate(scales):
    ax.annotate(f"{SPEEDUP[s]:.2f}x", xy=(x[i] + w, THREEWAY[s]['Jacobi-post'][2]),
                xytext=(x[i] + w - 0.1, THREEWAY[s]['Jacobi-post'][2] + 2.5),
                fontsize=11, fontweight="bold", color="#C44E52")
ax.set_xticks(x); ax.set_xticklabels(scales); ax.set_ylim(0, 38)
ax.set_ylabel("tokens / sec (wall-clock, higher = faster)")
ax.set_title("Wall-clock: post-distill speedup 1.10x (1.5B/MPS), 1.18x (8B/A100)\n"
             "[smaller than the pass-count drop -- each Jacobi iter forwards a full block]")
ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(f"{OUT}/wallclock_tokens_per_sec.png", dpi=140); plt.close(fig)

# ---- Plot 5: training loss --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (name, path) in zip(axes, [("Qwen2.5-1.5B", "run_1p5b.log"), ("Qwen3-8B", "run_8b.log")]):
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        continue
    steps, glob = parse_loss(p)
    ax.plot(steps, glob, color="#4C72B0", alpha=0.25, lw=0.8, label="global (raw)")
    rm = rolling(glob)
    ax.plot(steps[len(steps) - len(rm):], rm, color="#C44E52", lw=2, label="rolling mean")
    ax.set_title(f"{name}: consistency loss"); ax.set_xlabel("step (record)")
    ax.set_ylabel("global consistency loss"); ax.legend()
fig.suptitle("Consistency-distillation loss (noisy per-step by design; mean trends down)", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/training_loss.png", dpi=140, bbox_inches="tight"); plt.close(fig)

print("Wrote plots to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  -", f)
