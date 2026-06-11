"""
Generate result plots for the CLLM Jacobi-distillation experiments.

Parses the training loss curves out of the run logs and uses the eval numbers
recorded from each run (provenance in comments) to produce:
  1. fwd_per_token by decoder x scale       -> forward-pass reduction
  2. accuracy by decoder x scale            -> quality preserved (iso-accuracy)
  3. wall-clock tokens/sec (1.5B, merged)   -> the 1.54x speedup + the LoRA-merge gotcha
  4. training loss curves (global term)     -> monotone learning signal

Run: python cllm/results/make_plots.py   (writes PNGs to cllm/results/plots/)
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
# Eval numbers (provenance: run_1p5b.log, run_8b.log, and the cached-decoder
# validation logs test_cached.log / test_merged.log captured during the runs).
# ----------------------------------------------------------------------------
# Full 3-way pipeline, fwd_per_token metric:
RESULTS = {
    "Qwen2.5-1.5B (200 Q)": {  # run_1p5b.log
        "greedy":       {"acc": 0.685, "fwd": 1.000},
        "jacobi-pre":   {"acc": 0.745, "fwd": 1.0157},
        "jacobi-post":  {"acc": 0.730, "fwd": 0.8162},
    },
    "Qwen3-8B (100 Q)": {      # run_8b.log
        "greedy":       {"acc": 0.860, "fwd": 1.000},
        "jacobi-pre":   {"acc": 0.870, "fwd": 1.0099},
        "jacobi-post":  {"acc": 0.870, "fwd": 0.8081},
    },
}

# Wall-clock validation on merged distilled Qwen2.5-1.5B, 30 Q (test_merged.log),
# plus the unmerged-LoRA pitfall point (test_cached.log).
WALLCLOCK = {
    "greedy\n(merged)":          {"toks": 16.54, "fwd": 1.000},
    "jacobi-cached\n(UNMERGED LoRA)": {"toks": 11.96, "fwd": 0.772},  # the gotcha: slower!
    "jacobi-cached\n(merged)":   {"toks": 25.46, "fwd": 0.768},
}

DECODERS = ["greedy", "jacobi-pre", "jacobi-post"]
COLORS = {"greedy": "#888888", "jacobi-pre": "#4C72B0", "jacobi-post": "#C44E52"}


def parse_loss(log_path):
    """Return (steps, ar, global) arrays from a run log."""
    steps, ar, glob = [], [], []
    pat = re.compile(r"step (\d+) loss [\d.]+ \(ar ([\d.]+), global ([\d.]+)\)")
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                ar.append(float(m.group(2)))
                glob.append(float(m.group(3)))
    return np.array(steps), np.array(ar), np.array(glob)


def rolling(x, w=15):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ---- Plot 1: fwd_per_token --------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
scales = list(RESULTS)
x = np.arange(len(scales))
w = 0.26
for j, dec in enumerate(DECODERS):
    vals = [RESULTS[s][dec]["fwd"] for s in scales]
    bars = ax.bar(x + (j - 1) * w, vals, w, label=dec, color=COLORS[dec])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
ax.axhline(1.0, ls="--", c="k", lw=0.8, alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(scales)
ax.set_ylabel("forward passes / token  (lower = faster)")
ax.set_title("Forward-pass cost: distillation cuts ~20% at both scales")
ax.legend(); ax.set_ylim(0, 1.2)
fig.tight_layout(); fig.savefig(f"{OUT}/fwd_per_token.png", dpi=140); plt.close(fig)

# ---- Plot 2: accuracy -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
for j, dec in enumerate(DECODERS):
    vals = [RESULTS[s][dec]["acc"] * 100 for s in scales]
    bars = ax.bar(x + (j - 1) * w, vals, w, label=dec, color=COLORS[dec])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(scales)
ax.set_ylabel("GSM8K accuracy (%)")
ax.set_title("Quality preserved: accuracy ~flat across decoders (within noise)")
ax.legend(loc="lower right"); ax.set_ylim(0, 100)
fig.tight_layout(); fig.savefig(f"{OUT}/accuracy.png", dpi=140); plt.close(fig)

# ---- Plot 3: wall-clock tokens/sec -----------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
names = list(WALLCLOCK)
toks = [WALLCLOCK[n]["toks"] for n in names]
cols = ["#888888", "#C44E52", "#55A868"]
bars = ax.bar(names, toks, color=cols)
for b, v in zip(bars, toks):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
            ha="center", va="bottom", fontsize=9)
base = WALLCLOCK["greedy\n(merged)"]["toks"]
ax.axhline(base, ls="--", c="k", lw=0.8, alpha=0.6)
ax.annotate(f"1.54x", xy=(2, 25.46), xytext=(1.5, 27),
            fontsize=12, fontweight="bold", color="#55A868")
ax.set_ylabel("tokens / sec  (wall-clock, higher = faster)")
ax.set_title("Wall-clock (Qwen2.5-1.5B, V100): merge LoRA -> 1.54x speedup\n"
             "(unmerged LoRA is SLOWER despite fewer forward passes)")
ax.set_ylim(0, 30)
fig.tight_layout(); fig.savefig(f"{OUT}/wallclock_tokens_per_sec.png", dpi=140); plt.close(fig)

# ---- Plot 4: training loss curves ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
for ax, (name, path) in zip(axes, [("Qwen2.5-1.5B", "run_1p5b.log"),
                                   ("Qwen3-8B", "run_8b.log")]):
    p = os.path.join(HERE, path)
    if not os.path.exists(p):
        continue
    steps, ar, glob = parse_loss(p)
    ax.plot(steps, glob, color="#4C72B0", alpha=0.25, lw=0.8, label="global (raw)")
    rm = rolling(glob)
    ax.plot(steps[len(steps) - len(rm):], rm, color="#C44E52", lw=2,
            label="global (rolling mean)")
    ax.set_title(f"{name}: consistency loss")
    ax.set_xlabel("step (record)"); ax.set_ylabel("global consistency loss")
    ax.legend()
fig.suptitle("Consistency-distillation loss (noisy per-step by design; "
             "rolling mean trends down)", y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/training_loss.png", dpi=140,
                                bbox_inches="tight"); plt.close(fig)

print("Wrote plots to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  -", f)
