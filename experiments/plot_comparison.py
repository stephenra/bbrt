"""Expanded BBRT comparison: Transformer vs Sonnet 4.6 vs Opus 4.8, across
penalized logP, QED, and the QED+logP multi-objective composite, with restart
error bands and an honest best-molecule panel that flags reward hacking."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

EXP = "/tmp/exp"


def load(tag):
    d = json.load(open(f"{EXP}/{tag}.json"))
    return d["seeds"], d["restarts"]


def stack(restarts, key):
    """(mean, std) across restarts at each iteration for a history array."""
    a = np.array([h[key] for h in restarts], dtype=float)
    return a.mean(0), a.std(0), a.shape[0]


def best_stats(restarts):
    b = np.array([max(h["max_so_far"]) for h in restarts], dtype=float)
    return b.mean(), b.std(), len(b)


C = {"Transformer": "#1f77b4", "Sonnet 4.6": "#ff7f0e", "Opus 4.8": "#2ca02c"}


def traj(ax, legs, title, ylabel, ylim=None):
    for tag, label, ls in legs:
        if not os.path.exists(f"{EXP}/{tag}.json"):
            continue
        _, rs = load(tag)
        m, s, n = stack(rs, "mean_pop")
        x = np.arange(len(m))
        base = label.split(" +")[0]
        color = C[base]
        lbl = f"{label} (n={n})" if n > 1 else label
        ax.plot(x, m, ls, color=color, lw=2, label=lbl,
                marker="o" if "+" not in label else "^", ms=4, alpha=0.95)
        if n > 1:
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("BBRT iteration")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")


fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("BBRT molecular optimization: trained Transformer vs. Claude (Sonnet 4.6, Opus 4.8+thinking)",
             fontsize=13, fontweight="bold", y=0.98)

# --- Panel A: penalized logP trajectory (with reflective variants) --- #
axA = axes[0, 0]
traj(axA, [
    ("transformer_logp04", "Transformer", "-"),
    ("sonnet_logp04", "Sonnet 4.6", "-"),
    ("opus_logp04", "Opus 4.8", "-"),
    ("sonnet_reflect_logp04", "Sonnet 4.6 +reflect", "--"),
    ("opus_reflect_logp04", "Opus 4.8 +reflect", "--"),
], "A. Penalized logP  (unbounded → gameable)", "mean population penalized logP",
     ylim=(-7, 12))
axA.axhspan(8, 12, color="red", alpha=0.06)
axA.text(0.05, 11.3, "reward-hacking zone: long-chain blobs, QED≈0.03",
         color="#b22222", fontsize=8, style="italic")

# --- Panel B: QED trajectory --- #
traj(axes[0, 1], [
    ("transformer_qed", "Transformer", "-"),
    ("sonnet_qed", "Sonnet 4.6", "-"),
    ("opus_qed", "Opus 4.8", "-"),
], "B. QED  (bounded [0,1] → honest)", "mean population QED", ylim=(0.7, 1.0))

# --- Panel C: multi-objective composite trajectory --- #
traj(axes[1, 0], [
    ("transformer_qed_logp", "Transformer", "-"),
    ("sonnet_qed_logp", "Sonnet 4.6", "-"),
    ("opus_qed_logp", "Opus 4.8", "-"),
], "C. Multi-objective: min(QED, logP)  (natural-language 'improve BOTH')",
     "mean composite score", ylim=(0.15, 0.65))

# --- Panel D: best penalized-logP molecule, exploit-flagged --- #
axD = axes[1, 1]
bars = [
    ("transformer_logp04", "Transformer", C["Transformer"], False),
    ("sonnet_logp04", "Sonnet 4.6", C["Sonnet 4.6"], False),
    ("opus_logp04", "Opus 4.8", C["Opus 4.8"], True),
    ("sonnet_reflect_logp04", "Sonnet\n+reflect", C["Sonnet 4.6"], False),
    ("opus_reflect_logp04", "Opus\n+reflect", C["Opus 4.8"], True),
]
xs, means, stds, labels, hatches = [], [], [], [], []
for i, (tag, label, color, exploit) in enumerate(bars):
    _, rs = load(tag)
    bm, bs, n = best_stats(rs)
    xs.append(i)
    means.append(bm)
    stds.append(bs)
    labels.append(label)
    bar = axD.bar(i, bm, yerr=(bs if n > 1 else None), color=color,
                  hatch="///" if exploit else None, edgecolor="black",
                  alpha=0.85, capsize=4)
    axD.text(i, bm + 0.6, f"{bm:.1f}", ha="center", fontsize=9, fontweight="bold")
    if exploit:
        axD.text(i, bm - 2.5, "QED\n≈0.03", ha="center", fontsize=7,
                 color="white", fontweight="bold")
axD.set_xticks(xs)
axD.set_xticklabels(labels, fontsize=9)
axD.set_ylabel("best penalized logP achieved")
axD.set_title("D. Best molecule  (⚠ hatched = reward-hacking artifact)",
              fontsize=11, fontweight="bold")
axD.axhline(5.5, color="gray", ls=":", lw=1)
axD.text(4.4, 5.7, "realistic ceiling", color="gray", fontsize=7, ha="right")
axD.grid(alpha=0.3, axis="y")
axD.legend(handles=[Patch(facecolor="gray", hatch="///", edgecolor="black",
                          label="reward-hacking (non-drug-like)")],
           fontsize=8, loc="upper left")

fig.tight_layout(rect=[0, 0.01, 1, 0.96])
os.makedirs("output", exist_ok=True)
os.makedirs("assets", exist_ok=True)
for path in ["output/bbrt_comparison_expanded.png", "assets/bbrt_comparison_expanded.png"]:
    fig.savefig(path, dpi=150, bbox_inches="tight")
fig.savefig("assets/bbrt_comparison_expanded.pdf", bbox_inches="tight")
print("saved output/ + assets/ bbrt_comparison_expanded.{png,pdf}")
