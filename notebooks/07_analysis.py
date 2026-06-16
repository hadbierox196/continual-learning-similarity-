# %% [markdown]
# # Week 7 — Analysis and Publication-Quality Figures
#
# Produces the four main paper figures:
#   Figure 1: BWT comparison — all methods × similarity conditions
#   Figure 2: CKA feature drift comparison
#   Figure 3: BWT vs. CKA scatter (120 individual runs)
#   Figure 4: INTERACTION plot — EWC minus ER gap vs. similarity (HEADLINE)
#
# All figures saved at 300 DPI as PNG + SVG.

# %% — Setup
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from pathlib import Path

from src.experiment import load_all_results

Path("../results/figures").mkdir(parents=True, exist_ok=True)

# ── Consistent colour palette (define once, use everywhere) ──────────────────
COLOR = {
    "finetune": "#9E9E9E",
    "ewc":      "#2196F3",
    "er":       "#FF5722",
    "joint":    "#4CAF50",
}
METHOD_LABELS = {
    "finetune": "Fine-tuning",
    "ewc":      "EWC",
    "er":       "Experience Replay",
    "joint":    "Joint Training",
}
SIM_ORDER  = ["low", "medium", "high"]
SIM_LABELS = ["Low", "Medium", "High"]
METHODS    = ["finetune", "ewc", "er", "joint"]

def save_fig(fig, name: str):
    """Save figure as both PNG (300 DPI) and SVG (for editing)."""
    for ext in ("png", "svg"):
        fig.savefig(f"../results/figures/{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved: results/figures/{name}.png / .svg")

# %% — Load all results
results = load_all_results("../results/raw")
print(f"Loaded {len(results)} results (expect 120)")

rows = []
for r in results:
    cka = r.get("cka_values", {}).get("task0_to_final", np.nan)
    rows.append({
        "method":     r["method"],
        "similarity": r["similarity"],
        "seed":       r["seed"],
        "bwt":        r["bwt"],
        "avg_accuracy": r["avg_accuracy"],
        "cka":        cka,
    })

df = pd.DataFrame(rows)
df["similarity"] = pd.Categorical(df["similarity"], categories=SIM_ORDER, ordered=True)
df["method"]     = pd.Categorical(df["method"],     categories=METHODS,   ordered=True)

print(df.groupby(["method","similarity"])[["bwt","avg_accuracy","cka"]].mean().round(4))

# %% — FIGURE 1: BWT comparison (grouped bar chart, all methods × conditions)
fig, ax = plt.subplots(figsize=(11, 5))

n_methods    = len(METHODS)
n_conditions = len(SIM_ORDER)
group_width  = 0.8
bar_width    = group_width / n_methods
x_base       = np.arange(n_conditions)

for mi, method in enumerate(METHODS):
    offsets = x_base + (mi - n_methods / 2 + 0.5) * bar_width
    sub = df[df["method"] == method].groupby("similarity")["bwt"]
    means = [sub.get_group(s).mean() for s in SIM_ORDER]
    stds  = [sub.get_group(s).std()  for s in SIM_ORDER]
    ax.bar(offsets, means, bar_width * 0.9, yerr=stds, capsize=4,
           label=METHOD_LABELS[method], color=COLOR[method], alpha=0.85,
           edgecolor="black", linewidth=0.5)

ax.set_xticks(x_base)
ax.set_xticklabels(SIM_LABELS, fontsize=12)
ax.set_xlabel("Inter-Task Similarity Condition", fontsize=13)
ax.set_ylabel("Backward Transfer (BWT) ↑ better", fontsize=13)
ax.set_title("Figure 1: Backward Transfer Across Methods and Similarity Conditions\n"
             "(Split-MNIST, Class-IL, mean ± SD, n=10 seeds per cell)", fontsize=13)
ax.legend(loc="lower right", fontsize=10, ncol=2)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.25, axis="y")
save_fig(fig, "fig1_bwt_comparison")
plt.show()

# %% — FIGURE 2: CKA feature drift (same layout as Figure 1)
df_seq = df[df["method"] != "joint"]   # CKA drift only meaningful for sequential methods

fig, ax = plt.subplots(figsize=(10, 5))
seq_methods = ["finetune", "ewc", "er"]
n_seq = len(seq_methods)

for mi, method in enumerate(seq_methods):
    offsets = x_base + (mi - n_seq / 2 + 0.5) * bar_width
    sub = df_seq[df_seq["method"] == method].groupby("similarity")["cka"]
    means = [sub.get_group(s).mean() for s in SIM_ORDER]
    stds  = [sub.get_group(s).std()  for s in SIM_ORDER]
    ax.bar(offsets, means, bar_width * 0.9, yerr=stds, capsize=4,
           label=METHOD_LABELS[method], color=COLOR[method], alpha=0.85,
           edgecolor="black", linewidth=0.5)

ax.set_xticks(x_base)
ax.set_xticklabels(SIM_LABELS, fontsize=12)
ax.set_xlabel("Inter-Task Similarity Condition", fontsize=13)
ax.set_ylabel("CKA (Task 0 → Task 4) ↑ = less drift", fontsize=13)
ax.set_title("Figure 2: Representational Drift (CKA) Across Methods and Similarity Conditions\n"
             "(Layer-2 activations, probe set n=500, mean ± SD)", fontsize=13)
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, label="No drift")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.25, axis="y")
save_fig(fig, "fig2_cka_drift")
plt.show()

# %% — FIGURE 3: BWT vs. CKA scatter (all 120 runs, colored by method)
fig, ax = plt.subplots(figsize=(8, 6))

for method in METHODS:
    sub = df[df["method"] == method]
    ax.scatter(sub["cka"], sub["bwt"], label=METHOD_LABELS[method],
               color=COLOR[method], alpha=0.65, s=45, edgecolors="white",
               linewidths=0.4)

# Pearson correlation across ALL runs
valid = df.dropna(subset=["cka", "bwt"])
r, p  = stats.pearsonr(valid["cka"], valid["bwt"])
print(f"\nPearson r (CKA vs BWT, all {len(valid)} runs): r={r:.4f}, p={p:.4e}")

# Regression line
m_coef, b_coef = np.polyfit(valid["cka"], valid["bwt"], 1)
x_line = np.linspace(valid["cka"].min(), valid["cka"].max(), 100)
ax.plot(x_line, m_coef * x_line + b_coef, "k--", linewidth=1.5, alpha=0.6,
        label=f"Regression (r={r:.3f}, p={p:.3f})")

ax.set_xlabel("CKA: Task 0 → Task 4 (Layer 2) ↑ less drift", fontsize=13)
ax.set_ylabel("Backward Transfer (BWT) ↑ better", fontsize=13)
ax.set_title("Figure 3: Representational Drift Predicts Forgetting\n"
             "(Each point = one experimental run, n=120)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
save_fig(fig, "fig3_bwt_cka_scatter")
plt.show()

# %% — FIGURE 4: INTERACTION — EWC minus ER gap vs. similarity (HEADLINE FIGURE)
print("\n" + "="*60)
print("GENERATING FIGURE 4 — THE HEADLINE INTERACTION PLOT")
print("="*60)

ewc_bwt = df[df["method"]=="ewc"].groupby("similarity")["bwt"].agg(["mean","std"])
er_bwt  = df[df["method"]=="er" ].groupby("similarity")["bwt"].agg(["mean","std"])

gap_means = ewc_bwt.loc[SIM_ORDER, "mean"].values - er_bwt.loc[SIM_ORDER, "mean"].values
# Propagate errors (approximate, assumes independence)
gap_stds  = np.sqrt(ewc_bwt.loc[SIM_ORDER, "std"].values**2 +
                     er_bwt.loc[SIM_ORDER, "std"].values**2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: raw BWT per method per condition
for method in ["ewc", "er"]:
    sub   = df[df["method"]==method].groupby("similarity")["bwt"]
    means = [sub.get_group(s).mean() for s in SIM_ORDER]
    stds  = [sub.get_group(s).std()  for s in SIM_ORDER]
    axes[0].errorbar(SIM_LABELS, means, yerr=stds, fmt="o-",
                     color=COLOR[method], linewidth=2.5, markersize=9,
                     capsize=5, label=METHOD_LABELS[method])

axes[0].set_xlabel("Similarity Condition", fontsize=13)
axes[0].set_ylabel("Backward Transfer (BWT)", fontsize=13)
axes[0].set_title("EWC vs. ER: BWT per Condition", fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Right panel: EWC - ER gap (the interaction)
axes[1].errorbar(SIM_LABELS, gap_means, yerr=gap_stds, fmt="D-",
                 color="#7B1FA2", linewidth=2.5, markersize=9, capsize=5)
axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.5,
                label="EWC = ER (gap = 0)")
for xi, (y, e, lab) in enumerate(zip(gap_means, gap_stds, SIM_LABELS)):
    axes[1].annotate(f"{y:+.3f}", (xi, y + e + 0.005),
                     ha="center", fontsize=11, fontweight="bold")

axes[1].set_xlabel("Similarity Condition", fontsize=13)
axes[1].set_ylabel("EWC BWT − ER BWT\n(positive = EWC better; negative = ER better)",
                   fontsize=12)
axes[1].set_title("Figure 4: EWC–ER Performance Gap vs. Similarity\n"
                  "(Positive → EWC better; Negative → ER better)", fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Method × Similarity Interaction — Primary Result", fontsize=14,
             fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "fig4_interaction")
plt.show()

# %% — Print all numbers for lab notebook
print("\n" + "="*60)
print("RECORD ALL OF THESE IN YOUR LAB NOTEBOOK:")
print("="*60)
print(f"\nPearson r (CKA vs BWT): r={r:.4f}, p={p:.4e}")
print("\nEWC-ER Gap by Condition:")
for lab, gm, gs in zip(SIM_LABELS, gap_means, gap_stds):
    direction = "EWC better" if gm > 0 else "ER better"
    print(f"  {lab}: gap={gm:+.4f} ± {gs:.4f}  ({direction})")

print("\nRead ANOVA results from: results/interaction_anova.txt")
print("Write your 200-word interpretation paragraph in your lab notebook now.")
