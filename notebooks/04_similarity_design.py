# %% [markdown]
# # Week 4 — Similarity Manipulation Design and Validation
#
# Goal: Build and statistically validate the three task-pairing conditions.
# This is the scientific heart of the project. If the ANOVA fails (p > 0.05),
# the manipulation is invalid — see Week 4 pivot trigger in the roadmap.
#
# Output:
#   results/digit_similarity_matrix.png      — 10×10 heatmap
#   results/similarity_conditions_validation.png — box plot of conditions
#   results/task_pairings.json               — used by ALL subsequent notebooks

# %% — Setup
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from pathlib import Path

from src.data       import load_mnist, save_task_pairings
from src.similarity import (compute_class_means, cosine_similarity_matrix,
                             greedy_pair, validate_conditions)

Path("../results/figures").mkdir(parents=True, exist_ok=True)

# %% — Compute class mean images
print("Loading MNIST and computing class means ...")
train_ds, _ = load_mnist("../data")
means = compute_class_means(train_ds)     # [10, 784]
print(f"Class means shape: {means.shape}")

# %% — Build 10×10 cosine similarity matrix
sim_mat = cosine_similarity_matrix(means)
print("Cosine similarity matrix computed.")
print(f"Diagonal (self-similarity): {sim_mat.diagonal().round(4)}")

# Find most and least similar pairs for reference
upper_tri_idx = [(i, j) for i in range(10) for j in range(i+1, 10)]
sims = [(sim_mat[i, j], i, j) for i, j in upper_tri_idx]
sims_sorted = sorted(sims, reverse=True)

print("\nTop-5 most similar digit pairs:")
for sim, a, b in sims_sorted[:5]:
    print(f"  Digits ({a}, {b}): cosine sim = {sim:.4f}")

print("\nTop-5 least similar digit pairs:")
for sim, a, b in sims_sorted[-5:]:
    print(f"  Digits ({a}, {b}): cosine sim = {sim:.4f}")

# %% — Find optimal pairings
print("\nFinding LOW similarity pairings (greedy minimisation) ...")
low_pairs, low_mean = greedy_pair(sim_mat, mode="low")
print(f"  Low pairs: {low_pairs}   mean sim={low_mean:.4f}")

# Medium: standard Split-MNIST
med_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
med_mean  = float(np.mean([sim_mat[a, b] for a, b in med_pairs]))
print(f"\nMedium pairs: {med_pairs}   mean sim={med_mean:.4f}")

print("\nFinding HIGH similarity pairings (greedy maximisation) ...")
high_pairs, high_mean = greedy_pair(sim_mat, mode="high")
print(f"  High pairs: {high_pairs}   mean sim={high_mean:.4f}")

# %% — Statistical validation via one-way ANOVA
print("\n--- ANOVA VALIDATION ---")
val = validate_conditions(low_pairs, med_pairs, high_pairs, sim_mat)

print(f"Low  condition mean sim:  {val['low_mean']:.4f}")
print(f"Med  condition mean sim:  {val['med_mean']:.4f}")
print(f"High condition mean sim:  {val['high_mean']:.4f}")
print(f"\nMonotone (Low < Med < High): {'✓' if val['monotone'] else '✗'}")
print(f"Gap (High - Low):            {val['gap_lo_hi']:.4f}  (target ≥ 0.05)")
print(f"ANOVA F-statistic:           {val['f_stat']:.4f}")
print(f"ANOVA p-value:               {val['p_val']:.4f}  {'✓ VALID' if val['valid'] else '✗ INVALID — see pivot trigger'}")

if not val["valid"]:
    print("\n⚠ PIVOT REQUIRED: ANOVA p > 0.05.")
    print("Switch to feature-space similarity using a pre-trained CNN.")
    print("See Week 4 Pivot Trigger in the roadmap.")

# %% — Plot 10×10 cosine similarity heatmap
fig, ax = plt.subplots(figsize=(9, 8))
mask = np.eye(10, dtype=bool)   # mask diagonal for clarity

sns.heatmap(sim_mat, ax=ax, cmap="YlOrRd", vmin=0.0, vmax=1.0,
            annot=True, fmt=".2f", linewidths=0.3,
            xticklabels=[f"Digit {i}" for i in range(10)],
            yticklabels=[f"Digit {i}" for i in range(10)])
ax.set_title("Pairwise Cosine Similarity Between MNIST Digit Class Means",
             fontsize=13, fontweight="bold")

# Annotate pairings
pair_styles = {
    "Low":    (low_pairs,  "#2196F3", "solid"),
    "Medium": (med_pairs,  "#4CAF50", "dashed"),
    "High":   (high_pairs, "#FF5722", "dotted"),
}
for label, (pairs, color, style) in pair_styles.items():
    for a, b in pairs:
        for (r, c) in [(a, b), (b, a)]:
            ax.add_patch(plt.Rectangle(
                (c, r), 1, 1, fill=False, edgecolor=color, lw=2.5, ls=style
            ))

patches = [mpatches.Patch(edgecolor=c, fill=False, label=lbl, lw=2)
           for lbl, (_, c, _) in pair_styles.items()]
ax.legend(handles=patches, loc="upper right", fontsize=10, title="Condition")

plt.tight_layout()
plt.savefig("../results/figures/digit_similarity_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: results/figures/digit_similarity_matrix.png")

# %% — Box plot of within-pair similarities
fig, ax = plt.subplots(figsize=(8, 5))
data = [val["low_sims"], val["med_sims"], val["high_sims"]]
bplot = ax.boxplot(data, labels=["Low", "Medium", "High"],
                   patch_artist=True, notch=False, widths=0.5)
colors = ["#2196F3", "#4CAF50", "#FF5722"]
for patch, color in zip(bplot["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.scatter([1]*5, val["low_sims"],  color="#2196F3", zorder=5, s=60)
ax.scatter([2]*5, val["med_sims"],  color="#4CAF50", zorder=5, s=60)
ax.scatter([3]*5, val["high_sims"], color="#FF5722", zorder=5, s=60)

ax.set_ylabel("Within-pair cosine similarity", fontsize=12)
ax.set_xlabel("Similarity Condition", fontsize=12)
ax.set_title(f"Similarity Condition Validation (ANOVA: F={val['f_stat']:.2f}, p={val['p_val']:.3f})",
             fontsize=13)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("../results/figures/similarity_conditions_validation.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: results/figures/similarity_conditions_validation.png")

# %% — Visualise the class mean images
fig, axes = plt.subplots(1, 10, figsize=(16, 2))
for i, ax in enumerate(axes):
    img = means[i].reshape(28, 28).numpy()
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Digit {i}", fontsize=9)
    ax.axis("off")
plt.suptitle("MNIST Class Mean Images", fontsize=12, y=1.05)
plt.tight_layout()
plt.savefig("../results/figures/class_mean_images.png", dpi=200, bbox_inches="tight")
plt.show()

# %% — Save task pairings JSON for all subsequent notebooks
save_task_pairings(
    low=low_pairs,
    medium=med_pairs,
    high=high_pairs,
    pairings_file="../results/task_pairings.json"
)

# %% — Lab notebook summary
print("\n" + "=" * 60)
print("RECORD THESE IN YOUR LAB NOTEBOOK:")
print("=" * 60)
print(f"Low  pairings:  {low_pairs}  (mean sim={val['low_mean']:.4f})")
print(f"Med  pairings:  {med_pairs}  (mean sim={val['med_mean']:.4f})")
print(f"High pairings:  {high_pairs} (mean sim={val['high_mean']:.4f})")
print(f"Gap (H-L):      {val['gap_lo_hi']:.4f}")
print(f"ANOVA:          F={val['f_stat']:.4f}, p={val['p_val']:.4f}")
