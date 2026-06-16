# %% [markdown]
# # Reproduce Results — Generate All Figures from Saved Checkpoints
#
# This notebook regenerates all four main paper figures from pre-saved .pkl
# checkpoint files WITHOUT re-running any experiments.
#
# Runtime: ~5 minutes (load + plot, no training)
# Prerequisites: 120 .pkl files in results/raw/ (from notebook 06)
#
# Instructions for a fresh Colab session:
#   1. Mount Drive
#   2. Navigate to continual-learning-similarity/
#   3. Run all cells top to bottom

# %% — Setup
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# Mount Drive (Colab)
# from google.colab import drive
# drive.mount('/content/drive')
# os.chdir('/content/drive/MyDrive/continual-learning-similarity')

# Install if needed
# !pip install torch torchvision numpy matplotlib seaborn scipy statsmodels pandas tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path

from src.experiment import load_all_results

Path("../results/figures").mkdir(parents=True, exist_ok=True)

# ── Palette (must match 07_analysis.py) ──────────────────────────────────────
COLOR = {"finetune":"#9E9E9E","ewc":"#2196F3","er":"#FF5722","joint":"#4CAF50"}
METHOD_LABELS = {"finetune":"Fine-tuning","ewc":"EWC",
                 "er":"Experience Replay","joint":"Joint Training"}
SIM_ORDER = ["low","medium","high"]
SIM_LABELS = ["Low","Medium","High"]
METHODS = ["finetune","ewc","er","joint"]

def save_fig(fig, name):
    for ext in ("png","svg"):
        fig.savefig(f"../results/figures/{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved results/figures/{name}.png + .svg")

# %% — Load all results
print("Loading checkpoint files ...")
results = load_all_results("../results/raw")
print(f"Loaded: {len(results)} results (expect 120)")
if len(results) < 120:
    print(f"⚠ Warning: only {len(results)}/120 files found. Figures may be incomplete.")

rows = []
for r in results:
    cka = r.get("cka_values", {}).get("task0_to_final", np.nan)
    rows.append({"method":r["method"],"similarity":r["similarity"],
                 "seed":r["seed"],"bwt":r["bwt"],
                 "avg_accuracy":r["avg_accuracy"],"cka":cka})

df = pd.DataFrame(rows)
df["similarity"] = pd.Categorical(df["similarity"], categories=SIM_ORDER, ordered=True)
df["method"]     = pd.Categorical(df["method"],     categories=METHODS,   ordered=True)
print(f"\nData loaded: {len(df)} rows")
print(df.groupby(["method","similarity"])[["bwt","cka"]].mean().round(4))

# %% — Figure 1: BWT comparison
print("\nGenerating Figure 1 ...")
fig, ax = plt.subplots(figsize=(11, 5))
x_base = np.arange(len(SIM_ORDER))
bar_width = 0.8 / len(METHODS)
for mi, method in enumerate(METHODS):
    offsets = x_base + (mi - len(METHODS)/2 + 0.5) * bar_width
    sub = df[df["method"]==method].groupby("similarity")["bwt"]
    means = [sub.get_group(s).mean() for s in SIM_ORDER]
    stds  = [sub.get_group(s).std()  for s in SIM_ORDER]
    ax.bar(offsets, means, bar_width*0.9, yerr=stds, capsize=4,
           label=METHOD_LABELS[method], color=COLOR[method], alpha=0.85,
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x_base); ax.set_xticklabels(SIM_LABELS, fontsize=12)
ax.set_xlabel("Similarity Condition", fontsize=13)
ax.set_ylabel("Backward Transfer (BWT) ↑ better", fontsize=13)
ax.set_title("Figure 1: Backward Transfer Across Methods and Similarity Conditions\n"
             "(mean ± SD, n=10 seeds)", fontsize=13)
ax.legend(fontsize=10, ncol=2, loc="lower right")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.25, axis="y")
save_fig(fig, "fig1_bwt_comparison"); plt.show()

# %% — Figure 2: CKA feature drift
print("Generating Figure 2 ...")
seq_methods = ["finetune","ewc","er"]
fig, ax = plt.subplots(figsize=(10, 5))
bar_width2 = 0.8 / len(seq_methods)
for mi, method in enumerate(seq_methods):
    offsets = x_base + (mi - len(seq_methods)/2 + 0.5) * bar_width2
    sub = df[df["method"]==method].groupby("similarity")["cka"]
    means = [sub.get_group(s).mean() for s in SIM_ORDER]
    stds  = [sub.get_group(s).std()  for s in SIM_ORDER]
    ax.bar(offsets, means, bar_width2*0.9, yerr=stds, capsize=4,
           label=METHOD_LABELS[method], color=COLOR[method], alpha=0.85,
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x_base); ax.set_xticklabels(SIM_LABELS, fontsize=12)
ax.set_xlabel("Similarity Condition", fontsize=13)
ax.set_ylabel("CKA (Task 0 → Task 4) ↑ less drift", fontsize=13)
ax.set_title("Figure 2: Representational Drift (CKA) Across Methods\n"
             "(Layer-2 activations, mean ± SD)", fontsize=13)
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, label="No drift")
ax.legend(fontsize=10, loc="lower right"); ax.grid(True, alpha=0.25, axis="y")
save_fig(fig, "fig2_cka_drift"); plt.show()

# %% — Figure 3: BWT vs CKA scatter
print("Generating Figure 3 ...")
valid = df.dropna(subset=["cka","bwt"])
r_val, p_val = stats.pearsonr(valid["cka"], valid["bwt"])
m_c, b_c = np.polyfit(valid["cka"], valid["bwt"], 1)
x_line = np.linspace(valid["cka"].min(), valid["cka"].max(), 100)
fig, ax = plt.subplots(figsize=(8, 6))
for method in METHODS:
    sub = valid[valid["method"]==method]
    ax.scatter(sub["cka"], sub["bwt"], label=METHOD_LABELS[method],
               color=COLOR[method], alpha=0.65, s=45, edgecolors="white", linewidths=0.4)
ax.plot(x_line, m_c*x_line+b_c, "k--", linewidth=1.5, alpha=0.6,
        label=f"Regression (r={r_val:.3f}, p={p_val:.3f})")
ax.set_xlabel("CKA: Task 0 → Task 4 ↑ less drift", fontsize=13)
ax.set_ylabel("Backward Transfer (BWT) ↑ better", fontsize=13)
ax.set_title("Figure 3: Representational Drift Predicts Forgetting\n(n=120 runs)",
             fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
save_fig(fig, "fig3_bwt_cka_scatter"); plt.show()
print(f"\nPearson r={r_val:.4f}, p={p_val:.4e}")

# %% — Figure 4: Interaction plot
print("Generating Figure 4 ...")
ewc_bwt = df[df["method"]=="ewc"].groupby("similarity")["bwt"].agg(["mean","std"])
er_bwt  = df[df["method"]=="er" ].groupby("similarity")["bwt"].agg(["mean","std"])
gap_means = ewc_bwt.loc[SIM_ORDER,"mean"].values - er_bwt.loc[SIM_ORDER,"mean"].values
gap_stds  = np.sqrt(ewc_bwt.loc[SIM_ORDER,"std"].values**2 +
                     er_bwt.loc[SIM_ORDER,"std"].values**2)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for method in ["ewc","er"]:
    sub = df[df["method"]==method].groupby("similarity")["bwt"]
    means = [sub.get_group(s).mean() for s in SIM_ORDER]
    stds  = [sub.get_group(s).std()  for s in SIM_ORDER]
    axes[0].errorbar(SIM_LABELS, means, yerr=stds, fmt="o-",
                     color=COLOR[method], linewidth=2.5, markersize=9,
                     capsize=5, label=METHOD_LABELS[method])
axes[0].set_xlabel("Similarity Condition", fontsize=13)
axes[0].set_ylabel("Backward Transfer (BWT)", fontsize=13)
axes[0].set_title("EWC vs. ER: BWT per Condition", fontsize=13)
axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)
axes[1].errorbar(SIM_LABELS, gap_means, yerr=gap_stds, fmt="D-",
                 color="#7B1FA2", linewidth=2.5, markersize=9, capsize=5)
axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.5)
for xi,(y,e) in enumerate(zip(gap_means, gap_stds)):
    axes[1].annotate(f"{y:+.3f}", (xi, y+e+0.005), ha="center",
                     fontsize=11, fontweight="bold")
axes[1].set_xlabel("Similarity Condition", fontsize=13)
axes[1].set_ylabel("EWC BWT − ER BWT", fontsize=13)
axes[1].set_title("Figure 4: EWC–ER Performance Gap vs. Similarity\n(Headline Result)",
                  fontsize=13)
axes[1].grid(True, alpha=0.3)
plt.suptitle("Method × Similarity Interaction", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save_fig(fig, "fig4_interaction"); plt.show()

# %% — ANOVA (reproduce from saved data)
print("\n--- TWO-WAY ANOVA (reproduced from saved data) ---")
df_a = df.copy()
df_a["method"]     = pd.Categorical(df_a["method"])
df_a["similarity"] = pd.Categorical(df_a["similarity"])
model = smf.ols("bwt ~ C(method) + C(similarity) + C(method):C(similarity)",
                data=df_a).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table.to_string())

# %% — Final summary
print("\n" + "="*60)
print("REPRODUCE COMPLETE — all figures regenerated")
print("="*60)
print(f"Pearson r (CKA vs BWT): {r_val:.4f}, p={p_val:.4e}")
print(f"EWC-ER gap by condition:")
for lab, g, s in zip(SIM_LABELS, gap_means, gap_stds):
    print(f"  {lab}: {g:+.4f} ± {s:.4f}")
