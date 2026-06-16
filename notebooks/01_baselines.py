# %% [markdown]
# # Week 1 — Baselines: Fine-tuning Lower Bound & Joint Training Upper Bound
#
# Goal: Validate the training pipeline against published numbers BEFORE building
# anything on top of it. Every downstream result is measured relative to these
# two anchors. A data pipeline error here will corrupt all 120 subsequent runs.
#
# Expected output:
#   Fine-tuning BWT  ≈ -0.45 to -0.60  (heavy forgetting confirmed)
#   Joint accuracy   ≈ 0.95+             (no forgetting upper bound)
#   results/baseline_accuracy_matrix.png — staircase heatmap
#   results/joint_accuracy_matrix.png    — uniformly hot heatmap

# %% — Setup
# Run this cell FIRST every session.

# Install dependencies (Colab only — skip if running locally)
# !pip install torch torchvision numpy matplotlib seaborn scipy statsmodels tqdm

# Mount Drive and navigate to your repo (Colab only)
# from google.colab import drive
# drive.mount('/content/drive')
# import os; os.chdir('/content/drive/MyDrive/continual-learning-similarity')

# OR: clone from GitHub
# !git clone https://github.com/YOUR_USERNAME/continual-learning-similarity
# import os; os.chdir('continual-learning-similarity')

import sys, os
sys.path.insert(0, os.path.abspath('..'))   # so `import src` works from notebooks/

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data    import get_split_mnist, get_joint_loader, load_mnist
from src.models  import MLP
from src.metrics import TaskAccuracyMatrix
from src.experiment import set_seeds, _train_epoch, DEFAULT_CONFIG

Path("../results").mkdir(exist_ok=True)
Path("../results/figures").mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# %% — Helper: train one sequential method
def train_sequential(method: str, similarity: str = "medium",
                     config: dict = None, seed: int = 0):
    """
    Train fine-tuning or joint-training baseline and return the accuracy matrix.
    method: "finetune" | "joint"
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    set_seeds(seed)
    import torch.nn as nn
    import torch.optim as optim

    train_loaders, test_loaders, task_pairs = get_split_mnist(
        similarity=similarity, data_dir=config["data_dir"],
        batch_size=config["batch_size"]
    )

    model     = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    matrix    = TaskAccuracyMatrix(config["n_tasks"])

    if method == "joint":
        joint_loader = get_joint_loader(train_loaders, config["batch_size"])
        total_epochs = config["epochs_per_task"] * config["n_tasks"]
        for epoch in range(total_epochs):
            loss = _train_epoch(model, joint_loader, optimizer, criterion,
                                device=DEVICE)
            if epoch % 5 == 0:
                print(f"  [Joint] epoch {epoch}/{total_epochs}, loss={loss:.4f}")
        for i in range(config["n_tasks"]):
            acc = matrix.evaluate_task(model, test_loaders[i], DEVICE)
            for j in range(config["n_tasks"]):
                matrix.R[i, j] = acc
    else:  # finetune
        for task_idx, train_loader in enumerate(train_loaders):
            print(f"  [Finetune] Training task {task_idx+1}/5 ...")
            for epoch in range(config["epochs_per_task"]):
                _train_epoch(model, train_loader, optimizer, criterion,
                             device=DEVICE)
            matrix.record_after_task(model, test_loaders, task_idx, DEVICE)

    return matrix

# %% — Run fine-tuning baseline
print("=" * 50)
print("Training FINE-TUNING baseline (medium similarity) ...")
ft_matrix = train_sequential("finetune", similarity="medium")

print(f"\nFine-tuning BWT:          {ft_matrix.bwt:.4f}")
print(f"Fine-tuning Avg Accuracy: {ft_matrix.average_accuracy:.4f}")

# Sanity check: after task 1, accuracy on task 1 should be ≥ 0.95
print(f"\nSanity check R[0,0] (after task 1, test task 1): {ft_matrix.R[0,0]:.4f}  (expect ≥ 0.95)")
print(f"Sanity check R[0,4] (after task 5, test task 1): {ft_matrix.R[0,4]:.4f}  (expect ≤ 0.30 — forgetting)")

# %% — Run joint training upper bound
print("=" * 50)
print("Training JOINT TRAINING upper bound ...")
joint_matrix = train_sequential("joint", similarity="medium")

print(f"\nJoint Training Avg Accuracy: {joint_matrix.average_accuracy:.4f}  (expect ≥ 0.95)")

# %% — Plot accuracy matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, matrix, title in zip(
    axes,
    [ft_matrix, joint_matrix],
    ["Fine-tuning (lower bound)", "Joint Training (upper bound)"]
):
    R_plot = matrix.R.copy()
    R_plot[np.isnan(R_plot)] = 0
    sns.heatmap(R_plot, ax=ax, vmin=0, vmax=1, cmap="RdYlGn",
                annot=True, fmt=".2f", linewidths=0.5,
                xticklabels=[f"After T{j+1}" for j in range(5)],
                yticklabels=[f"Task {i+1}" for i in range(5)])
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training stage", fontsize=11)
    ax.set_ylabel("Evaluated on", fontsize=11)

plt.suptitle("Class-IL Accuracy Matrix R[i, j] — Split-MNIST (Medium Similarity)",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("../results/figures/baseline_accuracy_matrices.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: results/figures/baseline_accuracy_matrices.png")

# %% — Record in lab notebook
print("\n" + "=" * 60)
print("RECORD THESE IN YOUR LAB NOTEBOOK:")
print("=" * 60)
print(f"Fine-tuning BWT:        {ft_matrix.bwt:.4f}")
print(f"Joint Avg Accuracy:     {joint_matrix.average_accuracy:.4f}")
print(f"Joint BWT (should=0):  {joint_matrix.bwt:.4f}")
print("")
print("PIPELINE VALID if:")
print(f"  BWT ∈ [-0.60, -0.35]:  {'✓' if -0.60 <= ft_matrix.bwt <= -0.30 else '✗ CHECK DATA PIPELINE'}")
print(f"  Joint acc ≥ 0.90:      {'✓' if joint_matrix.average_accuracy >= 0.90 else '✗ CHECK MODEL/TRAINING'}")
