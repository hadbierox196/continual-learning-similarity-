# %% [markdown]
# # Week 3 — Experience Replay: Buffer Size Sweep and Validation
#
# Goal: Validate the ReservoirBuffer implementation and determine the buffer
# size at which ER matches or exceeds best-lambda EWC BWT.
#
# Correctness checks:
#   1. Buffer class distribution after task 5: all 10 classes represented
#   2. ER BWT with 2000 samples significantly better than fine-tuning
#   3. BWT shows diminishing returns as buffer size grows

# %% — Setup
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data       import get_split_mnist
from src.models     import MLP
from src.metrics    import TaskAccuracyMatrix
from src.replay     import ReservoirBuffer
from src.experiment import set_seeds, _train_epoch, DEFAULT_CONFIG

Path("../results/figures").mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# %% — Buffer size sweep function
BUFFER_SIZES = [100, 500, 1000, 2000, 5000]

def run_er_sweep(buffer_sizes: list, similarity: str = "medium",
                 seed: int = 0, config: dict = None):
    """Run ER for each buffer size and return BWT per size."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    results = {}
    for buf_size in buffer_sizes:
        print(f"\n  Buffer size = {buf_size} ...")
        set_seeds(seed)

        train_loaders, test_loaders, _ = get_split_mnist(
            similarity=similarity, data_dir=config["data_dir"],
            batch_size=config["batch_size"]
        )

        model     = MLP().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        buffer    = ReservoirBuffer(capacity=buf_size, seed=seed)
        matrix    = TaskAccuracyMatrix(config["n_tasks"])

        for task_idx, train_loader in enumerate(train_loaders):
            for _ in range(config["epochs_per_task"]):
                _train_epoch(model, train_loader, optimizer, criterion,
                             replay_buffer=buffer,
                             er_ratio=config["er_replay_ratio"],
                             device=DEVICE)
            # Populate buffer AFTER training on this task
            for x_b, y_b in train_loader:
                buffer.update(x_b, y_b)
            matrix.record_after_task(model, test_loaders, task_idx, DEVICE)

        bwt = matrix.bwt
        dist = buffer.class_distribution()
        results[buf_size] = {
            "bwt":       bwt,
            "avg_acc":   matrix.average_accuracy,
            "class_dist": dist,
            "n_classes_in_buffer": len(dist),
        }
        print(f"    BWT={bwt:.4f}   AvgAcc={matrix.average_accuracy:.4f}")
        print(f"    Class distribution: {dist}")
        print(f"    Classes in buffer: {len(dist)}/10")

    return results

# %% — Run sweep
print("Running ER buffer size sweep ...")
er_results = run_er_sweep(BUFFER_SIZES)

# %% — Diagnostic: class distribution validation
print("\n--- CLASS DISTRIBUTION DIAGNOSTIC ---")
print("All 10 classes should appear in buffer after task 5.")
print("If early classes are missing → reservoir sampling is broken.\n")
for size, res in er_results.items():
    ok = res["n_classes_in_buffer"] == 10
    print(f"  Buffer={size:>5}: {res['n_classes_in_buffer']}/10 classes  {'✓' if ok else '✗ BROKEN'}")

# %% — Plot BWT vs buffer size
buf_sizes = list(er_results.keys())
bwt_vals  = [er_results[s]["bwt"] for s in buf_sizes]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(buf_sizes, bwt_vals, "s-", color="#FF5722", linewidth=2, markersize=8,
        label="Experience Replay")
ax.set_xlabel("Replay Buffer Size (n examples)", fontsize=12)
ax.set_ylabel("Backward Transfer (BWT) ↑ better", fontsize=12)
ax.set_title("ER: BWT vs. Buffer Size\n(Split-MNIST, Medium Similarity)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/er_buffer_sweep.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: results/figures/er_buffer_sweep.png")

# %% — Preliminary comparison plot (EWC vs ER vs Finetune)
# Load EWC best result from Week 2 (update these manually from your Week 2 output)
import json
try:
    with open("../results/ewc_best_lambda.json") as f:
        ewc_config = json.load(f)
    BEST_EWC_LAMBDA = ewc_config["ewc_lambda"]
    print(f"\nLoaded best EWC lambda: {BEST_EWC_LAMBDA}")
except FileNotFoundError:
    BEST_EWC_LAMBDA = 100.0
    print(f"\nWARNING: ewc_best_lambda.json not found. Using default λ={BEST_EWC_LAMBDA}")

# Run a quick fine-tuning baseline for comparison
from src.experiment import run_experiment
ft_result = run_experiment("finetune", "medium", seed=0)

# %% — Select best ER buffer size for Week 6 factorial
best_buf   = buf_sizes[int(np.argmax(bwt_vals))]
best_er_bwt = max(bwt_vals)
print(f"\nBest buffer size: {best_buf}   (BWT={best_er_bwt:.4f})")
print(f"Fine-tuning BWT:  {ft_result['bwt']:.4f}")
print(f"ER improvement:   {best_er_bwt - ft_result['bwt']:+.4f}")

# Save best buffer size
config_update = {"er_buffer_size": best_buf}
with open("../results/er_best_config.json", "w") as f:
    json.dump(config_update, f, indent=2)
print("Saved: results/er_best_config.json")
print("→ Update DEFAULT_CONFIG['er_buffer_size'] in src/experiment.py before Week 6.")

# %% — Lab notebook summary
print("\n" + "=" * 60)
print("RECORD THESE IN YOUR LAB NOTEBOOK:")
print("=" * 60)
for size, res in er_results.items():
    print(f"  Buffer={size:>5}: BWT={res['bwt']:+.4f}  AvgAcc={res['avg_acc']:.4f}  "
          f"Classes={res['n_classes_in_buffer']}/10")
print(f"\nBest buffer size: {best_buf}, BWT={best_er_bwt:.4f}")
