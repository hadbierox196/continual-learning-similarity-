# %% [markdown]
# # Week 2 — EWC Implementation and Lambda Sweep
#
# Goal: Validate EWC against published Split-MNIST numbers (van de Ven & Tolias
# 2019). Best-lambda EWC BWT must exceed fine-tuning BWT by at least 0.10.
#
# Expected output:
#   results/ewc_lambda_sweep.png   — BWT vs log(λ), U-shaped curve
#   results/ewc_accuracy_matrix.png
#   Lab notebook: best λ and its BWT

# %% — Setup (same as notebook 01 — paste into first Colab cell)
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
from src.ewc        import EWC
from src.cka        import validate_cka
from src.experiment import set_seeds, _train_epoch, DEFAULT_CONFIG

Path("../results/figures").mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# %% — Lambda sweep function
LAMBDA_RANGE = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

def run_ewc_sweep(lambdas: list, similarity: str = "medium",
                  seed: int = 0, config: dict = None):
    """Run EWC for each lambda value and return BWT per lambda."""
    if config is None:
        config = DEFAULT_CONFIG.copy()

    results = {}
    for lam in lambdas:
        print(f"\n  λ = {lam} ...")
        set_seeds(seed)

        train_loaders, test_loaders, _ = get_split_mnist(
            similarity=similarity, data_dir=config["data_dir"],
            batch_size=config["batch_size"]
        )

        model     = MLP().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        ewc_agent = EWC(model, lambda_ewc=lam)
        matrix    = TaskAccuracyMatrix(config["n_tasks"])

        for task_idx, train_loader in enumerate(train_loaders):
            for _ in range(config["epochs_per_task"]):
                _train_epoch(model, train_loader, optimizer, criterion,
                             ewc_agent=ewc_agent, device=DEVICE)
            ewc_agent.compute_fisher(train_loader,
                                     n_samples=config["fisher_n_samples"],
                                     device=DEVICE)
            matrix.record_after_task(model, test_loaders, task_idx, DEVICE)

        bwt = matrix.bwt
        results[lam] = {"bwt": bwt, "avg_acc": matrix.average_accuracy}
        print(f"    BWT={bwt:.4f}   AvgAcc={matrix.average_accuracy:.4f}")

        # Print Fisher diagnostics after last task
        stats = ewc_agent.fisher_stats()
        print(f"    Fisher: mean={stats['mean']:.4e}, frac_near_zero={stats['frac_near_zero']:.2%}")

    return results

# %% — Run lambda sweep
print("Running EWC lambda sweep ...")
sweep_results = run_ewc_sweep(LAMBDA_RANGE)

# %% — Plot BWT vs lambda
lambdas  = list(sweep_results.keys())
bwt_vals = [sweep_results[l]["bwt"] for l in lambdas]

best_lam  = lambdas[int(np.argmax(bwt_vals))]   # least negative = best
best_bwt  = max(bwt_vals)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(lambdas, bwt_vals, "o-", color="#2196F3", linewidth=2, markersize=8)
ax.axhline(bwt_vals[0], color="gray", linestyle="--", alpha=0.5, label="fine-tuning BWT (ref)")
ax.axvline(best_lam, color="#FF5722", linestyle=":", alpha=0.7, label=f"Best λ={best_lam}")
ax.set_xlabel("EWC λ (log scale)", fontsize=12)
ax.set_ylabel("Backward Transfer (BWT) ↑ better", fontsize=12)
ax.set_title("EWC: BWT vs. Regularisation Strength λ\n(Split-MNIST, Medium Similarity)",
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/ewc_lambda_sweep.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: results/figures/ewc_lambda_sweep.png")

# %% — Save best lambda to config and run with it
print(f"\nBest λ = {best_lam}   (BWT={best_bwt:.4f})")
print("→ Update DEFAULT_CONFIG['ewc_lambda'] in src/experiment.py before Week 6.")

# Update the config file automatically
import json
config_update = {"ewc_lambda": best_lam}
with open("../results/ewc_best_lambda.json", "w") as f:
    json.dump(config_update, f, indent=2)
print("Saved: results/ewc_best_lambda.json")

# %% — Fisher diagnostic on best-lambda model
print("\n--- Fisher Diagnostic (best lambda) ---")
print("If frac_near_zero > 0.20 → Fisher computation is broken")
print("  See FAILURE CONDITION in roadmap Week 2.")

# %% — Lab notebook summary
print("\n" + "=" * 60)
print("RECORD THESE IN YOUR LAB NOTEBOOK:")
print("=" * 60)
for lam, res in sweep_results.items():
    print(f"  λ={lam:>8}  BWT={res['bwt']:+.4f}  AvgAcc={res['avg_acc']:.4f}")
print(f"\nBest λ = {best_lam}, BWT = {best_bwt:.4f}")
