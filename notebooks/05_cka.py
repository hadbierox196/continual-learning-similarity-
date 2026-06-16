# %% [markdown]
# # Week 5 — CKA Implementation Validation and Feature Drift Measurement
#
# Goal: Validate the CKA implementation against known-answer test cases, then
# measure representational drift under fine-tuning across all 3 similarity
# conditions. This is the first hypothesis check: does feature drift increase
# with task similarity under fine-tuning?
#
# Output:
#   results/cka_validation.txt
#   results/figures/cka_finetune_conditions.png

# %% — Setup
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data       import get_split_mnist, get_probe_set
from src.models     import MLP
from src.cka        import validate_cka, linear_cka, extract_activations
from src.experiment import set_seeds, _train_epoch, DEFAULT_CONFIG

Path("../results/figures").mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# %% — Step 1: Validate CKA implementation
print("=" * 50)
print("CKA VALIDATION (must pass before proceeding)")
print("=" * 50)
val = validate_cka(n=500, d=256, seed=42)

print(f"\nTest 1 — Identical matrices  : CKA={val['cka_identical']:.6f}  "
      f"(target=1.0)  {'✓' if val['test1_pass'] else '✗ FAIL'}")
print(f"Test 2 — Random matrices     : CKA={val['cka_random']:.6f}  "
      f"(target≈0.0)  {'✓' if val['test2_pass'] else '✗ FAIL'}")
print(f"Test 3 — Orthogonal rotation : CKA={val['cka_rotated']:.6f}  "
      f"(target={val['cka_identical']:.6f})  {'✓' if val['test3_pass'] else '✗ FAIL'}")
print(f"\nAll tests passed: {'✓ YES — proceed' if val['all_pass'] else '✗ NO — fix before Week 6'}")

if not val["all_pass"]:
    print("\n⚠ Most common error: missing double-centering of Gram matrix.")
    print("  Check _center_gram() in src/cka.py — the formula is H @ K @ H.")
    raise AssertionError("CKA validation failed. Do not proceed to Week 6.")

# Save validation results
with open("../results/cka_validation.txt", "w") as f:
    f.write(f"CKA Validation Results\n")
    f.write(f"======================\n")
    f.write(f"Test 1 (identical):  {val['cka_identical']:.6f}  {'PASS' if val['test1_pass'] else 'FAIL'}\n")
    f.write(f"Test 2 (random):     {val['cka_random']:.6f}  {'PASS' if val['test2_pass'] else 'FAIL'}\n")
    f.write(f"Test 3 (rotated):    {val['cka_rotated']:.6f}  {'PASS' if val['test3_pass'] else 'FAIL'}\n")
    f.write(f"All pass:            {'YES' if val['all_pass'] else 'NO'}\n")
print("\nSaved: results/cka_validation.txt")

# %% — Step 2: Measure CKA feature drift under fine-tuning
print("\n" + "=" * 50)
print("CKA FEATURE DRIFT — Fine-tuning across similarity conditions")
print("=" * 50)
print("Prediction: drift should INCREASE with similarity (lower CKA = more drift)")
print("If this prediction holds, it validates the similarity manipulation.\n")

CONDITIONS = ["low", "medium", "high"]
SEEDS      = [0, 1, 2]    # average over 3 seeds for stability
config     = DEFAULT_CONFIG.copy()

drift_results = {}   # condition → list of CKA(task0 → task4) values

for condition in CONDITIONS:
    cka_vals = []
    print(f"  Condition: {condition} ...")

    for seed in SEEDS:
        set_seeds(seed)

        train_loaders, test_loaders, _ = get_split_mnist(
            similarity=condition, data_dir=config["data_dir"],
            batch_size=config["batch_size"]
        )
        probe_x, _ = get_probe_set(test_loaders, n_per_task=100, seed=seed)

        model     = MLP().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        snapshots = {}    # task_idx → layer-2 activations

        for task_idx, train_loader in enumerate(train_loaders):
            for _ in range(config["epochs_per_task"]):
                _train_epoch(model, train_loader, optimizer, criterion,
                             device=DEVICE)
            snapshots[task_idx] = extract_activations(model, probe_x, DEVICE)

        # Compute CKA between task 0 and task 4 representations
        cka_drift = linear_cka(snapshots[0], snapshots[4])
        cka_vals.append(cka_drift)
        print(f"    Seed {seed}: CKA(T0→T4) = {cka_drift:.4f}")

    drift_results[condition] = cka_vals
    print(f"    Mean CKA(T0→T4) = {np.mean(cka_vals):.4f} ± {np.std(cka_vals):.4f}")

# %% — Plot CKA drift by condition
means  = [np.mean(drift_results[c]) for c in CONDITIONS]
stds   = [np.std(drift_results[c])  for c in CONDITIONS]
colors = ["#2196F3", "#4CAF50", "#FF5722"]   # Low/Medium/High

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(CONDITIONS, means, yerr=stds, capsize=6,
              color=colors, alpha=0.8, edgecolor="black", linewidth=0.8)
ax.set_xlabel("Similarity Condition", fontsize=12)
ax.set_ylabel("CKA (Task 0 → Task 4) — Layer 2\n↑ higher = less drift", fontsize=12)
ax.set_title("Representational Drift Under Fine-tuning\n"
             "Lower CKA = more feature drift after sequential training", fontsize=13)
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4, label="No drift (CKA=1)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("../results/figures/cka_finetune_conditions.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: results/figures/cka_finetune_conditions.png")

# %% — First hypothesis check
print("\n--- FIRST HYPOTHESIS CHECK ---")
print("Expected: CKA decreases (drift increases) from Low → Medium → High condition")
print(f"  Low  : {np.mean(drift_results['low']):.4f}")
print(f"  Med  : {np.mean(drift_results['medium']):.4f}")
print(f"  High : {np.mean(drift_results['high']):.4f}")
monotone = (np.mean(drift_results['low']) > np.mean(drift_results['medium']) >
            np.mean(drift_results['high']))
print(f"  Monotone drift pattern: {'✓ YES' if monotone else '✗ NO — check similarity manipulation'}")
