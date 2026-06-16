"""
Data pipeline for Split-MNIST continual learning experiments.
Supports Low / Medium / High inter-task similarity conditions.

IMPORTANT — Class-IL protocol:
  - All tasks share a single 10-class output head.
  - Original digit labels (0-9) are preserved; never remapped to 0/1.
  - Test evaluation uses all 10 logits simultaneously — no task identity.
"""

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
import json
from pathlib import Path

# ── Default task pairings ─────────────────────────────────────────────────────
# Medium: standard Split-MNIST ordering (van de Ven & Tolias 2019)
# Low / High: computed empirically in 04_similarity_design.py and stored in
#             results/task_pairings.json. Placeholders here are overwritten
#             after Week 4. Update these manually if you don't use the JSON.
_DEFAULT_PAIRINGS = {
    "medium": [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)],
    "low":    None,   # populated by 04_similarity_design.py
    "high":   None,   # populated by 04_similarity_design.py
}

_PAIRINGS_FILE = "results/task_pairings.json"


def load_task_pairings(pairings_file: str = _PAIRINGS_FILE) -> dict:
    """
    Load task pairings from JSON (written by 04_similarity_design.py).
    Falls back to defaults if file not found (medium only works at that point).
    """
    pairings = _DEFAULT_PAIRINGS.copy()
    path = Path(pairings_file)
    if path.exists():
        with open(path) as f:
            saved = json.load(f)
        for cond in ("low", "medium", "high"):
            if cond in saved:
                pairings[cond] = [tuple(p) for p in saved[cond]]
    return pairings


def save_task_pairings(low: list, medium: list, high: list,
                       pairings_file: str = _PAIRINGS_FILE):
    """Save computed task pairings to JSON for downstream notebooks."""
    Path(pairings_file).parent.mkdir(parents=True, exist_ok=True)
    data = {"low": low, "medium": medium, "high": high}
    with open(pairings_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Task pairings saved to {pairings_file}")


# ── MNIST loading ─────────────────────────────────────────────────────────────

def load_mnist(data_dir: str = "./data"):
    """Download and return raw MNIST train and test datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_ds, test_ds


# ── Task extraction ───────────────────────────────────────────────────────────

def _extract_classes(dataset, class_a: int, class_b: int):
    """
    Return (x, y) tensors for examples belonging to class_a or class_b.
    Labels are the ORIGINAL digit values — never remapped — to support the
    shared output head in Class-IL.
    """
    indices = [i for i, (_, y) in enumerate(dataset) if int(y) in (class_a, class_b)]
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    return x, y


def get_split_mnist(
    similarity:  str = "medium",
    data_dir:    str = "./data",
    batch_size:  int = 256,
    pairings_file: str = _PAIRINGS_FILE,
):
    """
    Build the Split-MNIST task sequence for a given similarity condition.

    Args:
        similarity:  "low" | "medium" | "high"
        data_dir:    where MNIST will be downloaded
        batch_size:  DataLoader batch size
        pairings_file: path to JSON produced by 04_similarity_design.py

    Returns:
        train_loaders  : list[DataLoader] — one per task (length 5)
        test_loaders   : list[DataLoader] — one per task (length 5)
        task_pairs     : list[tuple] — (class_a, class_b) for each task
    """
    pairings = load_task_pairings(pairings_file)
    task_pairs = pairings.get(similarity)
    if task_pairs is None:
        raise ValueError(
            f"Task pairings for similarity='{similarity}' not found. "
            f"Run 04_similarity_design.py first to generate {pairings_file}."
        )

    train_ds, test_ds = load_mnist(data_dir)
    train_loaders, test_loaders = [], []

    for class_a, class_b in task_pairs:
        x_tr, y_tr = _extract_classes(train_ds, class_a, class_b)
        x_te, y_te = _extract_classes(test_ds,  class_a, class_b)
        train_loaders.append(DataLoader(
            TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True,  num_workers=0
        ))
        test_loaders.append(DataLoader(
            TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False, num_workers=0
        ))

    return train_loaders, test_loaders, task_pairs


def get_joint_loader(train_loaders: list, batch_size: int = 256):
    """Combine all task train loaders into one shuffled loader (for joint training)."""
    all_datasets = [loader.dataset for loader in train_loaders]
    combined = ConcatDataset(all_datasets)
    return DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=0)


def get_probe_set(test_loaders: list, n_per_task: int = 100, seed: int = 42):
    """
    Build a fixed probe set for CKA measurements (500 examples total).

    The probe set must stay CONSTANT across all CKA measurements to ensure
    comparability. It is sampled once here and reused throughout.

    Returns:
        probe_x : FloatTensor [n_tasks * n_per_task, 1, 28, 28]
        probe_y : LongTensor  [n_tasks * n_per_task]
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for loader in test_loaders:
        x, y = next(iter(loader))
        n    = min(n_per_task, len(x))
        idx  = rng.choice(len(x), size=n, replace=False)
        xs.append(x[idx])
        ys.append(y[idx])
    return torch.cat(xs), torch.cat(ys)
