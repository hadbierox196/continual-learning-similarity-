"""
Unified experiment runner for the full factorial design.

Factorial: 4 methods × 3 similarity conditions × 10 seeds = 120 runs.

Each run returns a results dict containing:
  - R_matrix      : 5×5 accuracy matrix (Class-IL, shared head)
  - bwt           : Backward Transfer (negative = forgetting)
  - avg_accuracy  : mean per-task accuracy after final task
  - cka_values    : representational drift measurements (task0→final, etc.)
  - config        : full hyperparameter snapshot for reproducibility

Checkpointing: results are saved as individual .pkl files after every run.
A crashed session loses at most one run. Use check_completion() to resume.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
from pathlib import Path

from src.data      import get_split_mnist, get_joint_loader, get_probe_set
from src.models    import MLP
from src.metrics   import TaskAccuracyMatrix
from src.ewc       import EWC
from src.replay    import ReservoirBuffer
from src.cka       import extract_activations, linear_cka


# ── Default hyperparameters ────────────────────────────────────────────────────
# Update ewc_lambda after Week 2 sweep; update er_buffer_size after Week 3.

DEFAULT_CONFIG = {
    "lr":               1e-3,
    "batch_size":       256,
    "epochs_per_task":  10,
    "optimizer":        "adam",
    "ewc_lambda":       100.0,      # ← update after 02_ewc.py
    "er_buffer_size":   500,        # ← update after 03_experience_replay.py
    "er_replay_ratio":  1.0,        # current-task : replay batch size ratio
    "fisher_n_samples": 1000,
    "cka_probe_n":      100,        # samples per task for the probe set
    "n_tasks":          5,
    "input_size":       784,
    "hidden_size":      256,
    "output_size":      10,
    "clip_grad_norm":   1.0,        # gradient clipping; prevents divergence
    "data_dir":         "./data",
    "checkpoint_dir":   "results/raw",
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
}


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seeds(seed: int):
    """Fix ALL sources of randomness for a fully reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Single-epoch training ──────────────────────────────────────────────────────

def _train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    ewc_agent=None,
    replay_buffer=None,
    er_ratio: float = 1.0,
    clip_norm: float = 1.0,
    device: str = "cpu",
) -> float:
    """
    One training epoch. Handles EWC penalty and interleaved experience replay.

    Returns average loss for the epoch (diagnostic only).
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # ── Current task loss
        logits = model(x)
        loss   = criterion(logits, y)

        # ── EWC penalty (0 before any Fisher is computed)
        if ewc_agent is not None:
            loss = loss + ewc_agent.penalty()

        # ── Experience replay: interleave replay samples in the same backward pass
        if replay_buffer is not None and len(replay_buffer) > 0:
            n_replay = max(1, int(x.size(0) * er_ratio))
            rx, ry   = replay_buffer.sample(n_replay)
            if rx is not None:
                rx, ry = rx.to(device), ry.to(device)
                loss   = loss + criterion(model(rx), ry)

        loss.backward()
        if clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── Main experiment function ───────────────────────────────────────────────────

def run_experiment(
    method:     str,
    similarity: str,
    seed:       int,
    config:     dict = None,
) -> dict:
    """
    Run one continual learning experiment and return results.

    Args:
        method     : "finetune" | "ewc" | "er" | "joint"
        similarity : "low" | "medium" | "high"
        seed       : integer 0–9
        config     : hyperparameter dict (DEFAULT_CONFIG used if None)

    Returns:
        results dict — see module docstring for keys
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    set_seeds(seed)
    device = config["device"]

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loaders, test_loaders, task_pairs = get_split_mnist(
        similarity=similarity,
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
    )
    probe_x, _ = get_probe_set(
        test_loaders, n_per_task=config["cka_probe_n"], seed=seed
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MLP(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    # ── Method-specific objects ────────────────────────────────────────────────
    ewc_agent     = EWC(model, lambda_ewc=config["ewc_lambda"]) if method == "ewc" else None
    replay_buffer = ReservoirBuffer(capacity=config["er_buffer_size"], seed=seed) \
                    if method == "er" else None

    # ── Joint training (special case) ─────────────────────────────────────────
    if method == "joint":
        joint_loader = get_joint_loader(train_loaders, config["batch_size"])
        total_epochs = config["epochs_per_task"] * config["n_tasks"]
        for _ in range(total_epochs):
            _train_epoch(model, joint_loader, optimizer, criterion, device=device)

        matrix = TaskAccuracyMatrix(config["n_tasks"])
        for i in range(config["n_tasks"]):
            acc = matrix.evaluate_task(model, test_loaders[i], device)
            for j in range(config["n_tasks"]):
                matrix.R[i, j] = acc      # joint: evaluated once, same for all j

        acts_final = extract_activations(model, probe_x, device)
        return {
            "method": method, "similarity": similarity, "seed": seed,
            "R_matrix": matrix.R.tolist(),
            "bwt":      0.0,               # joint has no forgetting by definition
            "avg_accuracy": float(matrix.average_accuracy),
            "cka_values":   {"task0_to_final": 1.0},
            "task_pairs":   task_pairs,
            "config":       config,
        }

    # ── Sequential training: Finetune / EWC / ER ──────────────────────────────
    matrix      = TaskAccuracyMatrix(config["n_tasks"])
    cka_snapshots: dict = {}            # task_idx → ndarray of activations

    for task_idx, train_loader in enumerate(train_loaders):

        # Train for N epochs
        for _ in range(config["epochs_per_task"]):
            _train_epoch(
                model, train_loader, optimizer, criterion,
                ewc_agent=ewc_agent,
                replay_buffer=replay_buffer if method == "er" else None,
                er_ratio=config["er_replay_ratio"],
                clip_norm=config["clip_grad_norm"],
                device=device,
            )

        # Snapshot layer-2 activations on the fixed probe set
        cka_snapshots[task_idx] = extract_activations(model, probe_x, device)

        # Post-task: EWC Fisher update (must happen AFTER the task's training)
        if method == "ewc":
            ewc_agent.compute_fisher(
                train_loader,
                n_samples=config["fisher_n_samples"],
                device=device,
            )

        # Post-task: populate replay buffer
        if method == "er":
            for x_b, y_b in train_loader:
                replay_buffer.update(x_b, y_b)

        # Evaluate all tasks seen so far (Class-IL, shared head)
        matrix.record_after_task(model, test_loaders, task_idx, device)

    # ── CKA drift metrics ─────────────────────────────────────────────────────
    cka_values: dict = {}
    n = config["n_tasks"]
    if 0 in cka_snapshots and (n - 1) in cka_snapshots:
        cka_values["task0_to_final"] = linear_cka(
            cka_snapshots[0], cka_snapshots[n - 1]
        )
    for t in range(1, n):
        if (t - 1) in cka_snapshots and t in cka_snapshots:
            cka_values[f"task{t-1}_to_task{t}"] = linear_cka(
                cka_snapshots[t - 1], cka_snapshots[t]
            )

    return {
        "method":       method,
        "similarity":   similarity,
        "seed":         seed,
        "R_matrix":     matrix.R.tolist(),
        "bwt":          float(matrix.bwt),
        "avg_accuracy": float(matrix.average_accuracy),
        "cka_values":   cka_values,
        "task_pairs":   task_pairs,
        "config":       config,
    }


# ── Checkpoint I/O ─────────────────────────────────────────────────────────────

def save_result(result: dict, checkpoint_dir: str = "results/raw") -> str:
    """Persist one result dict as a .pkl file. Returns the file path."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    m, s, seed = result["method"], result["similarity"], result["seed"]
    path = f"{checkpoint_dir}/{m}_{s}_{seed:02d}.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return path


def load_all_results(checkpoint_dir: str = "results/raw") -> list:
    """Load every completed result dict from the checkpoint directory."""
    results = []
    for p in Path(checkpoint_dir).glob("*.pkl"):
        with open(p, "rb") as f:
            results.append(pickle.load(f))
    return results


def check_completion(
    methods:     list,
    similarities: list,
    seeds:       list,
    checkpoint_dir: str = "results/raw",
) -> list:
    """
    Scan checkpoint directory and print/return remaining (method, sim, seed) combos.
    Use this at the top of each Colab session to know exactly what to run.
    """
    completed = set()
    for p in Path(checkpoint_dir).glob("*.pkl"):
        parts = p.stem.split("_")
        # stem format: method_similarity_seed  (e.g. ewc_high_03)
        if len(parts) >= 3:
            completed.add("_".join(parts))

    remaining = []
    for m in methods:
        for sim in similarities:
            for seed in seeds:
                key = f"{m}_{sim}_{seed:02d}"
                if key not in completed:
                    remaining.append((m, sim, seed))

    total = len(methods) * len(similarities) * len(seeds)
    print(f"✓ Completed : {len(completed)} / {total}")
    print(f"↻ Remaining : {len(remaining)}")
    return remaining
