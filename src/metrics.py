"""
Continual learning evaluation metrics — Class-IL protocol.

Backward Transfer (BWT):
    BWT = (1 / T-1) * sum_{i=0}^{T-2} [ R[i, T-1] - R[i, i] ]
    where R[i, j] = accuracy on task i's test set after training on task j.
    Negative BWT indicates forgetting. More negative = more catastrophic.

All evaluation uses the shared 10-class output head with no task identity
information at test time, consistent with Class-IL (van de Ven & Tolias 2019).
"""

import numpy as np
import torch


class TaskAccuracyMatrix:
    """
    Records R[i, j]: accuracy on task i after training through task j.

    After sequential training on T tasks, the matrix has:
      - R[i, i] : accuracy immediately after learning task i (diagonal)
      - R[i, T-1]: accuracy on task i after all tasks (rightmost column)
      - R[i, i] - R[i, T-1]: forgetting for task i

    Class-IL convention: evaluate ALL 10 logits; argmax over all classes.
    No masking, no task-specific softmax — the model must distinguish all digits.
    """

    def __init__(self, n_tasks: int = 5):
        self.n_tasks = n_tasks
        self.R = np.full((n_tasks, n_tasks), np.nan)

    def evaluate_task(
        self,
        model,
        test_loader,
        device: str = "cpu",
    ) -> float:
        """
        Evaluate model accuracy on one task's test set (Class-IL protocol).
        Uses shared output head — no task masking.
        """
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)            # full 10-class logits
                preds  = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        return correct / max(total, 1)

    def record_after_task(
        self,
        model,
        test_loaders: list,
        current_task: int,
        device: str = "cpu",
    ):
        """
        Evaluate model on all tasks 0 … current_task and store in R.
        Call this immediately after finishing training on `current_task`.
        """
        for i in range(current_task + 1):
            acc = self.evaluate_task(model, test_loaders[i], device)
            self.R[i, current_task] = acc
        return self

    # ── Aggregate metrics ─────────────────────────────────────────────────────

    @property
    def bwt(self) -> float:
        """
        Backward Transfer. Negative = forgetting.
        Only computed once all tasks have been trained (uses final column).
        """
        T = self.n_tasks
        vals = []
        for i in range(T - 1):
            r_final  = self.R[i, T - 1]
            r_diag   = self.R[i, i]
            if not (np.isnan(r_final) or np.isnan(r_diag)):
                vals.append(r_final - r_diag)
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def average_accuracy(self) -> float:
        """Mean accuracy across all tasks after training on the last task."""
        T = self.n_tasks
        vals = [self.R[i, T - 1] for i in range(T) if not np.isnan(self.R[i, T - 1])]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def per_task_forgetting(self) -> np.ndarray:
        """Per-task forgetting: R[i,i] - R[i, T-1] for i in 0..T-2."""
        T = self.n_tasks
        return np.array([self.R[i, i] - self.R[i, T - 1] for i in range(T - 1)])

    def to_dict(self) -> dict:
        return {
            "R_matrix":        self.R.tolist(),
            "bwt":             self.bwt,
            "average_accuracy": self.average_accuracy,
            "per_task_forgetting": self.per_task_forgetting.tolist(),
        }
