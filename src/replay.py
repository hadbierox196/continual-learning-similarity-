"""
Experience Replay with reservoir sampling — Rolnick et al., NeurIPS 2019.

Correctness requirements (both are easy to violate):

1. ONLINE POPULATION: The buffer is filled *during* training as each example
   is seen. It must NOT be filled from the full previous task after training —
   that gives access to data unavailable in deployment and inflates ER's advantage.

2. INTERLEAVED TRAINING: Replay samples are mixed with current-task samples
   within the same batch and the same backward pass. They are NOT trained in
   separate phases. This is enforced by the training loop in experiment.py.

Reservoir sampling guarantee:
   After seeing k examples, each of the k examples has equal probability
   (min(B, k) / k) of being in the buffer of capacity B.
   This ensures uniform representation of all seen classes as long as
   class frequencies are roughly balanced — which holds for MNIST.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class ReservoirBuffer:
    """
    Fixed-capacity replay buffer using Vitter's reservoir sampling algorithm.

    The key invariant: for any example seen at position k (1-indexed),
    its probability of being in the buffer is exactly min(capacity, k) / k.
    """

    def __init__(self, capacity: int, seed: int = None):
        """
        Args:
            capacity : maximum number of (x, y) pairs stored
            seed     : RNG seed for reproducibility (set to the experiment seed)
        """
        self.capacity  = capacity
        self.rng       = np.random.default_rng(seed)
        self._xs: list = []
        self._ys: list = []
        self.n_seen    = 0           # total examples ever offered to the buffer

    # ── Core API ──────────────────────────────────────────────────────────────

    def update(self, x_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Offer a batch of examples to the buffer.
        Each example is individually accepted/rejected by reservoir rule.

        x_batch, y_batch : CPU tensors [batch, ...]
        """
        x_batch = x_batch.detach().cpu()
        y_batch = y_batch.detach().cpu()

        for xi, yi in zip(x_batch, y_batch):
            self.n_seen += 1
            if len(self._xs) < self.capacity:
                # Buffer not yet full — always accept
                self._xs.append(xi)
                self._ys.append(yi)
            else:
                # Replace with probability capacity / n_seen
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.capacity:
                    self._xs[j] = xi
                    self._ys[j] = yi

    def sample(self, n: int):
        """
        Draw n examples uniformly at random from the buffer (without replacement
        if n ≤ len(buffer), with replacement otherwise).

        Returns:
            (x_tensor, y_tensor) on CPU, or (None, None) if buffer is empty.
        """
        if len(self._xs) == 0:
            return None, None

        n   = min(n, len(self._xs))
        idx = self.rng.choice(len(self._xs), size=n, replace=False)
        x   = torch.stack([self._xs[i] for i in idx])
        y   = torch.stack([self._ys[i] for i in idx])
        return x, y

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def class_distribution(self) -> dict:
        """
        Returns count per class label currently in the buffer.
        All 10 classes should be roughly balanced after task 5.
        If early classes are absent, reservoir sampling is broken.
        """
        if not self._ys:
            return {}
        counts: dict = {}
        for y in self._ys:
            label = int(y.item())
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def to_dataloader(self, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
        """Export buffer contents as a DataLoader (for inspection or offline ER)."""
        x = torch.stack(self._xs)
        y = torch.stack(self._ys)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    def __len__(self) -> int:
        return len(self._xs)

    def __repr__(self) -> str:
        return (
            f"ReservoirBuffer(capacity={self.capacity}, "
            f"stored={len(self)}, seen={self.n_seen})"
        )
