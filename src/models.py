"""
MLP for Split-MNIST — 784 → 256 → 256 → 10.

Design constraints:
  - Single shared 10-class output head (mandatory for Class-IL).
  - Returns layer-2 activations alongside logits for CKA measurement.
  - Xavier init ensures stable training across seeds.
"""

import torch
import torch.nn as nn
import copy


class MLP(nn.Module):
    """
    Three-layer MLP with ReLU activations and a shared output head.

    Architecture:
        Input  : 784  (flattened 28×28 MNIST image)
        Layer 1: Linear(784 → 256) + ReLU
        Layer 2: Linear(256 → 256) + ReLU   ← activations extracted here for CKA
        Output : Linear(256 → 10)            ← shared head, all 10 digit classes

    The shared head is the defining constraint of Class-IL: during test time the
    model must assign one of 10 class labels with no knowledge of which task the
    example came from.
    """

    def __init__(
        self,
        input_size:  int = 784,
        hidden_size: int = 256,
        output_size: int = 10,
    ):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1  = nn.Linear(input_size, hidden_size)
        self.fc2  = nn.Linear(hidden_size, hidden_size)
        self.fc3  = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x               : [batch, 784] or [batch, 1, 28, 28]
            return_features : if True, also return layer-2 activations

        Returns:
            logits   : [batch, 10]
            features : [batch, 256] — only when return_features=True
        """
        x  = x.view(x.size(0), -1)
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        logits = self.fc3(h2)

        if return_features:
            return logits, h2
        return logits

    # ── Convenience methods ───────────────────────────────────────────────────

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return only layer-2 activations (used by CKA module)."""
        _, h2 = self.forward(x, return_features=True)
        return h2

    def clone(self) -> "MLP":
        """Deep-copy the model with all parameter values preserved."""
        return copy.deepcopy(self)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
