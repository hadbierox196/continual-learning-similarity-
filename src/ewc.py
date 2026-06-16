"""
Elastic Weight Consolidation — Kirkpatrick et al., PNAS 2017.

Implementation choices (each is a deliberate decision, not a default):

1. ONLINE EWC: Fisher matrices are *accumulated* (summed) across tasks.
   This prevents the penalty from referencing only the most recent task.
   Alternative (separate Fishers per task) scales linearly in memory.

2. EMPIRICAL FISHER: Fisher is estimated using the model's own predicted
   labels rather than the true labels. This is the standard practice in CL
   and equivalent to the gradient of the log-likelihood under the model's
   current distribution.

3. DIAGONAL APPROXIMATION: Only the diagonal of the FIM is stored, one
   scalar per parameter. Full FIM is quadratic in parameter count (~impossible
   for any non-trivial model).

4. NORMALIZATION: Fisher values are averaged over n_samples, not summed.
   This keeps the effective λ interpretable across different task dataset sizes.

Key diagnostic: after calling compute_fisher(), call fisher_stats() and confirm
that < 20% of Fisher values are near-zero. If >80% are near-zero, the gradient
computation is broken (most common cause: zero_grad() called at wrong point).
"""

import torch
import torch.nn.functional as F


class EWC:
    """Online EWC regularizer for a given model."""

    def __init__(self, model: torch.nn.Module, lambda_ewc: float = 100.0):
        """
        Args:
            model      : the MLP being trained (reference, not a copy)
            lambda_ewc : regularization strength; sweep this in Week 2
        """
        self.model      = model
        self.lambda_ewc = lambda_ewc

        # Accumulated diagonal Fisher across all tasks seen so far
        self.fisher: dict[str, torch.Tensor] = {}
        # Optimal parameters θ* after the most recent task
        self.optpar: dict[str, torch.Tensor] = {}

    # ── Fisher estimation ─────────────────────────────────────────────────────

    def compute_fisher(
        self,
        dataloader,
        n_samples: int = 1000,
        device:    str = "cpu",
    ) -> dict:
        """
        Estimate the diagonal Fisher IM for the *current* task and accumulate.

        MUST be called AFTER training on a task is complete, BEFORE the next
        task's training loop begins.

        Fisher_i ≈ (1/N) Σ_n [ ∂ log p(ŷ_n | x_n) / ∂ θ_i ]²
        where ŷ_n = argmax_c f(x_n) (empirical Fisher).

        Args:
            dataloader : train loader for the task just completed
            n_samples  : number of examples to use for estimation (default 1000)
            device     : "cuda" or "cpu"

        Returns:
            fisher_new : per-task (unaccumulated) Fisher dict for diagnostics
        """
        self.model.eval()

        # Initialise per-task Fisher accumulator
        fisher_new = {
            n: torch.zeros_like(p, device=device)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        n_seen = 0
        for x, _ in dataloader:            # true labels not used (empirical Fisher)
            if n_seen >= n_samples:
                break
            x = x.to(device)
            bs = x.size(0)

            self.model.zero_grad()
            logits    = self.model(x)
            log_probs = F.log_softmax(logits, dim=1)

            # Use model's own predictions as pseudo-labels
            pseudo_y = logits.detach().argmax(dim=1)
            loss = F.nll_loss(log_probs, pseudo_y)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_new[n] += (p.grad.detach() ** 2) * bs

            n_seen += bs

        # Normalise by number of samples processed
        n_seen = max(n_seen, 1)
        for n in fisher_new:
            fisher_new[n] /= n_seen

        # Accumulate into the global Fisher (online EWC)
        for n, f in fisher_new.items():
            if n in self.fisher:
                self.fisher[n] = self.fisher[n] + f
            else:
                self.fisher[n] = f.clone()

        # Snapshot current parameters as θ*
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.optpar[n] = p.data.detach().clone()

        self.model.train()
        return fisher_new

    # ── Penalty term ──────────────────────────────────────────────────────────

    def penalty(self) -> torch.Tensor:
        """
        EWC regularisation term to ADD to the cross-entropy loss.

        L_total = L_CE  +  (λ/2) Σ_i F_i * (θ_i - θ*_i)²

        Returns 0.0 (on the model's device) if no task has been trained yet,
        so it is safe to call before any Fisher has been computed.
        """
        if not self.fisher:
            return torch.tensor(0.0)

        # Find the device from the first parameter
        device = next(self.model.parameters()).device
        loss = torch.zeros(1, device=device)

        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.fisher:
                f  = self.fisher[n].to(device)
                p0 = self.optpar[n].to(device)
                loss += (f * (p - p0).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * loss.squeeze()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def fisher_stats(self) -> dict:
        """
        Sanity check: Fisher values should be non-negative and mostly non-zero.
        If frac_near_zero > 0.8 the Fisher computation is broken.
        """
        if not self.fisher:
            return {"error": "No Fisher computed yet"}

        all_vals = torch.cat([f.flatten().cpu() for f in self.fisher.values()])
        return {
            "mean":          float(all_vals.mean()),
            "std":           float(all_vals.std()),
            "max":           float(all_vals.max()),
            "frac_near_zero": float((all_vals < 1e-6).float().mean()),
            "n_params":      int(all_vals.numel()),
        }

    def n_tasks_seen(self) -> int:
        """Number of tasks for which Fisher has been accumulated."""
        return len(self.optpar) > 0
