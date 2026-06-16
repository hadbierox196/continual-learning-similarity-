"""
Inter-task similarity measurement and task-pairing design.

Approach:
  1. Compute the mean image for each of the 10 MNIST digit classes.
  2. Build a 10×10 pairwise cosine-similarity matrix from these means.
  3. Use a greedy matching algorithm to find the 5 digit pairs that
     maximise (HIGH condition) or minimise (LOW condition) within-pair
     similarity, exhausting all 10 digits with no repeats.
  4. Validate that the three conditions (Low / Medium / High) are
     statistically separable using a one-way ANOVA on within-pair
     similarities (target: p < 0.05, ideally p < 0.01).

Pixel-space cosine similarity is used here. If the ANOVA fails (Week 4
pivot trigger), switch to feature-space similarity by extracting class
means from a frozen pre-trained simple CNN.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader


# ── Class-mean computation ─────────────────────────────────────────────────────

def compute_class_means(dataset, n_classes: int = 10) -> torch.Tensor:
    """
    Compute per-class mean images from the full MNIST train set.

    Args:
        dataset   : torchvision MNIST dataset (train)
        n_classes : number of digit classes (10 for MNIST)

    Returns:
        means : FloatTensor [10, 784] — one flattened mean image per class
    """
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)

    sums   = torch.zeros(n_classes, 784)
    counts = torch.zeros(n_classes)

    for x, y in loader:
        x_flat = x.view(x.size(0), -1)         # [batch, 784]
        for c in range(n_classes):
            mask = (y == c)
            if mask.any():
                sums[c]   += x_flat[mask].sum(dim=0)
                counts[c] += mask.sum()

    means = sums / counts.unsqueeze(1).clamp(min=1)
    return means                                # [10, 784]


# ── Similarity matrix ──────────────────────────────────────────────────────────

def cosine_similarity_matrix(means: torch.Tensor) -> np.ndarray:
    """
    Build a 10×10 pairwise cosine-similarity matrix.

    Args:
        means : FloatTensor [10, 784]

    Returns:
        sim_mat : ndarray [10, 10], values in [-1, 1]
    """
    m = means.numpy()
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    m_norm = m / np.maximum(norms, 1e-8)
    return m_norm @ m_norm.T                    # [10, 10]


# ── Greedy pairing ─────────────────────────────────────────────────────────────

def greedy_pair(sim_mat: np.ndarray, mode: str = "high") -> tuple[list, float]:
    """
    Greedy matching to find 5 non-overlapping digit pairs.

    At each step select the globally highest (mode='high') or lowest
    (mode='low') similarity pair among remaining unpaired digits.

    Args:
        sim_mat : [10, 10] cosine similarity matrix
        mode    : "high" | "low"

    Returns:
        pairs        : list of 5 (class_a, class_b) tuples, class_a < class_b
        mean_sim     : float — average within-pair cosine similarity
    """
    work = sim_mat.copy().astype(float)
    np.fill_diagonal(work, np.nan)          # exclude self-pairs

    available = set(range(10))
    pairs     = []

    while len(available) >= 2:
        best_val = -np.inf if mode == "high" else np.inf
        best_pair = None

        for a in sorted(available):
            for b in sorted(available):
                if b <= a:
                    continue
                v = work[a, b]
                if np.isnan(v):
                    continue
                if (mode == "high" and v > best_val) or \
                   (mode == "low"  and v < best_val):
                    best_val  = v
                    best_pair = (a, b)

        if best_pair is None:
            break
        pairs.append(best_pair)
        available.discard(best_pair[0])
        available.discard(best_pair[1])

    mean_sim = float(np.mean([sim_mat[a, b] for a, b in pairs]))
    return pairs, mean_sim


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_conditions(
    low_pairs:  list,
    med_pairs:  list,
    high_pairs: list,
    sim_mat:    np.ndarray,
) -> dict:
    """
    One-way ANOVA on within-pair similarities across the three conditions.

    Target: F-statistic large, p < 0.05 (ideally p < 0.01), and
    low_mean < med_mean < high_mean with a gap of at least 0.05.

    Returns:
        dict with per-condition means, similarity lists, F-stat, and p-value.
    """
    from scipy import stats

    low_sims  = [sim_mat[a, b] for a, b in low_pairs]
    med_sims  = [sim_mat[a, b] for a, b in med_pairs]
    high_sims = [sim_mat[a, b] for a, b in high_pairs]

    f_stat, p_val = stats.f_oneway(low_sims, med_sims, high_sims)

    return {
        "low_mean":   float(np.mean(low_sims)),
        "med_mean":   float(np.mean(med_sims)),
        "high_mean":  float(np.mean(high_sims)),
        "low_sims":   low_sims,
        "med_sims":   med_sims,
        "high_sims":  high_sims,
        "f_stat":     float(f_stat),
        "p_val":      float(p_val),
        "monotone":   float(np.mean(low_sims)) < float(np.mean(med_sims)) < float(np.mean(high_sims)),
        "gap_lo_hi":  float(np.mean(high_sims)) - float(np.mean(low_sims)),
        "valid":      p_val < 0.05,
    }
