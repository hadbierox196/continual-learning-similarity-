"""
Linear Centered Kernel Alignment (CKA) — Kornblith et al., ICML 2019.
arXiv: 1905.00414

CKA measures the similarity between two representational geometries.
Unlike comparing weight matrices directly, CKA is invariant to orthogonal
transformations and isotropic scaling, making it a principled measure of
whether two sets of activations encode the *same information* regardless
of how individual neurons are oriented.

Linear CKA Formula:
    CKA(X, Y) = HSIC(XX^T, YY^T) / sqrt( HSIC(XX^T, XX^T) * HSIC(YY^T, YY^T) )

HSIC (Hilbert-Schmidt Independence Criterion):
    HSIC(K, L) = (1/n^2) tr(K_c L_c)
    where K_c = HKH, H = I - (1/n)11^T  (double-centering matrix)

Properties:
    CKA(X, X) = 1.0            ← identical geometry
    CKA(X, Y_random) ≈ 0.0    ← unrelated geometry
    CKA(X, XQ) = CKA(X, X)    ← invariant to orthogonal Q

Practical note: CKA requires ≥ 200 samples to be reliable. We use a
fixed probe set of 500 examples (100 per task × 5 tasks) for all measurements.
"""

import numpy as np
import torch


# ── Core math ─────────────────────────────────────────────────────────────────

def _center_gram(K: np.ndarray) -> np.ndarray:
    """Double-center a Gram matrix: K_c = H K H where H = I - (1/n)11^T."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic(K: np.ndarray, L: np.ndarray) -> float:
    """
    HSIC between two Gram matrices.
    HSIC(K, L) = (1/n²) Σ_{i,j} (K_c)_{ij} (L_c)_{ij}
    """
    K_c = _center_gram(K)
    L_c = _center_gram(L)
    return float(np.sum(K_c * L_c))   # = tr(K_c @ L_c), computed element-wise


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between activation matrices X and Y.

    Args:
        X : [n_samples, d1] — activations from representation 1
        Y : [n_samples, d2] — activations from representation 2
            (n_samples must match; d1 and d2 can differ)

    Returns:
        CKA value in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], \
        f"Sample count mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}"
    assert X.shape[0] >= 2, "Need at least 2 samples for CKA"

    K = X @ X.T    # [n, n] linear Gram matrix
    L = Y @ Y.T    # [n, n] linear Gram matrix

    hsic_kl = _hsic(K, L)
    hsic_kk = _hsic(K, K)
    hsic_ll = _hsic(L, L)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-12:
        return 0.0

    return float(np.clip(hsic_kl / denom, 0.0, 1.0))


# ── Activation extraction ─────────────────────────────────────────────────────

def extract_activations(
    model,
    probe_x: torch.Tensor,
    device:  str = "cpu",
) -> np.ndarray:
    """
    Extract layer-2 activations from the MLP on a fixed probe set.

    Args:
        model   : MLP instance (must implement get_features())
        probe_x : [n_samples, 1, 28, 28] or [n_samples, 784]
        device  : "cuda" | "cpu"

    Returns:
        activations : ndarray [n_samples, 256]
    """
    model.eval()
    with torch.no_grad():
        probe_x = probe_x.to(device)
        h2 = model.get_features(probe_x)   # [n, 256]
    return h2.cpu().numpy()


# ── Validation suite ──────────────────────────────────────────────────────────

def validate_cka(n: int = 500, d: int = 256, seed: int = 0) -> dict:
    """
    Three unit tests that every correct CKA implementation must pass.

    Test 1 — Identical matrices  : CKA(X, X) should equal 1.0 (±0.001)
    Test 2 — Random matrices     : CKA(X, Y) should be ≈ 0.0 (±0.05) for
                                   independent Gaussian X, Y
    Test 3 — Orthogonal rotation : CKA(X, XQ) should equal CKA(X, X) = 1.0
                                   for any orthogonal matrix Q

    If any test fails, the implementation is incorrect. Most common error:
    forgetting to double-center the Gram matrix (the H @ K @ H step).
    """
    rng = np.random.default_rng(seed)

    X        = rng.standard_normal((n, d)).astype(np.float64)
    Y_rand   = rng.standard_normal((n, d)).astype(np.float64)
    Q, _     = np.linalg.qr(rng.standard_normal((d, d)))
    X_rot    = X @ Q

    cka_identical = linear_cka(X, X)
    cka_random    = linear_cka(X, Y_rand)
    cka_rotated   = linear_cka(X, X_rot)

    return {
        "cka_identical": cka_identical,           # target: 1.0
        "cka_random":    cka_random,              # target: ~0.0
        "cka_rotated":   cka_rotated,             # target: same as cka_identical
        "test1_pass":    abs(cka_identical - 1.0) < 0.001,
        "test2_pass":    abs(cka_random)           < 0.05,
        "test3_pass":    abs(cka_rotated - cka_identical) < 0.001,
        "all_pass":      (
            abs(cka_identical - 1.0) < 0.001 and
            abs(cka_random)          < 0.05  and
            abs(cka_rotated - cka_identical) < 0.001
        ),
    }
