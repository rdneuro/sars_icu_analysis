# -*- coding: utf-8 -*-
"""
sars.tractogram_bootstrap.core
==================================================

Core engine for probabilistic connectomics via SIFT2 tractogram bootstrap.

The standard connectomics pipeline treats structural connectivity as
deterministic: SC_ab = Σ wᵢ for streamlines connecting parcels a↔b.
But this is one realization of a stochastic process (seeding, tracking,
filtering), and the uncertainty is real but universally ignored.

This module quantifies that uncertainty via weighted bootstrap of the
SIFT2 tractogram.  The bootstrap operates on (streamline, weight)
pairs, preserving the joint distribution of (endpoints, weight):

    For b = 1, …, B:
        Sample N streamlines with replacement from the tractogram
        Assign each sampled streamline its SIFT2 weight
        Build SC⁽ᵇ⁾ by aggregating into the parcellation

    Uncertainty at edge (a,b) = variability of SC⁽ᵇ⁾_{ab} across b

This is valid because streamlines are approximately independent
conditional on FODs, and SIFT2 weights are tied to individual
streamlines.

Compatible with any MRtrix3 pipeline:
    tckgen → tcksift2 → tck2connectome -out_assignments

Classes
-------
StreamlineAssignment
    Pre-computed mapping of streamlines to (parcel_a, parcel_b, weight).
EdgeStats
    Bootstrap statistics for a single SC edge.
BootstrapResult
    Complete result of the tractogram bootstrap.

Functions
---------
load_streamline_assignments
    Load from MRtrix3 output files (connectome, weights, assignments).
create_assignments_from_sc
    Create synthetic streamline assignments from an SC matrix.
build_sc_from_streamlines
    Build symmetric SC matrix from streamline assignments.
bootstrap_tractogram
    Run the weighted bootstrap to estimate SC uncertainty.
classify_edges
    Classify edges as robust / present / fragile / spurious.

References
----------
- Tournier et al. (2019). NeuroImage 202:116137. MRtrix3.
- Smith et al. (2015). NeuroImage 121:176-185. SIFT2.
- Efron & Tibshirani (1993). An Introduction to the Bootstrap.
  Chapman & Hall.
"""

import numpy as np
from scipy import stats
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StreamlineAssignment:
    """
    Pre-computed assignment of streamlines to parcel pairs.

    This is the key data structure: each streamline maps to
    (parcel_a, parcel_b, sift2_weight).  Built once from the
    ``tck2connectome -out_assignments`` file or computed from endpoints.

    Parameters
    ----------
    n_streamlines : int
        Total number of assigned streamlines.
    n_parcels : int
        Number of brain parcels.
    parcel_a : np.ndarray (n_streamlines,)
        Source parcel index (0-indexed).
    parcel_b : np.ndarray (n_streamlines,)
        Target parcel index (0-indexed).
    weights : np.ndarray (n_streamlines,)
        SIFT2 weights.
    parcel_labels : list
        Human-readable parcel names.
    """

    n_streamlines: int
    n_parcels: int
    parcel_a: np.ndarray
    parcel_b: np.ndarray
    weights: np.ndarray
    parcel_labels: list = field(default_factory=list)

    def __post_init__(self):
        assert len(self.parcel_a) == self.n_streamlines
        assert len(self.parcel_b) == self.n_streamlines
        assert len(self.weights) == self.n_streamlines


@dataclass
class EdgeStats:
    """
    Bootstrap statistics for a single SC edge.

    Parameters
    ----------
    mean : float
    std : float
    cv : float
        Coefficient of variation (std / mean).  High CV = unreliable.
    ci_low : float
        Lower bound of 95% CI.
    ci_high : float
        Upper bound of 95% CI.
    median : float
    iqr : float
        Interquartile range.
    prob_nonzero : float
        Fraction of bootstrap samples where this edge is nonzero.
        Low probability = edge may be spurious.
    """

    mean: float
    std: float
    cv: float
    ci_low: float
    ci_high: float
    median: float
    iqr: float
    prob_nonzero: float


@dataclass
class BootstrapResult:
    """
    Complete result of the tractogram bootstrap.

    Parameters
    ----------
    sc_mean : np.ndarray (N, N)
        Mean SC across bootstrap samples.
    sc_std : np.ndarray (N, N)
        Standard deviation.
    sc_cv : np.ndarray (N, N)
        Coefficient of variation.
    sc_ci_low : np.ndarray (N, N)
        Lower 95% CI bound.
    sc_ci_high : np.ndarray (N, N)
        Upper 95% CI bound.
    prob_nonzero : np.ndarray (N, N)
        Probability that edge is nonzero.
    n_bootstrap : int
    n_streamlines : int
    n_parcels : int
    sc_samples : np.ndarray (B, N, N), optional
        All bootstrap SC matrices (if stored).
    """

    sc_mean: np.ndarray
    sc_std: np.ndarray
    sc_cv: np.ndarray
    sc_ci_low: np.ndarray
    sc_ci_high: np.ndarray
    prob_nonzero: np.ndarray
    n_bootstrap: int
    n_streamlines: int
    n_parcels: int
    sc_samples: Optional[np.ndarray] = field(default=None, repr=False)


# =============================================================================
# LOADING FUNCTIONS (MRtrix3 interface)
# =============================================================================

def load_streamline_assignments(
    connectome_csv: str,
    weights_csv: str,
    assignments_txt: str,
    n_parcels: int,
    parcel_labels: Optional[list] = None,
    zero_indexed: bool = False,
) -> StreamlineAssignment:
    """
    Load streamline-to-parcel assignments from MRtrix3 output files.

    This uses the output of::

        tck2connectome tracks.tck parcellation.nii.gz connectome.csv \\
            -out_assignments assignments.txt

    And SIFT2 weights from::

        tcksift2 tracks.tck wmfod.mif tracks_sift2.tck \\
            -out_mu mu.txt -csv_output weights.csv

    Parameters
    ----------
    connectome_csv : str
        Path to connectome matrix CSV (for validation, not directly used).
    weights_csv : str
        Path to SIFT2 weights CSV.  One weight per line, per streamline.
    assignments_txt : str
        Path to ``tck2connectome -out_assignments`` file.
        Each line: two integers (parcel_a, parcel_b) for one streamline.
        Lines with "0 0" = streamline not assigned (both endpoints
        outside parcels).
    n_parcels : int
        Number of parcels in the atlas.
    parcel_labels : list, optional
        Parcel names.
    zero_indexed : bool
        If True, parcel indices in assignments are 0-indexed.
        MRtrix3 default is 1-indexed (1 to N), with 0 = unassigned.

    Returns
    -------
    StreamlineAssignment
    """
    # Load SIFT2 weights
    weights = np.loadtxt(weights_csv)

    # Load assignments
    parcel_pairs = np.loadtxt(assignments_txt, dtype=int)

    if parcel_pairs.ndim == 1:
        parcel_pairs = parcel_pairs.reshape(1, -1)

    assert len(weights) == len(parcel_pairs), (
        f"Mismatch: {len(weights)} weights vs {len(parcel_pairs)} assignments"
    )

    pa = parcel_pairs[:, 0]
    pb = parcel_pairs[:, 1]

    if not zero_indexed:
        # MRtrix3 convention: 1-indexed, 0 = unassigned
        valid = (pa > 0) & (pb > 0)
        pa = pa[valid] - 1
        pb = pb[valid] - 1
        weights = weights[valid]
    else:
        valid = (pa >= 0) & (pb >= 0)
        pa = pa[valid]
        pb = pb[valid]
        weights = weights[valid]

    # Ensure canonical order (a <= b) for undirected edges
    swap = pa > pb
    pa[swap], pb[swap] = pb[swap].copy(), pa[swap].copy()

    # Filter out self-connections
    not_self = pa != pb
    pa = pa[not_self]
    pb = pb[not_self]
    weights = weights[not_self]

    return StreamlineAssignment(
        n_streamlines=len(weights),
        n_parcels=n_parcels,
        parcel_a=pa,
        parcel_b=pb,
        weights=weights,
        parcel_labels=parcel_labels or [],
    )


def create_assignments_from_sc(
    sc_matrix: np.ndarray,
    expand_factor: int = 100,
    seed: int = 42,
    parcel_labels: Optional[list] = None,
) -> StreamlineAssignment:
    """
    Create synthetic streamline assignments from an SC matrix.

    Useful for testing/prototyping when the SC matrix is available but
    not the raw tractogram.  Each SC entry is decomposed into multiple
    "virtual streamlines" whose weights sum to the original entry.

    Parameters
    ----------
    sc_matrix : np.ndarray (N, N)
        Structural connectivity matrix.
    expand_factor : int
        How many streamlines per nonzero edge.  Higher = more precise
        bootstrap but slower.  100 is reasonable for prototyping.
    seed : int
    parcel_labels : list, optional

    Returns
    -------
    StreamlineAssignment
    """
    rng = np.random.default_rng(seed)
    N = sc_matrix.shape[0]

    # Symmetrize and get upper triangle
    sc = (sc_matrix + sc_matrix.T) / 2.0
    np.fill_diagonal(sc, 0)

    all_a, all_b, all_w = [], [], []

    for i in range(N):
        for j in range(i + 1, N):
            if sc[i, j] > 0:
                n_sl = max(1, int(expand_factor * sc[i, j] / sc.max()))
                w_raw = rng.exponential(1.0, n_sl)
                w_normalized = w_raw / w_raw.sum() * sc[i, j]

                all_a.extend([i] * n_sl)
                all_b.extend([j] * n_sl)
                all_w.extend(w_normalized)

    return StreamlineAssignment(
        n_streamlines=len(all_w),
        n_parcels=N,
        parcel_a=np.array(all_a, dtype=int),
        parcel_b=np.array(all_b, dtype=int),
        weights=np.array(all_w),
        parcel_labels=parcel_labels or [],
    )


# =============================================================================
# CORE: Bootstrap engine
# =============================================================================

def build_sc_from_streamlines(
    parcel_a: np.ndarray,
    parcel_b: np.ndarray,
    weights: np.ndarray,
    n_parcels: int,
) -> np.ndarray:
    """Build symmetric SC matrix from streamline assignments."""
    sc = np.zeros((n_parcels, n_parcels))
    np.add.at(sc, (parcel_a, parcel_b), weights)
    sc = sc + sc.T
    np.fill_diagonal(sc, 0)
    return sc


def bootstrap_tractogram(
    assignments: StreamlineAssignment,
    n_bootstrap: int = 1000,
    store_samples: bool = False,
    ci_level: float = 0.95,
    seed: int = 42,
    verbose: bool = True,
) -> BootstrapResult:
    """
    Bootstrap the SIFT2-weighted tractogram to estimate SC uncertainty.

    Algorithm::

        For b = 1, …, B:
            idx = sample n_streamlines indices with replacement
            SC⁽ᵇ⁾ = build_sc(parcel_a[idx], parcel_b[idx], weights[idx])

        Compute statistics across {SC⁽ᵇ⁾}

    Parameters
    ----------
    assignments : StreamlineAssignment
        Pre-computed streamline-to-parcel assignments with SIFT2 weights.
    n_bootstrap : int
        Number of bootstrap resamples.  1000 is standard.
        200–500 for quick exploration, 5000+ for publication CIs.
    store_samples : bool
        If True, store all B SC matrices (memory intensive!).
        Only needed for custom downstream analyses.
    ci_level : float
        Confidence interval level (default 0.95 for 95% CI).
    seed : int
    verbose : bool

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)

    N = assignments.n_parcels
    n_sl = assignments.n_streamlines
    pa = assignments.parcel_a
    pb = assignments.parcel_b
    w = assignments.weights

    alpha = (1 - ci_level) / 2

    if verbose:
        print(f"  Bootstrap: {n_bootstrap} resamples of {n_sl:,} streamlines")
        print(f"  Parcellation: {N} regions")

    # Running statistics to avoid storing all matrices
    sc_sum = np.zeros((N, N))
    sc_sumsq = np.zeros((N, N))
    sc_nonzero_count = np.zeros((N, N), dtype=int)

    # For percentiles — track upper triangle
    n_edges = N * (N - 1) // 2
    edge_samples = (
        np.zeros((n_bootstrap, n_edges), dtype=np.float32)
        if n_bootstrap <= 5000
        else None
    )

    if store_samples:
        all_sc = np.zeros((n_bootstrap, N, N), dtype=np.float32)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_sl, size=n_sl)
        sc_b = build_sc_from_streamlines(pa[idx], pb[idx], w[idx], N)

        sc_sum += sc_b
        sc_sumsq += sc_b ** 2
        sc_nonzero_count += (sc_b > 0).astype(int)

        if edge_samples is not None:
            triu_idx = np.triu_indices(N, k=1)
            edge_samples[b] = sc_b[triu_idx].astype(np.float32)

        if store_samples:
            all_sc[b] = sc_b.astype(np.float32)

        if verbose and (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"    {b+1}/{n_bootstrap} ({(b+1)/n_bootstrap*100:.0f}%)")

    # Statistics
    sc_mean = sc_sum / n_bootstrap
    sc_var = sc_sumsq / n_bootstrap - sc_mean ** 2
    sc_var = np.maximum(sc_var, 0)
    sc_std = np.sqrt(sc_var)

    sc_cv = np.zeros_like(sc_mean)
    nonzero_mean = sc_mean > 0
    sc_cv[nonzero_mean] = sc_std[nonzero_mean] / sc_mean[nonzero_mean]

    prob_nonzero = sc_nonzero_count / n_bootstrap

    # Confidence intervals
    if edge_samples is not None:
        ci_low_triu = np.percentile(edge_samples, alpha * 100, axis=0)
        ci_high_triu = np.percentile(
            edge_samples, (1 - alpha) * 100, axis=0
        )

        sc_ci_low = np.zeros((N, N))
        sc_ci_high = np.zeros((N, N))
        triu_idx = np.triu_indices(N, k=1)
        sc_ci_low[triu_idx] = ci_low_triu
        sc_ci_high[triu_idx] = ci_high_triu
        sc_ci_low = sc_ci_low + sc_ci_low.T
        sc_ci_high = sc_ci_high + sc_ci_high.T
    else:
        z = stats.norm.ppf(1 - alpha)
        sc_ci_low = np.maximum(0, sc_mean - z * sc_std)
        sc_ci_high = sc_mean + z * sc_std

    if verbose:
        mask_upper = np.triu_indices(N, k=1)
        mean_cv = np.mean(
            sc_cv[mask_upper][sc_mean[mask_upper] > 0]
        )
        n_edges_present = np.sum(prob_nonzero[mask_upper] > 0.5)
        n_edges_robust = np.sum(prob_nonzero[mask_upper] > 0.95)
        n_edges_fragile = np.sum(
            (prob_nonzero[mask_upper] > 0.05)
            & (prob_nonzero[mask_upper] < 0.5)
        )

        print(f"\n  Results:")
        print(f"    Mean CV across edges: {mean_cv:.3f}")
        print(f"    Edges present (>50% of bootstraps): {n_edges_present}")
        print(f"    Robust edges (>95%): {n_edges_robust}")
        print(f"    Fragile edges (5–50%): {n_edges_fragile}")

    return BootstrapResult(
        sc_mean=sc_mean,
        sc_std=sc_std,
        sc_cv=sc_cv,
        sc_ci_low=sc_ci_low,
        sc_ci_high=sc_ci_high,
        prob_nonzero=prob_nonzero,
        n_bootstrap=n_bootstrap,
        n_streamlines=n_sl,
        n_parcels=N,
        sc_samples=all_sc if store_samples else None,
    )


# =============================================================================
# EDGE RELIABILITY CLASSIFICATION
# =============================================================================

def classify_edges(
    result: BootstrapResult,
    robust_threshold: float = 0.95,
    fragile_range: Tuple[float, float] = (0.05, 0.50),
    cv_threshold: float = 0.5,
) -> Dict:
    """
    Classify SC edges by their bootstrap reliability.

    Categories:
        - **ROBUST**: present in >95% of bootstraps AND CV < 0.5
          → Use with confidence in any analysis
        - **PRESENT**: present in >50% of bootstraps
          → Likely real but weight is uncertain
        - **FRAGILE**: present in 5–50% of bootstraps
          → May be real but very uncertain; sensitive to tractography noise
        - **SPURIOUS**: present in <5% of bootstraps
          → Likely tractography artifact; consider removing

    Parameters
    ----------
    result : BootstrapResult
    robust_threshold : float
        Minimum P(nonzero) for "robust" classification.
    fragile_range : tuple of float
        (low, high) P(nonzero) range for "fragile" classification.
    cv_threshold : float
        Maximum CV for "robust" classification.

    Returns
    -------
    dict
        'labels' : np.ndarray (N, N)
            Integer array (0=absent, 1=spurious, 2=fragile,
            3=present, 4=robust).
        'n_robust', 'n_present', 'n_fragile', 'n_spurious' : int
        'robust_mask', 'present_mask', 'fragile_mask' : np.ndarray (N, N)
            Boolean masks.
    """
    N = result.n_parcels
    pnz = result.prob_nonzero
    cv = result.sc_cv

    labels = np.zeros((N, N), dtype=int)

    robust = (pnz >= robust_threshold) & (cv < cv_threshold)
    present = (pnz >= 0.50) & ~robust
    fragile = (pnz >= fragile_range[0]) & (pnz < fragile_range[1])
    spurious = (pnz > 0) & (pnz < fragile_range[0])

    labels[spurious] = 1
    labels[fragile] = 2
    labels[present] = 3
    labels[robust] = 4

    triu = np.triu_indices(N, k=1)

    return {
        "labels": labels,
        "n_robust": int(np.sum(labels[triu] == 4)),
        "n_present": int(np.sum(labels[triu] == 3)),
        "n_fragile": int(np.sum(labels[triu] == 2)),
        "n_spurious": int(np.sum(labels[triu] == 1)),
        "n_absent": int(np.sum(labels[triu] == 0)),
        "robust_mask": labels >= 4,
        "present_mask": labels >= 3,
        "fragile_mask": labels == 2,
    }
