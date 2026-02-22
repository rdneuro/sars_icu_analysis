# -*- coding: utf-8 -*-
"""
sars.eigenmorph.parcellation
==================================================

Parcel-level aggregation and comparison with classical morphometrics.

Provides two complementary analyses:

1. **Parcellation**: Summarize vertex-wise eigenvalue features per brain
   region (Schaefer, Brainnetome, Desikan-Killiany, etc.) to produce
   region × feature matrices suitable for group-level statistical
   analysis and clinical correlation.

2. **Classical comparison**: Quantify how much information eigenvalue
   features add beyond what standard FreeSurfer metrics (thickness,
   curvature, sulcal depth) already capture, via correlation analysis
   and unique-variance decomposition.

Functions
---------
parcellate_features
    Aggregate vertex-wise features into parcel-level summaries.
compare_with_classical
    Correlation and unique-variance analysis vs FreeSurfer metrics.

References
----------
- Schaefer et al. (2018). Cereb Cortex 28:3095-3114.
- Desikan et al. (2006). NeuroImage 31:968-980.
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Optional, Dict, Union

from .core import EigenFeatures, MultiScaleFeatures


# =============================================================================
# PARCELLATED SUMMARIES
# =============================================================================

def parcellate_features(
    features: Union[EigenFeatures, MultiScaleFeatures],
    vertex_labels: np.ndarray,
    n_parcels: int,
    parcel_names: Optional[list] = None,
    aggregation: str = "mean",
) -> Dict:
    """
    Summarize vertex-wise eigenvalue features per brain parcel.

    Parameters
    ----------
    features : EigenFeatures or MultiScaleFeatures
        Vertex-wise features from ``compute_eigenfeatures`` or
        ``compute_multiscale_eigenfeatures``.
    vertex_labels : np.ndarray (V,)
        Integer parcel label per vertex (1-indexed; 0 = medial wall).
        From ``nibabel.freesurfer.read_annot``, convert color-table IDs
        to sequential integers first (see Notes).
    n_parcels : int
        Number of parcels (e.g. 100 per hemisphere for Schaefer-200).
    parcel_names : list of str, optional
        Human-readable parcel names.  If None, defaults to
        ``parcel_1, parcel_2, ...``.
    aggregation : str
        Aggregation strategy: 'mean', 'median', 'std', or 'all'
        (returns mean, std, median, p25, p75).

    Returns
    -------
    dict
        Keys are feature names (e.g. ``linearity_r5mm``), values are
        ``(n_parcels,)`` arrays for simple aggregation or nested dicts
        for ``aggregation='all'``.  Special keys:
        ``_parcel_names``, ``_column_names``.

    Notes
    -----
    FreeSurfer ``.annot`` files encode parcel IDs as color-table values
    (e.g. 1376760), not sequential integers.  Convert before calling::

        labels, ctab, names = nib.freesurfer.read_annot('lh.annot')
        unique = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique)}
        labels_seq = np.array([label_map[l] for l in labels])
    """
    if isinstance(features, MultiScaleFeatures):
        feat_matrix = features.as_matrix()
        col_names = features.column_names()
    else:
        feat_matrix = features.as_matrix()
        col_names = [f"{fn}_r{features.radius:.0f}mm"
                     for fn in EigenFeatures.feature_names()]

    result = {}

    for col_idx, col_name in enumerate(col_names):
        values = feat_matrix[:, col_idx]

        if aggregation == "all":
            parcel_stats = {
                "mean": np.zeros(n_parcels),
                "std": np.zeros(n_parcels),
                "median": np.zeros(n_parcels),
                "p25": np.zeros(n_parcels),
                "p75": np.zeros(n_parcels),
            }

            for p in range(n_parcels):
                mask = vertex_labels == (p + 1)
                v = values[mask]
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    parcel_stats["mean"][p] = np.mean(v)
                    parcel_stats["std"][p] = np.std(v)
                    parcel_stats["median"][p] = np.median(v)
                    parcel_stats["p25"][p] = np.percentile(v, 25)
                    parcel_stats["p75"][p] = np.percentile(v, 75)

            result[col_name] = parcel_stats

        else:
            parcel_vals = np.zeros(n_parcels)
            for p in range(n_parcels):
                mask = vertex_labels == (p + 1)
                v = values[mask]
                v = v[~np.isnan(v)]
                if len(v) > 0:
                    if aggregation == "mean":
                        parcel_vals[p] = np.mean(v)
                    elif aggregation == "median":
                        parcel_vals[p] = np.median(v)
                    elif aggregation == "std":
                        parcel_vals[p] = np.std(v)

            result[col_name] = parcel_vals

    result["_parcel_names"] = (
        parcel_names or [f"parcel_{i+1}" for i in range(n_parcels)]
    )
    result["_column_names"] = col_names

    return result


# =============================================================================
# COMPARISON WITH CLASSICAL MORPHOMETRICS
# =============================================================================

def compare_with_classical(
    eigen_features: MultiScaleFeatures,
    classical_metrics: Dict[str, np.ndarray],
    verbose: bool = True,
) -> Dict:
    """
    Compare eigenvalue features with classical FreeSurfer morphometrics.

    Computes pairwise Pearson correlations between each eigenvalue
    feature and each classical metric, then estimates unique variance
    via OLS regression to quantify what eigenvalue features capture
    *beyond* what classical metrics already measure.

    Parameters
    ----------
    eigen_features : MultiScaleFeatures
        Multi-scale eigenvalue features.
    classical_metrics : dict
        ``{metric_name: (V,) array}``.  Typical keys:
        'thickness', 'curv', 'sulc', 'K1', 'K2'.
    verbose : bool
        Print summary table with unique variance markers.

    Returns
    -------
    dict
        'correlations' : np.ndarray (n_eigen, n_classical)
            Pairwise Pearson correlations.
        'eigen_names' : list of str
        'classical_names' : list of str
        'unique_variance' : np.ndarray (n_eigen,)
            Fraction of each eigenvalue feature's variance NOT explained
            by any linear combination of classical metrics.
            Values > 0.5 indicate predominantly novel information (★).
    """
    eigen_matrix = eigen_features.as_matrix()
    eigen_names = eigen_features.column_names()

    classical_names = list(classical_metrics.keys())
    classical_matrix = np.column_stack([
        classical_metrics[k] for k in classical_names
    ])

    n_eigen = eigen_matrix.shape[1]
    n_classical = classical_matrix.shape[1]

    # Remove NaN vertices
    valid = (~np.any(np.isnan(eigen_matrix), axis=1) &
             ~np.any(np.isnan(classical_matrix), axis=1))

    eigen_valid = eigen_matrix[valid]
    classical_valid = classical_matrix[valid]

    if eigen_valid.shape[0] < 10:
        if verbose:
            print(f"  WARNING: Only {eigen_valid.shape[0]} valid vertices "
                  f"— skipping comparison")
        return {
            "correlations": np.full((n_eigen, n_classical), np.nan),
            "eigen_names": eigen_names,
            "classical_names": classical_names,
            "unique_variance": np.full(n_eigen, np.nan),
        }

    # Correlation matrix
    correlations = np.zeros((n_eigen, n_classical))
    for i in range(n_eigen):
        for j in range(n_classical):
            r, _ = pearsonr(eigen_valid[:, i], classical_valid[:, j])
            correlations[i, j] = r

    # Unique variance via OLS: R² NOT explained by classical
    unique_variance = np.zeros(n_eigen)
    for i in range(n_eigen):
        y = eigen_valid[:, i]
        X = np.column_stack([np.ones(len(classical_valid)), classical_valid])

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            pred = X @ beta
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            unique_variance[i] = 1 - r_squared
        except np.linalg.LinAlgError:
            unique_variance[i] = np.nan

    if verbose:
        print("\n  Eigenvalue features vs classical morphometrics:")
        print(f"  {'Feature':<35s} {'Max |r|':>8s} {'Unique var':>12s}")
        print("  " + "-" * 58)
        for i, name in enumerate(eigen_names):
            max_corr = np.max(np.abs(correlations[i]))
            uv = unique_variance[i]
            marker = " ★" if uv > 0.5 else ""
            print(f"  {name:<35s} {max_corr:>8.3f} {uv:>10.3f}{marker}")

        print(f"\n  ★ = >50% unique variance (novel information)")
        print(f"  Features with high unique variance capture geometric")
        print(f"  information that thickness/curvature/sulcal depth miss.")

    return {
        "correlations": correlations,
        "eigen_names": eigen_names,
        "classical_names": classical_names,
        "unique_variance": unique_variance,
    }
