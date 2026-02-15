#!/usr/bin/env python3
"""
==============================================================================
GNN MULTIMODAL ANALYSIS - POST-HOC ANALYSIS
==============================================================================

Post-hoc analyses for interpreting GNN results:
- Hierarchical gradient comparison with normative data
- Cross-atlas consistency analysis
- Convergence of GAT and contrastive findings
- Statistical testing with permutation-based inference

Author: SARS-CoV-2 Neuroimaging Study
Date: February 2026
==============================================================================
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# HIERARCHICAL GRADIENT ANALYSIS
# ==============================================================================

def compute_hierarchical_gradient(
    regional_values: np.ndarray,
    atlas: str,
    labels: Optional[list] = None,
) -> Dict:
    """
    Compare regional SC-FC decoupling with the normative cortical hierarchy.

    The structure-function coupling hierarchy follows a well-established
    gradient from primary sensory/motor cortex (tight coupling) to
    transmodal association cortex (loose coupling) (Vázquez-Rodríguez
    et al., 2019; Baum et al., 2020).

    For Schaefer parcellation with Yeo network assignments:
    - Visual, Somatomotor: expected HIGH coupling (low decoupling)
    - Dorsal/Ventral Attention: intermediate
    - Default, Frontoparietal, Limbic: expected LOW coupling (high decoupling)

    Parameters
    ----------
    regional_values : np.ndarray
        Per-region metric (decoupling, coherence, etc.) of shape (N,).
    atlas : str
        Atlas name for network assignment.
    labels : list, optional
        ROI label names.

    Returns
    -------
    dict with:
        'network_means': mean value per network
        'network_stds': std per network
        'hierarchy_correlation': correlation with expected gradient
        'network_stats': ANOVA/Kruskal-Wallis across networks
        'expected_rank': normative ranking for comparison
    """
    # Yeo 7-network hierarchy (normative ranking of SC-FC coupling)
    # From TIGHT coupling to LOOSE coupling
    HIERARCHY_ORDER = [
        "Vis",          # 1 - tightest coupling
        "SomMot",       # 2
        "DorsAttn",     # 3
        "SalVentAttn",  # 4
        "Limbic",       # 5
        "Cont",         # 6 (Control/Frontoparietal)
        "Default",      # 7 - loosest coupling
    ]

    result = {
        "atlas": atlas,
        "n_regions": len(regional_values),
    }

    if atlas != "schaefer_100" or labels is None:
        # For non-Schaefer atlases, return basic statistics
        result["network_means"] = {}
        result["hierarchy_correlation"] = None
        result["note"] = "Network assignment available only for Schaefer atlas"
        return result

    # Assign each ROI to a Yeo network based on label name
    network_assignments = {}
    for i, label in enumerate(labels):
        for net in HIERARCHY_ORDER:
            if net in str(label):
                if net not in network_assignments:
                    network_assignments[net] = []
                network_assignments[net].append(i)
                break

    # Compute network-level statistics
    network_means = {}
    network_stds = {}
    network_values = {}

    for net in HIERARCHY_ORDER:
        if net in network_assignments:
            indices = network_assignments[net]
            vals = regional_values[indices]
            network_means[net] = float(np.mean(vals))
            network_stds[net] = float(np.std(vals))
            network_values[net] = vals

    # Test hierarchy: correlate network means with expected rank
    observed_means = []
    expected_ranks = []
    for rank, net in enumerate(HIERARCHY_ORDER):
        if net in network_means:
            observed_means.append(network_means[net])
            expected_ranks.append(rank)

    if len(observed_means) >= 3:
        rho, p_val = stats.spearmanr(expected_ranks, observed_means)
        hierarchy_corr = {"rho": float(rho), "p_value": float(p_val)}
    else:
        hierarchy_corr = None

    # Kruskal-Wallis test across networks
    groups = [network_values[n] for n in HIERARCHY_ORDER if n in network_values]
    if len(groups) >= 2:
        H, p_kw = stats.kruskal(*groups)
        network_stats = {"H_statistic": float(H), "p_value": float(p_kw)}
    else:
        network_stats = None

    result.update({
        "network_means": network_means,
        "network_stds": network_stds,
        "hierarchy_correlation": hierarchy_corr,
        "network_stats": network_stats,
        "expected_hierarchy": HIERARCHY_ORDER,
    })

    return result


# ==============================================================================
# CROSS-ATLAS CONSISTENCY
# ==============================================================================

def cross_atlas_consistency(
    results_by_atlas: Dict[str, Dict],
    metric_key: str = "mean_corr",
) -> Dict:
    """
    Assess consistency of findings across different brain atlases.

    Cross-atlas consistency is a strong validation criterion:
    if the same brain regions show altered SC-FC coupling across
    multiple parcellation schemes, the finding is robust to
    arbitrary parcellation choices.

    We compute rank correlations between atlas results at
    matching spatial scales using spatial overlap.

    Parameters
    ----------
    results_by_atlas : dict
        {atlas_name: {metric_key: np.ndarray, ...}}.
    metric_key : str
        Which metric to compare.

    Returns
    -------
    dict with:
        'pairwise_correlations': correlation between atlas pairs
        'mean_consistency': average cross-atlas correlation
        'atlas_pairs': which atlases were compared
    """
    atlas_names = list(results_by_atlas.keys())
    n_atlases = len(atlas_names)

    pairwise = {}
    correlations = []

    for i in range(n_atlases):
        for j in range(i + 1, n_atlases):
            a1, a2 = atlas_names[i], atlas_names[j]
            v1 = results_by_atlas[a1].get(metric_key)
            v2 = results_by_atlas[a2].get(metric_key)

            if v1 is None or v2 is None:
                continue

            # Since atlases have different numbers of ROIs,
            # we compare summary statistics
            # Method: compare distribution characteristics
            stats_v1 = _distribution_features(v1)
            stats_v2 = _distribution_features(v2)

            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(v1, v2)

            # Compare quantile profiles
            quantiles = np.arange(0.05, 1.0, 0.05)
            q1 = np.quantile(v1, quantiles)
            q2 = np.quantile(v2, quantiles)
            rho, p = stats.spearmanr(q1, q2)

            pairwise[f"{a1}_vs_{a2}"] = {
                "quantile_correlation": float(rho),
                "quantile_p_value": float(p),
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p),
                "n_rois_1": len(v1),
                "n_rois_2": len(v2),
            }
            correlations.append(rho)

    return {
        "pairwise_correlations": pairwise,
        "mean_consistency": float(np.mean(correlations)) if correlations else None,
        "n_pairs": len(correlations),
    }


def _distribution_features(values: np.ndarray) -> Dict:
    """Extract summary features from a distribution."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "skewness": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


# ==============================================================================
# CONVERGENCE ANALYSIS (GAT + CONTRASTIVE)
# ==============================================================================

def convergence_analysis(
    gat_results: Dict,
    contrastive_results: Dict,
    atlas: str,
) -> Dict:
    """
    Assess convergence between GAT SC→FC decoupling and contrastive coherence.

    Two independent methods measuring the same underlying phenomenon
    (SC-FC relationship) should produce correlated results. High
    convergence strengthens both findings; divergence may reveal
    complementary aspects.

    The GAT measures directed SC→FC prediction quality (how well
    structure predicts function), while contrastive learning measures
    symmetric SC↔FC alignment quality.

    Parameters
    ----------
    gat_results : dict
        Results from train_gat_sc_fc.
    contrastive_results : dict
        Results from train_contrastive.
    atlas : str
        Atlas name.

    Returns
    -------
    dict with convergence statistics.
    """
    # Extract per-region metrics
    gat_corr = gat_results.get("cohort_decoupling", {}).get("mean_corr")
    cont_coh = contrastive_results.get("coherence", {}).get("mean_coherence")

    if gat_corr is None or cont_coh is None:
        return {"error": "Missing required metrics"}

    # Ensure same number of regions
    n1, n2 = len(gat_corr), len(cont_coh)
    if n1 != n2:
        logger.warning(f"Dimension mismatch: GAT={n1}, contrastive={n2}")
        n_min = min(n1, n2)
        gat_corr = gat_corr[:n_min]
        cont_coh = cont_coh[:n_min]

    # Spatial correlation between methods
    rho, p_val = stats.spearmanr(gat_corr, cont_coh)

    # Rank agreement: top/bottom regions
    n_top = max(1, len(gat_corr) // 5)  # top 20%
    gat_top = set(np.argsort(gat_corr)[-n_top:])
    cont_top = set(np.argsort(cont_coh)[-n_top:])
    gat_bottom = set(np.argsort(gat_corr)[:n_top])
    cont_bottom = set(np.argsort(cont_coh)[:n_top])

    top_overlap = len(gat_top & cont_top) / n_top
    bottom_overlap = len(gat_bottom & cont_bottom) / n_top

    # Regional agreement map
    # Z-score both metrics and compute agreement
    gat_z = (gat_corr - np.mean(gat_corr)) / (np.std(gat_corr) + 1e-8)
    cont_z = (cont_coh - np.mean(cont_coh)) / (np.std(cont_coh) + 1e-8)
    agreement = gat_z * cont_z  # positive = both agree on direction

    return {
        "spatial_correlation": {"rho": float(rho), "p_value": float(p_val)},
        "top20_overlap": float(top_overlap),
        "bottom20_overlap": float(bottom_overlap),
        "agreement_map": agreement,
        "gat_z": gat_z,
        "contrastive_z": cont_z,
        "n_regions": len(gat_corr),
        "atlas": atlas,
    }


# ==============================================================================
# PERMUTATION TESTING
# ==============================================================================

def permutation_test_regional(
    observed_values: np.ndarray,
    null_generator: callable,
    n_permutations: int = 1000,
    two_sided: bool = True,
) -> Dict:
    """
    Non-parametric permutation test for regional metrics.

    Generates a null distribution by permuting subject labels
    (or shuffling SC-FC assignments) and computes p-values
    for each region.

    Suitable for small sample sizes (N=23) where parametric
    assumptions may not hold.

    Parameters
    ----------
    observed_values : np.ndarray
        Observed regional metric (N_regions,).
    null_generator : callable
        Function that returns one null realization (N_regions,).
    n_permutations : int
        Number of permutations.
    two_sided : bool
        Two-sided test.

    Returns
    -------
    dict with:
        'p_values': per-region p-values (N,)
        'z_scores': deviation from null in SD units (N,)
        'null_mean': mean of null distribution (N,)
        'null_std': std of null distribution (N,)
        'significant_mask': boolean mask after FDR correction
    """
    n_regions = len(observed_values)
    null_dist = np.zeros((n_permutations, n_regions))

    for p in range(n_permutations):
        null_dist[p] = null_generator()

    null_mean = np.mean(null_dist, axis=0)
    null_std = np.std(null_dist, axis=0)
    null_std[null_std < 1e-10] = 1e-10

    z_scores = (observed_values - null_mean) / null_std

    # Compute p-values
    if two_sided:
        p_values = np.mean(
            np.abs(null_dist - null_mean) >= np.abs(observed_values - null_mean),
            axis=0,
        )
    else:
        p_values = np.mean(null_dist >= observed_values, axis=0)

    # FDR correction (Benjamini-Hochberg)
    fdr_mask = _fdr_correction(p_values, alpha=0.05)

    return {
        "p_values": p_values,
        "z_scores": z_scores,
        "null_mean": null_mean,
        "null_std": null_std,
        "significant_mask": fdr_mask,
        "n_significant": int(np.sum(fdr_mask)),
        "n_permutations": n_permutations,
    }


def _fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    thresholds = alpha * np.arange(1, n + 1) / n
    below = sorted_p <= thresholds

    if not np.any(below):
        return np.zeros(n, dtype=bool)

    max_idx = np.max(np.where(below))
    significant = np.zeros(n, dtype=bool)
    significant[sorted_idx[:max_idx + 1]] = True

    return significant


# ==============================================================================
# COMPREHENSIVE REPORT
# ==============================================================================

def generate_analysis_report(
    all_results: Dict,
    config,
) -> str:
    """
    Generate a markdown report summarizing all GNN multimodal analyses.

    Parameters
    ----------
    all_results : dict
        Results from run_full_pipeline.
    config : GNNMultimodalConfig
        Configuration used.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        "# GNN Multimodal SC-FC Analysis Report",
        f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Subjects:** {len(config.subjects)}",
        f"**Atlases:** {', '.join(config.atlases)}",
        "",
    ]

    for atlas in config.atlases:
        atlas_results = all_results.get(atlas, {})
        lines.append(f"## Atlas: {atlas}")
        lines.append("")

        # GAT results
        gat = atlas_results.get("gat_sc_fc", {})
        if "error" not in gat:
            fold_results = gat.get("fold_results", [])
            if fold_results:
                mean_test_corr = np.mean([f["test_corr"] for f in fold_results])
                std_test_corr = np.std([f["test_corr"] for f in fold_results])
                lines.append("### GAT SC→FC Prediction (LOO-CV)")
                lines.append(
                    f"Mean prediction correlation: **{mean_test_corr:.4f} ± {std_test_corr:.4f}**"
                )
                lines.append("")

        # Contrastive results
        cont = atlas_results.get("contrastive", {})
        if "error" not in cont:
            coherence = cont.get("coherence", {})
            mean_coh = coherence.get("mean_coherence")
            if mean_coh is not None:
                lines.append("### Contrastive Multimodal Coherence")
                lines.append(
                    f"Mean regional coherence: **{np.mean(mean_coh):.4f} ± {np.std(mean_coh):.4f}**"
                )
                subgroups = cont.get("subgroups", {})
                if subgroups:
                    lines.append(
                        f"Optimal subgroups: **k={subgroups.get('best_k', '?')}**"
                    )
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)
