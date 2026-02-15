# -*- coding: utf-8 -*-
"""
sars.reservoir.reservoir_viz
============================

Publication-Quality Visualizations for Reservoir-Based SC-FC Decoupling
========================================================================

This module provides comprehensive visualization functions for all
analyses in the reservoir_dynamics module. Figures are designed to
meet the standards of high-impact neuroimaging journals (NeuroImage,
Human Brain Mapping, Nature Communications).

Design Principles
-----------------
- Clean, professional aesthetics (no gridlines unless informative)
- Consistent color schemes across related plots
- Diverging colormaps for SC-FC comparisons (coupled=blue, decoupled=red)
- Perceptually uniform colormaps for continuous metrics
- Multi-panel layouts for comprehensive dashboards
- Proper statistical annotation (effect sizes, corrected p-values)
- Font sizes suitable for publication (axis labels ≥ 10pt)

Author: Velho Mago
Project: SARS-CoV-2 Brain Connectivity Analysis Library
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Import result dataclasses
from reservoir_dynamics import (
    SCFCCouplingResult,
    SDIResult,
    PerturbationResult,
    GroupComparisonResult,
    ReservoirDynamicsResult,
)


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Journal-quality defaults
JOURNAL_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Color palettes
COUPLING_CMAP = "RdBu_r"  # Diverging: blue=coupled, red=decoupled
SDI_CMAP = "YlOrRd"  # Sequential: low SDI (yellow) to high SDI (red)
NETWORK_COLORS = {
    "Visual": "#781286",
    "Somatomotor": "#4682B4",
    "DorsalAttention": "#00760E",
    "VentralAttention": "#C43AFA",
    "Limbic": "#DCF8A4",
    "Frontoparietal": "#E69422",
    "Default": "#CD3E4E",
}
GROUP_COLORS = {"COVID": "#D32F2F", "Control": "#1976D2"}


def _apply_style():
    """Apply journal-quality matplotlib style."""
    plt.rcParams.update(JOURNAL_STYLE)


def _add_significance_stars(ax, p_value, x, y, fontsize=11):
    """Add significance stars annotation."""
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "n.s."
    ax.text(x, y, stars, ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


# =============================================================================
# COUPLING VISUALIZATIONS
# =============================================================================

def plot_sc_fc_coupling(
    result: SCFCCouplingResult,
    region_labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-panel figure showing SC-FC coupling analysis results.

    Panels:
    A) SC matrix
    B) Empirical FC
    C) Simulated FC (reservoir-predicted)
    D) SC vs FC scatter (upper triangle)
    E) Simulated vs Empirical FC scatter
    F) Regional coupling profile (bar plot)

    Parameters
    ----------
    result : SCFCCouplingResult
        Coupling analysis results from ConnectomeReservoir.predict_fc().
    region_labels : list of str, optional
        Labels for brain regions.
    figsize : tuple
        Figure size in inches.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    n = result.sc_matrix.shape[0]
    tick_step = max(1, n // 10)

    # A) SC matrix
    ax_sc = fig.add_subplot(gs[0, 0])
    im_sc = ax_sc.imshow(
        result.sc_matrix, cmap="hot", aspect="equal", interpolation="nearest"
    )
    ax_sc.set_title("A. Structural Connectivity", fontweight="bold", pad=8)
    ax_sc.set_xlabel("Region")
    ax_sc.set_ylabel("Region")
    plt.colorbar(im_sc, ax=ax_sc, fraction=0.046, pad=0.04, label="Weight")

    # B) Empirical FC
    ax_fc = fig.add_subplot(gs[0, 1])
    vmax_fc = np.percentile(np.abs(result.fc_empirical), 95)
    im_fc = ax_fc.imshow(
        result.fc_empirical,
        cmap=COUPLING_CMAP,
        aspect="equal",
        vmin=-vmax_fc,
        vmax=vmax_fc,
        interpolation="nearest",
    )
    ax_fc.set_title("B. Empirical FC", fontweight="bold", pad=8)
    ax_fc.set_xlabel("Region")
    ax_fc.set_ylabel("Region")
    plt.colorbar(im_fc, ax=ax_fc, fraction=0.046, pad=0.04, label="Correlation")

    # C) Simulated FC
    ax_sim = fig.add_subplot(gs[0, 2])
    vmax_sim = np.percentile(np.abs(result.fc_simulated), 95)
    im_sim = ax_sim.imshow(
        result.fc_simulated,
        cmap=COUPLING_CMAP,
        aspect="equal",
        vmin=-vmax_sim,
        vmax=vmax_sim,
        interpolation="nearest",
    )
    ax_sim.set_title("C. Reservoir-Simulated FC", fontweight="bold", pad=8)
    ax_sim.set_xlabel("Region")
    ax_sim.set_ylabel("Region")
    plt.colorbar(im_sim, ax=ax_sim, fraction=0.046, pad=0.04, label="Correlation")

    # D) SC vs FC scatter
    ax_scatter1 = fig.add_subplot(gs[1, 0])
    idx = np.triu_indices(n, k=1)
    sc_upper = result.sc_matrix[idx]
    fc_upper = result.fc_empirical[idx]

    # Subsample for visibility if needed
    if len(sc_upper) > 5000:
        rng = np.random.default_rng(42)
        sample = rng.choice(len(sc_upper), 5000, replace=False)
        sc_plot, fc_plot = sc_upper[sample], fc_upper[sample]
    else:
        sc_plot, fc_plot = sc_upper, fc_upper

    ax_scatter1.scatter(
        sc_plot, fc_plot, alpha=0.15, s=4, c="#2196F3", edgecolors="none"
    )
    # Regression line
    if np.std(sc_plot) > 0:
        z = np.polyfit(sc_plot, fc_plot, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sc_plot.min(), sc_plot.max(), 100)
        ax_scatter1.plot(x_line, p(x_line), "r-", linewidth=1.5, alpha=0.8)
    ax_scatter1.set_xlabel("SC Weight")
    ax_scatter1.set_ylabel("FC (Pearson r)")
    ax_scatter1.set_title(
        f"D. SC–FC Correlation (r = {result.coupling_global:.3f})",
        fontweight="bold",
        pad=8,
    )

    # E) Simulated vs Empirical FC scatter
    ax_scatter2 = fig.add_subplot(gs[1, 1])
    sim_upper = result.fc_simulated[idx]
    if len(sim_upper) > 5000:
        sim_plot, emp_plot = sim_upper[sample], fc_upper[sample]
    else:
        sim_plot, emp_plot = sim_upper, fc_upper

    ax_scatter2.scatter(
        sim_plot, emp_plot, alpha=0.15, s=4, c="#4CAF50", edgecolors="none"
    )
    if np.std(sim_plot) > 0:
        z = np.polyfit(sim_plot, emp_plot, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sim_plot.min(), sim_plot.max(), 100)
        ax_scatter2.plot(x_line, p(x_line), "r-", linewidth=1.5, alpha=0.8)
    ax_scatter2.set_xlabel("Simulated FC")
    ax_scatter2.set_ylabel("Empirical FC")
    ax_scatter2.set_title(
        f"E. Reservoir Prediction (r = {result.coupling_reservoir:.3f})",
        fontweight="bold",
        pad=8,
    )

    # F) Regional coupling bar plot
    ax_bar = fig.add_subplot(gs[1, 2])
    sorted_idx = np.argsort(result.regional_decoupling)
    colors = plt.cm.RdYlBu_r(result.regional_decoupling[sorted_idx])
    ax_bar.barh(range(n), result.regional_decoupling[sorted_idx], color=colors, height=0.8)
    ax_bar.set_xlabel("Decoupling Index")
    ax_bar.set_title("F. Regional Decoupling Profile", fontweight="bold", pad=8)
    ax_bar.set_yticks([0, n // 4, n // 2, 3 * n // 4, n - 1])
    ax_bar.axvline(x=np.mean(result.regional_decoupling), color="k", linestyle="--",
                   linewidth=0.8, alpha=0.5, label="Mean")
    ax_bar.legend(fontsize=8)

    # Summary text box
    summary_text = (
        f"Global SC-FC: r = {result.coupling_global:.3f}\n"
        f"Reservoir coupling: r = {result.coupling_reservoir:.3f}\n"
        f"Decoupling index: {result.decoupling_index_global:.3f}\n"
        f"Readout R²: {result.prediction_r2:.3f}\n"
        f"Spectral radius: {result.spectral_radius:.2f}"
    )
    fig.text(
        0.02, 0.98, summary_text, transform=fig.transFigure,
        fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.8),
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# SDI VISUALIZATIONS
# =============================================================================

def plot_sdi(
    sdi_result: SDIResult,
    region_labels: Optional[List[str]] = None,
    network_labels: Optional[np.ndarray] = None,
    network_names: Optional[Dict[int, str]] = None,
    figsize: Tuple[float, float] = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-panel figure showing Structural Decoupling Index results.

    Panels:
    A) SDI regional bar plot (sorted, color-coded by network)
    B) FC coupled component
    C) FC decoupled component
    D) SC Laplacian eigenvalue spectrum
    E) SDI by network (violin/box plot)
    F) Coupling ratio pie chart

    Parameters
    ----------
    sdi_result : SDIResult
        SDI analysis results.
    region_labels : list of str, optional
        Labels for brain regions.
    network_labels : np.ndarray, optional
        Network assignment per node.
    network_names : dict, optional
        Network label-to-name mapping.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)
    n = len(sdi_result.sdi)

    # A) SDI regional profile
    ax_sdi = fig.add_subplot(gs[0, 0])
    sorted_idx = np.argsort(sdi_result.sdi)

    if network_labels is not None and network_names is not None:
        colors = []
        net_color_map = {}
        unique_nets = np.unique(network_labels)
        cmap = plt.cm.tab10
        for i, net in enumerate(unique_nets):
            name = network_names.get(net, f"Net_{net}")
            net_color_map[net] = (
                NETWORK_COLORS.get(name, cmap(i / len(unique_nets)))
            )
        for idx in sorted_idx:
            colors.append(net_color_map.get(network_labels[idx], "#888888"))
    else:
        colors = plt.cm.YlOrRd(sdi_result.sdi[sorted_idx])

    ax_sdi.barh(range(n), sdi_result.sdi[sorted_idx], color=colors, height=0.8)
    ax_sdi.set_xlabel("SDI (normalized)")
    ax_sdi.set_title("A. Structural Decoupling Index", fontweight="bold", pad=8)
    ax_sdi.axvline(x=0.5, color="k", linestyle="--", linewidth=0.6, alpha=0.3)
    ax_sdi.set_yticks([])

    # B) Coupled FC
    ax_coupled = fig.add_subplot(gs[0, 1])
    vmax = np.percentile(np.abs(sdi_result.fc_coupled), 95)
    im_c = ax_coupled.imshow(
        sdi_result.fc_coupled, cmap=COUPLING_CMAP,
        vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest"
    )
    ax_coupled.set_title("B. Coupled FC Component", fontweight="bold", pad=8)
    ax_coupled.set_xlabel("Region")
    ax_coupled.set_ylabel("Region")
    plt.colorbar(im_c, ax=ax_coupled, fraction=0.046, pad=0.04)

    # C) Decoupled FC
    ax_decoupled = fig.add_subplot(gs[0, 2])
    vmax_d = np.percentile(np.abs(sdi_result.fc_decoupled), 95)
    if vmax_d < 1e-10:
        vmax_d = 1.0
    im_d = ax_decoupled.imshow(
        sdi_result.fc_decoupled, cmap=COUPLING_CMAP,
        vmin=-vmax_d, vmax=vmax_d, aspect="equal", interpolation="nearest"
    )
    ax_decoupled.set_title("C. Decoupled FC Component", fontweight="bold", pad=8)
    ax_decoupled.set_xlabel("Region")
    ax_decoupled.set_ylabel("Region")
    plt.colorbar(im_d, ax=ax_decoupled, fraction=0.046, pad=0.04)

    # D) Eigenvalue spectrum
    ax_eigen = fig.add_subplot(gs[1, 0])
    k = sdi_result.n_coupled_components
    eigenvalues = sdi_result.sc_eigenvalues

    ax_eigen.fill_between(
        range(k), eigenvalues[:k], alpha=0.3, color="#1976D2", label="Coupled"
    )
    ax_eigen.fill_between(
        range(k, n), eigenvalues[k:], alpha=0.3, color="#D32F2F", label="Decoupled"
    )
    ax_eigen.plot(eigenvalues, "k-", linewidth=0.8)
    ax_eigen.axvline(x=k, color="k", linestyle="--", linewidth=0.8)
    ax_eigen.set_xlabel("Eigenmode Index")
    ax_eigen.set_ylabel("Eigenvalue (λ)")
    ax_eigen.set_title("D. Graph Laplacian Spectrum", fontweight="bold", pad=8)
    ax_eigen.legend(fontsize=8)

    # E) SDI by network
    ax_net = fig.add_subplot(gs[1, 1])
    if sdi_result.network_sdi is not None:
        names = list(sdi_result.network_sdi.keys())
        values = [sdi_result.network_sdi[n_] for n_ in names]
        sorted_pairs = sorted(zip(values, names))
        values_sorted = [v for v, _ in sorted_pairs]
        names_sorted = [n_ for _, n_ in sorted_pairs]

        bar_colors = [NETWORK_COLORS.get(n_, "#888888") for n_ in names_sorted]
        ax_net.barh(names_sorted, values_sorted, color=bar_colors, height=0.6)
        ax_net.set_xlabel("Mean SDI")
        ax_net.set_title("E. SDI by Network", fontweight="bold", pad=8)
        ax_net.axvline(x=np.mean(values), color="k", linestyle="--",
                       linewidth=0.6, alpha=0.5)
    else:
        ax_net.text(0.5, 0.5, "No network labels\nprovided",
                   transform=ax_net.transAxes, ha="center", va="center",
                   fontsize=11, color="#999")
        ax_net.set_title("E. SDI by Network", fontweight="bold", pad=8)

    # F) Coupling ratio
    ax_pie = fig.add_subplot(gs[1, 2])
    coupled_pct = sdi_result.coupling_ratio * 100
    decoupled_pct = 100 - coupled_pct
    wedges, texts, autotexts = ax_pie.pie(
        [coupled_pct, decoupled_pct],
        labels=["Coupled", "Decoupled"],
        colors=["#1976D2", "#D32F2F"],
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(linewidth=1, edgecolor="white"),
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax_pie.set_title(
        f"F. FC Energy Decomposition\n(k={sdi_result.n_coupled_components} modes)",
        fontweight="bold", pad=8,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# PERTURBATION ANALYSIS
# =============================================================================

def plot_perturbation(
    result: PerturbationResult,
    region_labels: Optional[List[str]] = None,
    top_k: int = 15,
    figsize: Tuple[float, float] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize virtual lesion analysis results.

    Parameters
    ----------
    result : PerturbationResult
        Perturbation analysis results.
    region_labels : list of str, optional
        Labels for brain regions.
    top_k : int
        Number of top vulnerable/resilient nodes to highlight.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    n = len(result.vulnerability_index)

    # A) Vulnerability landscape
    ax = axes[0]
    sorted_idx = np.argsort(result.vulnerability_index)[::-1]
    colors = plt.cm.RdYlGn_r(
        np.linspace(0, 1, n)
    )
    ax.bar(range(n), result.vulnerability_index[sorted_idx], color=colors, width=0.8)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.set_xlabel("Nodes (sorted by vulnerability)")
    ax.set_ylabel("Vulnerability Index")
    ax.set_title("A. Node Vulnerability Landscape", fontweight="bold")

    # B) Top vulnerable vs resilient
    ax2 = axes[1]
    top_vuln = result.most_vulnerable_nodes[:top_k]
    top_res = result.most_resilient_nodes[:top_k]

    if region_labels is not None:
        vuln_labels = [region_labels[i] for i in top_vuln]
        res_labels = [region_labels[i] for i in top_res]
    else:
        vuln_labels = [f"R{i}" for i in top_vuln]
        res_labels = [f"R{i}" for i in top_res]

    all_labels = vuln_labels + res_labels
    all_values = np.concatenate([
        result.vulnerability_index[top_vuln],
        result.vulnerability_index[top_res],
    ])
    all_colors = ["#D32F2F"] * len(top_vuln) + ["#1976D2"] * len(top_res)

    y_pos = range(len(all_labels))
    ax2.barh(y_pos, all_values, color=all_colors, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(all_labels, fontsize=7)
    ax2.axvline(x=0, color="k", linewidth=0.5)
    ax2.set_xlabel("Vulnerability Index")
    ax2.set_title("B. Most Vulnerable (red) vs Resilient (blue)", fontweight="bold")

    # Add baseline annotation
    fig.text(
        0.5, 0.02,
        f"Baseline coupling: r = {result.baseline_coupling:.3f}",
        ha="center", fontsize=9, style="italic",
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# GROUP COMPARISON
# =============================================================================

def plot_group_comparison(
    result: GroupComparisonResult,
    region_labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-panel figure for group comparison of SC-FC coupling/decoupling.

    Panels:
    A) Regional effect sizes (Manhattan-style plot)
    B) Significant regions highlighted
    C) Network-level comparison (if available)
    D) Global comparison boxplot

    Parameters
    ----------
    result : GroupComparisonResult
        Group comparison results.
    region_labels : list of str, optional
        Labels for brain regions.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    n = len(result.regional_tstat)
    has_networks = result.network_results is not None

    if has_networks:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
        ax_manhattan = fig.add_subplot(gs[0, :])
        ax_effect = fig.add_subplot(gs[1, 0])
        ax_network = fig.add_subplot(gs[1, 1])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_manhattan = axes[0]
        ax_effect = axes[1]
        ax_network = None

    # A) Manhattan-style plot (-log10 p-values)
    neg_log_p = -np.log10(np.clip(result.regional_pvalue, 1e-20, 1.0))
    neg_log_p_fdr = -np.log10(np.clip(result.regional_pvalue_fdr, 1e-20, 1.0))

    colors_manh = np.where(result.significant_regions, "#D32F2F", "#BBBBBB")
    ax_manhattan.scatter(
        range(n), neg_log_p, c=colors_manh, s=20, edgecolors="none", alpha=0.7
    )
    ax_manhattan.axhline(
        y=-np.log10(0.05), color="#666", linestyle="--", linewidth=0.6,
        label="p = 0.05 (uncorrected)"
    )

    # FDR threshold line
    if np.any(result.significant_regions):
        fdr_threshold = np.min(neg_log_p[result.significant_regions])
        ax_manhattan.axhline(
            y=fdr_threshold, color="#D32F2F", linestyle="-.", linewidth=0.6,
            label="FDR threshold"
        )

    ax_manhattan.set_xlabel("Brain Region Index")
    ax_manhattan.set_ylabel("−log₁₀(p)")
    n_sig = np.sum(result.significant_regions)
    ax_manhattan.set_title(
        f"A. Regional Differences: {result.group1_name} vs {result.group2_name}"
        f" ({n_sig} significant / {n} regions)",
        fontweight="bold",
    )
    ax_manhattan.legend(fontsize=8)

    # B) Effect size plot (Cohen's d)
    sorted_idx = np.argsort(np.abs(result.effect_size_cohen_d))[::-1][:30]
    d_values = result.effect_size_cohen_d[sorted_idx]

    if region_labels is not None:
        labels_sorted = [region_labels[i] for i in sorted_idx]
    else:
        labels_sorted = [f"R{i}" for i in sorted_idx]

    bar_colors = ["#D32F2F" if d > 0 else "#1976D2" for d in d_values]
    ax_effect.barh(range(len(d_values)), d_values, color=bar_colors, height=0.6)
    ax_effect.set_yticks(range(len(labels_sorted)))
    ax_effect.set_yticklabels(labels_sorted, fontsize=6)
    ax_effect.axvline(x=0, color="k", linewidth=0.5)
    ax_effect.axvline(x=0.8, color="#999", linestyle=":", linewidth=0.5, label="|d|=0.8")
    ax_effect.axvline(x=-0.8, color="#999", linestyle=":", linewidth=0.5)
    ax_effect.set_xlabel("Cohen's d")
    ax_effect.set_title("B. Top 30 Effect Sizes", fontweight="bold")
    ax_effect.invert_yaxis()
    ax_effect.legend(fontsize=7)

    # C) Network-level comparison
    if ax_network is not None and has_networks:
        net_names = list(result.network_results.keys())
        net_g1 = [result.network_results[n_]["mean_group1"] for n_ in net_names]
        net_g2 = [result.network_results[n_]["mean_group2"] for n_ in net_names]
        net_p = [result.network_results[n_]["p_value"] for n_ in net_names]

        x = np.arange(len(net_names))
        width = 0.35
        bars1 = ax_network.bar(
            x - width / 2, net_g1, width,
            label=result.group1_name,
            color=GROUP_COLORS.get(result.group1_name, "#D32F2F"),
            alpha=0.8,
        )
        bars2 = ax_network.bar(
            x + width / 2, net_g2, width,
            label=result.group2_name,
            color=GROUP_COLORS.get(result.group2_name, "#1976D2"),
            alpha=0.8,
        )

        # Significance annotations
        for i, p_val in enumerate(net_p):
            max_val = max(net_g1[i], net_g2[i])
            _add_significance_stars(ax_network, p_val, i, max_val * 1.05, fontsize=9)

        ax_network.set_xticks(x)
        ax_network.set_xticklabels(net_names, rotation=45, ha="right", fontsize=7)
        ax_network.set_ylabel("Mean Metric")
        ax_network.set_title("C. Network-Level Comparison", fontweight="bold")
        ax_network.legend(fontsize=8)

    # Global result annotation
    fig.text(
        0.02, 0.01,
        f"Global: t = {result.global_tstat:.2f}, "
        f"p = {result.global_pvalue:.4f}",
        fontsize=8, style="italic",
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# SPECTRAL RADIUS SWEEP
# =============================================================================

def plot_spectral_sweep(
    sweep_result: Dict,
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot SC-FC coupling as a function of spectral radius.

    Shows the edge-of-chaos phenomenon: coupling peaks at intermediate
    spectral radius values.

    Parameters
    ----------
    sweep_result : dict
        Results from spectral_radius_sweep().
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    radii = sweep_result["radii"]
    coupling = sweep_result["coupling"]
    optimal_r = sweep_result["optimal_radius"]
    optimal_c = sweep_result["optimal_coupling"]

    ax.plot(radii, coupling, "o-", color="#1976D2", linewidth=1.5, markersize=5)
    ax.axvline(x=1.0, color="#999", linestyle="--", linewidth=0.6,
               label="ρ = 1 (edge of chaos)")
    ax.plot(optimal_r, optimal_c, "*", color="#D32F2F", markersize=14,
            label=f"Optimal: ρ={optimal_r:.2f}, r={optimal_c:.3f}")

    ax.fill_between(radii, coupling, alpha=0.1, color="#1976D2")
    ax.set_xlabel("Spectral Radius (ρ)")
    ax.set_ylabel("SC-FC Coupling (Pearson r)")
    ax.set_title("Edge of Chaos: Spectral Radius vs SC-FC Coupling", fontweight="bold")
    ax.legend(fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# PLASTICITY ANALYSIS
# =============================================================================

def plot_plasticity_analysis(
    plasticity_result: Dict,
    region_labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize Hebbian plasticity effects on SC-FC coupling.

    Parameters
    ----------
    plasticity_result : dict
        Results from HebbianAdaptiveReservoir.plasticity_analysis().
    region_labels : list of str, optional
        Labels for brain regions.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    n = plasticity_result["fc_before"].shape[0]

    # A) FC before plasticity
    ax_before = fig.add_subplot(gs[0, 0])
    vmax = np.percentile(np.abs(plasticity_result["fc_before"]), 95)
    ax_before.imshow(
        plasticity_result["fc_before"], cmap=COUPLING_CMAP,
        vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest"
    )
    ax_before.set_title(
        f"A. FC Before Plasticity\n(coupling r={plasticity_result['coupling_before']:.3f})",
        fontweight="bold",
    )

    # B) FC after plasticity
    ax_after = fig.add_subplot(gs[0, 1])
    ax_after.imshow(
        plasticity_result["fc_after"], cmap=COUPLING_CMAP,
        vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest"
    )
    ax_after.set_title(
        f"B. FC After Plasticity\n(coupling r={plasticity_result['coupling_after']:.3f})",
        fontweight="bold",
    )

    # C) Plasticity map (per-node weight change)
    ax_plast = fig.add_subplot(gs[1, 0])
    sorted_idx = np.argsort(plasticity_result["plasticity_map"])[::-1]
    ax_plast.bar(
        range(n),
        plasticity_result["plasticity_map"][sorted_idx],
        color=plt.cm.plasma(
            plasticity_result["plasticity_map"][sorted_idx]
            / max(plasticity_result["plasticity_map"].max(), 1e-10)
        ),
        width=0.8,
    )
    ax_plast.set_xlabel("Nodes (sorted by plasticity)")
    ax_plast.set_ylabel("Total Weight Change")
    ax_plast.set_title("C. Regional Plasticity Map", fontweight="bold")

    # D) Summary metrics
    ax_summary = fig.add_subplot(gs[1, 1])
    ax_summary.axis("off")

    improvement = plasticity_result["coupling_improvement"]
    direction = "↑" if improvement > 0 else "↓"

    summary_lines = [
        f"Coupling before:  r = {plasticity_result['coupling_before']:.4f}",
        f"Coupling after:   r = {plasticity_result['coupling_after']:.4f}",
        f"Change:           {direction} {abs(improvement):.4f}",
        f"",
        f"Mean weight Δ:    {plasticity_result['weight_change_mean']:.6f}",
        f"Weights changed:  {plasticity_result['n_weights_changed']}",
    ]

    for i, line in enumerate(summary_lines):
        ax_summary.text(
            0.1, 0.85 - i * 0.12, line,
            transform=ax_summary.transAxes,
            fontsize=11, fontfamily="monospace",
            color="#333",
        )

    ax_summary.set_title("D. Plasticity Summary", fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# COMPREHENSIVE DASHBOARD
# =============================================================================

def plot_reservoir_dashboard(
    result: ReservoirDynamicsResult,
    region_labels: Optional[List[str]] = None,
    network_labels: Optional[np.ndarray] = None,
    network_names: Optional[Dict[int, str]] = None,
    figsize: Tuple[float, float] = (20, 16),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Comprehensive dashboard combining all reservoir dynamics analyses.

    Creates a publication-ready multi-panel figure summarizing:
    - SC-FC coupling matrices
    - SDI regional profile
    - Eigenspectrum decomposition
    - Perturbation analysis (if available)
    - Summary statistics

    Parameters
    ----------
    result : ReservoirDynamicsResult
        Full pipeline results.
    region_labels : list of str, optional
        Region labels.
    network_labels : np.ndarray, optional
        Network assignments.
    network_names : dict, optional
        Network names.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    n_rows = 3 if result.perturbation is not None else 2
    fig = plt.figure(figsize=(figsize[0], figsize[1] * n_rows / 2.5))
    gs = gridspec.GridSpec(n_rows, 4, hspace=0.4, wspace=0.35)

    n = result.n_regions
    coupling = result.coupling
    sdi = result.sdi

    # Row 1: SC, FC empirical, FC simulated, SC-FC scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(coupling.sc_matrix, cmap="hot", aspect="equal", interpolation="nearest")
    ax1.set_title("SC Matrix", fontweight="bold", fontsize=10)

    ax2 = fig.add_subplot(gs[0, 1])
    vmax = np.percentile(np.abs(coupling.fc_empirical), 95)
    ax2.imshow(
        coupling.fc_empirical, cmap=COUPLING_CMAP,
        vmin=-vmax, vmax=vmax, aspect="equal", interpolation="nearest"
    )
    ax2.set_title("Empirical FC", fontweight="bold", fontsize=10)

    ax3 = fig.add_subplot(gs[0, 2])
    vmax_s = np.percentile(np.abs(coupling.fc_simulated), 95)
    ax3.imshow(
        coupling.fc_simulated, cmap=COUPLING_CMAP,
        vmin=-vmax_s, vmax=vmax_s, aspect="equal", interpolation="nearest"
    )
    ax3.set_title(f"Simulated FC (ρ={coupling.spectral_radius})", fontweight="bold", fontsize=10)

    ax4 = fig.add_subplot(gs[0, 3])
    idx = np.triu_indices(n, k=1)
    sim_u = coupling.fc_simulated[idx]
    emp_u = coupling.fc_empirical[idx]
    if len(sim_u) > 3000:
        rng = np.random.default_rng(42)
        s = rng.choice(len(sim_u), 3000, replace=False)
        ax4.scatter(sim_u[s], emp_u[s], alpha=0.1, s=3, c="#4CAF50", edgecolors="none")
    else:
        ax4.scatter(sim_u, emp_u, alpha=0.1, s=3, c="#4CAF50", edgecolors="none")
    ax4.set_xlabel("Simulated FC", fontsize=9)
    ax4.set_ylabel("Empirical FC", fontsize=9)
    ax4.set_title(f"r = {coupling.coupling_reservoir:.3f}", fontweight="bold", fontsize=10)

    # Row 2: SDI bar, Coupled FC, Decoupled FC, Eigenspectrum
    ax5 = fig.add_subplot(gs[1, 0])
    sorted_sdi = np.sort(sdi.sdi)
    ax5.barh(range(n), sorted_sdi, color=plt.cm.YlOrRd(sorted_sdi), height=0.8)
    ax5.set_xlabel("SDI", fontsize=9)
    ax5.set_title("SDI Profile", fontweight="bold", fontsize=10)
    ax5.set_yticks([])

    ax6 = fig.add_subplot(gs[1, 1])
    vmax_c = np.percentile(np.abs(sdi.fc_coupled), 95)
    if vmax_c < 1e-10:
        vmax_c = 1.0
    ax6.imshow(
        sdi.fc_coupled, cmap=COUPLING_CMAP,
        vmin=-vmax_c, vmax=vmax_c, aspect="equal", interpolation="nearest"
    )
    ax6.set_title("Coupled FC", fontweight="bold", fontsize=10)

    ax7 = fig.add_subplot(gs[1, 2])
    vmax_d = np.percentile(np.abs(sdi.fc_decoupled), 95)
    if vmax_d < 1e-10:
        vmax_d = 1.0
    ax7.imshow(
        sdi.fc_decoupled, cmap=COUPLING_CMAP,
        vmin=-vmax_d, vmax=vmax_d, aspect="equal", interpolation="nearest"
    )
    ax7.set_title("Decoupled FC", fontweight="bold", fontsize=10)

    ax8 = fig.add_subplot(gs[1, 3])
    k = sdi.n_coupled_components
    ax8.fill_between(range(k), sdi.sc_eigenvalues[:k], alpha=0.3, color="#1976D2")
    ax8.fill_between(range(k, n), sdi.sc_eigenvalues[k:], alpha=0.3, color="#D32F2F")
    ax8.plot(sdi.sc_eigenvalues, "k-", linewidth=0.8)
    ax8.axvline(x=k, color="k", linestyle="--", linewidth=0.6)
    ax8.set_xlabel("Mode", fontsize=9)
    ax8.set_ylabel("λ", fontsize=9)
    ax8.set_title("Graph Spectrum", fontweight="bold", fontsize=10)

    # Row 3: Perturbation (if available)
    if result.perturbation is not None and n_rows == 3:
        pert = result.perturbation
        ax9 = fig.add_subplot(gs[2, :2])
        sorted_v = np.argsort(pert.vulnerability_index)[::-1]
        colors_v = plt.cm.RdYlGn_r(np.linspace(0, 1, n))
        ax9.bar(range(n), pert.vulnerability_index[sorted_v], color=colors_v, width=0.8)
        ax9.set_xlabel("Nodes (sorted)")
        ax9.set_ylabel("Vulnerability")
        ax9.set_title("Node Vulnerability", fontweight="bold", fontsize=10)

        ax10 = fig.add_subplot(gs[2, 2:])
        ax10.axis("off")
        summary = (
            f"Atlas: {result.atlas_name}\n"
            f"Regions: {result.n_regions}\n"
            f"────────────────────────\n"
            f"SC-FC (direct): r = {coupling.coupling_global:.4f}\n"
            f"SC-FC (reservoir): r = {coupling.coupling_reservoir:.4f}\n"
            f"Decoupling index: {coupling.decoupling_index_global:.4f}\n"
            f"Readout R²: {coupling.prediction_r2:.4f}\n"
            f"────────────────────────\n"
            f"SDI range: [{sdi.sdi.min():.3f}, {sdi.sdi.max():.3f}]\n"
            f"Coupling ratio: {sdi.coupling_ratio:.3f}\n"
            f"────────────────────────\n"
            f"Baseline coupling: {pert.baseline_coupling:.4f}\n"
            f"Most vulnerable: node {pert.most_vulnerable_nodes[0]}"
        )
        ax10.text(
            0.05, 0.95, summary, transform=ax10.transAxes,
            fontsize=9, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5"),
        )
        ax10.set_title("Summary", fontweight="bold", fontsize=10)

    fig.suptitle(
        f"Reservoir-Based SC-FC Decoupling Analysis — {result.atlas_name.upper()}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# MULTI-ATLAS COMPARISON
# =============================================================================

def plot_multi_atlas_comparison(
    results: Dict[str, ReservoirDynamicsResult],
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare SC-FC coupling metrics across multiple atlases.

    Parameters
    ----------
    results : dict
        Atlas name → ReservoirDynamicsResult mapping.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Path to save.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    atlas_names = list(results.keys())
    n_atlases = len(atlas_names)
    atlas_colors = plt.cm.Set2(np.linspace(0, 1, n_atlases))

    # A) Global coupling comparison
    ax = axes[0]
    coupling_vals = [results[a].coupling.coupling_reservoir for a in atlas_names]
    ax.bar(atlas_names, coupling_vals, color=atlas_colors, width=0.5)
    ax.set_ylabel("SC-FC Coupling (r)")
    ax.set_title("A. Reservoir Coupling", fontweight="bold")
    ax.set_xticklabels(atlas_names, rotation=30, ha="right")

    # B) SDI distribution comparison
    ax = axes[1]
    sdi_data = [results[a].sdi.sdi for a in atlas_names]
    bp = ax.boxplot(
        sdi_data, labels=atlas_names, patch_artist=True,
        medianprops=dict(color="black"),
    )
    for patch, color in zip(bp["boxes"], atlas_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("SDI")
    ax.set_title("B. SDI Distribution", fontweight="bold")
    ax.set_xticklabels(atlas_names, rotation=30, ha="right")

    # C) Coupling ratio
    ax = axes[2]
    ratio_vals = [results[a].sdi.coupling_ratio for a in atlas_names]
    ax.bar(atlas_names, ratio_vals, color=atlas_colors, width=0.5)
    ax.set_ylabel("Coupling Ratio")
    ax.set_title("C. FC Coupling Ratio", fontweight="bold")
    ax.set_xticklabels(atlas_names, rotation=30, ha="right")
    ax.axhline(y=0.5, color="k", linestyle="--", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "plot_sc_fc_coupling",
    "plot_sdi",
    "plot_perturbation",
    "plot_group_comparison",
    "plot_spectral_sweep",
    "plot_plasticity_analysis",
    "plot_reservoir_dashboard",
    "plot_multi_atlas_comparison",
]
