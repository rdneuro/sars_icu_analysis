# -*- coding: utf-8 -*-
"""
sars.viz.matrix_plots
====================================

Publication-quality visualizations for connectivity matrices, graph
metrics distributions, and statistical summaries.

All functions return (fig, axes) tuples for further customization.
Default parameters follow high-impact journal standards:
  - 300 DPI, sans-serif font (DejaVu Sans / Arial)
  - Clean axes, proper colorbar placement
  - Informative titles and labels

Usage
-----
    from sars.viz import plot_connectivity_matrix
    fig, ax = plot_connectivity_matrix(fc_matrix, atlas="schaefer_100")
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Union
import warnings


# =============================================================================
# PLOTTING STYLE DEFAULTS
# =============================================================================

_DEFAULT_STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style():
    """Apply publication-quality matplotlib style."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.rcParams.update(_DEFAULT_STYLE)


# =============================================================================
# CONNECTIVITY MATRIX
# =============================================================================

def plot_connectivity_matrix(
    matrix: np.ndarray,
    atlas: Optional[str] = None,
    roi_labels: Optional[List[str]] = None,
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric_cbar: bool = True,
    show_labels: bool = False,
    network_order: Optional[Dict[str, List[int]]] = None,
    network_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (8, 7),
    cbar_label: str = "Connectivity",
    annotate_networks: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a connectivity matrix with optional network-based ordering.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    atlas : str, optional
        If 'schaefer_100', auto-detect network ordering.
    roi_labels : list of str, optional
        ROI names for axis tick labels.
    title : str
    cmap : str
        Colormap. 'RdBu_r' for FC (diverging), 'hot'/'YlOrRd' for SC.
    vmin, vmax : float, optional
    symmetric_cbar : bool
        If True, center colorbar at 0 (for FC matrices).
    show_labels : bool
        If True, show ROI labels on axes (works best for N < 50).
    network_order : dict, optional
        {network_name: [roi_indices]}. If provided, reorders matrix.
    network_colors : dict, optional
        {network_name: color_hex}.
    figsize : tuple
    cbar_label : str
    annotate_networks : bool
        If True and network_order is given, draw network boundaries.

    Returns
    -------
    fig, ax
    """
    _apply_style()

    mat = matrix.copy()
    N = mat.shape[0]

    # Auto-detect network ordering for Schaefer
    if atlas and "schaefer" in atlas and network_order is None:
        try:
            from ..data import get_schaefer_network_indices
            network_order = get_schaefer_network_indices(atlas)
        except Exception:
            pass

    # Reorder by network
    if network_order is not None:
        order = []
        net_boundaries = []
        net_names = []
        for net_name, indices in network_order.items():
            net_boundaries.append(len(order))
            net_names.append(net_name)
            order.extend(indices)
        net_boundaries.append(len(order))

        # Apply reordering
        mat = mat[np.ix_(order, order)]
        if roi_labels:
            roi_labels = [roi_labels[i] for i in order]

    # Determine color limits
    if symmetric_cbar and vmin is None and vmax is None:
        abs_max = np.max(np.abs(mat[~np.eye(N, dtype=bool)]))
        vmin, vmax = -abs_max, abs_max
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        if vmin is None:
            vmin = np.nanmin(mat)
        if vmax is None:
            vmax = np.nanmax(mat)
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal",
                   interpolation="nearest")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        label=cbar_label)

    # Network boundaries and annotations
    if network_order is not None and annotate_networks:
        # Default Yeo 7-network colors
        if network_colors is None:
            network_colors = {
                "Visual": "#781286",
                "SomMot": "#4682B4",
                "DorsAttn": "#00760E",
                "SalVentAttn": "#C43AFA",
                "Limbic": "#DCF8A4",
                "Cont": "#E69422",
                "Default": "#CD3E4E",
            }

        for i, (name, start) in enumerate(zip(net_names, net_boundaries[:-1])):
            end = net_boundaries[i + 1]
            size = end - start
            color = network_colors.get(name, "#888888")

            # Draw boundary lines
            ax.axhline(y=start - 0.5, color="white", linewidth=0.5)
            ax.axvline(x=start - 0.5, color="white", linewidth=0.5)

            # Network color bars on axes
            rect_y = Rectangle((-3.5, start - 0.5), 2.5, size,
                               facecolor=color, edgecolor="none",
                               clip_on=False)
            rect_x = Rectangle((start - 0.5, -3.5), size, 2.5,
                               facecolor=color, edgecolor="none",
                               clip_on=False)
            ax.add_patch(rect_y)
            ax.add_patch(rect_x)

            # Network name labels
            mid = (start + end) / 2
            ax.text(-5, mid, name, ha="right", va="center", fontsize=7,
                    color=color, fontweight="bold", clip_on=False)

    # ROI labels
    if show_labels and roi_labels and N < 60:
        ax.set_xticks(range(N))
        ax.set_xticklabels(roi_labels, rotation=90, fontsize=5)
        ax.set_yticks(range(N))
        ax.set_yticklabels(roi_labels, fontsize=5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()
    return fig, ax


# =============================================================================
# MATRIX COMPARISON (FC vs SC)
# =============================================================================

def plot_matrix_comparison(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    title1: str = "Functional Connectivity",
    title2: str = "Structural Connectivity",
    cmap1: str = "RdBu_r",
    cmap2: str = "YlOrRd",
    figsize: Tuple[float, float] = (14, 6),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side comparison of two connectivity matrices.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Matrix 1 (FC-like, diverging)
    abs_max1 = np.max(np.abs(matrix1[~np.eye(matrix1.shape[0], dtype=bool)]))
    norm1 = TwoSlopeNorm(vmin=-abs_max1, vcenter=0, vmax=abs_max1)
    im1 = axes[0].imshow(matrix1, cmap=cmap1, norm=norm1, aspect="equal")
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title(title1, fontweight="bold")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Matrix 2 (SC-like, sequential)
    mat2 = matrix2.copy()
    np.fill_diagonal(mat2, 0)
    im2 = axes[1].imshow(mat2, cmap=cmap2, aspect="equal",
                         vmin=0, vmax=np.percentile(mat2[mat2 > 0], 98)
                         if np.any(mat2 > 0) else 1)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(title2, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()
    return fig, axes


# =============================================================================
# THRESHOLD COMPARISON PANEL
# =============================================================================

def plot_threshold_panel(
    matrix: np.ndarray,
    thresholds: Optional[List[float]] = None,
    cmap: str = "RdBu_r",
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: str = "Matrix at Different Density Thresholds",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Panel showing a connectivity matrix at multiple density thresholds.

    Useful for visualizing how network topology changes with threshold
    choice (van den Heuvel et al. 2017).
    """
    _apply_style()
    from ..data import threshold_matrix

    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    n_thr = len(thresholds)
    ncols = min(3, n_thr)
    nrows = int(np.ceil(n_thr / ncols))
    if figsize is None:
        figsize = (5 * ncols, 4.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_thr == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, thr in enumerate(thresholds):
        mat_thr = threshold_matrix(matrix, method="density", value=thr,
                                   absolute=True)
        n_edges = np.sum(mat_thr > 0) // 2
        abs_max = np.max(np.abs(mat_thr))
        if abs_max > 0:
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        else:
            norm = Normalize(vmin=-1, vmax=1)

        axes[i].imshow(mat_thr, cmap=cmap, norm=norm, aspect="equal")
        axes[i].set_title(f"Density = {thr:.0%}\n({n_edges} edges)",
                          fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, axes


# =============================================================================
# DEGREE / STRENGTH DISTRIBUTION
# =============================================================================

def plot_degree_distribution(
    matrix: np.ndarray,
    weighted: bool = True,
    log_scale: bool = False,
    title: str = "Degree Distribution",
    figsize: Tuple[float, float] = (10, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot degree (binary) or strength (weighted) distribution with
    histogram and complementary cumulative distribution (CCDF).
    """
    _apply_style()
    from ..graph_analysis.network_metrics import _prepare_matrix

    mat = _prepare_matrix(matrix)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if weighted:
        values = np.sum(mat, axis=1)
        xlabel = "Node Strength"
    else:
        values = np.sum(mat > 0, axis=1)
        xlabel = "Node Degree"

    # Histogram
    axes[0].hist(values, bins="auto", color="#4C72B0", edgecolor="white",
                 alpha=0.85, density=True)
    axes[0].axvline(np.mean(values), color="#C44E52", linestyle="--",
                    linewidth=1.5, label=f"Mean = {np.mean(values):.2f}")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution")
    axes[0].legend(frameon=False)

    # CCDF (complementary cumulative distribution)
    sorted_vals = np.sort(values)
    ccdf = 1 - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axes[1].plot(sorted_vals, ccdf, "o-", color="#4C72B0", markersize=4,
                 linewidth=1.2)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("P(X > x)")
    axes[1].set_title("CCDF")
    if log_scale:
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, axes


# =============================================================================
# GENERIC METRIC DISTRIBUTION
# =============================================================================

def plot_metric_distribution(
    values: np.ndarray,
    metric_name: str = "Metric",
    roi_labels: Optional[List[str]] = None,
    highlight_top_n: int = 5,
    figsize: Tuple[float, float] = (12, 5),
    color: str = "#4C72B0",
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot a nodal metric as a bar chart (ranked) + histogram.

    Highlights the top-N regions by metric value.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                              gridspec_kw={"width_ratios": [2, 1]})

    N = len(values)
    sorted_idx = np.argsort(values)[::-1]

    # Bar chart (ranked)
    bars = axes[0].bar(range(N), values[sorted_idx], color=color,
                       edgecolor="none", alpha=0.7)
    # Highlight top N
    for i in range(min(highlight_top_n, N)):
        bars[i].set_color("#C44E52")
        bars[i].set_alpha(1.0)
        if roi_labels:
            label = roi_labels[sorted_idx[i]]
            axes[0].text(i, values[sorted_idx[i]], f" {label}",
                        rotation=45, fontsize=6, va="bottom", ha="left")

    axes[0].set_xlabel("Nodes (ranked)")
    axes[0].set_ylabel(metric_name)
    axes[0].set_title(f"{metric_name} (Ranked)")

    # Histogram
    axes[1].hist(values[~np.isnan(values)], bins="auto", color=color,
                 edgecolor="white", alpha=0.85, density=True)
    axes[1].axvline(np.nanmean(values), color="#C44E52", linestyle="--",
                    linewidth=1.5, label=f"Mean = {np.nanmean(values):.3f}")
    axes[1].set_xlabel(metric_name)
    axes[1].set_ylabel("Density")
    axes[1].set_title("Distribution")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    return fig, axes


# =============================================================================
# SC-FC SCATTER PLOT
# =============================================================================

def plot_sc_fc_scatter(
    sc_matrix: np.ndarray,
    fc_matrix: np.ndarray,
    method: str = "spearman",
    title: str = "Structure-Function Coupling",
    figsize: Tuple[float, float] = (7, 6),
    alpha: float = 0.15,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of SC vs FC edge weights (upper triangle).

    Replicates the classic SC-FC coupling visualization from
    Sporns (2013), Figure 6.
    """
    _apply_style()
    from ..data import get_upper_triangle
    from scipy import stats as sp_stats

    sc_triu = get_upper_triangle(sc_matrix, k=1)
    fc_triu = get_upper_triangle(fc_matrix, k=1)

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter all edges
    ax.scatter(sc_triu, fc_triu, alpha=alpha, s=8, color="#4C72B0",
               edgecolor="none", rasterized=True)

    # Highlight structurally connected edges
    mask = sc_triu > 0
    if np.sum(mask) > 2:
        if method == "spearman":
            r, p = sp_stats.spearmanr(sc_triu[mask], fc_triu[mask])
        else:
            r, p = sp_stats.pearsonr(sc_triu[mask], fc_triu[mask])

        # Regression line on connected edges
        z = np.polyfit(sc_triu[mask], fc_triu[mask], 1)
        x_line = np.linspace(sc_triu[mask].min(), sc_triu[mask].max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color="#C44E52",
                linewidth=2, label=f"r = {r:.3f} (p = {p:.2e})")

    ax.set_xlabel("Structural Connectivity")
    ax.set_ylabel("Functional Connectivity")
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


# =============================================================================
# EDGE WEIGHT DISTRIBUTION
# =============================================================================

def plot_edge_weight_distribution(
    matrix: np.ndarray,
    title: str = "Edge Weight Distribution",
    log_y: bool = True,
    figsize: Tuple[float, float] = (8, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot distribution of non-zero edge weights.
    """
    _apply_style()
    from ..data import get_upper_triangle

    triu = get_upper_triangle(matrix, k=1)
    nonzero = triu[triu != 0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(nonzero, bins=80, color="#4C72B0", edgecolor="white",
            alpha=0.85)
    ax.axvline(np.mean(nonzero), color="#C44E52", linestyle="--",
               linewidth=1.5, label=f"Mean = {np.mean(nonzero):.3f}")
    ax.axvline(np.median(nonzero), color="#55A868", linestyle=":",
               linewidth=1.5, label=f"Median = {np.median(nonzero):.3f}")
    ax.set_xlabel("Edge Weight")
    ax.set_ylabel("Count")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


# =============================================================================
# COMMUNITY-ORDERED MATRIX
# =============================================================================

def plot_community_matrix(
    matrix: np.ndarray,
    community_labels: np.ndarray,
    title: str = "Community-Ordered Connectivity",
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (8, 7),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot connectivity matrix reordered by community assignment.
    Community boundaries are marked with lines.
    """
    _apply_style()

    N = matrix.shape[0]
    labels = np.asarray(community_labels)
    unique_modules = np.unique(labels)

    # Create reordering index
    order = []
    boundaries = [0]
    for mod in unique_modules:
        idx = np.where(labels == mod)[0]
        order.extend(idx)
        boundaries.append(len(order))

    mat = matrix[np.ix_(order, order)]

    # Plot
    abs_max = np.max(np.abs(mat[~np.eye(N, dtype=bool)]))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max) \
        if abs_max > 0 else Normalize(-1, 1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Draw community boundaries
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_modules)))
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax.axhline(y=start - 0.5, color="black", linewidth=0.8)
        ax.axvline(x=start - 0.5, color="black", linewidth=0.8)
        # Module label
        mid = (start + end) / 2
        ax.text(-2, mid, f"M{i}", ha="right", va="center", fontsize=8,
                fontweight="bold", color=colors[i], clip_on=False)

    ax.axhline(y=boundaries[-1] - 0.5, color="black", linewidth=0.8)
    ax.axvline(x=boundaries[-1] - 0.5, color="black", linewidth=0.8)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig, ax


# =============================================================================
# INTER-MODULE CONNECTIVITY
# =============================================================================

def plot_inter_module_connectivity(
    inter_mod_matrix: np.ndarray,
    module_names: Optional[List[str]] = None,
    title: str = "Inter-Module Connectivity",
    figsize: Tuple[float, float] = (6, 5),
    cmap: str = "YlOrRd",
    annotate: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot inter-module connectivity as an annotated heatmap.
    """
    _apply_style()

    M = inter_mod_matrix.shape[0]
    if module_names is None:
        module_names = [f"Module {i}" for i in range(M)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(inter_mod_matrix, cmap=cmap, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Mean Connectivity")

    if annotate and M <= 15:
        for i in range(M):
            for j in range(M):
                val = inter_mod_matrix[i, j]
                color = "white" if val > np.median(inter_mod_matrix) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_xticks(range(M))
    ax.set_xticklabels(module_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(M))
    ax.set_yticklabels(module_names, fontsize=8)
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig, ax


# =============================================================================
# RICH CLUB CURVE
# =============================================================================

def plot_rich_club_curve(
    rich_club_results: Dict,
    title: str = "Rich Club Organization",
    figsize: Tuple[float, float] = (10, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot rich club coefficient and normalized rich club.

    Two panels:
      Left: φ(k) empirical vs. null
      Right: φ_norm(k) with significance shading.
    """
    _apply_style()

    k = rich_club_results["k_levels"]
    phi = rich_club_results["phi"]
    phi_rand = rich_club_results["phi_rand_mean"]
    phi_rand_std = rich_club_results["phi_rand_std"]
    phi_norm = rich_club_results["phi_norm"]
    p_vals = rich_club_results["p_values"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A: empirical vs null
    axes[0].plot(k, phi, "o-", color="#C44E52", markersize=4,
                 linewidth=1.5, label="Empirical")
    axes[0].plot(k, phi_rand, "s--", color="#8C8C8C", markersize=3,
                 linewidth=1, label="Random (mean)")
    axes[0].fill_between(k, phi_rand - phi_rand_std,
                         phi_rand + phi_rand_std,
                         alpha=0.2, color="#8C8C8C")
    axes[0].set_xlabel("Degree (k)")
    axes[0].set_ylabel("Rich Club Coefficient φ(k)")
    axes[0].set_title("A) Rich Club vs. Null", fontweight="bold")
    axes[0].legend(frameon=False)

    # Panel B: normalized
    axes[1].plot(k, phi_norm, "o-", color="#4C72B0", markersize=4,
                 linewidth=1.5)
    axes[1].axhline(y=1.0, color="gray", linestyle=":", linewidth=1)
    # Shade significant k
    sig_mask = p_vals < 0.05
    if np.any(sig_mask):
        for ki, is_sig in zip(k, sig_mask):
            if is_sig:
                axes[1].axvspan(ki - 0.4, ki + 0.4, alpha=0.15,
                                color="#C44E52")
    axes[1].set_xlabel("Degree (k)")
    axes[1].set_ylabel("φ_norm(k)")
    axes[1].set_title("B) Normalized Rich Club", fontweight="bold")

    regime = rich_club_results.get("rich_club_regime")
    if regime:
        axes[1].text(0.05, 0.95,
                     f"Rich club: k ∈ [{regime[0]}, {regime[1]}]",
                     transform=axes[1].transAxes, fontsize=9,
                     va="top", ha="left",
                     bbox=dict(boxstyle="round", facecolor="wheat",
                               alpha=0.5))

    fig.tight_layout()
    return fig, axes


# =============================================================================
# SMALL-WORLD SUMMARY
# =============================================================================

def plot_small_world_summary(
    sw_results: Dict,
    figsize: Tuple[float, float] = (8, 4),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Summary visualization of small-world properties.

    Shows sigma, omega, and the gamma/lambda space with reference
    regions for small-world, lattice, and random networks.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A: omega-sigma bar
    metrics = {
        "σ (Humphries)": sw_results["sigma"],
        "ω (Telesford)": sw_results["omega"],
        "γ (C/C_rand)": sw_results["gamma"],
        "λ (L/L_rand)": sw_results["lambda_ratio"],
    }
    names = list(metrics.keys())
    vals = list(metrics.values())
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    bars = axes[0].barh(names, vals, color=colors, edgecolor="white",
                        height=0.6)
    axes[0].axvline(x=1.0, color="gray", linestyle=":", linewidth=1)
    axes[0].axvline(x=0.0, color="gray", linestyle="-", linewidth=0.5)
    for bar, val in zip(bars, vals):
        axes[0].text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=9)
    axes[0].set_xlabel("Value")
    axes[0].set_title("A) Small-World Metrics", fontweight="bold")

    # Panel B: gamma-lambda scatter with reference zones
    ax2 = axes[1]
    # Reference zones
    ax2.axhspan(0.8, 3.0, xmin=0, xmax=0.4, alpha=0.08, color="blue",
                label="Small-world zone")
    ax2.axhline(y=1, color="gray", linestyle=":", linewidth=0.8)
    ax2.axvline(x=1, color="gray", linestyle=":", linewidth=0.8)

    gamma = sw_results["gamma"]
    lam = sw_results["lambda_ratio"]
    ax2.scatter([lam], [gamma], s=120, color="#C44E52", edgecolor="black",
                linewidth=1.5, zorder=5)
    ax2.annotate(f"({lam:.2f}, {gamma:.2f})", (lam, gamma),
                 textcoords="offset points", xytext=(10, 10), fontsize=9)
    ax2.set_xlabel("λ = L/L_rand")
    ax2.set_ylabel("γ = C/C_rand")
    ax2.set_title("B) Small-World Space", fontweight="bold")

    # Interpretation
    interp = sw_results.get("interpretation", "")
    fig.text(0.5, -0.02, f"Interpretation: {interp}",
             ha="center", fontsize=10, fontstyle="italic")

    fig.tight_layout()
    return fig, axes


# =============================================================================
# AUC ACROSS THRESHOLDS
# =============================================================================

def plot_auc_across_thresholds(
    multi_threshold_results: Dict,
    metrics_to_plot: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot graph metrics as a function of density threshold with AUC.

    Each subplot shows one metric across thresholds, with the AUC
    value annotated.
    """
    _apply_style()

    thresholds = multi_threshold_results["thresholds"]
    metrics_list = multi_threshold_results["metrics_per_threshold"]
    auc = multi_threshold_results["auc"]

    if metrics_to_plot is None:
        metrics_to_plot = [
            "clustering_coeff", "global_efficiency",
            "char_path_length", "local_efficiency",
            "density", "assortativity",
        ]
    # Filter to available metrics
    metrics_to_plot = [m for m in metrics_to_plot if m in auc]

    n_met = len(metrics_to_plot)
    ncols = min(3, n_met)
    nrows = int(np.ceil(n_met / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, n_met))

    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        vals = [m.get(metric_name, np.nan) for m in metrics_list]
        ax.plot(thresholds, vals, "o-", color=colors[idx], linewidth=2,
                markersize=6)
        ax.fill_between(thresholds, 0, vals, alpha=0.15, color=colors[idx])
        ax.set_xlabel("Density Threshold")
        ax.set_ylabel(metric_name.replace("_", " ").title())

        auc_val = auc.get(metric_name, np.nan)
        ax.set_title(f"{metric_name.replace('_', ' ').title()}\nAUC = {auc_val:.4f}",
                     fontweight="bold", fontsize=10)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Graph Metrics Across Density Thresholds",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, axes
