# -*- coding: utf-8 -*-
"""
sars.eigenmorph.viz
==================================================

Publication-quality visualization for eigenvalue morphology features.

Provides surface plots, multi-scale profiles, ternary feature-space
diagrams, and comparison heatmaps against classical FreeSurfer metrics.
All plots follow journal-ready defaults (300 dpi, tight layout,
colorblind-safe palettes).

Functions
---------
plot_surface_feature
    3D cortical surface colored by a single scalar feature.
plot_feature_overview
    Seven-panel overview (one per eigenvalue feature).
plot_multiscale_profile
    Feature evolution across spatial scales.
plot_ternary_features
    Linearity–Planarity–Sphericity ternary diagram +
    eigenentropy vs surface variation.
plot_classical_comparison
    Correlation heatmap and unique-variance bar chart.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Dict, Tuple

from .core import SurfaceMesh, EigenFeatures, MultiScaleFeatures


# =============================================================================
# STYLE
# =============================================================================

def _setup_style():
    """Apply publication-ready matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# =============================================================================
# SURFACE PLOTS
# =============================================================================

def plot_surface_feature(
    mesh: SurfaceMesh,
    values: np.ndarray,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    view: Tuple[float, float] = (30, -60),
    figsize: Tuple[int, int] = (8, 6),
    colorbar: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a 3D cortical surface colored by a scalar feature.

    Parameters
    ----------
    mesh : SurfaceMesh
    values : np.ndarray (V,)
        Scalar value per vertex.
    title : str
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Colormap range.  Defaults to 5th / 95th percentile.
    ax : plt.Axes, optional
        Existing 3D axes.
    view : tuple (elevation, azimuth)
    figsize : tuple
    colorbar : bool

    Returns
    -------
    fig, ax
    """
    _setup_style()

    vals = values.copy()
    vals[np.isnan(vals)] = 0

    if vmin is None:
        vmin = np.nanpercentile(values, 5)
    if vmax is None:
        vmax = np.nanpercentile(values, 95)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    face_colors = np.zeros((mesh.n_faces, 4))
    for i, face in enumerate(mesh.faces):
        face_val = np.mean(vals[face])
        face_colors[i] = mapper.to_rgba(face_val)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    polys = Poly3DCollection(
        mesh.vertices[mesh.faces],
        facecolors=face_colors,
        edgecolors="none",
        linewidths=0,
    )
    ax.add_collection3d(polys)

    v = mesh.vertices
    margin = 5
    ax.set_xlim(v[:, 0].min() - margin, v[:, 0].max() + margin)
    ax.set_ylim(v[:, 1].min() - margin, v[:, 1].max() + margin)
    ax.set_zlim(v[:, 2].min() - margin, v[:, 2].max() + margin)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(title, fontweight="bold")
    ax.axis("off")

    if colorbar:
        fig.colorbar(mapper, ax=ax, shrink=0.6, pad=0.05)

    return fig, ax


def plot_feature_overview(
    mesh: SurfaceMesh,
    features: EigenFeatures,
    figsize: Tuple[int, int] = (24, 9),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Seven-panel overview: one surface per eigenvalue feature.

    Parameters
    ----------
    mesh : SurfaceMesh
    features : EigenFeatures
        Single-scale features.
    figsize : tuple
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()

    feature_info = [
        ("linearity", "Linearity\n(λ₁−λ₂)/λ₁", "YlOrRd"),
        ("planarity", "Planarity\n(λ₂−λ₃)/λ₁", "YlGnBu"),
        ("sphericity", "Sphericity\nλ₃/λ₁", "PuRd"),
        ("omnivariance", "Omnivariance\n(λ₁λ₂λ₃)^⅓", "viridis"),
        ("anisotropy", "Anisotropy\n(λ₁−λ₃)/λ₁", "magma"),
        ("eigenentropy", "Eigenentropy\n−Σλ̃ln(λ̃)", "cividis"),
        ("surface_variation", "Surf. Variation\nλ₃/Σλ", "plasma"),
    ]

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Eigenvalue Features (r = {features.radius:.0f} mm)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for idx, (attr, label, cmap) in enumerate(feature_info):
        ax = fig.add_subplot(2, 4, idx + 1, projection="3d")
        vals = getattr(features, attr)
        plot_surface_feature(
            mesh, vals, title=label, cmap=cmap,
            ax=ax, colorbar=True,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# MULTI-SCALE PROFILES
# =============================================================================

def plot_multiscale_profile(
    ms_features: MultiScaleFeatures,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature evolution across spatial scales.

    Shows mean ± std for each feature as a function of neighborhood
    radius, revealing how cortical geometry changes from local to
    global scales.

    Parameters
    ----------
    ms_features : MultiScaleFeatures
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()

    feat_names = EigenFeatures.feature_names()
    radii = ms_features.radii
    palette = plt.cm.Set2(np.linspace(0, 1, len(feat_names)))

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.ravel()

    for idx, fn in enumerate(feat_names):
        ax = axes[idx]
        means, stds = [], []
        for s in ms_features.scales:
            vals = getattr(s, fn)
            valid = vals[~np.isnan(vals)]
            means.append(np.mean(valid) if len(valid) > 0 else 0)
            stds.append(np.std(valid) if len(valid) > 0 else 0)

        means = np.array(means)
        stds = np.array(stds)

        ax.fill_between(radii, means - stds, means + stds,
                        alpha=0.25, color=palette[idx])
        ax.plot(radii, means, "o-", color=palette[idx],
                linewidth=2, markersize=6)
        ax.set_title(fn, fontweight="bold")
        ax.set_xlabel("Radius (mm)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    # Hide extra subplot
    if len(feat_names) < len(axes):
        for i in range(len(feat_names), len(axes)):
            axes[i].set_visible(False)

    fig.suptitle("Multi-scale Feature Profiles",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# TERNARY FEATURE SPACE
# =============================================================================

def plot_ternary_features(
    features: EigenFeatures,
    max_points: int = 10000,
    seed: int = 42,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Ternary diagram (L–P–S) and eigenentropy vs surface variation.

    Left panel: Linearity + Planarity + Sphericity ≈ 1, so they can be
    visualized in a ternary space. Gyral crowns cluster near the
    planarity vertex; sulcal fundi near linearity; sulcal pits near
    sphericity.

    Right panel: Eigenentropy vs surface variation, colored by
    anisotropy.

    Parameters
    ----------
    features : EigenFeatures
    max_points : int
        Subsample for performance.
    seed : int
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()

    valid = ~np.isnan(features.linearity)
    indices = np.where(valid)[0]

    rng = np.random.default_rng(seed)
    if len(indices) > max_points:
        indices = rng.choice(indices, max_points, replace=False)

    L = features.linearity[indices]
    P = features.planarity[indices]
    S = features.sphericity[indices]

    # Ternary → Cartesian
    total = L + P + S
    total[total == 0] = 1
    L_n, P_n, S_n = L / total, P / total, S / total

    x = 0.5 * (2 * P_n + S_n)
    y = (np.sqrt(3) / 2) * S_n

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: Ternary ---
    scatter = ax1.scatter(x, y, c=features.eigenentropy[indices],
                          s=1, alpha=0.3, cmap="viridis", rasterized=True)
    fig.colorbar(scatter, ax=ax1, label="Eigenentropy")

    # Triangle outline
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax1.plot(tri_x, tri_y, "k-", linewidth=1.5)

    ax1.text(0, -0.05, "Linearity\n(ridges)", ha="center", fontsize=9)
    ax1.text(1, -0.05, "Planarity\n(flat)", ha="center", fontsize=9)
    ax1.text(0.5, np.sqrt(3) / 2 + 0.05, "Sphericity\n(pits)",
             ha="center", fontsize=9)

    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.15, 1.05)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title(f"L–P–S Ternary (r={features.radius:.0f}mm)",
                  fontweight="bold")

    # --- Right: Eigenentropy vs Surface variation ---
    scatter2 = ax2.scatter(
        features.eigenentropy[indices],
        features.surface_variation[indices],
        c=features.anisotropy[indices],
        s=1, alpha=0.3, cmap="magma", rasterized=True,
    )
    fig.colorbar(scatter2, ax=ax2, label="Anisotropy")
    ax2.set_xlabel("Eigenentropy")
    ax2.set_ylabel("Surface Variation")
    ax2.set_title("Complexity vs Roughness", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# CLASSICAL COMPARISON HEATMAP
# =============================================================================

def plot_classical_comparison(
    comparison: Dict,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Correlation heatmap and unique-variance bar chart.

    Left panel: heatmap of Pearson correlations (eigenvalue features ×
    classical metrics).  Right panel: unique variance per eigenvalue
    feature, with the 50% threshold highlighted.

    Parameters
    ----------
    comparison : dict
        Output from ``compare_with_classical()``.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()

    corr = comparison["correlations"]
    eigen_names = comparison["eigen_names"]
    classical_names = comparison["classical_names"]
    unique_var = comparison["unique_variance"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # --- Left: Correlation heatmap ---
    im = ax1.imshow(corr, aspect="auto", cmap="RdBu_r",
                    vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax1, label="Pearson r", shrink=0.8)

    ax1.set_xticks(range(len(classical_names)))
    ax1.set_xticklabels(classical_names, rotation=45, ha="right")
    ax1.set_yticks(range(len(eigen_names)))
    ax1.set_yticklabels(eigen_names, fontsize=8)
    ax1.set_title("Eigenvalue vs Classical Correlations",
                  fontweight="bold")

    # Annotate
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            val = corr[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6, color=color)

    # --- Right: Unique variance bars ---
    y_pos = np.arange(len(eigen_names))
    bar_colors = ["#e74c3c" if uv > 0.5 else "#95a5a6"
                  for uv in unique_var]
    ax2.barh(y_pos, unique_var, color=bar_colors, edgecolor="none")
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=1,
                label="50% threshold")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(eigen_names, fontsize=8)
    ax2.set_xlabel("Unique Variance")
    ax2.set_title("Novel Information", fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=8)
    ax2.invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
