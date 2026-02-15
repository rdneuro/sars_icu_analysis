# -*- coding: utf-8 -*-
"""
sars.viz.brain_plots
===================================

Brain-based visualizations using nilearn's plotting functions.

Provides functions for:
  - Connectome plots (glass brain with edges)
  - Nodal metric overlays on brain surfaces
  - Community assignment brain maps
  - Hub identification visualizations
  - SC-FC coupling brain maps

All functions use nilearn's plotting API, which renders MNI-space
coordinates as glass-brain projections or surface plots.

Requirements
------------
- nilearn >= 0.10
- nibabel >= 4.0
- Atlas NIfTI files for coordinate extraction

References
----------
- Abraham et al. (2014). Front Neuroinform. nilearn.
- Sporns (2013). Dialogues Clin Neurosci 15:247.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Union
import warnings


# =============================================================================
# HELPER: GET ROI COORDINATES
# =============================================================================

def _get_roi_coordinates(
    atlas: str,
) -> np.ndarray:
    """
    Extract MNI coordinates (centroids) for each ROI in the atlas.

    Parameters
    ----------
    atlas : str
        Atlas name from config.ATLASES.

    Returns
    -------
    np.ndarray (N, 3): MNI coordinates for each ROI.
    """
    from .. import config

    atlas_info = config.ATLASES[atlas]
    if atlas_info.nifti_file is None or not atlas_info.nifti_file.exists():
        raise FileNotFoundError(
            f"NIfTI file for atlas '{atlas}' not found: {atlas_info.nifti_file}\n"
            "Brain plots require atlas NIfTI files for coordinate extraction."
        )

    from nilearn.image import load_img
    from nilearn.plotting import find_parcellation_cut_coords

    atlas_img = load_img(str(atlas_info.nifti_file))
    coords = find_parcellation_cut_coords(atlas_img)

    expected_n = atlas_info.n_rois
    if len(coords) != expected_n:
        warnings.warn(
            f"Atlas '{atlas}': expected {expected_n} ROIs but found "
            f"{len(coords)} coordinates. Using available coordinates."
        )

    return coords


# =============================================================================
# CONNECTOME PLOT
# =============================================================================

def plot_connectome(
    matrix: np.ndarray,
    atlas: str,
    coords: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    node_size: Union[float, np.ndarray] = 20,
    node_color: Union[str, np.ndarray] = "auto",
    edge_cmap: str = "RdBu_r",
    edge_vmin: Optional[float] = None,
    edge_vmax: Optional[float] = None,
    title: str = "",
    display_mode: str = "lzr",
    colorbar: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    alpha: float = 0.7,
) -> Tuple[plt.Figure, object]:
    """
    Plot a brain connectome (glass brain with edges and nodes).

    Parameters
    ----------
    matrix : np.ndarray (N, N)
        Connectivity matrix. Can contain negative values.
    atlas : str
        Atlas name for coordinate extraction.
    coords : np.ndarray (N, 3), optional
        Pre-computed MNI coordinates. If None, extracted from atlas.
    threshold : float, optional
        Only show edges with |weight| > threshold.
        If None, shows top 5% of edges.
    node_size : float or np.ndarray
        Size of nodes. Can be a scalar or per-node array.
    node_color : str or np.ndarray
        Color of nodes. 'auto' uses a default blue.
    edge_cmap : str
        Colormap for edges.
    edge_vmin, edge_vmax : float
    title : str
    display_mode : str
        nilearn display mode: 'lzr', 'ortho', 'lyrz', 'x', 'y', 'z'.
    colorbar : bool
    figsize : tuple, optional
    alpha : float
        Edge transparency.

    Returns
    -------
    fig, display : matplotlib Figure and nilearn display object
    """
    from nilearn import plotting
    from ..graph_analysis.network_metrics import _prepare_matrix

    if coords is None:
        coords = _get_roi_coordinates(atlas)

    mat = _prepare_matrix(matrix, remove_negative=False, remove_diagonal=True)
    N = mat.shape[0]

    # Ensure coords match matrix
    assert len(coords) == N, \
        f"Coords ({len(coords)}) must match matrix size ({N})"

    # Default threshold: top 5% of edges
    if threshold is None:
        triu = np.abs(mat[np.triu_indices(N, k=1)])
        nonzero = triu[triu > 0]
        if len(nonzero) > 0:
            threshold = np.percentile(nonzero, 95)
        else:
            threshold = 0

    # Node color
    if isinstance(node_color, str) and node_color == "auto":
        node_color = "#4C72B0"

    if figsize is None:
        figsize = (12, 4) if "l" in display_mode else (6, 5)

    fig = plt.figure(figsize=figsize)

    display = plotting.plot_connectome(
        mat, coords,
        edge_threshold=threshold,
        node_size=node_size,
        node_color=node_color,
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        display_mode=display_mode,
        colorbar=colorbar,
        alpha=alpha,
        title=title,
        figure=fig,
    )

    return fig, display


# =============================================================================
# METRIC ON BRAIN
# =============================================================================

def plot_metric_on_brain(
    values: np.ndarray,
    atlas: str,
    coords: Optional[np.ndarray] = None,
    title: str = "",
    cmap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    display_mode: str = "lzr",
    symmetric_cbar: bool = False,
    threshold: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    marker_size_scale: float = 150,
) -> Tuple[plt.Figure, object]:
    """
    Overlay a nodal metric on a glass brain (marker-based).

    Node size and color are scaled by metric values.
    Useful for visualizing betweenness centrality, hub scores,
    clustering coefficient, participation coefficient, etc.

    Parameters
    ----------
    values : np.ndarray (N,)
        Per-node metric values.
    atlas : str
    coords : np.ndarray (N, 3), optional
    title : str
    cmap : str
    vmin, vmax : float
    display_mode : str
    symmetric_cbar : bool
        If True, center colorbar at 0.
    threshold : float, optional
        Only show nodes with |value| > threshold.
    marker_size_scale : float
        Controls maximum marker size.

    Returns
    -------
    fig, display
    """
    from nilearn import plotting

    if coords is None:
        coords = _get_roi_coordinates(atlas)

    values = np.asarray(values, dtype=float)
    N = len(values)
    assert len(coords) == N

    # Handle NaN
    valid = ~np.isnan(values)
    if not np.any(valid):
        warnings.warn("All values are NaN, nothing to plot.")
        fig, ax = plt.subplots()
        return fig, ax

    # Threshold
    if threshold is not None:
        mask = np.abs(values) > threshold
        values = values.copy()
        values[~mask] = 0
    else:
        mask = valid

    # Scale marker sizes
    abs_vals = np.abs(values)
    max_val = np.nanmax(abs_vals) if np.nanmax(abs_vals) > 0 else 1
    node_sizes = (abs_vals / max_val) * marker_size_scale
    node_sizes[~mask] = 0

    # Color limits
    if symmetric_cbar:
        abs_max = np.nanmax(np.abs(values[mask])) if np.any(mask) else 1
        vmin = vmin or -abs_max
        vmax = vmax or abs_max
    else:
        vmin = vmin or np.nanmin(values[mask]) if np.any(mask) else 0
        vmax = vmax or np.nanmax(values[mask]) if np.any(mask) else 1

    if figsize is None:
        figsize = (12, 4)

    # Create a null adjacency matrix (no edges, only nodes)
    adjacency = np.zeros((N, N))

    fig = plt.figure(figsize=figsize)
    display = plotting.plot_connectome(
        adjacency, coords,
        node_size=node_sizes,
        node_color=values,
        node_kwargs={"cmap": cmap, "vmin": vmin, "vmax": vmax},
        display_mode=display_mode,
        colorbar=True,
        title=title,
        figure=fig,
    )

    return fig, display


# =============================================================================
# GLASS BRAIN WITH HIGHLIGHTED EDGES
# =============================================================================

def plot_glass_brain_edges(
    matrix: np.ndarray,
    atlas: str,
    edge_list: Optional[List[Tuple[int, int]]] = None,
    coords: Optional[np.ndarray] = None,
    title: str = "",
    edge_color: str = "#C44E52",
    node_color: str = "#4C72B0",
    display_mode: str = "lzr",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, object]:
    """
    Plot specific edges on a glass brain (e.g., NBS significant edges).

    Parameters
    ----------
    matrix : np.ndarray (N, N)
        Full connectivity matrix (used for edge weights).
    atlas : str
    edge_list : list of (i, j) tuples, optional
        Edges to highlight. If None, shows all non-zero edges.
    coords : np.ndarray, optional
    title : str
    edge_color : str
    node_color : str
    display_mode : str
    figsize : tuple

    Returns
    -------
    fig, display
    """
    from nilearn import plotting
    from ..graph_analysis.network_metrics import _prepare_matrix

    if coords is None:
        coords = _get_roi_coordinates(atlas)

    mat = _prepare_matrix(matrix, remove_negative=False)
    N = mat.shape[0]

    # Build a matrix with only the specified edges
    if edge_list is not None:
        highlight_mat = np.zeros((N, N))
        for (i, j) in edge_list:
            highlight_mat[i, j] = mat[i, j]
            highlight_mat[j, i] = mat[j, i]
    else:
        highlight_mat = mat.copy()

    # Nodes involved in highlighted edges
    involved = np.any(highlight_mat != 0, axis=0)
    node_sizes = np.where(involved, 40, 8)

    if figsize is None:
        figsize = (12, 4)

    fig = plt.figure(figsize=figsize)
    display = plotting.plot_connectome(
        highlight_mat, coords,
        node_size=node_sizes,
        node_color=node_color,
        edge_cmap=None,
        display_mode=display_mode,
        colorbar=False,
        title=title,
        figure=fig,
    )

    return fig, display


# =============================================================================
# COMMUNITY ON BRAIN
# =============================================================================

def plot_community_on_brain(
    community_labels: np.ndarray,
    atlas: str,
    coords: Optional[np.ndarray] = None,
    title: str = "Community Structure",
    display_mode: str = "lzr",
    node_size: float = 60,
    cmap: str = "Set3",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, object]:
    """
    Visualize community assignments on a glass brain.

    Each community is shown in a different color.

    Parameters
    ----------
    community_labels : np.ndarray (N,)
    atlas : str
    coords : np.ndarray, optional
    title : str
    display_mode : str
    node_size : float
    cmap : str
    figsize : tuple

    Returns
    -------
    fig, display
    """
    from nilearn import plotting

    if coords is None:
        coords = _get_roi_coordinates(atlas)

    labels = np.asarray(community_labels)
    N = len(labels)
    assert len(coords) == N

    # Map community labels to colors
    unique_mods = np.unique(labels)
    n_mods = len(unique_mods)
    cm = plt.get_cmap(cmap)
    colors = [cm(i / max(n_mods - 1, 1)) for i in range(n_mods)]
    label_to_color = {mod: colors[i] for i, mod in enumerate(unique_mods)}
    node_colors = [label_to_color[l] for l in labels]

    # Null adjacency (no edges)
    adjacency = np.zeros((N, N))

    if figsize is None:
        figsize = (12, 4)

    fig = plt.figure(figsize=figsize)
    display = plotting.plot_connectome(
        adjacency, coords,
        node_size=node_size,
        node_color=node_colors,
        display_mode=display_mode,
        colorbar=False,
        title=title,
        figure=fig,
    )

    return fig, display


# =============================================================================
# HUB NODES ON BRAIN
# =============================================================================

def plot_hub_nodes(
    hub_scores: np.ndarray,
    atlas: str,
    roles: Optional[List[str]] = None,
    coords: Optional[np.ndarray] = None,
    title: str = "Hub Identification",
    hub_threshold: float = 1.0,
    display_mode: str = "lzr",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, object]:
    """
    Visualize hub nodes on a glass brain, with node size proportional
    to hub score and color coding by role.

    Parameters
    ----------
    hub_scores : np.ndarray (N,)
        Composite hub score (z-score of centrality measures).
    atlas : str
    roles : list of str, optional
        Node role labels from classify_nodes.
    coords : np.ndarray, optional
    title : str
    hub_threshold : float
    display_mode : str
    figsize : tuple

    Returns
    -------
    fig, display
    """
    from nilearn import plotting

    if coords is None:
        coords = _get_roi_coordinates(atlas)

    scores = np.asarray(hub_scores, dtype=float)
    N = len(scores)

    # Node sizes (hubs larger)
    node_sizes = np.maximum(scores * 40 + 10, 5)

    # Node colors by role
    role_color_map = {
        "connector_hub": "#C44E52",       # red
        "provincial_hub": "#DD8452",       # orange
        "connector_non_hub": "#55A868",    # green
        "peripheral": "#BBBBBB",           # gray
    }

    if roles is not None:
        node_colors = [role_color_map.get(r, "#BBBBBB") for r in roles]
    else:
        # Color by hub score
        node_colors = scores

    adjacency = np.zeros((N, N))

    if figsize is None:
        figsize = (12, 4)

    fig = plt.figure(figsize=figsize)
    display = plotting.plot_connectome(
        adjacency, coords,
        node_size=node_sizes,
        node_color=node_colors,
        display_mode=display_mode,
        colorbar=roles is None,
        title=title,
        figure=fig,
    )

    # Add legend for role colors
    if roles is not None:
        import matplotlib.patches as mpatches
        legend_handles = []
        for role, color in role_color_map.items():
            count = sum(1 for r in roles if r == role)
            if count > 0:
                label = f"{role.replace('_', ' ').title()} (n={count})"
                legend_handles.append(mpatches.Patch(color=color, label=label))
        if legend_handles:
            fig.legend(handles=legend_handles, loc="lower center",
                       ncol=len(legend_handles), frameon=False, fontsize=8)

    return fig, display


# =============================================================================
# SC-FC COUPLING BRAIN MAP
# =============================================================================

def plot_sc_fc_coupling_brain(
    regional_r: np.ndarray,
    atlas: str,
    coords: Optional[np.ndarray] = None,
    title: str = "Regional SC-FC Coupling",
    display_mode: str = "lzr",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, object]:
    """
    Visualize regional SC-FC coupling on a glass brain.

    Nodes are colored and sized by their SC-FC correlation,
    showing which brain regions have strong vs weak
    structure-function correspondence.

    Parameters
    ----------
    regional_r : np.ndarray (N,)
        Per-node SC-FC correlation coefficients.
    atlas : str
    coords : np.ndarray, optional
    title : str
    display_mode : str
    figsize : tuple

    Returns
    -------
    fig, display
    """
    return plot_metric_on_brain(
        values=regional_r,
        atlas=atlas,
        coords=coords,
        title=title,
        cmap="RdYlBu_r",
        symmetric_cbar=True,
        display_mode=display_mode,
        figsize=figsize,
        marker_size_scale=200,
    )
