# -*- coding: utf-8 -*-
"""
sars.eigenmorph.viz_interactive
==================================================

Showcase visualizations for eigenvalue morphology features.

Provides high-impact, publication-quality and interactive figures
designed to communicate the novelty and power of eigenvalue-based
cortical morphometrics.  Uses FURY (VTK) for 3D surface rendering
and matplotlib for 2D analytical plots.

Visual catalogue
----------------
**Surface renders (FURY)**

plot_rgb_identity
    L→Red, P→Green, S→Blue per-vertex RGB mapping — each vertex gets a
    unique color encoding its geometric identity.
plot_feature_landscape
    Inflated/deformed surface where the radial axis encodes a scalar
    feature (e.g. eigenentropy), creating a topographic metaphor.
plot_exploded_view
    Cortical vertices clustered by geometric type and spatially
    separated, revealing discrete morphological communities.
plot_neighborhood_explorer
    Single vertex highlighted with its KD-tree neighborhood rendered
    at multiple scales as semi-transparent shells.
render_scale_sweep
    Animated GIF sweeping from fine (3 mm) to coarse (20 mm) scale.

**Analytical plots (matplotlib)**

plot_dual_view
    Side-by-side classical metric vs eigenvalue feature on the same
    surface and orientation.
plot_morphological_radar
    Radar / spider chart showing the 7-feature fingerprint per parcel.
plot_feature_embedding
    UMAP or t-SNE of the 28-D feature space, colored by parcellation
    or functional network.

Dependencies
------------
Required:  numpy, matplotlib, scipy
Optional:  fury (VTK rendering), imageio (GIF export), umap-learn or
           scikit-learn (embedding)

If FURY is not installed, surface renders gracefully fall back to
matplotlib Poly3DCollection (lower quality but functional).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import cKDTree
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass

from .core import SurfaceMesh, EigenFeatures, MultiScaleFeatures

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------

_HAS_FURY = False
try:
    import fury
    from fury import actor, window, colormap as fury_cmap
    _HAS_FURY = True
except ImportError:
    pass

_HAS_IMAGEIO = False
try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    pass


def _require_fury(func_name: str):
    """Raise informative error if FURY is not available."""
    if not _HAS_FURY:
        raise ImportError(
            f"{func_name} requires FURY (pip install fury). "
            f"Install it for interactive 3D rendering, or use the "
            f"matplotlib-based functions in sars.eigenmorph.viz instead."
        )


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _setup_style():
    """Publication-ready matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ═══════════════════════════════════════════════════════════════════════════
#  1.  RGB MORPHOLOGICAL IDENTITY MAP
# ═══════════════════════════════════════════════════════════════════════════

def _lps_to_rgb(
    linearity: np.ndarray,
    planarity: np.ndarray,
    sphericity: np.ndarray,
    gamma: float = 0.7,
) -> np.ndarray:
    """
    Map Linearity–Planarity–Sphericity to RGB channels.

    Parameters
    ----------
    linearity, planarity, sphericity : (V,) arrays
    gamma : float
        Gamma correction (< 1 brightens midtones).  0.7 gives a
        vibrant, print-friendly result.

    Returns
    -------
    np.ndarray (V, 3)
        RGB values in [0, 1].
    """
    L = np.nan_to_num(linearity, nan=0.0)
    P = np.nan_to_num(planarity, nan=0.0)
    S = np.nan_to_num(sphericity, nan=0.0)

    total = L + P + S
    total[total < 1e-12] = 1.0

    r = np.clip(L / total, 0, 1)
    g = np.clip(P / total, 0, 1)
    b = np.clip(S / total, 0, 1)

    # Gamma correction for vibrancy
    r = np.power(r, gamma)
    g = np.power(g, gamma)
    b = np.power(b, gamma)

    return np.column_stack([r, g, b])


def plot_rgb_identity(
    mesh: SurfaceMesh,
    features: EigenFeatures,
    gamma: float = 0.7,
    size: Tuple[int, int] = (1200, 900),
    offscreen: bool = False,
    save_path: Optional[str] = None,
    views: Optional[List[Tuple[float, float, float]]] = None,
) -> Optional[np.ndarray]:
    """
    Render the cortical surface with L→R, P→G, S→B per-vertex coloring.

    Each vertex receives a unique RGB color encoding its geometric
    identity: sulcal fundi glow red (high linearity), gyral crowns
    green (high planarity), sulcal pits blue (high sphericity), with
    intermediate colors in transition zones.

    Parameters
    ----------
    mesh : SurfaceMesh
    features : EigenFeatures
    gamma : float
        Gamma correction (< 1 = brighter midtones).
    size : tuple
        Window size in pixels.
    offscreen : bool
        If True, render to array without opening a window.
    save_path : str, optional
        Save the rendered image to this path (PNG).
    views : list of (azimuth, elevation, roll), optional
        Camera positions.  If multiple views given AND save_path
        is provided, saves a multi-panel composite.

    Returns
    -------
    np.ndarray or None
        RGBA image array if offscreen, else None (interactive window).
    """
    rgb = _lps_to_rgb(features.linearity, features.planarity,
                       features.sphericity, gamma=gamma)

    if _HAS_FURY:
        return _rgb_identity_fury(mesh, rgb, size, offscreen, save_path, views)
    else:
        return _rgb_identity_mpl(mesh, rgb, save_path)


def _rgb_identity_fury(mesh, rgb, size, offscreen, save_path, views):
    """FURY backend for RGB identity map."""
    rgba = np.column_stack([rgb, np.ones(len(rgb))]).astype(np.float64)
    colors_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

    surface = actor.surface(
        vertices=mesh.vertices,
        faces=mesh.faces,
        colors=colors_uint8,
    )

    scene = window.Scene()
    scene.add(surface)
    scene.set_camera(
        position=(0, -300, 50),
        focal_point=(0, 0, 0),
        view_up=(0, 0, 1),
    )
    scene.background((1, 1, 1))

    if offscreen or save_path:
        arr = window.snapshot(
            scene, fname=save_path, size=size, offscreen=True,
        )
        return arr
    else:
        window.show(scene, size=size, title="Eigenmorph RGB Identity")
        return None


def _rgb_identity_mpl(mesh, rgb, save_path):
    """Matplotlib fallback for RGB identity map."""
    _setup_style()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    face_rgb = rgb[mesh.faces].mean(axis=1)
    face_rgba = np.column_stack([face_rgb, np.ones(len(face_rgb))])

    polys = Poly3DCollection(
        mesh.vertices[mesh.faces],
        facecolors=face_rgba,
        edgecolors="none",
        linewidths=0,
    )
    ax.add_collection3d(polys)

    v = mesh.vertices
    margin = 5
    ax.set_xlim(v[:, 0].min() - margin, v[:, 0].max() + margin)
    ax.set_ylim(v[:, 1].min() - margin, v[:, 1].max() + margin)
    ax.set_zlim(v[:, 2].min() - margin, v[:, 2].max() + margin)
    ax.view_init(elev=30, azim=-60)
    ax.axis("off")
    ax.set_title(
        "Geometric Identity: L→R  P→G  S→B",
        fontsize=14, fontweight="bold",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", label="Linearity (sulcal fundi)"),
        Patch(facecolor="green", label="Planarity (gyral crowns)"),
        Patch(facecolor="blue", label="Sphericity (sulcal pits)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    if save_path:
        fig.savefig(save_path)
    plt.tight_layout()
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  2.  FEATURE LANDSCAPE (topographic deformation)
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_landscape(
    mesh: SurfaceMesh,
    values: np.ndarray,
    title: str = "Eigenentropy Landscape",
    cmap: str = "inferno",
    deformation_scale: float = 5.0,
    size: Tuple[int, int] = (1200, 900),
    offscreen: bool = False,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Topographic surface deformed radially by a scalar feature.

    The cortical surface is inflated outward in proportion to the
    scalar value, creating a terrain where peaks represent regions of
    high feature values and valleys represent low values.  Combined
    with color mapping, this provides a powerful dual-coded
    visualization.

    Parameters
    ----------
    mesh : SurfaceMesh
    values : np.ndarray (V,)
        Feature to encode as radial deformation.
    title : str
    cmap : str
    deformation_scale : float
        Amplitude of deformation in mm.
    size : tuple
    offscreen : bool
    save_path : str, optional

    Returns
    -------
    np.ndarray or None
    """
    vals = np.nan_to_num(values, nan=0.0)
    vmin, vmax = np.nanpercentile(values[~np.isnan(values)], [5, 95])
    vals_norm = (vals - vmin) / max(vmax - vmin, 1e-12)
    vals_norm = np.clip(vals_norm, 0, 1)

    # Deform along vertex normals
    normals = mesh.vertex_normals
    if normals is None:
        normals = _estimate_normals(mesh)

    deformed = mesh.vertices + normals * (vals_norm * deformation_scale)[:, np.newaxis]

    # Color mapping
    mapper = cm.ScalarMappable(
        norm=colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap,
    )
    rgba = mapper.to_rgba(vals)

    if _HAS_FURY:
        colors_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

        surface = actor.surface(
            vertices=deformed.astype(np.float64),
            faces=mesh.faces,
            colors=colors_uint8,
        )

        scene = window.Scene()
        scene.add(surface)
        scene.set_camera(
            position=(0, -350, 80),
            focal_point=(0, 0, 0),
            view_up=(0, 0, 1),
        )
        scene.background((1, 1, 1))

        if offscreen or save_path:
            return window.snapshot(scene, fname=save_path, size=size,
                                   offscreen=True)
        else:
            window.show(scene, size=size, title=title)
            return None
    else:
        return _landscape_mpl(deformed, mesh.faces, rgba, title, save_path)


def _estimate_normals(mesh: SurfaceMesh) -> np.ndarray:
    """Compute per-vertex normals if missing."""
    normals = np.zeros_like(mesh.vertices)
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for i in range(3):
        np.add.at(normals, mesh.faces[:, i], fn)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return normals / norms


def _landscape_mpl(vertices, faces, rgba, title, save_path):
    """Matplotlib fallback for feature landscape."""
    _setup_style()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    face_rgba = rgba[faces].mean(axis=1)
    polys = Poly3DCollection(
        vertices[faces], facecolors=face_rgba,
        edgecolors="none", linewidths=0,
    )
    ax.add_collection3d(polys)

    v = vertices
    margin = 10
    ax.set_xlim(v[:, 0].min() - margin, v[:, 0].max() + margin)
    ax.set_ylim(v[:, 1].min() - margin, v[:, 1].max() + margin)
    ax.set_zlim(v[:, 2].min() - margin, v[:, 2].max() + margin)
    ax.view_init(elev=30, azim=-60)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  3.  ANIMATED SCALE SWEEP (GIF)
# ═══════════════════════════════════════════════════════════════════════════

def render_scale_sweep(
    mesh: SurfaceMesh,
    ms_features: MultiScaleFeatures,
    feature_name: str = "eigenentropy",
    cmap: str = "viridis",
    n_interp_frames: int = 10,
    size: Tuple[int, int] = (800, 600),
    save_path: str = "scale_sweep.gif",
    fps: int = 8,
) -> Optional[str]:
    """
    Animated GIF sweeping a feature from fine to coarse scale.

    Interpolates smoothly between computed scales (e.g. 3→5→10→20 mm)
    to create a fluid animation showing how the cortical geometry
    evolves across spatial scales.

    Parameters
    ----------
    mesh : SurfaceMesh
    ms_features : MultiScaleFeatures
    feature_name : str
        Which of the 7 features to animate.
    cmap : str
    n_interp_frames : int
        Frames interpolated between each computed scale.
    size : tuple
    save_path : str
        Output GIF path.
    fps : int

    Returns
    -------
    str or None
        Path to the saved GIF, or None if dependencies are missing.
    """
    if not _HAS_IMAGEIO:
        raise ImportError(
            "render_scale_sweep requires imageio (pip install imageio)."
        )

    scales_data = []
    for s in ms_features.scales:
        vals = getattr(s, feature_name)
        scales_data.append(np.nan_to_num(vals, nan=0.0))

    # Global color range for consistency across frames
    all_vals = np.concatenate(scales_data)
    vmin, vmax = np.nanpercentile(all_vals, [2, 98])
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Build interpolated frames
    frames_data = []
    for i in range(len(scales_data) - 1):
        for t in np.linspace(0, 1, n_interp_frames, endpoint=False):
            interp = (1 - t) * scales_data[i] + t * scales_data[i + 1]
            frames_data.append(interp)
    frames_data.append(scales_data[-1])  # Final frame

    # Render each frame
    images = []
    radii = ms_features.radii

    if _HAS_FURY:
        for idx, frame_vals in enumerate(frames_data):
            rgba = mapper.to_rgba(frame_vals)
            colors_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

            surface = actor.surface(
                vertices=mesh.vertices,
                faces=mesh.faces,
                colors=colors_uint8,
            )

            # Compute current radius for annotation
            n_total = len(frames_data)
            n_scales = len(radii)
            progress = idx / max(n_total - 1, 1)
            current_r = np.interp(
                progress,
                np.linspace(0, 1, n_scales),
                radii,
            )

            scene = window.Scene()
            scene.add(surface)
            scene.set_camera(
                position=(0, -300, 50),
                focal_point=(0, 0, 0),
                view_up=(0, 0, 1),
            )
            scene.background((1, 1, 1))

            # Add text annotation
            label = actor.text_3d(
                f"{feature_name}  r = {current_r:.1f} mm",
                position=(-80, -120, 80),
                color=(0.1, 0.1, 0.1),
                font_size=18,
            )
            scene.add(label)

            arr = window.snapshot(scene, size=size, offscreen=True)
            images.append(arr)
    else:
        # Matplotlib fallback (slower but works)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for idx, frame_vals in enumerate(frames_data):
            progress = idx / max(len(frames_data) - 1, 1)
            current_r = np.interp(
                progress, np.linspace(0, 1, len(radii)), radii,
            )

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

            rgba = mapper.to_rgba(frame_vals)
            face_rgba = rgba[mesh.faces].mean(axis=1)

            polys = Poly3DCollection(
                mesh.vertices[mesh.faces],
                facecolors=face_rgba, edgecolors="none", linewidths=0,
            )
            ax.add_collection3d(polys)

            v = mesh.vertices
            ax.set_xlim(v[:, 0].min() - 5, v[:, 0].max() + 5)
            ax.set_ylim(v[:, 1].min() - 5, v[:, 1].max() + 5)
            ax.set_zlim(v[:, 2].min() - 5, v[:, 2].max() + 5)
            ax.view_init(elev=30, azim=-60)
            ax.axis("off")
            ax.set_title(
                f"{feature_name}  r = {current_r:.1f} mm",
                fontsize=14, fontweight="bold",
            )

            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            arr = buf.reshape(h, w, 4)[:, :, :3]
            images.append(arr.copy())
            plt.close(fig)

    imageio.mimsave(save_path, images, fps=fps, loop=0)
    return save_path


# ═══════════════════════════════════════════════════════════════════════════
#  4.  EXPLODED VIEW BY GEOMETRIC COMMUNITY
# ═══════════════════════════════════════════════════════════════════════════

def plot_exploded_view(
    mesh: SurfaceMesh,
    ms_features: MultiScaleFeatures,
    n_clusters: int = 6,
    explosion_factor: float = 30.0,
    seed: int = 42,
    size: Tuple[int, int] = (1200, 900),
    offscreen: bool = False,
    save_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Cluster vertices by geometric type and separate spatially.

    Runs k-means in the full multi-scale feature space (28-D with
    default radii) to identify morphological communities, then
    displaces each cluster outward from the centroid, creating an
    exploded-view rendering that reveals discrete geometric types.

    Parameters
    ----------
    mesh : SurfaceMesh
    ms_features : MultiScaleFeatures
    n_clusters : int
        Number of geometric communities.
    explosion_factor : float
        Displacement amplitude in mm.
    seed : int
    size : tuple
    offscreen : bool
    save_path : str, optional

    Returns
    -------
    cluster_labels : np.ndarray (V,)
        Cluster assignment per vertex.
    image : np.ndarray or None
        Rendered image if offscreen.
    """
    from scipy.cluster.vq import kmeans2

    feat_matrix = ms_features.as_matrix()
    # Replace NaN with 0 for clustering
    feat_clean = np.nan_to_num(feat_matrix, nan=0.0)

    # Standardize for k-means
    means = feat_clean.mean(axis=0)
    stds = feat_clean.std(axis=0)
    stds[stds < 1e-12] = 1.0
    feat_std = (feat_clean - means) / stds

    _, cluster_labels = kmeans2(feat_std, n_clusters, minit="++", seed=seed)

    # Compute cluster centroids in physical space
    global_centroid = mesh.vertices.mean(axis=0)
    cluster_directions = np.zeros((n_clusters, 3))
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() > 0:
            cluster_centroid = mesh.vertices[mask].mean(axis=0)
            direction = cluster_centroid - global_centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                cluster_directions[c] = direction / norm
            else:
                cluster_directions[c] = np.array([1, 0, 0])

    # Displace vertices
    displaced = mesh.vertices.copy()
    for c in range(n_clusters):
        mask = cluster_labels == c
        displaced[mask] += cluster_directions[c] * explosion_factor

    # Color by cluster
    palette = plt.cm.Set1(np.linspace(0, 1, n_clusters))[:, :3]
    vertex_colors = palette[cluster_labels]

    if _HAS_FURY:
        rgba_uint8 = np.column_stack([
            vertex_colors, np.ones(len(vertex_colors))
        ])
        rgba_uint8 = (np.clip(rgba_uint8, 0, 1) * 255).astype(np.uint8)

        surface = actor.surface(
            vertices=displaced.astype(np.float64),
            faces=mesh.faces,
            colors=rgba_uint8,
        )

        scene = window.Scene()
        scene.add(surface)
        scene.set_camera(
            position=(0, -400, 100),
            focal_point=(0, 0, 0),
            view_up=(0, 0, 1),
        )
        scene.background((1, 1, 1))

        if offscreen or save_path:
            arr = window.snapshot(scene, fname=save_path, size=size,
                                  offscreen=True)
            return cluster_labels, arr
        else:
            window.show(scene, size=size, title="Exploded Geometric Communities")
            return cluster_labels, None
    else:
        _setup_style()
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        face_colors = vertex_colors[mesh.faces].mean(axis=1)
        face_rgba = np.column_stack([
            face_colors, np.ones(len(face_colors))
        ])

        polys = Poly3DCollection(
            displaced[mesh.faces],
            facecolors=face_rgba,
            edgecolors="none",
            linewidths=0,
        )
        ax.add_collection3d(polys)

        v = displaced
        margin = 20
        ax.set_xlim(v[:, 0].min() - margin, v[:, 0].max() + margin)
        ax.set_ylim(v[:, 1].min() - margin, v[:, 1].max() + margin)
        ax.set_zlim(v[:, 2].min() - margin, v[:, 2].max() + margin)
        ax.view_init(elev=25, azim=-45)
        ax.axis("off")
        ax.set_title(
            f"Exploded View: {n_clusters} Geometric Communities",
            fontsize=14, fontweight="bold",
        )

        # Legend
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=palette[c], label=f"Type {c+1} "
                  f"({(cluster_labels == c).sum():,} vertices)")
            for c in range(n_clusters)
        ]
        ax.legend(handles=handles, loc="lower left", fontsize=8)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
        return cluster_labels, None


# ═══════════════════════════════════════════════════════════════════════════
#  5.  NEIGHBORHOOD EXPLORER
# ═══════════════════════════════════════════════════════════════════════════

def plot_neighborhood_explorer(
    mesh: SurfaceMesh,
    vertex_idx: int,
    radii: List[float] = None,
    size: Tuple[int, int] = (1200, 900),
    offscreen: bool = False,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Visualize the KD-tree neighborhoods at multiple scales.

    Shows a single vertex highlighted in red, with its neighborhood at
    each radius rendered as a semi-transparent colored shell.  This
    reveals exactly what the algorithm 'sees' at each spatial scale.

    Parameters
    ----------
    mesh : SurfaceMesh
    vertex_idx : int
        Index of the center vertex.
    radii : list of float
        Scales to visualize.  Default [3, 5, 10, 20].
    size : tuple
    offscreen : bool
    save_path : str, optional

    Returns
    -------
    np.ndarray or None
    """
    if radii is None:
        radii = [3.0, 5.0, 10.0, 20.0]

    tree = cKDTree(mesh.vertices)

    palette = plt.cm.viridis(np.linspace(0.2, 0.9, len(radii)))[:, :3]

    if _HAS_FURY:
        scene = window.Scene()

        # Base surface (gray, semi-transparent)
        n_verts = mesh.n_vertices
        gray = np.full((n_verts, 4), [180, 180, 180, 60], dtype=np.uint8)

        surface = actor.surface(
            vertices=mesh.vertices,
            faces=mesh.faces,
            colors=gray,
        )
        scene.add(surface)

        # Center vertex (large red sphere)
        center = mesh.vertices[vertex_idx]
        center_sphere = actor.sphere(
            centers=np.array([center]),
            colors=np.array([[255, 0, 0, 255]], dtype=np.uint8),
            radii=1.5,
        )
        scene.add(center_sphere)

        # Neighborhood shells
        for r_idx, radius in enumerate(radii):
            neighbors = tree.query_ball_point(center, radius)
            if len(neighbors) == 0:
                continue

            pts = mesh.vertices[neighbors]
            col = (palette[r_idx] * 255).astype(np.uint8)
            col_rgba = np.array(
                [[col[0], col[1], col[2], 140]] * len(pts),
                dtype=np.uint8,
            )

            neighbor_pts = actor.sphere(
                centers=pts.astype(np.float64),
                colors=col_rgba,
                radii=0.4,
            )
            scene.add(neighbor_pts)

        scene.set_camera(
            position=tuple(center + np.array([0, -80, 30])),
            focal_point=tuple(center),
            view_up=(0, 0, 1),
        )
        scene.background((1, 1, 1))

        if offscreen or save_path:
            return window.snapshot(scene, fname=save_path, size=size,
                                   offscreen=True)
        else:
            window.show(scene, size=size, title="Neighborhood Explorer")
            return None
    else:
        # Matplotlib fallback
        _setup_style()
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        center = mesh.vertices[vertex_idx]

        for r_idx, radius in enumerate(reversed(radii)):
            neighbors = tree.query_ball_point(center, radius)
            pts = mesh.vertices[neighbors]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=3, alpha=0.3,
                color=palette[len(radii) - 1 - r_idx],
                label=f"r={radius:.0f}mm ({len(neighbors)} pts)",
            )

        ax.scatter(
            [center[0]], [center[1]], [center[2]],
            s=100, c="red", marker="*", zorder=10,
            label="Center vertex",
        )

        margin = max(radii) + 5
        ax.set_xlim(center[0] - margin, center[0] + margin)
        ax.set_ylim(center[1] - margin, center[1] + margin)
        ax.set_zlim(center[2] - margin, center[2] + margin)
        ax.legend(fontsize=8)
        ax.set_title(
            f"Neighborhood Explorer — vertex {vertex_idx}",
            fontweight="bold",
        )
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path)
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  6.  DUAL VIEW: CLASSICAL vs EIGENMORPH
# ═══════════════════════════════════════════════════════════════════════════

def plot_dual_view(
    mesh: SurfaceMesh,
    classical_values: np.ndarray,
    eigen_values: np.ndarray,
    classical_name: str = "Curvature",
    eigen_name: str = "Eigenentropy (5 mm)",
    classical_cmap: str = "coolwarm",
    eigen_cmap: str = "viridis",
    view: Tuple[float, float] = (30, -60),
    figsize: Tuple[int, int] = (20, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side: classical FreeSurfer metric vs eigenvalue feature.

    Same surface, same orientation, same colorbar range logic.  The
    visual contrast speaks for itself.

    Parameters
    ----------
    mesh : SurfaceMesh
    classical_values : np.ndarray (V,)
    eigen_values : np.ndarray (V,)
    classical_name, eigen_name : str
    classical_cmap, eigen_cmap : str
    view : tuple (elevation, azimuth)
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize,
        subplot_kw={"projection": "3d"},
    )

    for ax, vals, name, cmap_name in [
        (ax1, classical_values, classical_name, classical_cmap),
        (ax2, eigen_values, eigen_name, eigen_cmap),
    ]:
        vals_clean = np.nan_to_num(vals, nan=0.0)
        valid = vals[~np.isnan(vals)]
        vmin, vmax = np.percentile(valid, [5, 95]) if len(valid) > 0 else (0, 1)

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap_name)
        face_vals = vals_clean[mesh.faces].mean(axis=1)
        face_rgba = mapper.to_rgba(face_vals)

        polys = Poly3DCollection(
            mesh.vertices[mesh.faces],
            facecolors=face_rgba,
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
        ax.axis("off")
        ax.set_title(name, fontsize=14, fontweight="bold")
        fig.colorbar(mapper, ax=ax, shrink=0.5, pad=0.05)

    fig.suptitle(
        "Classical vs Eigenmorph", fontsize=16, fontweight="bold", y=0.95,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  7.  MORPHOLOGICAL FINGERPRINT RADAR
# ═══════════════════════════════════════════════════════════════════════════

def plot_morphological_radar(
    parcellated: Dict,
    parcel_indices: List[int] = None,
    parcel_names: Optional[List[str]] = None,
    scale_idx: int = 1,
    n_cols: int = 4,
    figsize_per_plot: Tuple[float, float] = (3.5, 3.5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Radar chart showing the 7-feature fingerprint per brain region.

    Each radar axis corresponds to one of the seven eigenvalue features.
    The shape of each radar encodes the region's morphological identity:
    elongated shapes = dominant geometry type; round shapes = balanced.

    Parameters
    ----------
    parcellated : dict
        Output from ``parcellate_features()``.  Keys are feature names
        (e.g. ``linearity_r5mm``), values are ``(n_parcels,)`` arrays.
    parcel_indices : list of int
        Which parcels to plot.  Default: first 8.
    parcel_names : list of str, optional
        Override parcel names from parcellated['_parcel_names'].
    scale_idx : int
        Which scale to use if multi-scale parcellation.  Index into
        the radii list (0 = finest, -1 = coarsest).
    n_cols : int
    figsize_per_plot : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()

    feature_labels = EigenFeatures.feature_names()
    col_names = parcellated.get("_column_names", [])
    names = parcel_names or parcellated.get("_parcel_names", [])

    # Find columns matching the requested scale
    # Strategy: group columns by feature, pick the scale_idx-th occurrence
    feature_cols = {}
    for fn in feature_labels:
        matching = [c for c in col_names if c.startswith(fn)]
        if scale_idx < len(matching):
            feature_cols[fn] = matching[scale_idx]
        elif matching:
            feature_cols[fn] = matching[-1]

    if not feature_cols:
        raise ValueError(
            f"Could not find features at scale_idx={scale_idx}. "
            f"Available columns: {col_names}"
        )

    n_features = len(feature_labels)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Determine parcels to plot
    sample_col = list(feature_cols.values())[0]
    n_parcels = len(parcellated[sample_col])
    if parcel_indices is None:
        parcel_indices = list(range(min(8, n_parcels)))

    n_plots = len(parcel_indices)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols,
                 figsize_per_plot[1] * n_rows),
        subplot_kw=dict(polar=True),
    )
    if n_plots == 1:
        axes = np.array([axes])
    axes_flat = axes.ravel()

    # Compute global range for normalization
    global_max = {}
    for fn in feature_labels:
        col = feature_cols[fn]
        global_max[fn] = max(parcellated[col].max(), 1e-12)

    palette = plt.cm.Set2(np.linspace(0, 1, n_plots))

    for plot_idx, parcel_idx in enumerate(parcel_indices):
        ax = axes_flat[plot_idx]

        values = []
        for fn in feature_labels:
            col = feature_cols[fn]
            val = parcellated[col][parcel_idx] / global_max[fn]
            values.append(val)
        values += values[:1]  # close

        ax.fill(angles, values, alpha=0.25, color=palette[plot_idx])
        ax.plot(angles, values, "o-", linewidth=2, markersize=4,
                color=palette[plot_idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [fn[:4] for fn in feature_labels], fontsize=7,
        )
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([], fontsize=6)

        label = names[parcel_idx] if parcel_idx < len(names) else f"Parcel {parcel_idx}"
        if isinstance(label, bytes):
            label = label.decode()
        ax.set_title(label, fontsize=9, fontweight="bold", pad=12)

    # Hide unused axes
    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    scale_col = list(feature_cols.values())[0]
    scale_label = scale_col.split("_r")[-1] if "_r" in scale_col else ""
    fig.suptitle(
        f"Morphological Fingerprints  ({scale_label})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  8.  FEATURE EMBEDDING (UMAP / t-SNE)
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_embedding(
    ms_features: MultiScaleFeatures,
    vertex_labels: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    method: str = "umap",
    max_points: int = 20000,
    seed: int = 42,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    UMAP or t-SNE embedding of the multi-scale feature space.

    Projects the 28-D feature space (7 features × 4 scales) to 2D,
    revealing clusters of geometrically similar vertices.  Colored by
    parcellation labels to show alignment (or misalignment) between
    geometric and anatomical organization.

    Parameters
    ----------
    ms_features : MultiScaleFeatures
    vertex_labels : np.ndarray (V,), optional
        Integer labels for coloring (e.g. from parcellation or
        functional network assignment).  If None, colors by
        eigenentropy.
    label_names : list of str, optional
        Names for unique labels (for legend).
    method : str
        'umap' or 'tsne'.
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

    feat_matrix = ms_features.as_matrix()
    valid_mask = ~np.any(np.isnan(feat_matrix), axis=1)
    valid_idx = np.where(valid_mask)[0]

    rng = np.random.default_rng(seed)
    if len(valid_idx) > max_points:
        sample_idx = rng.choice(valid_idx, max_points, replace=False)
    else:
        sample_idx = valid_idx

    X = feat_matrix[sample_idx]

    # Standardize
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-12] = 1.0
    X_std = (X - means) / stds

    # Compute embedding
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2, random_state=seed,
                n_neighbors=30, min_dist=0.3,
            )
            embedding = reducer.fit_transform(X_std)
        except ImportError:
            print("  umap-learn not found, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2, random_state=seed,
            perplexity=min(30, len(X_std) - 1),
        )
        embedding = reducer.fit_transform(X_std)

    # Prepare colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: colored by labels or eigenentropy ---
    if vertex_labels is not None:
        c = vertex_labels[sample_idx]
        n_unique = len(np.unique(c[c > 0]))

        if n_unique <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.nipy_spectral

        sc = ax1.scatter(
            embedding[:, 0], embedding[:, 1],
            c=c, s=1, alpha=0.4, cmap=cmap, rasterized=True,
        )
        ax1.set_title("Colored by Parcellation", fontweight="bold")
    else:
        # Color by eigenentropy at middle scale
        mid_scale = len(ms_features.scales) // 2
        entropy = ms_features.scales[mid_scale].eigenentropy[sample_idx]
        sc = ax1.scatter(
            embedding[:, 0], embedding[:, 1],
            c=entropy, s=1, alpha=0.4, cmap="viridis", rasterized=True,
        )
        fig.colorbar(sc, ax=ax1, label="Eigenentropy", shrink=0.8)
        ax1.set_title("Colored by Eigenentropy", fontweight="bold")

    ax1.set_xlabel(f"{method.upper()} 1")
    ax1.set_ylabel(f"{method.upper()} 2")

    # --- Right: colored by L-P-S RGB ---
    mid_scale = len(ms_features.scales) // 2
    s = ms_features.scales[mid_scale]
    rgb = _lps_to_rgb(s.linearity[sample_idx],
                       s.planarity[sample_idx],
                       s.sphericity[sample_idx])

    ax2.scatter(
        embedding[:, 0], embedding[:, 1],
        c=rgb, s=1, alpha=0.4, rasterized=True,
    )
    ax2.set_xlabel(f"{method.upper()} 1")
    ax2.set_ylabel(f"{method.upper()} 2")
    ax2.set_title("Colored by L→R  P→G  S→B", fontweight="bold")

    # RGB legend
    from matplotlib.patches import Patch
    rgb_legend = [
        Patch(facecolor="red", label="Linearity"),
        Patch(facecolor="green", label="Planarity"),
        Patch(facecolor="blue", label="Sphericity"),
    ]
    ax2.legend(handles=rgb_legend, fontsize=8, loc="lower right")

    fig.suptitle(
        f"{method.upper()} Embedding of 28-D Eigenmorph Features  "
        f"({len(sample_idx):,} vertices)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  9.  MULTI-PANEL COMPOSITE (the "hero figure")
# ═══════════════════════════════════════════════════════════════════════════

def plot_hero_figure(
    mesh: SurfaceMesh,
    features: EigenFeatures,
    ms_features: MultiScaleFeatures,
    classical_metrics: Optional[Dict[str, np.ndarray]] = None,
    comparison: Optional[Dict] = None,
    figsize: Tuple[int, int] = (24, 16),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel composite figure for graphical abstract or README.

    Panel layout (3 × 3):
        [RGB Identity]  [Ternary L-P-S]  [Scale Profiles]
        [Dual view ×2 ]  [Unique Variance]
        [Radar ×4    ]  [UMAP Embedding ]

    Falls back to matplotlib for all panels to ensure compatibility.

    Parameters
    ----------
    mesh : SurfaceMesh
    features : EigenFeatures
        Single-scale features (for RGB and ternary).
    ms_features : MultiScaleFeatures
    classical_metrics : dict, optional
        If provided, includes dual-view panel.
    comparison : dict, optional
        If provided, includes unique-variance panel.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    v = mesh.vertices
    margin = 5

    # ---- Panel A: RGB Identity (3D) ----
    ax_a = fig.add_subplot(gs[0, 0:2], projection="3d")
    rgb = _lps_to_rgb(features.linearity, features.planarity,
                       features.sphericity)
    face_rgb = rgb[mesh.faces].mean(axis=1)
    face_rgba = np.column_stack([face_rgb, np.ones(len(face_rgb))])

    polys = Poly3DCollection(
        v[mesh.faces], facecolors=face_rgba, edgecolors="none", linewidths=0,
    )
    ax_a.add_collection3d(polys)
    ax_a.set_xlim(v[:, 0].min() - margin, v[:, 0].max() + margin)
    ax_a.set_ylim(v[:, 1].min() - margin, v[:, 1].max() + margin)
    ax_a.set_zlim(v[:, 2].min() - margin, v[:, 2].max() + margin)
    ax_a.view_init(elev=30, azim=-60)
    ax_a.axis("off")
    ax_a.set_title("A.  Geometric Identity (L→R  P→G  S→B)",
                    fontweight="bold", fontsize=11)

    # ---- Panel B: Ternary L-P-S ----
    ax_b = fig.add_subplot(gs[0, 2])
    valid = ~np.isnan(features.linearity)
    indices = np.where(valid)[0]
    rng = np.random.default_rng(42)
    if len(indices) > 10000:
        indices = rng.choice(indices, 10000, replace=False)

    L = features.linearity[indices]
    P = features.planarity[indices]
    S = features.sphericity[indices]
    total = L + P + S
    total[total == 0] = 1
    L_n, P_n, S_n = L / total, P / total, S / total
    x = 0.5 * (2 * P_n + S_n)
    y = (np.sqrt(3) / 2) * S_n

    ax_b.scatter(x, y, c=rgb[indices], s=0.5, alpha=0.3, rasterized=True)
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3) / 2, 0]
    ax_b.plot(tri_x, tri_y, "k-", linewidth=1.5)
    ax_b.text(0, -0.06, "Linear", ha="center", fontsize=8)
    ax_b.text(1, -0.06, "Planar", ha="center", fontsize=8)
    ax_b.text(0.5, np.sqrt(3) / 2 + 0.04, "Spherical", ha="center", fontsize=8)
    ax_b.set_xlim(-0.1, 1.1)
    ax_b.set_ylim(-0.12, 1.0)
    ax_b.set_aspect("equal")
    ax_b.axis("off")
    ax_b.set_title("B.  Ternary Feature Space", fontweight="bold", fontsize=11)

    # ---- Panel C: Multi-scale profiles ----
    ax_c = fig.add_subplot(gs[0, 3])
    feat_names = EigenFeatures.feature_names()
    radii = ms_features.radii
    palette_c = plt.cm.Set2(np.linspace(0, 1, len(feat_names)))

    for idx, fn in enumerate(feat_names):
        means = []
        for s in ms_features.scales:
            vals = getattr(s, fn)
            v_clean = vals[~np.isnan(vals)]
            means.append(np.mean(v_clean) if len(v_clean) > 0 else 0)
        ax_c.plot(radii, means, "o-", color=palette_c[idx], linewidth=1.5,
                  markersize=4, label=fn[:5])

    ax_c.set_xlabel("Radius (mm)", fontsize=9)
    ax_c.set_ylabel("Mean value", fontsize=9)
    ax_c.legend(fontsize=6, ncol=2, loc="upper left")
    ax_c.grid(True, alpha=0.3)
    ax_c.set_title("C.  Multi-scale Profiles", fontweight="bold", fontsize=11)

    # ---- Panel D: Unique variance (if comparison provided) ----
    if comparison is not None:
        ax_d = fig.add_subplot(gs[1, 0:2])
        eigen_names = comparison["eigen_names"]
        unique_var = comparison["unique_variance"]
        y_pos = np.arange(len(eigen_names))

        bar_colors = ["#e74c3c" if uv > 0.5 else "#95a5a6"
                      for uv in unique_var]
        ax_d.barh(y_pos, unique_var, color=bar_colors, edgecolor="none",
                  height=0.7)
        ax_d.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax_d.set_yticks(y_pos)
        ax_d.set_yticklabels(eigen_names, fontsize=6)
        ax_d.set_xlabel("Unique Variance (1 − R²)")
        ax_d.set_xlim(0, 1)
        ax_d.invert_yaxis()
        ax_d.set_title("D.  Novel Information vs Classical Metrics",
                        fontweight="bold", fontsize=11)

        n_novel = sum(1 for uv in unique_var if uv > 0.5)
        ax_d.text(
            0.98, 0.02,
            f"{n_novel}/{len(unique_var)} features >50% unique",
            transform=ax_d.transAxes, ha="right", va="bottom",
            fontsize=9, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
        )
    else:
        # Placeholder: correlation heatmap of eigenfeatures across scales
        ax_d = fig.add_subplot(gs[1, 0:2])
        feat_mat = ms_features.as_matrix()
        valid_feat = feat_mat[~np.any(np.isnan(feat_mat), axis=1)]
        if len(valid_feat) > 5000:
            valid_feat = valid_feat[rng.choice(len(valid_feat), 5000, replace=False)]
        corr = np.corrcoef(valid_feat.T)
        ax_d.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax_d.set_title("D.  Feature Cross-correlation",
                        fontweight="bold", fontsize=11)
        ax_d.set_xticks(range(len(ms_features.column_names())))
        ax_d.set_xticklabels(ms_features.column_names(), rotation=90, fontsize=5)
        ax_d.set_yticks(range(len(ms_features.column_names())))
        ax_d.set_yticklabels(ms_features.column_names(), fontsize=5)

    # ---- Panel E: Feature surface at multiple scales ----
    for s_idx, scale in enumerate(ms_features.scales[:4]):
        ax_s = fig.add_subplot(gs[1, 2 + s_idx // 2 * 0 + s_idx % 4],
                                projection="3d") if s_idx < 2 else None
        if s_idx >= 2:
            break  # Only show 2 scales in this row

    # Simpler: show eigenentropy at finest and coarsest scale
    for panel_idx, (s, label) in enumerate([
        (ms_features.scales[0], f"r={ms_features.radii[0]:.0f}mm"),
        (ms_features.scales[-1], f"r={ms_features.radii[-1]:.0f}mm"),
    ]):
        ax_e = fig.add_subplot(gs[1, 2 + panel_idx], projection="3d")
        vals = np.nan_to_num(s.eigenentropy, nan=0.0)
        valid_vals = s.eigenentropy[~np.isnan(s.eigenentropy)]
        vmin_e = np.percentile(valid_vals, 5) if len(valid_vals) > 0 else 0
        vmax_e = np.percentile(valid_vals, 95) if len(valid_vals) > 0 else 1
        mapper = cm.ScalarMappable(
            norm=colors.Normalize(vmin=vmin_e, vmax=vmax_e), cmap="viridis",
        )
        face_vals = vals[mesh.faces].mean(axis=1)
        face_rgba_e = mapper.to_rgba(face_vals)

        polys_e = Poly3DCollection(
            mesh.vertices[mesh.faces],
            facecolors=face_rgba_e, edgecolors="none", linewidths=0,
        )
        ax_e.add_collection3d(polys_e)
        ax_e.set_xlim(v[:, 0].min() - margin, v[:, 0].max() + margin)
        ax_e.set_ylim(v[:, 1].min() - margin, v[:, 1].max() + margin)
        ax_e.set_zlim(v[:, 2].min() - margin, v[:, 2].max() + margin)
        ax_e.view_init(elev=30, azim=-60)
        ax_e.axis("off")
        title_letter = "E" if panel_idx == 0 else "F"
        ax_e.set_title(
            f"{title_letter}.  Eigenentropy {label}",
            fontweight="bold", fontsize=11,
        )

    # ---- Bottom row: radar fingerprints ----
    for r_idx in range(4):
        ax_r = fig.add_subplot(gs[2, r_idx], polar=True)

        # Use eigenfeatures at middle scale
        mid = len(ms_features.scales) // 2
        s = ms_features.scales[mid]

        # Random parcel-like regions (split into quadrants)
        angles_r = np.linspace(0, 2 * np.pi, len(feat_names),
                                endpoint=False).tolist()
        angles_r += angles_r[:1]

        # Compute mean feature values in a spatial quadrant
        centroids = np.array([
            [v[:, 0].mean() + 30, v[:, 1].mean(), v[:, 2].mean()],
            [v[:, 0].mean() - 30, v[:, 1].mean(), v[:, 2].mean()],
            [v[:, 0].mean(), v[:, 1].mean() + 30, v[:, 2].mean()],
            [v[:, 0].mean(), v[:, 1].mean() - 30, v[:, 2].mean()],
        ])
        tree = cKDTree(mesh.vertices)
        _, closest = tree.query(centroids[r_idx])
        neighbors = tree.query_ball_point(mesh.vertices[closest], 15.0)

        values_r = []
        for fn in feat_names:
            region_vals = getattr(s, fn)[neighbors]
            region_vals = region_vals[~np.isnan(region_vals)]
            values_r.append(np.mean(region_vals) if len(region_vals) > 0 else 0)

        # Normalize
        max_vals = [max(abs(vr), 1e-12) for vr in values_r]
        values_norm = [vr / mv for vr, mv in zip(values_r, max_vals)]
        values_norm += values_norm[:1]

        color_r = plt.cm.Set1(r_idx / 4)
        ax_r.fill(angles_r, values_norm, alpha=0.25, color=color_r)
        ax_r.plot(angles_r, values_norm, "o-", linewidth=2,
                  markersize=3, color=color_r)
        ax_r.set_xticks(angles_r[:-1])
        ax_r.set_xticklabels([fn[:4] for fn in feat_names], fontsize=7)
        ax_r.set_ylim(0, 1.2)
        ax_r.set_yticks([])

        quadrant_names = ["Anterior", "Posterior", "Superior", "Inferior"]
        ax_r.set_title(
            f"G{r_idx+1}. {quadrant_names[r_idx]} Region",
            fontsize=10, fontweight="bold", pad=12,
        )

    fig.suptitle(
        "Eigenmorph: Multi-scale Geometric Features for Cortical Morphology",
        fontsize=16, fontweight="bold", y=0.98,
    )

    if save_path:
        fig.savefig(save_path)
    return fig
