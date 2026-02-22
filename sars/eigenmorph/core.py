# -*- coding: utf-8 -*-
"""
sars.eigenmorph.core
==================================================

Core eigenvalue geometric feature computation for cortical surfaces.

Implements the computational engine that treats FreeSurfer cortical
surface meshes as 3D point clouds and extracts eigenvalue-based
geometric descriptors from local vertex neighborhoods.  The covariance
matrix of each vertex's spatial neighborhood is decomposed into
eigenvalues λ₁ ≥ λ₂ ≥ λ₃, from which seven complementary features
are derived:

    - **Linearity**   = (λ₁ − λ₂) / λ₁  → ridge-like (sulcal fundi)
    - **Planarity**   = (λ₂ − λ₃) / λ₁  → planar (gyral crowns)
    - **Sphericity**  = λ₃ / λ₁          → isotropic (sulcal pits)
    - **Omnivariance** = (λ₁·λ₂·λ₃)^⅓  → spatial dispersion
    - **Anisotropy**  = (λ₁ − λ₃) / λ₁  → directional bias
    - **Eigenentropy** = −Σ λ̃ᵢ ln(λ̃ᵢ)  → shape complexity
    - **Surface var.** = λ₃ / Σλᵢ        → local roughness

Classes
-------
SurfaceMesh
    Container for cortical surface geometry (vertices + faces).
EigenFeatures
    Single-scale eigenvalue features for all vertices.
MultiScaleFeatures
    Multi-scale features across multiple neighborhood radii.

Functions
---------
compute_eigenfeatures
    Compute features at a single spatial scale.
compute_multiscale_eigenfeatures
    Compute features across multiple scales (e.g. 3, 5, 10, 20 mm).

References
----------
- Poux (2024). 3D Point Cloud Processing with Python. Springer.
- Weinmann et al. (2015). ISPRS J Photogramm Remote Sens.
- Fischl (2012). NeuroImage 62:774-781. FreeSurfer.
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Optional, List
from dataclasses import dataclass, field


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SurfaceMesh:
    """
    A cortical surface mesh.

    Parameters
    ----------
    vertices : np.ndarray (V, 3)
        Vertex coordinates in mm (e.g., FreeSurfer RAS space).
    faces : np.ndarray (F, 3)
        Triangle face indices (0-indexed).
    vertex_normals : np.ndarray (V, 3), optional
        Per-vertex normals.  Computed automatically if not provided.
    hemisphere : str
        'lh', 'rh', or 'both'.
    surface_type : str
        'white', 'pial', 'inflated', 'sphere', 'synthetic', etc.
    """

    vertices: np.ndarray
    faces: np.ndarray
    vertex_normals: Optional[np.ndarray] = None
    n_vertices: int = 0
    n_faces: int = 0
    hemisphere: str = "unknown"
    surface_type: str = "unknown"

    def __post_init__(self):
        self.n_vertices = self.vertices.shape[0]
        self.n_faces = self.faces.shape[0]
        if self.vertex_normals is None:
            self.vertex_normals = self._compute_normals()

    def _compute_normals(self) -> np.ndarray:
        """Compute per-vertex normals from face normals (area-weighted)."""
        normals = np.zeros_like(self.vertices)

        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)

        for i in range(3):
            np.add.at(normals, self.faces[:, i], face_normals)

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normals /= norms

        return normals


@dataclass
class EigenFeatures:
    """
    Eigenvalue geometric features at a single spatial scale.

    Parameters
    ----------
    radius : float
        Neighborhood radius in mm.
    linearity : np.ndarray (V,)
        Ridge-like geometry.  High in sulcal fundi.
    planarity : np.ndarray (V,)
        Planar geometry.  High on gyral crowns.
    sphericity : np.ndarray (V,)
        Isotropic geometry.  High at sulcal intersections / pits.
    omnivariance : np.ndarray (V,)
        Overall spatial dispersion.
    anisotropy : np.ndarray (V,)
        Directional bias.  linearity + planarity ≈ anisotropy.
    eigenentropy : np.ndarray (V,)
        Shape complexity / disorder.
    surface_variation : np.ndarray (V,)
        Local roughness (change of curvature).
    eigenvalues : np.ndarray (V, 3)
        Raw eigenvalues [λ₁, λ₂, λ₃] per vertex.
    n_neighbors : np.ndarray (V,)
        Number of neighbors within radius for each vertex.
    """

    radius: float
    linearity: np.ndarray
    planarity: np.ndarray
    sphericity: np.ndarray
    omnivariance: np.ndarray
    anisotropy: np.ndarray
    eigenentropy: np.ndarray
    surface_variation: np.ndarray
    eigenvalues: np.ndarray = field(repr=False)
    n_neighbors: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        """Return all features as a dict {name: (V,) array}."""
        return {
            "linearity": self.linearity,
            "planarity": self.planarity,
            "sphericity": self.sphericity,
            "omnivariance": self.omnivariance,
            "anisotropy": self.anisotropy,
            "eigenentropy": self.eigenentropy,
            "surface_variation": self.surface_variation,
        }

    def as_matrix(self) -> np.ndarray:
        """Return (V, 7) feature matrix."""
        return np.column_stack([
            self.linearity, self.planarity, self.sphericity,
            self.omnivariance, self.anisotropy, self.eigenentropy,
            self.surface_variation,
        ])

    @staticmethod
    def feature_names() -> list:
        """Canonical feature name order."""
        return [
            "linearity", "planarity", "sphericity", "omnivariance",
            "anisotropy", "eigenentropy", "surface_variation",
        ]


@dataclass
class MultiScaleFeatures:
    """
    Multi-scale eigenvalue features across multiple neighborhood radii.

    Parameters
    ----------
    scales : list of EigenFeatures
        One EigenFeatures per radius.
    radii : list of float
        Radii used (in mm).
    n_vertices : int
        Number of surface vertices.
    """

    scales: list
    radii: list
    n_vertices: int

    def as_matrix(self) -> np.ndarray:
        """
        Concatenated (V, 7 × n_scales) feature matrix.

        Column order: [feat1_r1, feat2_r1, ..., feat7_r1, feat1_r2, ...]
        """
        return np.hstack([s.as_matrix() for s in self.scales])

    def column_names(self) -> list:
        """Feature column names including scale information."""
        names = []
        for s in self.scales:
            for fn in EigenFeatures.feature_names():
                names.append(f"{fn}_r{s.radius:.0f}mm")
        return names

    def get_feature(self, name: str) -> np.ndarray:
        """
        Get a specific feature across all scales.

        Parameters
        ----------
        name : str
            Feature name (e.g. 'linearity').

        Returns
        -------
        np.ndarray (V, n_scales)
        """
        return np.column_stack([
            getattr(s, name) for s in self.scales
        ])


# =============================================================================
# CORE COMPUTATION
# =============================================================================

def compute_eigenfeatures(
    mesh: SurfaceMesh,
    radius: float = 5.0,
    min_neighbors: int = 6,
    verbose: bool = True,
) -> EigenFeatures:
    """
    Compute eigenvalue geometric features for each vertex.

    For each vertex, finds all neighbors within ``radius`` mm via a
    KD-tree, computes the 3×3 covariance matrix of their coordinates,
    extracts eigenvalues λ₁ ≥ λ₂ ≥ λ₃, and derives the seven
    normalized geometric features.

    Parameters
    ----------
    mesh : SurfaceMesh
        Cortical surface mesh.
    radius : float
        Neighborhood radius in mm.  Typical values:
            3 mm — fine cortical folds (sulcal branches)
            5 mm — individual gyri / sulci
            10 mm — regional geometry
            20 mm — lobar shape
    min_neighbors : int
        Minimum neighbors for valid computation.  Vertices with fewer
        neighbors receive NaN features.
    verbose : bool
        Print progress and summary statistics.

    Returns
    -------
    EigenFeatures
        Features for all vertices at the given scale.
    """
    V = mesh.n_vertices
    coords = mesh.vertices

    if verbose:
        print(f"  Computing eigenfeatures (radius={radius}mm, V={V:,})...")

    tree = cKDTree(coords)

    eigenvalues = np.full((V, 3), np.nan)
    n_neighbors = np.zeros(V, dtype=int)

    linearity = np.full(V, np.nan)
    planarity = np.full(V, np.nan)
    sphericity = np.full(V, np.nan)
    omnivariance = np.full(V, np.nan)
    anisotropy = np.full(V, np.nan)
    eigenentropy = np.full(V, np.nan)
    surface_variation = np.full(V, np.nan)

    all_neighbors = tree.query_ball_tree(tree, r=radius)

    for v in range(V):
        neighbors = all_neighbors[v]
        n_neigh = len(neighbors)
        n_neighbors[v] = n_neigh

        if n_neigh < min_neighbors:
            continue

        pts = coords[neighbors]
        centroid = pts.mean(axis=0)
        pts_centered = pts - centroid

        cov = (pts_centered.T @ pts_centered) / n_neigh

        evals, _ = np.linalg.eigh(cov)
        evals = evals[::-1]
        evals = np.maximum(evals, 0)

        eigenvalues[v] = evals
        l1, l2, l3 = evals

        if l1 < 1e-12:
            continue

        sum_l = l1 + l2 + l3

        linearity[v] = (l1 - l2) / l1
        planarity[v] = (l2 - l3) / l1
        sphericity[v] = l3 / l1
        omnivariance[v] = (l1 * l2 * l3) ** (1.0 / 3.0)
        anisotropy[v] = (l1 - l3) / l1

        l_norm = evals / sum_l
        l_norm = l_norm[l_norm > 1e-12]
        eigenentropy[v] = -np.sum(l_norm * np.log(l_norm))

        surface_variation[v] = l3 / sum_l

    if verbose:
        valid = ~np.isnan(linearity)
        print(f"    Valid vertices: {valid.sum():,} / {V:,} "
              f"({valid.sum()/V*100:.1f}%)")
        print(f"    Mean neighbors: {n_neighbors[valid].mean():.0f}")
        print(f"    Linearity:  {np.nanmean(linearity):.3f} "
              f"± {np.nanstd(linearity):.3f}")
        print(f"    Planarity:  {np.nanmean(planarity):.3f} "
              f"± {np.nanstd(planarity):.3f}")
        print(f"    Sphericity: {np.nanmean(sphericity):.3f} "
              f"± {np.nanstd(sphericity):.3f}")

    return EigenFeatures(
        radius=radius,
        linearity=linearity,
        planarity=planarity,
        sphericity=sphericity,
        omnivariance=omnivariance,
        anisotropy=anisotropy,
        eigenentropy=eigenentropy,
        surface_variation=surface_variation,
        eigenvalues=eigenvalues,
        n_neighbors=n_neighbors,
    )


def compute_multiscale_eigenfeatures(
    mesh: SurfaceMesh,
    radii: List[float] = None,
    min_neighbors: int = 6,
    verbose: bool = True,
) -> MultiScaleFeatures:
    """
    Compute eigenvalue features at multiple spatial scales.

    Parameters
    ----------
    mesh : SurfaceMesh
        Cortical surface mesh.
    radii : list of float, optional
        Neighborhood radii in mm.  Default: [3, 5, 10, 20].
        Each captures a different spatial scale:
            3 mm — fine folds, sulcal branches
            5 mm — individual gyri / sulci
            10 mm — regional geometry
            20 mm — lobar-scale shape
    min_neighbors : int
        Minimum neighbors for valid computation.
    verbose : bool
        Print progress.

    Returns
    -------
    MultiScaleFeatures
        Contains 7 features × len(radii) scales per vertex
        (28 features with default 4 radii).
    """
    if radii is None:
        radii = [3.0, 5.0, 10.0, 20.0]

    if verbose:
        print(f"\n  Multi-scale eigenfeatures: {len(radii)} scales")
        print(f"  Radii: {radii} mm")

    scales = []
    for r in radii:
        feat = compute_eigenfeatures(
            mesh, radius=r, min_neighbors=min_neighbors, verbose=verbose,
        )
        scales.append(feat)

    return MultiScaleFeatures(
        scales=scales,
        radii=radii,
        n_vertices=mesh.n_vertices,
    )
