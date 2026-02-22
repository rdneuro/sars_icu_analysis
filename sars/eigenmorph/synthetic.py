# -*- coding: utf-8 -*-
"""
sars.eigenmorph.synthetic
==================================================

Synthetic cortical surface generation for testing and demonstration.

Generates an icosphere-based cortical surface with sinusoidal radial
deformations that mimic gyral and sulcal folds, plus mock classical
morphometrics (thickness, curvature, sulcal depth) for validating the
eigenvalue feature pipeline without requiring FreeSurfer data.

Functions
---------
generate_synthetic_cortex
    Create a deformed icosphere with mock morphometrics.
generate_vertex_parcellation
    Quick k-means parcellation for testing (use FreeSurfer annots for
    real analysis).
"""

import numpy as np
from scipy.cluster.vq import kmeans2
from typing import Tuple, Dict

from .core import SurfaceMesh


# =============================================================================
# ICOSPHERE GENERATION
# =============================================================================

def _icosphere(subdivisions: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an icosphere via recursive subdivision of an icosahedron.

    Parameters
    ----------
    subdivisions : int
        Number of subdivision iterations:
            4 → 2,562 vertices
            5 → 10,242 vertices
            6 → 40,962 vertices

    Returns
    -------
    vertices : np.ndarray (V, 3)
        Unit-sphere coordinates.
    faces : np.ndarray (F, 3)
        Triangle face indices.
    """
    t = (1 + np.sqrt(5)) / 2.0

    verts_list = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]

    for i in range(len(verts_list)):
        v = verts_list[i]
        norm = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        verts_list[i] = [v[0] / norm, v[1] / norm, v[2] / norm]

    faces_list = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    for _ in range(subdivisions):
        edge_cache = {}
        new_faces = []

        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in edge_cache:
                return edge_cache[key]
            v1 = verts_list[i]
            v2 = verts_list[j]
            mid = [(v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2,
                   (v1[2] + v2[2]) / 2]
            norm = (mid[0] ** 2 + mid[1] ** 2 + mid[2] ** 2) ** 0.5
            mid = [mid[0] / norm, mid[1] / norm, mid[2] / norm]
            idx = len(verts_list)
            verts_list.append(mid)
            edge_cache[key] = idx
            return idx

        for tri in faces_list:
            a, b, c = tri
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])

        faces_list = new_faces

    return np.array(verts_list), np.array(faces_list, dtype=int)


# =============================================================================
# SYNTHETIC CORTEX
# =============================================================================

def generate_synthetic_cortex(
    n_vertices: int = 10000,
    n_gyri: int = 8,
    seed: int = 42,
) -> Tuple[SurfaceMesh, Dict[str, np.ndarray]]:
    """
    Generate a synthetic cortical surface with gyri and sulci.

    Uses icosphere subdivision for a proper triangle mesh with uniform
    vertex density, then deforms radially with sinusoidal folds to
    mimic cortical geometry.

    Parameters
    ----------
    n_vertices : int
        Approximate target vertex count.  Actual count determined by
        icosphere subdivision level:
            ≤ 3000  → sub=4 (2,562 vertices)
            ≤ 15000 → sub=5 (10,242 vertices)
            > 15000 → sub=6 (40,962 vertices)
    n_gyri : int
        Number of major gyral folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    mesh : SurfaceMesh
    classical_metrics : dict
        {'thickness': (V,), 'curv': (V,), 'sulc': (V,)} mock metrics.
    """
    rng = np.random.default_rng(seed)

    if n_vertices <= 3000:
        subdiv = 4
    elif n_vertices <= 15000:
        subdiv = 5
    else:
        subdiv = 6

    unit_verts, faces = _icosphere(subdiv)
    actual_n = len(unit_verts)
    base_radius = 80.0

    x, y, z = unit_verts[:, 0], unit_verts[:, 1], unit_verts[:, 2]
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z, -1, 1))

    # Major gyral folds
    radial_deformation = np.zeros(actual_n)
    for _ in range(n_gyri):
        amp = rng.uniform(2.0, 5.0)
        freq_theta = rng.uniform(1, 4)
        freq_phi = rng.uniform(1, 3)
        phase = rng.uniform(0, 2 * np.pi)
        radial_deformation += (amp * np.sin(freq_theta * theta + phase) *
                               np.cos(freq_phi * phi))

    # Fine folds
    for _ in range(20):
        amp = rng.uniform(0.5, 1.5)
        freq = rng.uniform(4, 12)
        phase = rng.uniform(0, 2 * np.pi)
        if rng.random() < 0.5:
            radial_deformation += amp * np.sin(freq * theta + phase)
        else:
            radial_deformation += amp * np.sin(freq * phi + phase)

    r = base_radius + radial_deformation
    vertices = unit_verts * r[:, np.newaxis]

    mesh = SurfaceMesh(
        vertices=vertices, faces=faces,
        hemisphere="lh", surface_type="synthetic",
    )

    # Mock classical morphometrics
    rd_std = max(radial_deformation.std(), 1e-6)
    thickness = 2.5 + 0.3 * (radial_deformation - radial_deformation.mean()) / rd_std
    thickness += rng.normal(0, 0.1, actual_n)
    thickness = np.clip(thickness, 1.0, 4.5)

    curv = -np.gradient(radial_deformation)
    curv += rng.normal(0, 0.05, actual_n)

    sulc = -(radial_deformation - radial_deformation.max())
    sulc = sulc / max(sulc.max(), 1e-6)
    sulc += rng.normal(0, 0.02, actual_n)

    classical = {"thickness": thickness, "curv": curv, "sulc": sulc}

    return mesh, classical


def generate_vertex_parcellation(
    mesh: SurfaceMesh,
    n_parcels: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a mock vertex-to-parcel assignment via k-means.

    For real data, use FreeSurfer annotation files instead.

    Parameters
    ----------
    mesh : SurfaceMesh
    n_parcels : int
    seed : int

    Returns
    -------
    np.ndarray (V,)
        Integer labels, 1-indexed (0 = unassigned / medial wall).
    """
    _, labels = kmeans2(mesh.vertices, n_parcels, minit="points",
                        seed=seed)
    return labels + 1
