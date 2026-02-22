# -*- coding: utf-8 -*-
"""
sars.eigenmorph
==============================

Eigenvalue-based geometric features for cortical morphology analysis.

Bridges point cloud analysis (Poux, 2024) with computational neuroanatomy
by treating FreeSurfer cortical surface meshes as 3D point clouds and
computing eigenvalue-based geometric features from local vertex
neighborhoods at multiple spatial scales.

Standard cortical morphometrics (thickness, curvature, sulcal depth)
capture 2D differential properties of the cortical sheet.  Eigenvalue
features capture the full 3D shape of local vertex neighborhoods via
the covariance matrix eigenvalues λ₁ ≥ λ₂ ≥ λ₃, yielding seven
complementary descriptors: linearity, planarity, sphericity,
omnivariance, anisotropy, eigenentropy, and surface variation.

Multi-scale computation (e.g. 3, 5, 10, 20 mm radii) provides a rich
morphological fingerprint per vertex, from fine sulcal branches to
lobar-scale geometry — capturing information that classical FreeSurfer
metrics miss (>50% unique variance in empirical validation).

Modules
-------
core
    SurfaceMesh container, EigenFeatures / MultiScaleFeatures data
    classes, single- and multi-scale eigenvalue computation via
    KD-tree neighborhood queries.
parcellation
    Parcel-level aggregation (Schaefer, Brainnetome, etc.) and
    comparison with classical FreeSurfer morphometrics via correlation
    and unique-variance analysis.
viz
    Publication-quality surface plots, multi-scale profiles, ternary
    feature-space visualization, and classical comparison heatmaps.
viz_interactive
    High-impact showcase visualizations using FURY (VTK) for 3D
    rendering: RGB identity maps, feature landscapes, exploded views,
    animated scale sweeps, neighborhood explorers, morphological
    radar fingerprints, UMAP embeddings, and composite hero figures.
synthetic
    Synthetic cortical surface generation (icosphere + sinusoidal folds)
    for testing and demonstration.

Pipeline
--------
For a single-subject analysis:

    from sars.eigenmorph import (
        SurfaceMesh,
        compute_multiscale_eigenfeatures,
        parcellate_features,
        compare_with_classical,
    )
    import nibabel as nib

    v, f = nib.freesurfer.read_geometry('lh.white')
    mesh = SurfaceMesh(vertices=v, faces=f)
    ms = compute_multiscale_eigenfeatures(mesh, radii=[3, 5, 10, 20])

    labels, _, names = nib.freesurfer.read_annot('lh.Schaefer200.annot')
    parc = parcellate_features(ms, labels, n_parcels=100)

For comparison with classical metrics:

    classical = {
        'thickness': nib.freesurfer.read_morph_data('lh.thickness'),
        'curv': nib.freesurfer.read_morph_data('lh.curv'),
        'sulc': nib.freesurfer.read_morph_data('lh.sulc'),
    }
    comp = compare_with_classical(ms, classical)

References
----------
- Poux (2024). 3D Point Cloud Processing with Python. Springer.
- Weinmann et al. (2015). ISPRS J Photogramm Remote Sens. Semantic
  point cloud interpretation based on optimal neighborhoods, relevant
  features and efficient classifiers.
- Schaefer et al. (2018). Cereb Cortex 28:3095-3114. Local-Global
  Parcellation of the Human Cerebral Cortex.
- Fischl (2012). NeuroImage 62:774-781. FreeSurfer.
"""

# === core ===
from .core import (
    SurfaceMesh,
    EigenFeatures,
    MultiScaleFeatures,
    compute_eigenfeatures,
    compute_multiscale_eigenfeatures,
)

# === parcellation ===
from .parcellation import (
    parcellate_features,
    compare_with_classical,
)

# === viz ===
from .viz import (
    plot_surface_feature,
    plot_feature_overview,
    plot_multiscale_profile,
    plot_ternary_features,
    plot_classical_comparison,
)

# === synthetic ===
from .synthetic import (
    generate_synthetic_cortex,
    generate_vertex_parcellation,
)

# === viz_interactive (optional — requires fury) ===
try:
    from .viz_interactive import (
        plot_rgb_identity,
        plot_feature_landscape,
        render_scale_sweep,
        plot_exploded_view,
        plot_neighborhood_explorer,
        plot_dual_view,
        plot_morphological_radar,
        plot_feature_embedding,
        plot_hero_figure,
    )
    _HAS_VIZ_INTERACTIVE = True
except ImportError:
    _HAS_VIZ_INTERACTIVE = False


__all__ = [
    # --- core ---
    "SurfaceMesh",
    "EigenFeatures",
    "MultiScaleFeatures",
    "compute_eigenfeatures",
    "compute_multiscale_eigenfeatures",
    # --- parcellation ---
    "parcellate_features",
    "compare_with_classical",
    # --- viz ---
    "plot_surface_feature",
    "plot_feature_overview",
    "plot_multiscale_profile",
    "plot_ternary_features",
    "plot_classical_comparison",
    # --- viz_interactive ---
    "plot_rgb_identity",
    "plot_feature_landscape",
    "render_scale_sweep",
    "plot_exploded_view",
    "plot_neighborhood_explorer",
    "plot_dual_view",
    "plot_morphological_radar",
    "plot_feature_embedding",
    "plot_hero_figure",
    # --- synthetic ---
    "generate_synthetic_cortex",
    "generate_vertex_parcellation",
]
