# -*- coding: utf-8 -*-
"""
sars.tractogram_bootstrap
==============================

Probabilistic connectomics via SIFT2 tractogram bootstrap.

The standard connectomics pipeline treats structural connectivity (SC)
as deterministic: SC_ab = Σ wᵢ for streamlines connecting parcels a↔b.
But this is one realization of a stochastic process (seeding, tracking,
filtering), and the uncertainty is real but universally ignored.

This module quantifies that uncertainty via weighted bootstrap of the
SIFT2 tractogram, enabling:

    1. Confidence intervals on every SC edge
    2. Edge reliability classification (robust vs fragile connections)
    3. Probabilistic community detection (module assignment stability)
    4. Graph metric distributions (not point estimates)
    5. Subject-level uncertainty maps (no controls needed)

The bootstrap operates on (streamline, weight) pairs:
    For b = 1, …, B:
        Sample N streamlines with replacement from the tractogram
        Assign each sampled streamline its SIFT2 weight
        Build SC⁽ᵇ⁾ by aggregating into the parcellation

    Uncertainty at edge (a,b) = variability of SC⁽ᵇ⁾_{ab} across b

Compatible with any MRtrix3 pipeline:
    tckgen → tcksift2 → tck2connectome -out_assignments

Modules
-------
core
    StreamlineAssignment, EdgeStats, BootstrapResult data structures;
    loading from MRtrix3 files; bootstrap engine; edge classification.
community
    Probabilistic community detection across bootstrap samples,
    co-assignment matrices, consensus partition, node stability,
    and graph-theoretic metrics with bootstrap CIs.
viz
    Publication-quality figures: SC uncertainty maps, edge classification,
    co-assignment probability, node stability, graph metric CIs, and
    clinical correlation plots.

Pipeline
--------
From MRtrix3 outputs:

    from sars.tractogram_bootstrap import (
        load_streamline_assignments,
        bootstrap_tractogram,
        classify_edges,
        probabilistic_community_detection,
    )

    assignments = load_streamline_assignments(
        connectome_csv='connectome.csv',
        weights_csv='sift2_weights.csv',
        assignments_txt='assignments.txt',
        n_parcels=100,
    )
    result = bootstrap_tractogram(assignments, n_bootstrap=1000)
    edge_class = classify_edges(result)
    communities = probabilistic_community_detection(result)

For prototyping without raw tractograms:

    from sars.tractogram_bootstrap import create_assignments_from_sc
    assignments = create_assignments_from_sc(sc_matrix)

References
----------
- Tournier et al. (2019). NeuroImage 202:116137. MRtrix3.
- Smith et al. (2015). NeuroImage 121:176-185. SIFT2.
- Efron & Tibshirani (1993). An Introduction to the Bootstrap.
  Chapman & Hall.
- Rubinov & Sporns (2010). NeuroImage 52:1059-1069.
- Lancichinetti & Fortunato (2012). Sci Rep 2:336.
"""

# === core ===
from .core import (
    StreamlineAssignment,
    EdgeStats,
    BootstrapResult,
    load_streamline_assignments,
    create_assignments_from_sc,
    build_sc_from_streamlines,
    bootstrap_tractogram,
    classify_edges,
)

# === community ===
from .community import (
    probabilistic_community_detection,
    graph_metrics_with_ci,
)

# === viz ===
from .viz import (
    plot_sc_uncertainty,
    plot_edge_classification,
    plot_community_results,
    plot_graph_metrics_ci,
    plot_stability_vs_clinical,
)

__all__ = [
    # --- core ---
    "StreamlineAssignment",
    "EdgeStats",
    "BootstrapResult",
    "load_streamline_assignments",
    "create_assignments_from_sc",
    "build_sc_from_streamlines",
    "bootstrap_tractogram",
    "classify_edges",
    # --- community ---
    "probabilistic_community_detection",
    "graph_metrics_with_ci",
    # --- viz ---
    "plot_sc_uncertainty",
    "plot_edge_classification",
    "plot_community_results",
    "plot_graph_metrics_ci",
    "plot_stability_vs_clinical",
]
