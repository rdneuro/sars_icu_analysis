# -*- coding: utf-8 -*-
"""
sars.graph_analysis
===================

Graph-theoretic analysis of brain connectivity networks.

Modules
-------
network_metrics
    Global and nodal graph metrics (Rubinov & Sporns 2010): segregation
    (clustering, transitivity, modularity), integration (path length,
    efficiency), influence (degree, betweenness, eigenvector centrality),
    small-world indices, rich-club, SC-FC coupling, multi-threshold AUC.
community
    Community detection (Louvain, Leiden, consensus clustering), modular
    roles (participation coefficient, within-module degree), and
    Network-Based Statistic (NBS).
"""

# === network_metrics ===
from .network_metrics import (
    compute_global_metrics,
    compute_nodal_metrics,
    compute_rich_club,
    compute_small_world,
    compute_sc_fc_coupling,
    compute_metrics_across_thresholds,
    classify_nodes,
    analyze_network,
)

# === community ===
from .community import (
    detect_communities_louvain,
    detect_communities_leiden,
    consensus_clustering,
    compute_participation_coefficient,
    compute_within_module_degree,
    compute_nbs,
    analyze_communities,
)

__all__ = [
    # network_metrics
    "compute_global_metrics",
    "compute_nodal_metrics",
    "compute_rich_club",
    "compute_small_world",
    "compute_sc_fc_coupling",
    "compute_metrics_across_thresholds",
    "classify_nodes",
    "analyze_network",
    # community
    "detect_communities_louvain",
    "detect_communities_leiden",
    "consensus_clustering",
    "compute_participation_coefficient",
    "compute_within_module_degree",
    "compute_nbs",
    "analyze_communities",
]
