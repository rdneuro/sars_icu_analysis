# -*- coding: utf-8 -*-
"""
sars.viz
========

Publication-quality visualizations for the SARS-CoV-2 neuroimaging library.

Modules
-------
matrix_plots
    Connectivity matrices, degree distributions, metric distributions,
    SC-FC scatter, edge-weight histograms, community matrices, rich-club
    curves, small-world summaries, AUC across thresholds.
brain_plots
    Glass-brain connectomes, nodal metric overlays, community maps,
    hub identification, SC-FC coupling on cortical surfaces.
"""

# === matrix_plots ===
from .matrix_plots import (
    plot_connectivity_matrix,
    plot_matrix_comparison,
    plot_threshold_panel,
    plot_degree_distribution,
    plot_metric_distribution,
    plot_sc_fc_scatter,
    plot_edge_weight_distribution,
    plot_community_matrix,
    plot_inter_module_connectivity,
    plot_rich_club_curve,
    plot_small_world_summary,
    plot_auc_across_thresholds,
)

# === brain_plots ===
from .brain_plots import (
    plot_connectome,
    plot_metric_on_brain,
    plot_glass_brain_edges,
    plot_community_on_brain,
    plot_hub_nodes,
    plot_sc_fc_coupling_brain,
)

__all__ = [
    # matrix_plots
    "plot_connectivity_matrix",
    "plot_matrix_comparison",
    "plot_threshold_panel",
    "plot_degree_distribution",
    "plot_metric_distribution",
    "plot_sc_fc_scatter",
    "plot_edge_weight_distribution",
    "plot_community_matrix",
    "plot_inter_module_connectivity",
    "plot_rich_club_curve",
    "plot_small_world_summary",
    "plot_auc_across_thresholds",
    # brain_plots
    "plot_connectome",
    "plot_metric_on_brain",
    "plot_glass_brain_edges",
    "plot_community_on_brain",
    "plot_hub_nodes",
    "plot_sc_fc_coupling_brain",
]
