#!/usr/bin/env python3
"""
==============================================================================
GNN MULTIMODAL ANALYSIS SUBPACKAGE
==============================================================================

Graph Neural Network analyses combining structural and functional
connectivity for the SARS-CoV-2 Brain Connectivity Analysis Library.

Implements two complementary frameworks:
1. GAT SC→FC Prediction: predicts functional connectivity from structural
   connectivity using Graph Attention Networks, revealing learned attention
   weights (effective structural connectivity) and regional SC-FC decoupling.
2. Contrastive Multimodal Learning: learns aligned SC-FC representations
   via self-supervised contrastive optimization, revealing regional
   multimodal coherence and data-driven patient subgroups.

Quick Start:
-----------
    from gnn_multimodal import GNNMultimodalConfig, run_full_pipeline

    config = GNNMultimodalConfig()
    results = run_full_pipeline(config)

Individual Analyses:
-------------------
    from gnn_multimodal.training import train_gat_sc_fc, train_contrastive
    from gnn_multimodal.analysis import convergence_analysis

    gat_results = train_gat_sc_fc(config, atlas="schaefer_100")
    cont_results = train_contrastive(config, atlas="schaefer_100")
    conv = convergence_analysis(gat_results, cont_results, "schaefer_100")

Requirements:
------------
    torch >= 2.0
    torch_geometric >= 2.4
    numpy, scipy, scikit-learn, matplotlib, seaborn

References:
----------
    Wu & Li (2023). Human Brain Mapping, 44(9), 3885-3896.
    Chen et al. (2020). SimCLR. ICML 2020.
    Baum et al. (2020). NeuroImage, 210, 116612.
    Vázquez-Rodríguez et al. (2019). PNAS, 116(42), 21219-21227.

Author: SARS-CoV-2 Neuroimaging Study
Date: February 2026
==============================================================================
"""

__version__ = "0.1.0"
__author__ = "Velho Mago"

# Configuration
from .config import (
    GNNMultimodalConfig,
    GATConfig,
    ContrastiveConfig,
    ATLAS_REGISTRY,
    SUBJECTS,
)

# Data loading
from .data_loader import (
    BrainConnectomeDataset,
    MultimodalPairedDataset,
    load_sc_matrix,
    load_fc_matrix,
    load_atlas_labels,
    build_subject_graph,
    build_paired_graphs,
)

# Models
from .gat_sc_fc import (
    GATSCFC,
    SCFCPredictionLoss,
    compute_regional_decoupling,
    compute_cohort_decoupling,
    extract_attention_matrix,
)

from .contrastive import (
    MultimodalContrastiveModel,
    NTXentLoss,
    NodeContrastiveLoss,
    compute_regional_coherence,
    discover_subgroups,
)

# Training
from .training import (
    train_gat_sc_fc,
    train_contrastive,
    run_full_pipeline,
)

# Analysis
from .analysis import (
    compute_hierarchical_gradient,
    cross_atlas_consistency,
    convergence_analysis,
    permutation_test_regional,
    generate_analysis_report,
)

# Visualization
from .viz import (
    plot_scfc_prediction_overview,
    plot_contrastive_overview,
    plot_convergence,
    plot_hierarchical_gradient,
)

from .gnn_connectome import (
    # Utilities
    connectivity_to_pyg,
    build_multimodal_features,
    compute_r2_nodal,
    # Models
    SCFCPredictor,
    BrainVGAE,
    MultimodalHeteroGNN,
    NodeAnomalyDetector,
    GraphLevelEmbedder,
    # Pipeline
    BrainGNNPipeline,
    # Result dataclasses
    SCFCPredictionResult,
    VAEResult,
    HeterogeneousResult,
    AnomalyResult,
    GraphEmbeddingResult,
)

__all__ = [
    # Config
    "GNNMultimodalConfig", "GATConfig", "ContrastiveConfig",
    "ATLAS_REGISTRY", "SUBJECTS",
    # Data
    "BrainConnectomeDataset", "MultimodalPairedDataset",
    "load_sc_matrix", "load_fc_matrix", "load_atlas_labels",
    "build_subject_graph", "build_paired_graphs",
    # Models
    "GATSCFC", "SCFCPredictionLoss",
    "MultimodalContrastiveModel", "NTXentLoss", "NodeContrastiveLoss",
    # Inference
    "compute_regional_decoupling", "compute_cohort_decoupling",
    "extract_attention_matrix", "compute_regional_coherence",
    "discover_subgroups",
    # Training
    "train_gat_sc_fc", "train_contrastive", "run_full_pipeline",
    # Analysis
    "compute_hierarchical_gradient", "cross_atlas_consistency",
    "convergence_analysis", "permutation_test_regional",
    "generate_analysis_report",
    # Visualization
    "plot_scfc_prediction_overview", "plot_contrastive_overview",
    "plot_convergence", "plot_hierarchical_gradient",
]
