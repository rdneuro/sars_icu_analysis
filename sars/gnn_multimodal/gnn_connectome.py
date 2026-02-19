#!/usr/bin/env python3
"""
===============================================================================
SARS Library — Graph Neural Network Module for Brain Connectomics
===============================================================================

PyTorch Geometric-based GNN models for multimodal brain network analysis.
Designed for the SARS-CoV-2 neuroimaging study (N=23, post-ICU patients).

Models implemented:
    1. SC→FC Predictor (GATv2) — Predicts functional connectivity from 
       structural connectivity; residuals yield learned SC-FC decoupling.
    2. Graph Variational Autoencoder (VGAE) — Unsupervised latent embeddings
       for nodes; reconstruction error identifies atypical connections.
    3. Multimodal Heterogeneous Graph — Integrates SC + FC as a two-layer
       heterogeneous graph; inter-layer weights learn coupling strength.
    4. Node Anomaly Detector — Trains per-subject GAE and identifies regions
       with systematically high reconstruction error across subjects.
    5. Graph-Level Embedding — Produces whole-brain representations via 
       attention pooling for subject fingerprinting and clinical correlation.

References:
    - Chen et al. (Imaging Neuroscience, 2024) — SC-FC coupling via GNN
    - Xia et al. (Medical Image Analysis, 2025) — MS-Inter-GCN
    - Safai et al. (Frontiers in Neuroscience, 2022) — GAT multimodal PD
    - Kipf & Welling (2016) — Variational Graph Autoencoder
    - Li et al. (BrainGNN, Medical Image Analysis, 2021)

Requirements:
    pip install torch torch-geometric scipy numpy pandas scikit-learn

Author: SARS-1 Project / Velho Mago
Date: February 2026
===============================================================================
"""

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# PYTORCH / TORCH GEOMETRIC IMPORTS
# =============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import (
        GATv2Conv,
        GCNConv,
        SAGEConv,
        GraphNorm,
        LayerNorm,
        global_mean_pool,
        global_add_pool,
    )
    from torch_geometric.utils import (
        dense_to_sparse,
        to_dense_adj,
        negative_sampling,
        add_self_loops,
        degree,
    )
    from torch_geometric.nn import VGAE as PyG_VGAE, GAE as PyG_GAE

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn(
        "torch / torch_geometric not found. "
        "Install via: pip install torch torch-geometric"
    )


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

DEVICE = torch.device("cuda" if HAS_PYG and torch.cuda.is_available() else "cpu")

ATLASES = ["synthseg", "schaefer_100", "aal3", "brainnetome"]
ATLAS_SIZES = {"synthseg": 86, "schaefer_100": 100, "aal3": 170, "brainnetome": 246}

# Default node feature sets — expanded at runtime based on available data
FMRI_NODE_FEATURES = [
    "alff", "falff", "reho",
    "strength_fc", "degree_fc", "clustering_fc",
    "eigenvector_centrality", "betweenness_centrality",
    "participation_coef", "within_module_degree_z",
    "gradient_1", "gradient_2", "gradient_3",
    "bold_variance", "bold_mean",
]

DMRI_NODE_FEATURES = [
    "fa_mean", "md_mean", "ad_mean", "rd_mean",
    "strength_sc", "degree_sc", "clustering_sc",
    "betweenness_sc",
]

SC_EDGE_FEATURES = [
    "streamline_count", "sift2_weight",
    "mean_fa", "mean_length",
]


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class SCFCPredictionResult:
    """Results from SC→FC prediction model."""
    fc_predicted: np.ndarray          # (n_nodes, n_nodes) predicted FC
    fc_true: np.ndarray               # (n_nodes, n_nodes) true FC
    nodal_error: np.ndarray           # (n_nodes,) per-node prediction error
    decoupling_score: np.ndarray      # (n_nodes,) normalized decoupling
    edge_error: np.ndarray            # (n_nodes, n_nodes) per-edge error
    train_losses: List[float]         # Training loss history
    val_losses: List[float]           # Validation loss history
    r2_global: float                  # Global R² of FC prediction
    r2_nodal: np.ndarray              # (n_nodes,) per-node R²
    model_state: Optional[dict] = None
    attention_weights: Optional[np.ndarray] = None  # (n_edges,) GAT attention


@dataclass
class VAEResult:
    """Results from Graph Variational Autoencoder."""
    embeddings: np.ndarray            # (n_nodes, latent_dim) node embeddings
    reconstructed_adj: np.ndarray     # (n_nodes, n_nodes) reconstructed adjacency
    edge_recon_error: np.ndarray      # (n_nodes, n_nodes) reconstruction error
    nodal_recon_error: np.ndarray     # (n_nodes,) mean recon error per node
    mu: np.ndarray                    # (n_nodes, latent_dim) mean vectors
    logstd: np.ndarray                # (n_nodes, latent_dim) log std vectors
    train_losses: List[float]
    auc: float                        # Link prediction AUC
    ap: float                         # Average precision


@dataclass
class HeterogeneousResult:
    """Results from Multimodal Heterogeneous Graph model."""
    coupling_weights: np.ndarray      # (n_nodes,) learned SC-FC coupling per node
    sc_embeddings: np.ndarray         # (n_nodes, dim) SC-layer embeddings
    fc_embeddings: np.ndarray         # (n_nodes, dim) FC-layer embeddings
    fused_embeddings: np.ndarray      # (n_nodes, dim) fused representation
    embedding_similarity: np.ndarray  # (n_nodes,) cosine sim SC↔FC embeddings
    train_losses: List[float]


@dataclass
class AnomalyResult:
    """Results from Node-level Anomaly Detection."""
    nodal_anomaly_score: np.ndarray   # (n_nodes,) mean anomaly across subjects
    nodal_anomaly_std: np.ndarray     # (n_nodes,) std of anomaly across subjects
    per_subject_scores: np.ndarray    # (n_subjects, n_nodes) anomaly per subj
    z_scores: np.ndarray              # (n_nodes,) z-scored anomaly
    flagged_nodes: np.ndarray         # Indices of anomalous nodes (z > threshold)
    threshold: float


@dataclass
class GraphEmbeddingResult:
    """Results from Graph-level Embedding."""
    subject_embeddings: np.ndarray    # (n_subjects, embed_dim) 
    subject_ids: List[str]
    similarity_matrix: np.ndarray     # (n_subjects, n_subjects) pairwise sim
    node_importance: np.ndarray       # (n_nodes,) attention-based importance
    cluster_labels: Optional[np.ndarray] = None
    clinical_correlations: Optional[Dict] = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _check_pyg():
    """Raise informative error if PyG is not available."""
    if not HAS_PYG:
        raise ImportError(
            "PyTorch Geometric is required for GNN models.\n"
            "Install: pip install torch torch-geometric\n"
            "See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
        )


def connectivity_to_pyg(
    sc_matrix: np.ndarray,
    fc_matrix: Optional[np.ndarray] = None,
    node_features: Optional[np.ndarray] = None,
    edge_threshold: float = 0.0,
    normalize_edges: bool = True,
    normalize_features: bool = True,
    self_loops: bool = False,
) -> Data:
    """
    Convert connectivity matrices to PyTorch Geometric Data object.

    Parameters
    ----------
    sc_matrix : np.ndarray, shape (n_nodes, n_nodes)
        Structural connectivity matrix (used as graph topology).
    fc_matrix : np.ndarray, optional, shape (n_nodes, n_nodes)
        Functional connectivity matrix (used as node-level targets).
    node_features : np.ndarray, optional, shape (n_nodes, n_features)
        Multimodal node feature matrix. If None, uses node strength.
    edge_threshold : float
        Remove edges with weight below this value (after normalization).
    normalize_edges : bool
        If True, log-transform and min-max normalize edge weights.
    normalize_features : bool
        If True, z-score normalize node features.
    self_loops : bool
        Whether to add self-loops to the graph.

    Returns
    -------
    torch_geometric.data.Data
        Graph object with x, edge_index, edge_attr, and optional y fields.
    """
    _check_pyg()

    n = sc_matrix.shape[0]

    # --- Edge construction from SC ---
    sc = sc_matrix.copy().astype(np.float32)
    np.fill_diagonal(sc, 0)

    if normalize_edges and sc.max() > 0:
        sc_pos = np.maximum(sc, 0)
        sc_pos = np.log1p(sc_pos)
        sc_max = sc_pos.max()
        if sc_max > 0:
            sc_pos /= sc_max
        sc = sc_pos

    # Apply threshold
    sc[sc < edge_threshold] = 0

    # Convert to sparse
    edge_index, edge_attr = dense_to_sparse(torch.tensor(sc))
    edge_attr = edge_attr.float().unsqueeze(-1)  # (n_edges, 1)

    # --- Node features ---
    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float32)
        if normalize_features:
            mu = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + 1e-8
            x = (x - mu) / std
    else:
        # Default: node strength as feature
        strength = sc_matrix.sum(axis=1).astype(np.float32)
        if strength.max() > 0:
            strength = (strength - strength.mean()) / (strength.std() + 1e-8)
        x = torch.tensor(strength, dtype=torch.float32).unsqueeze(-1)

    # --- Self-loops ---
    if self_loops:
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr.squeeze(-1), fill_value=1.0, num_nodes=n
        )
        edge_attr = edge_attr.unsqueeze(-1)

    # --- Build Data object ---
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = n

    # --- FC as target ---
    if fc_matrix is not None:
        fc = fc_matrix.copy().astype(np.float32)
        np.fill_diagonal(fc, 0)
        data.y = torch.tensor(fc, dtype=torch.float32)  # full FC matrix
        data.fc_matrix = torch.tensor(fc, dtype=torch.float32)

    return data


def build_multimodal_features(
    fmri_metrics: Optional[Dict[str, np.ndarray]] = None,
    dmri_metrics: Optional[Dict[str, np.ndarray]] = None,
    n_nodes: Optional[int] = None,
) -> np.ndarray:
    """
    Assemble multimodal node feature matrix from available metrics.

    Parameters
    ----------
    fmri_metrics : dict, optional
        Dictionary mapping feature names to (n_nodes,) arrays from rs-fMRI.
    dmri_metrics : dict, optional
        Dictionary mapping feature names to (n_nodes,) arrays from dMRI.
    n_nodes : int, optional
        Number of nodes (inferred from data if not given).

    Returns
    -------
    np.ndarray, shape (n_nodes, n_features)
        Concatenated feature matrix.
    """
    features = []
    feature_names = []

    for metrics, prefix in [(fmri_metrics, "fmri"), (dmri_metrics, "dmri")]:
        if metrics is None:
            continue
        for name, vals in metrics.items():
            arr = np.asarray(vals, dtype=np.float32).ravel()
            if n_nodes is not None and len(arr) != n_nodes:
                warnings.warn(
                    f"Feature '{name}' has {len(arr)} values, expected {n_nodes}. Skipping."
                )
                continue
            if n_nodes is None:
                n_nodes = len(arr)
            # Handle NaN/Inf
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(arr)
            feature_names.append(f"{prefix}_{name}" if prefix else name)

    if not features:
        raise ValueError("No valid features provided. Supply fmri_metrics or dmri_metrics.")

    X = np.column_stack(features)
    return X


def compute_r2_nodal(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute per-node R² between true and predicted FC rows."""
    n = y_true.shape[0]
    r2 = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i  # exclude diagonal
        yt = y_true[i, mask]
        yp = y_pred[i, mask]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2[i] = 1.0 - ss_res / (ss_tot + 1e-12)
    return r2


# =============================================================================
# MODEL 1: SC → FC PREDICTOR (GATv2)
# =============================================================================

class _SCFCEncoder(nn.Module):
    """
    Graph Attention Network encoder for SC → FC prediction.

    Architecture:
        Input features → GATv2 (multi-head) → GraphNorm → GATv2 → GraphNorm
        → Linear → FC row prediction per node.

    The attention mechanism learns which structural neighbors are most
    informative for predicting each node's functional profile.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 100,
        n_heads: int = 4,
        dropout: float = 0.3,
        n_layers: int = 3,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.dropout = dropout
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_ch = hidden_channels if i == 0 else hidden_channels * n_heads
            self.convs.append(
                GATv2Conv(
                    in_ch, hidden_channels,
                    heads=n_heads,
                    dropout=dropout,
                    concat=True,
                    add_self_loops=True,
                    share_weights=False,
                )
            )
            self.norms.append(GraphNorm(hidden_channels * n_heads))

        # Output head: predict FC row for each node
        final_dim = hidden_channels * n_heads
        self.fc_head = nn.Sequential(
            nn.Linear(final_dim, hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr=None, return_attention=False):
        # Input projection
        h = self.input_proj(x)

        attention_weights = []

        for i in range(self.n_layers):
            h_in = h
            if return_attention:
                h, (ei, aw) = self.convs[i](
                    h, edge_index, edge_attr=edge_attr,
                    return_attention_weights=True,
                )
                attention_weights.append(aw)
            else:
                h = self.convs[i](h, edge_index, edge_attr=edge_attr)

            h = self.norms[i](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Residual connection (after first layer, dimensions match)
            if self.residual and i > 0:
                h = h + h_in

        # Predict FC row per node
        fc_pred = self.fc_head(h)

        if return_attention:
            return fc_pred, attention_weights
        return fc_pred


class SCFCPredictor:
    """
    SC → FC Prediction via Graph Attention Network.

    Trains a GATv2 model on structural connectivity graph topology with
    multimodal node features to predict each node's functional connectivity
    profile. The per-node prediction error constitutes a learned, non-linear
    measure of SC-FC decoupling.

    Parameters
    ----------
    hidden_channels : int
        Hidden dimension in GAT layers.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of GAT layers.
    dropout : float
        Dropout rate.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    epochs : int
        Maximum training epochs.
    patience : int
        Early stopping patience.
    device : str or torch.device
        Compute device.

    Example
    -------
    >>> predictor = SCFCPredictor(hidden_channels=64, n_heads=4)
    >>> result = predictor.fit_predict(sc_matrix, fc_matrix, node_features)
    >>> print(result.decoupling_score)  # per-node SC-FC decoupling
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 500,
        patience: int = 50,
        device: Optional[Union[str, torch.device]] = None,
    ):
        _check_pyg()
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(device) if device else DEVICE
        self.model = None

    def fit_predict(
        self,
        sc_matrix: np.ndarray,
        fc_matrix: np.ndarray,
        node_features: Optional[np.ndarray] = None,
        edge_threshold: float = 0.0,
        val_ratio: float = 0.15,
        verbose: bool = True,
    ) -> SCFCPredictionResult:
        """
        Train SC→FC model and compute decoupling scores.

        Parameters
        ----------
        sc_matrix : np.ndarray, (n_nodes, n_nodes)
            Structural connectivity matrix.
        fc_matrix : np.ndarray, (n_nodes, n_nodes)
            Functional connectivity matrix (target).
        node_features : np.ndarray, optional, (n_nodes, n_features)
            Multimodal node features.
        edge_threshold : float
            SC edge threshold (post-normalization).
        val_ratio : float
            Fraction of nodes used for validation.
        verbose : bool
            Print training progress.

        Returns
        -------
        SCFCPredictionResult
        """
        n_nodes = sc_matrix.shape[0]

        # Build PyG data
        data = connectivity_to_pyg(
            sc_matrix, fc_matrix, node_features,
            edge_threshold=edge_threshold,
        )
        data = data.to(self.device)

        in_channels = data.x.shape[1]

        # Validation mask: random subset of nodes
        perm = torch.randperm(n_nodes)
        n_val = max(1, int(n_nodes * val_ratio))
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask[perm[:n_val]] = True
        train_mask = ~val_mask

        # Initialize model
        self.model = _SCFCEncoder(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=n_nodes,
            n_heads=self.n_heads,
            dropout=self.dropout,
            n_layers=self.n_layers,
        ).to(self.device)

        optimizer = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=20, factor=0.5)

        fc_target = data.fc_matrix  # (n_nodes, n_nodes)

        # --- Training loop ---
        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            fc_pred = self.model(data.x, data.edge_index, data.edge_attr)

            # Masked MSE loss (train nodes only)
            loss_train = F.mse_loss(fc_pred[train_mask], fc_target[train_mask])
            loss_train.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                fc_pred_val = self.model(data.x, data.edge_index, data.edge_attr)
                loss_val = F.mse_loss(
                    fc_pred_val[val_mask], fc_target[val_mask]
                ).item()

            train_losses.append(loss_train.item())
            val_losses.append(loss_val)
            scheduler.step(loss_val)

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

            if verbose and epoch % 50 == 0:
                print(
                    f"  Epoch {epoch:4d} | "
                    f"Train MSE: {loss_train.item():.6f} | "
                    f"Val MSE: {loss_val:.6f}"
                )

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        # --- Final prediction with attention ---
        self.model.eval()
        with torch.no_grad():
            fc_pred, attn_list = self.model(
                data.x, data.edge_index, data.edge_attr,
                return_attention=True,
            )

        fc_pred_np = fc_pred.cpu().numpy()
        fc_true_np = fc_target.cpu().numpy()

        # --- Compute decoupling metrics ---
        edge_error = np.abs(fc_true_np - fc_pred_np)
        nodal_error = edge_error.mean(axis=1)

        # Normalize decoupling score to [0, 1]
        decoupling = (nodal_error - nodal_error.min()) / (
            nodal_error.max() - nodal_error.min() + 1e-12
        )

        # Per-node R²
        r2_nodal = compute_r2_nodal(fc_true_np, fc_pred_np)

        # Global R²
        mask_triu = np.triu_indices(n_nodes, k=1)
        yt = fc_true_np[mask_triu]
        yp = fc_pred_np[mask_triu]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_global = 1.0 - ss_res / (ss_tot + 1e-12)

        # Attention weights (last layer)
        attn_np = None
        if attn_list:
            attn_np = attn_list[-1].cpu().numpy()

        return SCFCPredictionResult(
            fc_predicted=fc_pred_np,
            fc_true=fc_true_np,
            nodal_error=nodal_error,
            decoupling_score=decoupling,
            edge_error=edge_error,
            train_losses=train_losses,
            val_losses=val_losses,
            r2_global=r2_global,
            r2_nodal=r2_nodal,
            model_state=best_state,
            attention_weights=attn_np,
        )


# =============================================================================
# MODEL 2: GRAPH VARIATIONAL AUTOENCODER (VGAE)
# =============================================================================

class _GCNEncoder(nn.Module):
    """GCN-based encoder for VGAE (produces mu and logstd)."""

    def __init__(self, in_channels: int, hidden_channels: int, latent_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_channels)
        self.conv_logstd = GCNConv(hidden_channels, latent_channels)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


class _GATEncoder(nn.Module):
    """GAT-based encoder for VGAE — richer attention-based encoding."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_channels: int,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=n_heads, concat=True)
        self.norm1 = GraphNorm(hidden_channels * n_heads)
        self.conv_mu = GATv2Conv(
            hidden_channels * n_heads, latent_channels, heads=1, concat=False
        )
        self.conv_logstd = GATv2Conv(
            hidden_channels * n_heads, latent_channels, heads=1, concat=False
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


class BrainVGAE:
    """
    Graph Variational Autoencoder for unsupervised brain network analysis.

    Learns latent node embeddings that capture the topological structure
    of the connectome. The reconstruction error per edge identifies
    atypical connections; the latent space can be compared to gradient
    analysis (BrainSpace) for cross-validation.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of latent space.
    hidden_channels : int
        Hidden layer size.
    encoder_type : str
        'gcn' or 'gat'.
    lr, weight_decay, epochs, patience : training hyperparameters.
    modality : str
        Which matrix to use as adjacency: 'sc', 'fc', or 'both'.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_channels: int = 64,
        encoder_type: str = "gat",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 300,
        patience: int = 40,
        device: Optional[Union[str, torch.device]] = None,
    ):
        _check_pyg()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.encoder_type = encoder_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(device) if device else DEVICE
        self.model = None

    def fit_transform(
        self,
        adj_matrix: np.ndarray,
        node_features: Optional[np.ndarray] = None,
        edge_threshold: float = 0.0,
        verbose: bool = True,
    ) -> VAEResult:
        """
        Train VGAE and extract latent embeddings.

        Parameters
        ----------
        adj_matrix : np.ndarray, (n_nodes, n_nodes)
            Adjacency matrix (SC or FC).
        node_features : np.ndarray, optional
            Node feature matrix.
        edge_threshold : float
            Threshold for adjacency.
        verbose : bool
            Print progress.

        Returns
        -------
        VAEResult
        """
        data = connectivity_to_pyg(
            adj_matrix, node_features=node_features,
            edge_threshold=edge_threshold,
        )
        data = data.to(self.device)

        in_channels = data.x.shape[1]

        # Build encoder
        if self.encoder_type == "gat":
            encoder = _GATEncoder(
                in_channels, self.hidden_channels, self.latent_dim
            )
        else:
            encoder = _GCNEncoder(
                in_channels, self.hidden_channels, self.latent_dim
            )

        self.model = PyG_VGAE(encoder).to(self.device)
        optimizer = Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # --- Training loop ---
        train_losses = []
        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        n_nodes = data.num_nodes

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()

            z = self.model.encode(data.x, data.edge_index)

            # Reconstruction loss + KL divergence
            recon_loss = self.model.recon_loss(z, data.edge_index)
            kl_loss = (1.0 / n_nodes) * self.model.kl_loss()
            loss = recon_loss + kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if verbose:
                    print(f"  VGAE early stopping at epoch {epoch}")
                break

            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss.item():.4f}")

        # Load best
        if best_state:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        # --- Extract results ---
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(data.x, data.edge_index)
            mu = self.model.__mu__
            logstd = self.model.__logstd__

            # Reconstructed adjacency via inner product
            adj_recon = torch.sigmoid(torch.mm(z, z.t()))

            # Link prediction evaluation
            auc, ap = self.model.test(z, data.edge_index, data.edge_index)

        embeddings = z.cpu().numpy()
        mu_np = mu.cpu().numpy()
        logstd_np = logstd.cpu().numpy()
        adj_recon_np = adj_recon.cpu().numpy()

        # Reconstruction error
        adj_true = to_dense_adj(data.edge_index, max_num_nodes=n_nodes)[0].cpu().numpy()
        edge_recon_error = np.abs(adj_true - adj_recon_np)
        nodal_recon_error = edge_recon_error.mean(axis=1)

        return VAEResult(
            embeddings=embeddings,
            reconstructed_adj=adj_recon_np,
            edge_recon_error=edge_recon_error,
            nodal_recon_error=nodal_recon_error,
            mu=mu_np,
            logstd=logstd_np,
            train_losses=train_losses,
            auc=auc,
            ap=ap,
        )


# =============================================================================
# MODEL 3: MULTIMODAL HETEROGENEOUS GRAPH (SC + FC)
# =============================================================================

class _ModalityEncoder(nn.Module):
    """Single-modality GNN encoder branch."""

    def __init__(self, in_channels: int, hidden_channels: int, n_heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=n_heads, concat=True)
        self.norm1 = GraphNorm(hidden_channels * n_heads)
        self.conv2 = GATv2Conv(
            hidden_channels * n_heads, hidden_channels, heads=n_heads, concat=True
        )
        self.norm2 = GraphNorm(hidden_channels * n_heads)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.elu(self.norm1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.elu(self.norm2(self.conv2(h, edge_index)))
        return h


class _InteractionModule(nn.Module):
    """
    Learns inter-modality coupling weights per node.
    Given embeddings from SC and FC branches, produces a coupling
    weight in [0, 1] reflecting structure-function alignment.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_sc, z_fc):
        combined = torch.cat([z_sc, z_fc], dim=-1)
        coupling = self.mlp(combined)  # (n_nodes, 1)
        return coupling.squeeze(-1)


class MultimodalHeteroGNN:
    """
    Multimodal Heterogeneous Graph Neural Network for SC-FC integration.

    Constructs a two-layer graph:
        - Layer 1: ROIs connected by structural connectivity
        - Layer 2: ROIs connected by functional connectivity
        - Inter-layer: learned coupling weights per node

    The model is trained to reconstruct both modalities from a fused
    representation, with the inter-layer weights providing an interpretable
    measure of regional SC-FC coupling.

    Inspired by MS-Inter-GCN (Xia et al., Medical Image Analysis, 2025).
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        n_heads: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 400,
        patience: int = 40,
        reconstruction_weight: float = 1.0,
        coupling_reg: float = 0.01,
        device: Optional[Union[str, torch.device]] = None,
    ):
        _check_pyg()
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.reconstruction_weight = reconstruction_weight
        self.coupling_reg = coupling_reg
        self.device = torch.device(device) if device else DEVICE

    def fit_transform(
        self,
        sc_matrix: np.ndarray,
        fc_matrix: np.ndarray,
        node_features: Optional[np.ndarray] = None,
        sc_threshold: float = 0.0,
        fc_threshold: float = 0.1,
        verbose: bool = True,
    ) -> HeterogeneousResult:
        """
        Train multimodal heterogeneous model.

        Parameters
        ----------
        sc_matrix, fc_matrix : np.ndarray
            Connectivity matrices.
        node_features : np.ndarray, optional
            Shared node features.
        sc_threshold, fc_threshold : float
            Edge thresholds for each modality.
        verbose : bool
            Print progress.

        Returns
        -------
        HeterogeneousResult
        """
        n_nodes = sc_matrix.shape[0]

        # Build separate graphs for each modality
        data_sc = connectivity_to_pyg(
            sc_matrix, node_features=node_features,
            edge_threshold=sc_threshold,
        ).to(self.device)

        # For FC: use absolute values, threshold
        fc_abs = np.abs(fc_matrix.copy())
        np.fill_diagonal(fc_abs, 0)
        data_fc = connectivity_to_pyg(
            fc_abs, node_features=node_features,
            edge_threshold=fc_threshold, normalize_edges=False,
        ).to(self.device)

        in_channels = data_sc.x.shape[1]
        embed_dim = self.hidden_channels * self.n_heads

        # Build model components
        sc_encoder = _ModalityEncoder(
            in_channels, self.hidden_channels, self.n_heads
        ).to(self.device)
        fc_encoder = _ModalityEncoder(
            in_channels, self.hidden_channels, self.n_heads
        ).to(self.device)
        interaction = _InteractionModule(embed_dim).to(self.device)

        # Decoder: reconstruct adjacency from fused embeddings
        sc_decoder = nn.Linear(embed_dim, n_nodes).to(self.device)
        fc_decoder = nn.Linear(embed_dim, n_nodes).to(self.device)

        all_params = (
            list(sc_encoder.parameters())
            + list(fc_encoder.parameters())
            + list(interaction.parameters())
            + list(sc_decoder.parameters())
            + list(fc_decoder.parameters())
        )
        optimizer = Adam(all_params, lr=self.lr, weight_decay=self.weight_decay)

        # Targets
        sc_target = torch.tensor(
            np.log1p(np.abs(sc_matrix)).astype(np.float32), device=self.device
        )
        sc_target = sc_target / (sc_target.max() + 1e-8)
        fc_target = torch.tensor(fc_matrix.astype(np.float32), device=self.device)

        # --- Training loop ---
        train_losses = []
        best_loss = float("inf")
        best_states = None
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            sc_encoder.train(); fc_encoder.train(); interaction.train()

            optimizer.zero_grad()

            z_sc = sc_encoder(data_sc.x, data_sc.edge_index, data_sc.edge_attr)
            z_fc = fc_encoder(data_fc.x, data_fc.edge_index, data_fc.edge_attr)

            # Coupling weights
            coupling = interaction(z_sc, z_fc)  # (n_nodes,)

            # Fused embedding: weighted combination
            coupling_expanded = coupling.unsqueeze(-1)  # (n_nodes, 1)
            z_fused = coupling_expanded * z_sc + (1 - coupling_expanded) * z_fc

            # Reconstruct
            sc_recon = sc_decoder(z_fused)
            fc_recon = fc_decoder(z_fused)

            # Loss
            loss_sc = F.mse_loss(sc_recon, sc_target)
            loss_fc = F.mse_loss(fc_recon, fc_target)
            loss_recon = self.reconstruction_weight * (loss_sc + loss_fc)

            # Coupling regularization (encourage diversity, not all 0 or 1)
            coupling_entropy = -(
                coupling * torch.log(coupling + 1e-8)
                + (1 - coupling) * torch.log(1 - coupling + 1e-8)
            ).mean()
            loss_reg = -self.coupling_reg * coupling_entropy

            loss = loss_recon + loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_states = {
                    "sc_enc": {k: v.cpu().clone() for k, v in sc_encoder.state_dict().items()},
                    "fc_enc": {k: v.cpu().clone() for k, v in fc_encoder.state_dict().items()},
                    "interaction": {k: v.cpu().clone() for k, v in interaction.state_dict().items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if verbose:
                    print(f"  Hetero-GNN early stopping at epoch {epoch}")
                break

            if verbose and epoch % 50 == 0:
                print(
                    f"  Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                    f"Coupling mean: {coupling.mean().item():.3f}"
                )

        # Restore best
        if best_states:
            sc_encoder.load_state_dict(best_states["sc_enc"])
            fc_encoder.load_state_dict(best_states["fc_enc"])
            interaction.load_state_dict(best_states["interaction"])

        # --- Extract results ---
        sc_encoder.eval(); fc_encoder.eval(); interaction.eval()
        with torch.no_grad():
            z_sc = sc_encoder(data_sc.x, data_sc.edge_index, data_sc.edge_attr)
            z_fc = fc_encoder(data_fc.x, data_fc.edge_index, data_fc.edge_attr)
            coupling = interaction(z_sc, z_fc)

            coupling_expanded = coupling.unsqueeze(-1)
            z_fused = coupling_expanded * z_sc + (1 - coupling_expanded) * z_fc

        # Cosine similarity between SC and FC embeddings
        cos_sim = F.cosine_similarity(z_sc, z_fc, dim=-1).cpu().numpy()

        return HeterogeneousResult(
            coupling_weights=coupling.cpu().numpy(),
            sc_embeddings=z_sc.cpu().numpy(),
            fc_embeddings=z_fc.cpu().numpy(),
            fused_embeddings=z_fused.cpu().numpy(),
            embedding_similarity=cos_sim,
            train_losses=train_losses,
        )


# =============================================================================
# MODEL 4: NODE-LEVEL ANOMALY DETECTION
# =============================================================================

class NodeAnomalyDetector:
    """
    Identifies anomalous brain regions via GAE reconstruction error.

    Without a control group, this approach defines the "expected pattern"
    from the cohort itself. Regions with consistently high reconstruction
    error across subjects are candidates for COVID-related alterations.

    Strategy:
        1. Train a GAE per subject (or a shared GAE on all subjects).
        2. Compute per-node reconstruction error for each subject.
        3. Aggregate across subjects to find systematically anomalous nodes.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_channels: int = 64,
        epochs: int = 200,
        lr: float = 1e-3,
        z_threshold: float = 2.0,
        strategy: str = "per_subject",
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Parameters
        ----------
        strategy : str
            'per_subject' — train separate GAE per subject.
            'shared' — train single GAE on pooled graphs.
        z_threshold : float
            Z-score threshold for flagging anomalous nodes.
        """
        _check_pyg()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.epochs = epochs
        self.lr = lr
        self.z_threshold = z_threshold
        self.strategy = strategy
        self.device = torch.device(device) if device else DEVICE

    def detect(
        self,
        adjacency_matrices: List[np.ndarray],
        node_features_list: Optional[List[np.ndarray]] = None,
        subject_ids: Optional[List[str]] = None,
        modality: str = "sc",
        verbose: bool = True,
    ) -> AnomalyResult:
        """
        Run anomaly detection across subjects.

        Parameters
        ----------
        adjacency_matrices : list of np.ndarray
            One adjacency matrix per subject.
        node_features_list : list of np.ndarray, optional
            Node features per subject.
        subject_ids : list of str, optional
            Subject identifiers.
        modality : str
            Label for logging ('sc' or 'fc').
        verbose : bool
            Print progress.

        Returns
        -------
        AnomalyResult
        """
        n_subjects = len(adjacency_matrices)
        n_nodes = adjacency_matrices[0].shape[0]

        if subject_ids is None:
            subject_ids = [f"sub-{i:02d}" for i in range(1, n_subjects + 1)]

        per_subject_scores = np.zeros((n_subjects, n_nodes))

        for i, adj in enumerate(adjacency_matrices):
            if verbose:
                print(f"  [{modality.upper()}] Processing {subject_ids[i]}...")

            nf = node_features_list[i] if node_features_list else None

            data = connectivity_to_pyg(
                adj, node_features=nf, edge_threshold=0.0
            ).to(self.device)

            in_ch = data.x.shape[1]

            encoder = _GCNEncoder(in_ch, self.hidden_channels, self.latent_dim)
            gae = PyG_GAE(encoder).to(self.device)
            optimizer = Adam(gae.parameters(), lr=self.lr)

            # Train
            gae.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                z = gae.encode(data.x, data.edge_index)
                loss = gae.recon_loss(z, data.edge_index)
                loss.backward()
                optimizer.step()

            # Evaluate: per-node reconstruction error
            gae.eval()
            with torch.no_grad():
                z = gae.encode(data.x, data.edge_index)
                adj_recon = torch.sigmoid(torch.mm(z, z.t()))

            adj_true = to_dense_adj(
                data.edge_index, max_num_nodes=n_nodes
            )[0].cpu().numpy()
            adj_recon_np = adj_recon.cpu().numpy()

            node_error = np.abs(adj_true - adj_recon_np).mean(axis=1)
            per_subject_scores[i] = node_error

        # Aggregate across subjects
        mean_scores = per_subject_scores.mean(axis=0)
        std_scores = per_subject_scores.std(axis=0)

        # Z-score relative to node-level distribution
        global_mean = mean_scores.mean()
        global_std = mean_scores.std() + 1e-8
        z_scores = (mean_scores - global_mean) / global_std

        flagged = np.where(z_scores > self.z_threshold)[0]

        if verbose:
            print(f"  Anomaly detection complete: {len(flagged)} nodes flagged "
                  f"(z > {self.z_threshold})")

        return AnomalyResult(
            nodal_anomaly_score=mean_scores,
            nodal_anomaly_std=std_scores,
            per_subject_scores=per_subject_scores,
            z_scores=z_scores,
            flagged_nodes=flagged,
            threshold=self.z_threshold,
        )


# =============================================================================
# MODEL 5: GRAPH-LEVEL EMBEDDING + SUBJECT FINGERPRINTING
# =============================================================================

class _AttentionPooling(nn.Module):
    """
    Attention-based graph pooling for graph-level readout.
    Learns which nodes are most important for the global representation.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),
            nn.Linear(in_channels // 2, 1),
        )

    def forward(self, x, batch=None):
        # x: (n_nodes, in_channels) or (total_nodes, in_channels) if batched
        attn_scores = self.attention(x)  # (n_nodes, 1)
        if batch is not None:
            # Softmax within each graph in the batch
            from torch_geometric.utils import softmax as pyg_softmax
            attn_weights = pyg_softmax(attn_scores, batch)
        else:
            attn_weights = F.softmax(attn_scores, dim=0)

        # Weighted sum
        weighted = x * attn_weights
        if batch is not None:
            from torch_geometric.nn import global_add_pool
            return global_add_pool(weighted, batch), attn_weights
        return weighted.sum(dim=0, keepdim=True), attn_weights


class _GraphEncoder(nn.Module):
    """Full graph encoder with attention pooling for graph-level embeddings."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        embed_dim: int = 32,
        n_heads: int = 4,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=n_heads, concat=True)
        self.norm1 = GraphNorm(hidden_channels * n_heads)
        self.conv2 = GATv2Conv(
            hidden_channels * n_heads, hidden_channels, heads=n_heads, concat=True
        )
        self.norm2 = GraphNorm(hidden_channels * n_heads)

        self.pool = _AttentionPooling(hidden_channels * n_heads)
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels * n_heads, embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = F.elu(self.norm1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.elu(self.norm2(self.conv2(h, edge_index)))

        graph_emb, attn_weights = self.pool(h, batch)
        graph_emb = self.projector(graph_emb)

        return graph_emb, h, attn_weights


class GraphLevelEmbedder:
    """
    Produces whole-brain graph-level representations for each subject.

    Uses contrastive/reconstruction objectives to learn embeddings that
    capture global network properties. The resulting subject vectors enable:
        - Subject fingerprinting / identification
        - Phenotype subgroup clustering
        - Correlation with clinical variables
        - Test-retest reliability assessment

    Training strategy for N=23:
        Uses a graph-level autoencoder objective: encode graph → embed →
        decode node features. No labels required.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        hidden_channels: int = 64,
        n_heads: int = 4,
        lr: float = 1e-3,
        epochs: int = 300,
        patience: int = 40,
        device: Optional[Union[str, torch.device]] = None,
    ):
        _check_pyg()
        self.embed_dim = embed_dim
        self.hidden_channels = hidden_channels
        self.n_heads = n_heads
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device(device) if device else DEVICE

    def fit_transform(
        self,
        adjacency_matrices: List[np.ndarray],
        node_features_list: Optional[List[np.ndarray]] = None,
        subject_ids: Optional[List[str]] = None,
        clinical_data: Optional[pd.DataFrame] = None,
        n_clusters: int = 3,
        verbose: bool = True,
    ) -> GraphEmbeddingResult:
        """
        Compute graph-level embeddings for all subjects.

        Parameters
        ----------
        adjacency_matrices : list of np.ndarray
            Adjacency matrices (one per subject).
        node_features_list : list of np.ndarray, optional
            Node features per subject.
        subject_ids : list of str
            Subject identifiers.
        clinical_data : pd.DataFrame, optional
            Clinical variables for correlation analysis.
            Index should be subject IDs, columns are variables.
        n_clusters : int
            Number of clusters for phenotyping.
        verbose : bool
            Print progress.

        Returns
        -------
        GraphEmbeddingResult
        """
        n_subjects = len(adjacency_matrices)
        n_nodes = adjacency_matrices[0].shape[0]

        if subject_ids is None:
            subject_ids = [f"sub-{i:02d}" for i in range(1, n_subjects + 1)]

        # Build PyG dataset
        data_list = []
        for i in range(n_subjects):
            nf = node_features_list[i] if node_features_list else None
            d = connectivity_to_pyg(
                adjacency_matrices[i], node_features=nf
            )
            d.graph_id = i
            data_list.append(d)

        in_channels = data_list[0].x.shape[1]

        # Model
        encoder = _GraphEncoder(
            in_channels, self.hidden_channels, self.embed_dim, self.n_heads
        ).to(self.device)

        # Decoder: reconstruct mean node features from graph embedding
        decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_channels),
            nn.ELU(),
            nn.Linear(self.hidden_channels, in_channels),
        ).to(self.device)

        all_params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = Adam(all_params, lr=self.lr)

        # --- Training: graph autoencoder objective ---
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        train_losses = []

        for epoch in range(1, self.epochs + 1):
            encoder.train(); decoder.train()
            epoch_loss = 0.0

            for data in data_list:
                data = data.to(self.device)
                optimizer.zero_grad()

                graph_emb, node_emb, attn_w = encoder(
                    data.x, data.edge_index, data.edge_attr
                )

                # Reconstruct mean node features
                target = data.x.mean(dim=0, keepdim=True)
                recon = decoder(graph_emb)
                loss = F.mse_loss(recon, target)

                # Additional loss: node feature reconstruction from broadcast
                node_recon = decoder(graph_emb.expand(data.x.shape[0], -1))
                loss += 0.5 * F.mse_loss(node_recon, data.x)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= n_subjects
            train_losses.append(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {
                    "enc": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                    "dec": {k: v.cpu().clone() for k, v in decoder.state_dict().items()},
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if verbose:
                    print(f"  Graph embedder early stopping at epoch {epoch}")
                break

            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch:4d} | Loss: {epoch_loss:.6f}")

        # Restore best
        if best_state:
            encoder.load_state_dict(best_state["enc"])
            decoder.load_state_dict(best_state["dec"])
        encoder = encoder.to(self.device)

        # --- Extract embeddings ---
        encoder.eval()
        embeddings = np.zeros((n_subjects, self.embed_dim))
        all_attn = np.zeros(n_nodes)

        with torch.no_grad():
            for i, data in enumerate(data_list):
                data = data.to(self.device)
                graph_emb, node_emb, attn_w = encoder(
                    data.x, data.edge_index, data.edge_attr
                )
                embeddings[i] = graph_emb.squeeze().cpu().numpy()
                all_attn += attn_w.squeeze().cpu().numpy()

        node_importance = all_attn / n_subjects

        # Pairwise similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)

        # Clustering
        cluster_labels = None
        try:
            from sklearn.cluster import KMeans
            if n_subjects >= n_clusters:
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = km.fit_predict(embeddings)
        except ImportError:
            pass

        # Clinical correlations
        clinical_correlations = None
        if clinical_data is not None:
            clinical_correlations = {}
            for col in clinical_data.columns:
                vals = clinical_data[col].values
                if len(vals) == n_subjects and np.issubdtype(vals.dtype, np.number):
                    valid = ~np.isnan(vals)
                    if valid.sum() >= 5:
                        for dim in range(self.embed_dim):
                            r, p = stats.pearsonr(
                                embeddings[valid, dim], vals[valid]
                            )
                            if p < 0.05:
                                if col not in clinical_correlations:
                                    clinical_correlations[col] = []
                                clinical_correlations[col].append(
                                    {"dim": dim, "r": r, "p": p}
                                )

        return GraphEmbeddingResult(
            subject_embeddings=embeddings,
            subject_ids=subject_ids,
            similarity_matrix=sim_matrix,
            node_importance=node_importance,
            cluster_labels=cluster_labels,
            clinical_correlations=clinical_correlations,
        )


# =============================================================================
# ORCHESTRATOR: RUN ALL MODELS
# =============================================================================

class BrainGNNPipeline:
    """
    Orchestrator for running all GNN models on a single atlas/subject set.

    This class provides a unified interface to run all five models and
    aggregate their results into a comprehensive analysis.

    Example
    -------
    >>> pipeline = BrainGNNPipeline(atlas='schaefer_100')
    >>> results = pipeline.run_all(
    ...     sc_matrices=sc_list,
    ...     fc_matrices=fc_list,
    ...     node_features_list=features_list,
    ...     subject_ids=subjects,
    ... )
    >>> print(results.keys())
    # dict_keys(['scfc', 'vgae_sc', 'vgae_fc', 'hetero', 'anomaly', 'embedding'])
    """

    def __init__(
        self,
        atlas: str = "schaefer_100",
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True,
    ):
        _check_pyg()
        self.atlas = atlas
        self.device = torch.device(device) if device else DEVICE
        self.verbose = verbose

    def run_all(
        self,
        sc_matrices: List[np.ndarray],
        fc_matrices: List[np.ndarray],
        node_features_list: Optional[List[np.ndarray]] = None,
        subject_ids: Optional[List[str]] = None,
        clinical_data: Optional[pd.DataFrame] = None,
        run_scfc: bool = True,
        run_vgae: bool = True,
        run_hetero: bool = True,
        run_anomaly: bool = True,
        run_embedding: bool = True,
        scfc_kwargs: Optional[Dict] = None,
        vgae_kwargs: Optional[Dict] = None,
        hetero_kwargs: Optional[Dict] = None,
        anomaly_kwargs: Optional[Dict] = None,
        embed_kwargs: Optional[Dict] = None,
    ) -> Dict:
        """
        Run selected GNN models on the dataset.

        Parameters
        ----------
        sc_matrices : list of np.ndarray
            Structural connectivity matrices.
        fc_matrices : list of np.ndarray
            Functional connectivity matrices.
        node_features_list : list of np.ndarray, optional
            Node features per subject.
        subject_ids : list of str
            Subject identifiers.
        clinical_data : pd.DataFrame, optional
            Clinical variables for embedding correlation.
        run_* : bool
            Flags to enable/disable each model.
        *_kwargs : dict, optional
            Extra keyword arguments for each model.

        Returns
        -------
        dict
            Keys: 'scfc', 'vgae_sc', 'vgae_fc', 'hetero', 'anomaly', 'embedding'
        """
        results = {}
        n_subjects = len(sc_matrices)

        if subject_ids is None:
            subject_ids = [f"sub-{i:02d}" for i in range(1, n_subjects + 1)]

        # --- 1. SC → FC Prediction (per subject) ---
        if run_scfc:
            if self.verbose:
                print("=" * 60)
                print("MODEL 1: SC → FC Prediction (GATv2)")
                print("=" * 60)

            kwargs = scfc_kwargs or {}
            predictor = SCFCPredictor(device=self.device, **kwargs)
            scfc_results = {}

            for i in range(n_subjects):
                if self.verbose:
                    print(f"\n  [{subject_ids[i]}]")
                nf = node_features_list[i] if node_features_list else None
                scfc_results[subject_ids[i]] = predictor.fit_predict(
                    sc_matrices[i], fc_matrices[i], nf,
                    verbose=self.verbose,
                )

            results["scfc"] = scfc_results

        # --- 2. Graph VAE (one per modality, per subject) ---
        if run_vgae:
            if self.verbose:
                print("\n" + "=" * 60)
                print("MODEL 2: Graph Variational Autoencoder (VGAE)")
                print("=" * 60)

            kwargs = vgae_kwargs or {}

            vgae_sc_results = {}
            vgae_fc_results = {}

            for i in range(n_subjects):
                if self.verbose:
                    print(f"\n  [{subject_ids[i]}] SC-VGAE")
                nf = node_features_list[i] if node_features_list else None
                vgae = BrainVGAE(device=self.device, **kwargs)
                vgae_sc_results[subject_ids[i]] = vgae.fit_transform(
                    sc_matrices[i], nf, verbose=self.verbose,
                )

                if self.verbose:
                    print(f"  [{subject_ids[i]}] FC-VGAE")
                fc_abs = np.abs(fc_matrices[i])
                np.fill_diagonal(fc_abs, 0)
                vgae_fc_results[subject_ids[i]] = vgae.fit_transform(
                    fc_abs, nf, verbose=self.verbose,
                )

            results["vgae_sc"] = vgae_sc_results
            results["vgae_fc"] = vgae_fc_results

        # --- 3. Multimodal Heterogeneous Graph (per subject) ---
        if run_hetero:
            if self.verbose:
                print("\n" + "=" * 60)
                print("MODEL 3: Multimodal Heterogeneous Graph (SC + FC)")
                print("=" * 60)

            kwargs = hetero_kwargs or {}
            hetero_results = {}

            for i in range(n_subjects):
                if self.verbose:
                    print(f"\n  [{subject_ids[i]}]")
                nf = node_features_list[i] if node_features_list else None
                hetero = MultimodalHeteroGNN(device=self.device, **kwargs)
                hetero_results[subject_ids[i]] = hetero.fit_transform(
                    sc_matrices[i], fc_matrices[i], nf,
                    verbose=self.verbose,
                )

            results["hetero"] = hetero_results

        # --- 4. Anomaly Detection ---
        if run_anomaly:
            if self.verbose:
                print("\n" + "=" * 60)
                print("MODEL 4: Node-level Anomaly Detection")
                print("=" * 60)

            kwargs = anomaly_kwargs or {}
            detector = NodeAnomalyDetector(device=self.device, **kwargs)
            results["anomaly"] = detector.detect(
                sc_matrices, node_features_list, subject_ids,
                modality="sc", verbose=self.verbose,
            )

        # --- 5. Graph-level Embedding ---
        if run_embedding:
            if self.verbose:
                print("\n" + "=" * 60)
                print("MODEL 5: Graph-level Embedding + Fingerprinting")
                print("=" * 60)

            kwargs = embed_kwargs or {}
            embedder = GraphLevelEmbedder(device=self.device, **kwargs)
            results["embedding"] = embedder.fit_transform(
                sc_matrices, node_features_list, subject_ids,
                clinical_data=clinical_data,
                verbose=self.verbose,
            )

        if self.verbose:
            print("\n" + "=" * 60)
            print("ALL MODELS COMPLETE")
            print("=" * 60)

        return results

    @staticmethod
    def summarize(results: Dict, labels: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a summary DataFrame combining key metrics from all models.

        Parameters
        ----------
        results : dict
            Output from run_all().
        labels : list of str, optional
            ROI labels for the atlas.

        Returns
        -------
        pd.DataFrame
            Rows = ROIs, columns = metrics from each model.
        """
        dfs = []

        # SC-FC decoupling (mean across subjects)
        if "scfc" in results:
            decoupling_all = np.stack([
                r.decoupling_score for r in results["scfc"].values()
            ])
            r2_all = np.stack([r.r2_nodal for r in results["scfc"].values()])
            df = pd.DataFrame({
                "scfc_decoupling_mean": decoupling_all.mean(axis=0),
                "scfc_decoupling_std": decoupling_all.std(axis=0),
                "scfc_r2_mean": r2_all.mean(axis=0),
            })
            dfs.append(df)

        # VGAE reconstruction error
        if "vgae_sc" in results:
            recon_all = np.stack([
                r.nodal_recon_error for r in results["vgae_sc"].values()
            ])
            df = pd.DataFrame({
                "vgae_sc_recon_error": recon_all.mean(axis=0),
            })
            dfs.append(df)

        if "vgae_fc" in results:
            recon_all = np.stack([
                r.nodal_recon_error for r in results["vgae_fc"].values()
            ])
            df = pd.DataFrame({
                "vgae_fc_recon_error": recon_all.mean(axis=0),
            })
            dfs.append(df)

        # Hetero coupling
        if "hetero" in results:
            coupling_all = np.stack([
                r.coupling_weights for r in results["hetero"].values()
            ])
            sim_all = np.stack([
                r.embedding_similarity for r in results["hetero"].values()
            ])
            df = pd.DataFrame({
                "hetero_coupling_mean": coupling_all.mean(axis=0),
                "hetero_coupling_std": coupling_all.std(axis=0),
                "hetero_embed_sim": sim_all.mean(axis=0),
            })
            dfs.append(df)

        # Anomaly
        if "anomaly" in results:
            anom = results["anomaly"]
            df = pd.DataFrame({
                "anomaly_score": anom.nodal_anomaly_score,
                "anomaly_zscore": anom.z_scores,
                "anomaly_flagged": np.isin(
                    np.arange(len(anom.z_scores)), anom.flagged_nodes
                ).astype(int),
            })
            dfs.append(df)

        # Graph embedding node importance
        if "embedding" in results:
            emb = results["embedding"]
            df = pd.DataFrame({
                "attention_importance": emb.node_importance,
            })
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        summary = pd.concat(dfs, axis=1)

        if labels is not None and len(labels) == len(summary):
            summary.index = labels

        return summary


# =============================================================================
# MODULE-LEVEL EXPORTS
# =============================================================================

__all__ = [
    # Utilities
    "connectivity_to_pyg",
    "build_multimodal_features",
    "compute_r2_nodal",
    # Models
    "SCFCPredictor",
    "BrainVGAE",
    "MultimodalHeteroGNN",
    "NodeAnomalyDetector",
    "GraphLevelEmbedder",
    # Pipeline
    "BrainGNNPipeline",
    # Result dataclasses
    "SCFCPredictionResult",
    "VAEResult",
    "HeterogeneousResult",
    "AnomalyResult",
    "GraphEmbeddingResult",
    # Constants
    "FMRI_NODE_FEATURES",
    "DMRI_NODE_FEATURES",
    "SC_EDGE_FEATURES",
    "DEVICE",
]
