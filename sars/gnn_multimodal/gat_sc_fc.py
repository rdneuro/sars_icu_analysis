#!/usr/bin/env python3
"""
==============================================================================
GNN MULTIMODAL ANALYSIS - GAT SC→FC PREDICTION
==============================================================================

Graph Attention Network for predicting functional connectivity from
structural connectivity at the individual level.

Scientific Rationale:
--------------------
Functional connectivity emerges from the underlying structural connectome
through multi-hop information propagation (Honey et al., 2009). The
structure-function relationship follows a hierarchical gradient: tightly
coupled in unimodal sensory cortices and progressively decoupled in
transmodal association areas (Vázquez-Rodríguez et al., 2019; Baum et al.,
2020). The GAT captures this by learning attention-weighted message passing
over structural edges, where multi-layer propagation naturally models
indirect (multi-hop) structural pathways.

Key Innovation:
--------------
The learned attention coefficients α_{ij} provide a novel, data-driven
measure of which structural connections are most critical for supporting
functional communication — a form of "effective structural connectivity"
distinct from raw streamline counts or FA values.

The regional prediction error ε_i = ||F̂_i - F_i||² serves as an
individual-level SC-FC decoupling index that can be compared against
the normative hierarchical gradient from healthy populations.

Architecture:
------------
Input:   SC matrix → Graph (edge_index, edge_attr)
         SC row profiles → Node features (x)
GAT:     Multi-head attention × N layers → learned representations
Output:  Predicted FC profile per node → reconstruction of F̂

Loss:    L = MSE(F̂, F) + λ·||attention||₁ (optional sparsity)

References:
-----------
- Wu & Li (2023). Human Brain Mapping, 44(9), 3885-3896.
- Veličković et al. (2018). Graph Attention Networks. ICLR 2018.
- Baum et al. (2020). NeuroImage, 210, 116612.

Author: SARS-CoV-2 Neuroimaging Study
Date: February 2026
==============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

from .config import GATConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class GATLayer(nn.Module):
    """
    Single GAT layer with optional residual connection and layer normalization.

    Uses GATv2 (Brody et al., 2022) which computes dynamic attention:
        e_{ij} = a^T · LeakyReLU(W·[h_i || h_j])

    rather than static attention in the original GAT, providing more
    expressive attention patterns suited for heterogeneous brain networks.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        dropout: float = 0.3,
        residual: bool = True,
        layer_norm: bool = True,
        edge_dim: int = 1,
    ):
        super().__init__()

        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // n_heads,
            heads=n_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
            concat=True,
        )

        self.residual = residual
        if residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = None

        self.layer_norm = nn.LayerNorm(out_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features (N, in_dim).
        edge_index : torch.Tensor
            Edge indices (2, E).
        edge_attr : torch.Tensor, optional
            Edge weights (E, 1).
        return_attention : bool
            If True, return attention coefficients.

        Returns
        -------
        out : torch.Tensor
            Updated node features (N, out_dim).
        attention : tuple or None
            (edge_index, attention_weights) if requested.
        """
        identity = x

        if return_attention:
            out, (attn_edge_index, attn_weights) = self.conv(
                x, edge_index, edge_attr=edge_attr,
                return_attention_weights=True,
            )
            attention = (attn_edge_index, attn_weights)
        else:
            out = self.conv(x, edge_index, edge_attr=edge_attr)
            attention = None

        out = F.elu(out)
        out = self.dropout(out)

        # Residual connection
        if self.residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            out = out + identity

        # Layer normalization
        if self.layer_norm is not None:
            out = self.layer_norm(out)

        return out, attention


class GATSCFC(nn.Module):
    """
    Graph Attention Network for SC→FC prediction.

    Architecture:
    1. Input projection: SC profile (N features) → hidden_dim
    2. GAT layers × n_layers: message passing with attention
    3. Output projection: hidden_dim → N (predicted FC row)

    The model predicts each region's functional connectivity profile
    based on information propagated through the structural connectome.

    Multi-layer propagation captures multi-hop structural effects:
    - Layer 1: direct structural connections
    - Layer 2: 2-hop indirect connections
    - Layer k: k-hop connections

    This implements the principle from Wu & Li (2023) that indirect
    structural pathways are essential for capturing the individual
    specificity of the SC-FC relationship.
    """

    def __init__(
        self,
        n_rois: int,
        config: GATConfig,
    ):
        """
        Parameters
        ----------
        n_rois : int
            Number of brain regions (nodes).
        config : GATConfig
            Model configuration.
        """
        super().__init__()

        self.n_rois = n_rois
        self.config = config
        hidden = config.hidden_dim
        n_heads = config.n_heads

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_rois, hidden),
            nn.ELU(),
            nn.Dropout(config.dropout),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(config.n_layers):
            in_dim = hidden
            out_dim = hidden
            self.gat_layers.append(
                GATLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    n_heads=n_heads,
                    dropout=config.dropout,
                    residual=config.residual,
                    layer_norm=config.layer_norm,
                )
            )

        # Output projection: predict FC profile for each node
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, n_rois),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        data: Data,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: predict FC from SC.

        Parameters
        ----------
        data : Data
            PyG Data with x, edge_index, edge_attr.
        return_attention : bool
            If True, collect attention weights from all layers.

        Returns
        -------
        dict with:
            'fc_pred': Predicted FC matrix (N, N)
            'embeddings': Node embeddings from last GAT layer (N, hidden)
            'attention_weights': List of attention per layer (if requested)
        """
        x = data.x  # (N, N_rois) — SC row profiles
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Project input features
        h = self.input_proj(x)  # (N, hidden)

        # GAT message passing
        attention_all = []
        for layer in self.gat_layers:
            h, attn = layer(
                h, edge_index, edge_attr,
                return_attention=return_attention,
            )
            if attn is not None:
                attention_all.append(attn)

        # Store embeddings
        embeddings = h  # (N, hidden)

        # Predict FC profile
        fc_pred = self.output_proj(h)  # (N, N_rois)

        # Symmetrize prediction
        fc_pred = (fc_pred + fc_pred.T) / 2.0

        result = {
            "fc_pred": fc_pred,
            "embeddings": embeddings,
        }
        if return_attention:
            result["attention_weights"] = attention_all

        return result


# ==============================================================================
# LOSS FUNCTIONS
# ==============================================================================

class SCFCPredictionLoss(nn.Module):
    """
    Loss function for SC→FC prediction.

    Combines:
    1. MSE loss on predicted vs actual FC
    2. Optional correlation loss (Pearson correlation between rows)
    3. Optional attention sparsity regularization

    L = MSE(F̂, F) + λ_corr · (1 - corr(F̂, F)) + λ_sparse · ||α||₁
    """

    def __init__(
        self,
        lambda_corr: float = 0.5,
        lambda_sparse: float = 0.01,
    ):
        super().__init__()
        self.lambda_corr = lambda_corr
        self.lambda_sparse = lambda_sparse

    def forward(
        self,
        fc_pred: torch.Tensor,
        fc_target: torch.Tensor,
        attention_weights: Optional[List] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Parameters
        ----------
        fc_pred : torch.Tensor
            Predicted FC (N, N).
        fc_target : torch.Tensor
            Actual FC (N, N).
        attention_weights : list, optional
            Attention coefficients from GAT layers.

        Returns
        -------
        dict with 'total', 'mse', 'corr', 'sparse' losses.
        """
        # MSE loss
        mse = F.mse_loss(fc_pred, fc_target)

        # Correlation loss (row-wise Pearson correlation)
        # Measures how well the predicted FC profile matches the actual one
        pred_centered = fc_pred - fc_pred.mean(dim=1, keepdim=True)
        target_centered = fc_target - fc_target.mean(dim=1, keepdim=True)

        pred_norm = pred_centered / (pred_centered.norm(dim=1, keepdim=True) + 1e-8)
        target_norm = target_centered / (target_centered.norm(dim=1, keepdim=True) + 1e-8)

        corr_per_row = (pred_norm * target_norm).sum(dim=1)
        corr_loss = 1.0 - corr_per_row.mean()

        # Attention sparsity
        sparse_loss = torch.tensor(0.0, device=fc_pred.device)
        if attention_weights is not None and self.lambda_sparse > 0:
            for _, attn_w in attention_weights:
                sparse_loss = sparse_loss + attn_w.abs().mean()
            sparse_loss = sparse_loss / len(attention_weights)

        # Total
        total = (
            mse
            + self.lambda_corr * corr_loss
            + self.lambda_sparse * sparse_loss
        )

        return {
            "total": total,
            "mse": mse,
            "corr_loss": corr_loss,
            "sparse_loss": sparse_loss,
            "mean_corr": corr_per_row.mean(),
        }


# ==============================================================================
# SC-FC DECOUPLING ANALYSIS
# ==============================================================================

def compute_regional_decoupling(
    model: GATSCFC,
    data: Data,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Compute regional SC-FC decoupling indices for a single subject.

    The decoupling index ε_i quantifies how much region i's functional
    connectivity deviates from what is predicted by the structural
    connectome. High ε_i indicates that the region's functional
    connectivity is "liberated" from structural constraints.

    In the normative hierarchy (Vázquez-Rodríguez et al., 2019):
    - Primary sensory/motor cortices: low ε (tight coupling)
    - Association/transmodal cortices: high ε (loose coupling)

    Parameters
    ----------
    model : GATSCFC
        Trained model.
    data : Data
        Subject's graph data.
    device : torch.device
        Computation device.

    Returns
    -------
    dict with:
        'prediction_error': per-region MSE (N,)
        'prediction_corr': per-region Pearson r (N,)
        'fc_pred': predicted FC matrix (N, N)
        'fc_actual': actual FC matrix (N, N)
        'attention_maps': attention weights per layer
        'embeddings': learned node representations (N, hidden)
    """
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        output = model(data, return_attention=True)

    fc_pred = output["fc_pred"].cpu().numpy()
    fc_actual = data.y.cpu().numpy()
    embeddings = output["embeddings"].cpu().numpy()

    n = fc_pred.shape[0]

    # Per-region prediction error (MSE per row)
    prediction_error = np.mean((fc_pred - fc_actual) ** 2, axis=1)

    # Per-region Pearson correlation
    prediction_corr = np.zeros(n)
    for i in range(n):
        # Exclude self-connection (diagonal)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred_row = fc_pred[i, mask]
        actual_row = fc_actual[i, mask]

        if np.std(pred_row) > 1e-8 and np.std(actual_row) > 1e-8:
            prediction_corr[i] = np.corrcoef(pred_row, actual_row)[0, 1]
        else:
            prediction_corr[i] = 0.0

    # Extract attention maps
    attention_maps = []
    if "attention_weights" in output:
        for edge_idx, attn_w in output["attention_weights"]:
            attention_maps.append({
                "edge_index": edge_idx.cpu().numpy(),
                "weights": attn_w.cpu().numpy(),
            })

    return {
        "prediction_error": prediction_error,
        "prediction_corr": prediction_corr,
        "fc_pred": fc_pred,
        "fc_actual": fc_actual,
        "attention_maps": attention_maps,
        "embeddings": embeddings,
    }


def extract_attention_matrix(
    attention_maps: List[Dict],
    n_rois: int,
    layer: int = -1,
    aggregate: str = "mean",
) -> np.ndarray:
    """
    Convert sparse attention weights to a dense attention matrix.

    This matrix represents the "effective structural connectivity"
    learned by the GAT — a data-driven measure of which structural
    connections are most critical for generating functional activity.

    Parameters
    ----------
    attention_maps : list
        Attention maps from compute_regional_decoupling.
    n_rois : int
        Number of ROIs.
    layer : int
        Which layer's attention to extract (-1 for last).
    aggregate : str
        How to aggregate multi-head attention: 'mean' or 'max'.

    Returns
    -------
    np.ndarray
        Dense attention matrix (N, N).
    """
    if not attention_maps:
        return np.zeros((n_rois, n_rois))

    attn = attention_maps[layer]
    edge_index = attn["edge_index"]
    weights = attn["weights"]

    # If multi-head, aggregate
    if weights.ndim == 2:
        if aggregate == "mean":
            weights = weights.mean(axis=1)
        elif aggregate == "max":
            weights = weights.max(axis=1)

    # Build dense matrix
    mat = np.zeros((n_rois, n_rois))
    rows, cols = edge_index[0], edge_index[1]

    # Filter out self-loops from edge_index
    valid = rows != cols
    if valid.any():
        mat[rows[valid], cols[valid]] = weights[valid]

    return mat


def compute_cohort_decoupling(
    model: GATSCFC,
    dataset,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Compute SC-FC decoupling across the entire cohort.

    Returns group-level maps that can be compared with the normative
    hierarchy from the literature.

    Parameters
    ----------
    model : GATSCFC
        Trained model.
    dataset : BrainConnectomeDataset
        Full cohort dataset.
    device : torch.device
        Computation device.

    Returns
    -------
    dict with:
        'mean_error': group-mean prediction error (N,)
        'std_error': group-std prediction error (N,)
        'mean_corr': group-mean prediction correlation (N,)
        'std_corr': group-std (N,)
        'all_errors': per-subject errors (S, N)
        'all_corrs': per-subject correlations (S, N)
        'mean_attention': group-mean attention matrix (N, N)
        'subjects': list of subject IDs
    """
    all_errors = []
    all_corrs = []
    all_attention = []
    subjects = []

    for i in range(len(dataset)):
        data = dataset[i]
        result = compute_regional_decoupling(model, data, device)

        all_errors.append(result["prediction_error"])
        all_corrs.append(result["prediction_corr"])
        subjects.append(data.subject)

        if result["attention_maps"]:
            attn_mat = extract_attention_matrix(
                result["attention_maps"],
                n_rois=data.n_rois,
            )
            all_attention.append(attn_mat)

    all_errors = np.array(all_errors)  # (S, N)
    all_corrs = np.array(all_corrs)    # (S, N)

    result = {
        "mean_error": np.mean(all_errors, axis=0),
        "std_error": np.std(all_errors, axis=0),
        "mean_corr": np.mean(all_corrs, axis=0),
        "std_corr": np.std(all_corrs, axis=0),
        "all_errors": all_errors,
        "all_corrs": all_corrs,
        "subjects": subjects,
    }

    if all_attention:
        result["mean_attention"] = np.mean(all_attention, axis=0)
        result["std_attention"] = np.std(all_attention, axis=0)

    return result
