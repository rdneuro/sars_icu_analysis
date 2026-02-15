#!/usr/bin/env python3
"""
==============================================================================
GNN MULTIMODAL ANALYSIS - CONTRASTIVE MULTIMODAL LEARNING
==============================================================================

Self-supervised contrastive learning framework treating structural (SC)
and functional (FC) connectivity as complementary "views" of the same
brain, learning aligned representations without external labels.

Scientific Rationale:
--------------------
SC and FC provide complementary information about brain organization.
Their alignment (or misalignment) at the regional level reveals the
degree of structure-function coherence — a fundamental property of
brain network organization that is disrupted in neurological conditions.

By training a contrastive model to align SC and FC representations
of the same subject while separating representations from different
subjects, we obtain:

1. A learned measure of multimodal coherence per brain region
2. Subject-level "fingerprints" in a shared embedding space
3. Data-driven discovery of patient subgroups without clinical labels

Loss Function (NT-Xent — Normalized Temperature-scaled Cross-Entropy):
----------------------------------------------------------------------
For a batch of N subjects, each providing z_SC^i and z_FC^i:

    L_i = -log( exp(sim(z_SC^i, z_FC^i)/τ) / Σ_{j≠i} exp(sim(z_SC^i, z_FC^j)/τ) )

where sim(a,b) = a·b / (||a||·||b||) is cosine similarity and τ is temperature.

The symmetric version averages L(SC→FC) and L(FC→SC).

References:
-----------
- Chen et al. (2020). SimCLR: A Simple Framework for Contrastive
  Learning of Visual Representations. ICML 2020.
- Choi et al. (2023). A Generative Self-Supervised Framework using
  Functional Connectivity in fMRI Data. arXiv:2312.01994.
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
from torch_geometric.nn import GATv2Conv, GINConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

from .config import ContrastiveConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# GRAPH ENCODERS
# ==============================================================================

class GATEncoder(nn.Module):
    """
    GAT-based graph encoder producing node-level and graph-level embeddings.

    Uses GATv2 for dynamic attention, with a readout layer aggregating
    node embeddings into a global graph representation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // n_heads,
                    heads=n_heads,
                    dropout=dropout,
                    edge_dim=1,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Node-level embedding
        self.node_proj = nn.Linear(hidden_dim, embedding_dim)

        # Graph-level readout
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode graph into node and graph embeddings.

        Returns
        -------
        dict with:
            'node_emb': Node embeddings (N, embedding_dim)
            'graph_emb': Graph embedding (B, embedding_dim)
            'hidden': Last hidden states (N, hidden_dim)
        """
        h = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)
            h = norm(h + h_new)  # residual

        # Node embeddings
        node_emb = self.node_proj(h)

        # Graph embedding (global mean pooling)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        graph_emb = self.graph_proj(global_mean_pool(h, batch))

        return {
            "node_emb": node_emb,
            "graph_emb": graph_emb,
            "hidden": h,
        }


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) encoder.

    GIN is provably as powerful as the Weisfeiler-Lehman graph
    isomorphism test, making it ideal for distinguishing different
    graph topologies (i.e., different brain connectomes).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        n_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.node_proj = nn.Linear(hidden_dim, embedding_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)
            h = norm(h + h_new)

        node_emb = self.node_proj(h)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        graph_emb = self.graph_proj(global_mean_pool(h, batch))

        return {"node_emb": node_emb, "graph_emb": graph_emb, "hidden": h}


def build_encoder(
    encoder_type: str,
    in_dim: int,
    hidden_dim: int,
    embedding_dim: int,
    n_layers: int,
    n_heads: int = 4,
    dropout: float = 0.2,
) -> nn.Module:
    """Factory function for graph encoders."""
    if encoder_type == "GAT":
        return GATEncoder(in_dim, hidden_dim, embedding_dim, n_layers, n_heads, dropout)
    elif encoder_type == "GIN":
        return GINEncoder(in_dim, hidden_dim, embedding_dim, n_layers, dropout)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ==============================================================================
# PROJECTION HEAD
# ==============================================================================

class ProjectionHead(nn.Module):
    """
    Non-linear projection head mapping embeddings to the contrastive space.

    Following SimCLR (Chen et al., 2020), a 2-layer MLP projection
    improves contrastive learning quality by separating the
    representation space from the contrastive objective space.
    """

    def __init__(self, embedding_dim: int, projection_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ELU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ==============================================================================
# CONTRASTIVE MODEL
# ==============================================================================

class MultimodalContrastiveModel(nn.Module):
    """
    Contrastive learning model for SC-FC multimodal alignment.

    Two separate encoders process SC and FC graphs independently,
    producing aligned embeddings in a shared latent space via
    contrastive optimization.

    The model operates at two levels:
    1. Graph-level: aligns global brain representations SC ↔ FC
    2. Node-level: aligns per-region representations SC ↔ FC

    After training, the alignment quality per region provides a
    measure of "multimodal coherence" — regions where SC and FC
    representations are well-aligned have coherent structure-function
    relationships, while poorly aligned regions show decoupling.
    """

    def __init__(
        self,
        n_rois: int,
        config: ContrastiveConfig,
    ):
        super().__init__()

        self.n_rois = n_rois
        self.config = config

        # SC encoder
        self.sc_encoder = build_encoder(
            encoder_type=config.encoder_type,
            in_dim=n_rois,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # FC encoder
        self.fc_encoder = build_encoder(
            encoder_type=config.encoder_type,
            in_dim=n_rois,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Projection heads
        self.sc_proj = ProjectionHead(config.embedding_dim, config.projection_dim)
        self.fc_proj = ProjectionHead(config.embedding_dim, config.projection_dim)

        # Node-level projection (for regional alignment analysis)
        self.sc_node_proj = ProjectionHead(config.embedding_dim, config.projection_dim)
        self.fc_node_proj = ProjectionHead(config.embedding_dim, config.projection_dim)

    def forward(
        self,
        sc_graphs: List[Data],
        fc_graphs: List[Data],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a batch of paired SC-FC graphs.

        Parameters
        ----------
        sc_graphs : list of Data
            Structural connectivity graphs.
        fc_graphs : list of Data
            Functional connectivity graphs.

        Returns
        -------
        dict with graph and node level embeddings and projections.
        """
        device = next(self.parameters()).device

        # Process all SC graphs
        sc_graph_embs = []
        sc_node_embs = []
        for g in sc_graphs:
            g = g.to(device)
            out = self.sc_encoder(g.x, g.edge_index, g.edge_attr)
            sc_graph_embs.append(out["graph_emb"])
            sc_node_embs.append(out["node_emb"])

        # Process all FC graphs
        fc_graph_embs = []
        fc_node_embs = []
        for g in fc_graphs:
            g = g.to(device)
            out = self.fc_encoder(g.x, g.edge_index, g.edge_attr)
            fc_graph_embs.append(out["graph_emb"])
            fc_node_embs.append(out["node_emb"])

        # Stack graph embeddings (B, emb_dim)
        sc_graph_embs = torch.cat(sc_graph_embs, dim=0)
        fc_graph_embs = torch.cat(fc_graph_embs, dim=0)

        # Project to contrastive space
        z_sc = self.sc_proj(sc_graph_embs)  # (B, proj_dim)
        z_fc = self.fc_proj(fc_graph_embs)  # (B, proj_dim)

        # Node-level projections (for regional analysis)
        sc_node_projs = [self.sc_node_proj(ne) for ne in sc_node_embs]
        fc_node_projs = [self.fc_node_proj(ne) for ne in fc_node_embs]

        return {
            "z_sc": z_sc,
            "z_fc": z_fc,
            "sc_graph_embs": sc_graph_embs,
            "fc_graph_embs": fc_graph_embs,
            "sc_node_embs": sc_node_embs,
            "fc_node_embs": fc_node_embs,
            "sc_node_projs": sc_node_projs,
            "fc_node_projs": fc_node_projs,
        }


# ==============================================================================
# CONTRASTIVE LOSSES
# ==============================================================================

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).

    For batch size B with positive pairs (z_SC^i, z_FC^i):

        L_i = -log( exp(sim(z_SC^i, z_FC^i)/τ) /
                     Σ_{k=1}^{B} [k≠i] exp(sim(z_SC^i, z_FC^k)/τ) )

    The symmetric version averages SC→FC and FC→SC directions.

    Parameters
    ----------
    temperature : float
        Scaling temperature τ. Lower values create sharper distributions.
    symmetric : bool
        If True, compute bidirectional loss.
    """

    def __init__(self, temperature: float = 0.1, symmetric: bool = True):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        z_sc: torch.Tensor,
        z_fc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute NT-Xent loss.

        Parameters
        ----------
        z_sc : torch.Tensor
            SC projections (B, D), L2-normalized.
        z_fc : torch.Tensor
            FC projections (B, D), L2-normalized.

        Returns
        -------
        dict with 'total', 'sc_to_fc', 'fc_to_sc' losses.
        """
        B = z_sc.size(0)

        # Cosine similarity matrix
        sim = torch.mm(z_sc, z_fc.T) / self.temperature  # (B, B)

        # Labels: positive pair is on the diagonal
        labels = torch.arange(B, device=z_sc.device)

        # SC → FC direction
        loss_sc_fc = F.cross_entropy(sim, labels)

        if self.symmetric:
            # FC → SC direction
            loss_fc_sc = F.cross_entropy(sim.T, labels)
            total = (loss_sc_fc + loss_fc_sc) / 2.0
        else:
            loss_fc_sc = torch.tensor(0.0, device=z_sc.device)
            total = loss_sc_fc

        return {
            "total": total,
            "sc_to_fc": loss_sc_fc,
            "fc_to_sc": loss_fc_sc,
        }


class NodeContrastiveLoss(nn.Module):
    """
    Node-level contrastive loss for regional alignment.

    For each node i across subjects, aligns SC and FC representations
    of the same node from the same subject.

    This provides per-region alignment quality after training.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        sc_node_projs: List[torch.Tensor],
        fc_node_projs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute average node-level contrastive loss.

        Parameters
        ----------
        sc_node_projs : list of Tensor
            Per-subject SC node projections, each (N, D).
        fc_node_projs : list of Tensor
            Per-subject FC node projections, each (N, D).

        Returns
        -------
        torch.Tensor
            Mean node contrastive loss.
        """
        losses = []
        for sc_nodes, fc_nodes in zip(sc_node_projs, fc_node_projs):
            N = sc_nodes.size(0)
            # Similarity between all nodes SC vs FC within subject
            sim = torch.mm(sc_nodes, fc_nodes.T) / self.temperature  # (N, N)
            labels = torch.arange(N, device=sc_nodes.device)
            loss = F.cross_entropy(sim, labels)
            losses.append(loss)

        return torch.stack(losses).mean()


# ==============================================================================
# GRAPH AUGMENTATION
# ==============================================================================

def augment_graph(
    data: Data,
    edge_drop_prob: float = 0.1,
    feat_mask_prob: float = 0.1,
) -> Data:
    """
    Apply stochastic augmentation to a graph for contrastive robustness.

    Augmentations:
    1. Edge dropping: randomly remove edges with probability p
    2. Feature masking: randomly zero-out node features with probability p

    These augmentations force the model to learn robust representations
    that capture essential topological patterns rather than memorizing
    exact connectivity values.

    Parameters
    ----------
    data : Data
        Input graph.
    edge_drop_prob : float
        Probability of dropping each edge.
    feat_mask_prob : float
        Probability of masking each feature dimension.

    Returns
    -------
    Data
        Augmented graph (new object, original unchanged).
    """
    augmented = data.clone()

    # Edge dropping
    if edge_drop_prob > 0 and augmented.edge_index.size(1) > 0:
        n_edges = augmented.edge_index.size(1)
        mask = torch.rand(n_edges) > edge_drop_prob
        augmented.edge_index = augmented.edge_index[:, mask]
        if augmented.edge_attr is not None:
            augmented.edge_attr = augmented.edge_attr[mask]

    # Feature masking
    if feat_mask_prob > 0:
        mask = torch.rand_like(augmented.x) > feat_mask_prob
        augmented.x = augmented.x * mask.float()

    return augmented


# ==============================================================================
# MULTIMODAL COHERENCE ANALYSIS
# ==============================================================================

def compute_regional_coherence(
    model: MultimodalContrastiveModel,
    sc_graphs: List[Data],
    fc_graphs: List[Data],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Compute per-region multimodal coherence across the cohort.

    Regional coherence is measured as the cosine similarity between
    SC and FC node embeddings for the same region and subject.
    High coherence indicates tight structure-function alignment;
    low coherence indicates decoupling.

    Parameters
    ----------
    model : MultimodalContrastiveModel
        Trained contrastive model.
    sc_graphs, fc_graphs : list of Data
        Paired graphs for all subjects.
    device : torch.device
        Computation device.

    Returns
    -------
    dict with:
        'coherence_per_subject': (S, N) cosine similarity
        'mean_coherence': (N,) group mean
        'std_coherence': (N,) group std
        'graph_embeddings_sc': (S, D) subject-level SC embeddings
        'graph_embeddings_fc': (S, D) subject-level FC embeddings
    """
    model.eval()

    all_coherence = []
    graph_embs_sc = []
    graph_embs_fc = []

    with torch.no_grad():
        for sc_g, fc_g in zip(sc_graphs, fc_graphs):
            sc_g = sc_g.to(device)
            fc_g = fc_g.to(device)

            sc_out = model.sc_encoder(sc_g.x, sc_g.edge_index, sc_g.edge_attr)
            fc_out = model.fc_encoder(fc_g.x, fc_g.edge_index, fc_g.edge_attr)

            # Node-level coherence (cosine similarity per region)
            sc_node = F.normalize(sc_out["node_emb"], dim=-1)
            fc_node = F.normalize(fc_out["node_emb"], dim=-1)
            coherence = (sc_node * fc_node).sum(dim=-1).cpu().numpy()
            all_coherence.append(coherence)

            # Graph-level embeddings
            graph_embs_sc.append(sc_out["graph_emb"].cpu().numpy().flatten())
            graph_embs_fc.append(fc_out["graph_emb"].cpu().numpy().flatten())

    all_coherence = np.array(all_coherence)  # (S, N)
    graph_embs_sc = np.array(graph_embs_sc)  # (S, D)
    graph_embs_fc = np.array(graph_embs_fc)  # (S, D)

    return {
        "coherence_per_subject": all_coherence,
        "mean_coherence": np.mean(all_coherence, axis=0),
        "std_coherence": np.std(all_coherence, axis=0),
        "graph_embeddings_sc": graph_embs_sc,
        "graph_embeddings_fc": graph_embs_fc,
    }


def discover_subgroups(
    graph_embeddings: np.ndarray,
    method: str = "spectral",
    n_clusters_range: Tuple[int, int] = (2, 5),
) -> Dict[str, any]:
    """
    Discover patient subgroups from learned graph embeddings.

    Uses unsupervised clustering on the joint SC+FC embedding space
    to identify potential subgroups of post-COVID patients with
    distinct network reorganization patterns.

    Parameters
    ----------
    graph_embeddings : np.ndarray
        Concatenated [SC; FC] embeddings (S, 2D).
    method : str
        Clustering method: 'spectral', 'kmeans', 'hdbscan'.
    n_clusters_range : tuple
        Range of cluster numbers to evaluate.

    Returns
    -------
    dict with:
        'labels': cluster assignments for best k
        'silhouette_scores': score per k
        'best_k': optimal number of clusters
        'embeddings_2d': UMAP/t-SNE coordinates for visualization
    """
    from sklearn.cluster import SpectralClustering, KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    # Normalize embeddings
    scaler = StandardScaler()
    X = scaler.fit_transform(graph_embeddings)

    # Test different k values
    silhouette_scores = {}
    all_labels = {}

    for k in range(n_clusters_range[0], n_clusters_range[1] + 1):
        if k >= X.shape[0]:
            break

        if method == "spectral":
            clusterer = SpectralClustering(n_clusters=k, random_state=42)
        elif method == "kmeans":
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            break

        labels = clusterer.fit_predict(X)
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            silhouette_scores[k] = score
            all_labels[k] = labels

    # Select best k
    if silhouette_scores:
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        best_labels = all_labels[best_k]
    else:
        best_k = 1
        best_labels = np.zeros(X.shape[0], dtype=int)

    # 2D embedding for visualization
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, X.shape[0]-1))
        emb_2d = tsne.fit_transform(X)
    except Exception:
        emb_2d = X[:, :2] if X.shape[1] >= 2 else X

    return {
        "labels": best_labels,
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "all_labels": all_labels,
        "embeddings_2d": emb_2d,
    }
