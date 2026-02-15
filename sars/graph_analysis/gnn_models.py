"""
Graph Neural Network Models
============================

Implementa modelos GNN modernos para análise de redes cerebrais:
- mBrainGT: Modular Brain Graph Transformer
- BrainGNN: Interpretable brain GNN
- Dynamic GNN para FC temporal

Baseado em:
- mBrainGT (IEEE TNNLS Feb 2025)
- BrainGNN (Medical Image Analysis 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Optional, Tuple, Dict
import numpy as np


class ModularBrainGraphTransformer(nn.Module):
    """
    Modular Brain Graph Transformer (mBrainGT).
    
    Analisa subnetworks modulares separadamente e integra via
    self-attention e cross-attention.
    
    Baseado em paper IEEE TNNLS Feb 2025.
    
    Parameters
    ----------
    num_node_features : int
        Features por nó
    hidden_channels : int
        Canais escondidos
    num_modules : int
        Número de módulos/subnetworks
    num_heads : int
        Número de attention heads
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        num_modules: int = 7,  # 7 networks Yeo
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super(ModularBrainGraphTransformer, self).__init__()
        
        self.num_modules = num_modules
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        
        # Encoders modulares (um para cada subnetwork)
        self.module_encoders = nn.ModuleList([
            nn.Sequential(
                GATConv(num_node_features, hidden_channels, heads=num_heads, concat=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False),
                nn.ReLU()
            )
            for _ in range(num_modules)
        ])
        
        # Self-attention dentro de cada módulo
        self.module_self_attention = nn.ModuleList([
            nn.MultiheadAttention(hidden_channels, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_modules)
        ])
        
        # Cross-attention entre módulos
        self.cross_attention = nn.MultiheadAttention(
            hidden_channels, num_heads, dropout=dropout, batch_first=True
        )
        
        # Adaptive fusion
        self.fusion_weights = nn.Parameter(torch.ones(num_modules) / num_modules)
        self.fusion_fc = nn.Linear(hidden_channels * num_modules, hidden_channels)
        
        # Classificador final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)  # Binary classification
        )
        
    def forward(
        self,
        data: Data,
        module_assignments: torch.Tensor
    ):
        """
        Forward pass.
        
        Parameters
        ----------
        data : Data
            PyG Data object
        module_assignments : torch.Tensor
            Assignment de cada nó a um módulo [num_nodes]
            
        Returns
        -------
        out : torch.Tensor
            Predições [batch_size, 2]
        module_embeddings : List[torch.Tensor]
            Embeddings de cada módulo
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        module_embeddings = []
        
        # Processa cada módulo separadamente
        for module_idx in range(self.num_modules):
            # Seleciona nós deste módulo
            module_mask = (module_assignments == module_idx)
            
            if torch.sum(module_mask) == 0:
                # Módulo vazio, pula
                continue
            
            # Extrai subgraph para este módulo
            module_nodes = torch.where(module_mask)[0]
            
            # Cria máscara de edges dentro do módulo
            edge_mask = torch.isin(edge_index[0], module_nodes) & torch.isin(edge_index[1], module_nodes)
            module_edges = edge_index[:, edge_mask]
            
            # Reindexação de nós para o subgraph
            node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(module_nodes)}
            reindexed_edges = torch.tensor([
                [node_mapping[edge_index[0, i].item()], node_mapping[edge_index[1, i].item()]]
                for i in range(module_edges.shape[1]) if edge_mask[i]
            ]).T
            
            if reindexed_edges.shape[1] == 0:
                continue
            
            # Features do módulo
            module_x = x[module_mask]
            
            # Encode módulo
            module_h = self.module_encoders[module_idx](module_x, reindexed_edges)
            
            # Self-attention dentro do módulo
            module_h_expanded = module_h.unsqueeze(0)  # [1, num_nodes_in_module, hidden]
            module_h_attn, _ = self.module_self_attention[module_idx](
                module_h_expanded, module_h_expanded, module_h_expanded
            )
            module_h = module_h_attn.squeeze(0)
            
            # Pool módulo para representação única
            module_rep = torch.mean(module_h, dim=0, keepdim=True)  # [1, hidden]
            module_embeddings.append(module_rep)
        
        if len(module_embeddings) == 0:
            # Fallback se nenhum módulo válido
            return torch.zeros((1, 2)), []
        
        # Stack embeddings de módulos
        all_modules = torch.cat(module_embeddings, dim=0).unsqueeze(0)  # [1, num_modules, hidden]
        
        # Cross-attention entre módulos
        cross_attn_out, _ = self.cross_attention(all_modules, all_modules, all_modules)
        
        # Adaptive fusion com pesos aprendidos
        fusion_weights = F.softmax(self.fusion_weights[:len(module_embeddings)], dim=0)
        weighted_modules = cross_attn_out.squeeze(0) * fusion_weights.unsqueeze(1)
        
        # Concatena e funde
        fused = torch.cat([m for m in weighted_modules], dim=-1)
        fused = self.fusion_fc(fused)
        
        # Global representation
        global_rep = torch.mean(fused, dim=0, keepdim=True)
        
        # Classificação
        out = self.classifier(global_rep)
        
        return out, module_embeddings


class ROIAwareGNN(nn.Module):
    """
    ROI-aware Graph Convolutional Network (BrainGNN style).
    
    Usa basis functions condicionadas em ROI assignments.
    
    Parameters
    ----------
    num_node_features : int
        Features por nó
    hidden_channels : int
        Canais escondidos
    num_communities : int
        Número de communities/modules
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        num_communities: int = 7,
        dropout: float = 0.2
    ):
        super(ROIAwareGNN, self).__init__()
        
        self.num_communities = num_communities
        
        # Basis weights para cada community
        self.basis_weights = nn.Parameter(
            torch.randn(num_communities, num_node_features, hidden_channels)
        )
        
        # GCN layers
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Pooling
        self.pool_fc = nn.Linear(hidden_channels, 1)
        
        # Classifier
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, 2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        data: Data,
        community_assignments: torch.Tensor
    ):
        """
        Forward pass com ROI-aware convolution.
        
        Parameters
        ----------
        data : Data
            PyG Data object
        community_assignments : torch.Tensor
            Community assignment para cada nó
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # ROI-aware embedding
        # Cada nó usa basis weight de sua community
        h = torch.zeros(x.shape[0], self.basis_weights.shape[2]).to(x.device)
        for i in range(x.shape[0]):
            community = community_assignments[i].item()
            h[i] = torch.matmul(x[i], self.basis_weights[community])
        
        # GCN layers com edge weights
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        h = self.conv1(h, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        
        # ROI-selection pooling
        scores = self.pool_fc(h)
        scores = torch.sigmoid(scores).squeeze(-1)
        
        # Seleciona top-k nodes
        k = max(1, int(0.5 * h.shape[0]))  # Mantém 50% dos nós
        top_k_indices = torch.topk(scores, k).indices
        
        h_pooled = h[top_k_indices]
        
        # Global pooling
        batch_pooled = batch[top_k_indices] if batch is not None else torch.zeros(k, dtype=torch.long)
        h_global = global_mean_pool(h_pooled, batch_pooled)
        
        # Classification
        h_global = self.fc1(h_global)
        h_global = F.relu(h_global)
        h_global = self.dropout(h_global)
        out = self.fc2(h_global)
        
        return out, top_k_indices


class DynamicGraphGNN(nn.Module):
    """
    Dynamic Graph GNN para análise temporal de conectividade.
    
    Usa LSTM para capturar dinâmica temporal + GNN espacial.
    
    Parameters
    ----------
    num_node_features : int
        Features por nó
    hidden_channels : int
        Canais escondidos
    num_timesteps : int
        Número de timesteps/windows
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int = 64,
        num_timesteps: int = 10,
        dropout: float = 0.2
    ):
        super(DynamicGraphGNN, self).__init__()
        
        self.num_timesteps = num_timesteps
        
        # GNN espacial para cada timestep
        self.spatial_gnn = nn.Sequential(
            GCNConv(num_node_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            GCNConv(hidden_channels, hidden_channels),
            nn.ReLU()
        )
        
        # LSTM temporal
        self.temporal_lstm = nn.LSTM(
            hidden_channels,
            hidden_channels,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)
        )
        
    def forward(
        self,
        graphs_sequence: List[Data]
    ):
        """
        Forward pass em sequência de graphs temporais.
        
        Parameters
        ----------
        graphs_sequence : List[Data]
            Lista de graphs para cada timestep
        """
        # Processa cada timestep com GNN espacial
        temporal_embeddings = []
        
        for graph in graphs_sequence:
            # GNN espacial
            h = self.spatial_gnn[0](graph.x, graph.edge_index)
            for layer in self.spatial_gnn[1:]:
                if isinstance(layer, GCNConv):
                    h = layer(h, graph.edge_index)
                else:
                    h = layer(h)
            
            # Pool para embedding do graph
            if hasattr(graph, 'batch'):
                h_graph = global_mean_pool(h, graph.batch)
            else:
                h_graph = torch.mean(h, dim=0, keepdim=True)
            
            temporal_embeddings.append(h_graph)
        
        # Stack temporal
        temporal_sequence = torch.stack(temporal_embeddings, dim=1)  # [batch, time, hidden]
        
        # LSTM temporal
        lstm_out, _ = self.temporal_lstm(temporal_sequence)
        
        # Usa último timestep
        final_embedding = lstm_out[:, -1, :]
        
        # Classification
        out = self.classifier(final_embedding)
        
        return out


def assign_nodes_to_modules(
    connectivity_matrix: np.ndarray,
    num_modules: int = 7,
    method: str = 'louvain'
) -> np.ndarray:
    """
    Atribui nós a módulos usando community detection.
    
    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz de conectividade
    num_modules : int
        Número alvo de módulos
    method : str
        Método de detecção ('louvain', 'spectral')
        
    Returns
    -------
    assignments : np.ndarray
        Assignment de cada nó [num_nodes]
    """
    import networkx as nx
    from sklearn.cluster import SpectralClustering
    
    if method == 'louvain':
        # Usa Louvain community detection
        G = nx.from_numpy_array(connectivity_matrix)
        communities = nx.community.louvain_communities(G, seed=42)
        
        # Converte para array
        assignments = np.zeros(connectivity_matrix.shape[0], dtype=int)
        for module_idx, community in enumerate(communities):
            for node in community:
                assignments[node] = module_idx
        
    elif method == 'spectral':
        # Spectral clustering
        clustering = SpectralClustering(
            n_clusters=num_modules,
            affinity='precomputed',
            random_state=42
        )
        assignments = clustering.fit_predict(connectivity_matrix)
    
    else:
        raise ValueError(f"Método {method} não reconhecido")
    
    return assignments
