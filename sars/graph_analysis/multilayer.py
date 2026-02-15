"""
Multilayer Network Analysis
============================

Implementa análise de redes multicamadas integrando:
- Conectividade estrutural (dMRI)
- Conectividade funcional (rs-fMRI)

Baseado em:
- Casas-Roma et al. (Network Neuroscience 2022)
- Multilayer network analysis (MIT Press 2025)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class MultilayerMetrics:
    """Métricas de multilayer network"""
    interlayer_correlation: float  # Correlação entre layers
    layer_specific_hubs: Dict[str, List[int]]  # Hubs específicos de cada layer
    multiplex_participation: np.ndarray  # Coeficiente de participação multiplex
    structural_functional_coupling: np.ndarray  # Acoplamento SC-FC por nó
    global_efficiency_diff: float  # Diferença de eficiência global entre layers


class MultilayerNetwork:
    """
    Análise de redes multicamadas integrando SC e FC.
    
    Implementa framework de multilayer networks para combinar
    informação estrutural e funcional.
    
    Parameters
    ----------
    structural_matrix : np.ndarray
        Matriz de conectividade estrutural (dMRI)
    functional_matrix : np.ndarray
        Matriz de conectividade funcional (fMRI)
    node_labels : List[str], optional
        Labels dos nós (ROIs)
    """
    
    def __init__(
        self,
        structural_matrix: np.ndarray,
        functional_matrix: np.ndarray,
        node_labels: Optional[List[str]] = None
    ):
        self.sc_matrix = structural_matrix
        self.fc_matrix = functional_matrix
        self.n_nodes = structural_matrix.shape[0]
        
        if node_labels is None:
            self.node_labels = [f"ROI_{i}" for i in range(self.n_nodes)]
        else:
            self.node_labels = node_labels
        
        # Normaliza matrizes
        self.sc_matrix = self._normalize_matrix(self.sc_matrix)
        self.fc_matrix = self._normalize_matrix(self.fc_matrix)
        
        # Cria graphs NetworkX
        self.sc_graph = self._create_graph(self.sc_matrix)
        self.fc_graph = self._create_graph(self.fc_matrix)
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normaliza matriz de conectividade"""
        matrix = matrix.copy()
        # Remove diagonal
        np.fill_diagonal(matrix, 0)
        # Normaliza
        if np.max(matrix) > 0:
            matrix = matrix / np.max(matrix)
        return matrix
    
    def _create_graph(self, matrix: np.ndarray) -> nx.Graph:
        """Cria NetworkX graph a partir de matriz"""
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        
        # Adiciona edges
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if matrix[i, j] > 0:
                    G.add_edge(i, j, weight=matrix[i, j])
        
        return G
    
    def calculate_interlayer_correlation(self) -> float:
        """
        Calcula correlação entre layers estrutural e funcional.
        
        Returns
        -------
        correlation : float
            Correlação de Pearson entre SC e FC
        """
        # Pega upper triangle (sem diagonal)
        triu_indices = np.triu_indices(self.n_nodes, k=1)
        sc_values = self.sc_matrix[triu_indices]
        fc_values = self.fc_matrix[triu_indices]
        
        # Correlação
        correlation, _ = stats.pearsonr(sc_values, fc_values)
        
        return correlation
    
    def identify_layer_specific_hubs(
        self,
        top_percentile: float = 90
    ) -> Dict[str, List[int]]:
        """
        Identifica hubs específicos de cada layer.
        
        Hub = nó com degree ou strength no top percentile.
        
        Parameters
        ----------
        top_percentile : float
            Percentil para definir hubs
            
        Returns
        -------
        hubs : Dict[str, List[int]]
            Hubs de cada layer e hubs comuns
        """
        # Calcula strength de cada nó
        sc_strength = np.sum(self.sc_matrix, axis=1)
        fc_strength = np.sum(self.fc_matrix, axis=1)
        
        # Define thresholds
        sc_threshold = np.percentile(sc_strength, top_percentile)
        fc_threshold = np.percentile(fc_strength, top_percentile)
        
        # Identifica hubs
        sc_hubs = np.where(sc_strength >= sc_threshold)[0].tolist()
        fc_hubs = np.where(fc_strength >= fc_threshold)[0].tolist()
        
        # Hubs comuns
        common_hubs = list(set(sc_hubs) & set(fc_hubs))
        
        # Hubs específicos
        sc_specific = list(set(sc_hubs) - set(common_hubs))
        fc_specific = list(set(fc_hubs) - set(common_hubs))
        
        return {
            'structural': sc_specific,
            'functional': fc_specific,
            'common': common_hubs
        }
    
    def multiplex_participation_coefficient(self) -> np.ndarray:
        """
        Calcula coeficiente de participação multiplex.
        
        Mede o quanto cada nó participa de ambos os layers.
        
        Returns
        -------
        participation : np.ndarray
            Coeficiente de participação para cada nó [0, 1]
        """
        # Strength normalizado em cada layer
        sc_strength_norm = np.sum(self.sc_matrix, axis=1)
        fc_strength_norm = np.sum(self.fc_matrix, axis=1)
        
        # Normaliza
        if np.max(sc_strength_norm) > 0:
            sc_strength_norm /= np.max(sc_strength_norm)
        if np.max(fc_strength_norm) > 0:
            fc_strength_norm /= np.max(fc_strength_norm)
        
        # Participação = produto normalizado
        participation = sc_strength_norm * fc_strength_norm
        
        return participation
    
    def structural_functional_coupling(
        self,
        method: str = 'correlation'
    ) -> np.ndarray:
        """
        Calcula acoplamento SC-FC para cada nó.
        
        Parameters
        ----------
        method : str
            Método para calcular coupling ('correlation', 'cosine')
            
        Returns
        -------
        coupling : np.ndarray
            Acoplamento SC-FC por nó
        """
        coupling = np.zeros(self.n_nodes)
        
        for i in range(self.n_nodes):
            sc_profile = self.sc_matrix[i, :]
            fc_profile = self.fc_matrix[i, :]
            
            if method == 'correlation':
                if np.std(sc_profile) > 0 and np.std(fc_profile) > 0:
                    coupling[i], _ = stats.pearsonr(sc_profile, fc_profile)
                else:
                    coupling[i] = 0
            elif method == 'cosine':
                # Similaridade cosseno
                norm_sc = np.linalg.norm(sc_profile)
                norm_fc = np.linalg.norm(fc_profile)
                if norm_sc > 0 and norm_fc > 0:
                    coupling[i] = np.dot(sc_profile, fc_profile) / (norm_sc * norm_fc)
                else:
                    coupling[i] = 0
        
        return coupling
    
    def global_efficiency_comparison(self) -> Tuple[float, float, float]:
        """
        Compara eficiência global entre layers.
        
        Returns
        -------
        sc_efficiency : float
            Eficiência global estrutural
        fc_efficiency : float
            Eficiência global funcional
        difference : float
            Diferença (FC - SC)
        """
        sc_eff = nx.global_efficiency(self.sc_graph)
        fc_eff = nx.global_efficiency(self.fc_graph)
        
        return sc_eff, fc_eff, fc_eff - sc_eff
    
    def rich_club_comparison(
        self,
        k_range: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compara rich club organization entre layers.
        
        Parameters
        ----------
        k_range : List[int], optional
            Range de degrees para calcular rich club
            
        Returns
        -------
        rich_club : Dict[str, np.ndarray]
            Coeficientes de rich club por layer
        """
        if k_range is None:
            max_k = min(
                max(dict(self.sc_graph.degree()).values()),
                max(dict(self.fc_graph.degree()).values())
            )
            k_range = list(range(1, max_k))
        
        sc_rc = []
        fc_rc = []
        
        for k in k_range:
            try:
                sc_rc.append(nx.rich_club_coefficient(self.sc_graph, normalized=False)[k])
            except:
                sc_rc.append(0)
            
            try:
                fc_rc.append(nx.rich_club_coefficient(self.fc_graph, normalized=False)[k])
            except:
                fc_rc.append(0)
        
        return {
            'structural': np.array(sc_rc),
            'functional': np.array(fc_rc),
            'k_values': np.array(k_range)
        }
    
    def analyze(self) -> MultilayerMetrics:
        """
        Pipeline completo de análise multilayer.
        
        Returns
        -------
        metrics : MultilayerMetrics
            Métricas completas
        """
        # Correlação inter-layer
        interlayer_corr = self.calculate_interlayer_correlation()
        
        # Hubs
        hubs = self.identify_layer_specific_hubs()
        
        # Participação multiplex
        multiplex_part = self.multiplex_participation_coefficient()
        
        # Acoplamento SC-FC
        sc_fc_coupling = self.structural_functional_coupling()
        
        # Eficiência global
        _, _, eff_diff = self.global_efficiency_comparison()
        
        return MultilayerMetrics(
            interlayer_correlation=interlayer_corr,
            layer_specific_hubs=hubs,
            multiplex_participation=multiplex_part,
            structural_functional_coupling=sc_fc_coupling,
            global_efficiency_diff=eff_diff
        )


class TemporalMultilayerNetwork:
    """
    Análise de redes multilayer temporais.
    
    Estende MultilayerNetwork para dados temporais (sliding windows).
    
    Parameters
    ----------
    timeseries : np.ndarray
        Séries temporais (timepoints x ROIs)
    structural_matrix : np.ndarray
        Conectividade estrutural
    window_size : int
        Tamanho da janela temporal
    overlap : float
        Fração de overlap entre janelas
    """
    
    def __init__(
        self,
        timeseries: np.ndarray,
        structural_matrix: np.ndarray,
        window_size: int = 50,
        overlap: float = 0.5
    ):
        self.timeseries = timeseries
        self.structural_matrix = structural_matrix
        self.window_size = window_size
        self.overlap = overlap
        self.n_nodes = timeseries.shape[1]
        
        # Calcula windows
        self.windows = self._create_windows()
        
    def _create_windows(self) -> List[Tuple[int, int]]:
        """Cria janelas temporais com overlap"""
        step = int(self.window_size * (1 - self.overlap))
        windows = []
        
        start = 0
        while start + self.window_size <= len(self.timeseries):
            windows.append((start, start + self.window_size))
            start += step
        
        return windows
    
    def calculate_dynamic_fc(
        self,
        method: str = 'correlation'
    ) -> List[np.ndarray]:
        """
        Calcula conectividade funcional dinâmica.
        
        Parameters
        ----------
        method : str
            Método de conectividade ('correlation', 'covariance')
            
        Returns
        -------
        fc_windows : List[np.ndarray]
            FC para cada janela temporal
        """
        fc_windows = []
        
        for start, end in self.windows:
            window_data = self.timeseries[start:end, :]
            
            if method == 'correlation':
                fc = np.corrcoef(window_data.T)
            elif method == 'covariance':
                fc = np.cov(window_data.T)
            else:
                raise ValueError(f"Método {method} não reconhecido")
            
            np.fill_diagonal(fc, 0)
            fc_windows.append(fc)
        
        return fc_windows
    
    def temporal_coupling_variability(self) -> np.ndarray:
        """
        Calcula variabilidade temporal do acoplamento SC-FC.
        
        Returns
        -------
        variability : np.ndarray
            Variabilidade de coupling por nó
        """
        fc_windows = self.calculate_dynamic_fc()
        
        # Calcula coupling para cada window
        coupling_windows = []
        for fc in fc_windows:
            mlnet = MultilayerNetwork(self.structural_matrix, fc)
            coupling = mlnet.structural_functional_coupling()
            coupling_windows.append(coupling)
        
        coupling_windows = np.array(coupling_windows)
        
        # Variabilidade = std temporal
        variability = np.std(coupling_windows, axis=0)
        
        return variability
