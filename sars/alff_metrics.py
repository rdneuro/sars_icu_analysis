"""
Specific COVID-19 Analysis
============================

Metrics and analyses focused on Long COVID studies:
- ALFF, fALFF, ReHo
- Network-based statistics
- Longitudinal changes
"""

import numpy as np
from scipy import signal, ndimage
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class COVIDMetrics:
    """Resultados de análise COVID"""
    alff: np.ndarray  # ALFF por ROI
    falff: np.ndarray  # fALFF por ROI
    reho: np.ndarray  # ReHo por ROI
    network_efficiency: Dict[str, float]  # Eficiências de rede
    altered_networks: Dict[str, str]  # Redes alteradas (hiper/hipo)


class COVIDAnalyzer:
    """
    Análise de neuroimagem focada em alterações pós-COVID.
    
    Implementa métricas reportadas na literatura:
    - ALFF aumentado em putamen, temporal (Lancet 2024)
    - DMN hiper-conectado (múltiplos estudos)
    - FPN relacionado a déficits de memória
    
    Parameters
    ----------
    low_freq : float
        Frequência mínima para ALFF (Hz)
    high_freq : float
        Frequência máxima para ALFF (Hz)
    tr : float
        Repetition time em segundos
    """
    
    def __init__(
        self,
        low_freq: float = 0.01,
        high_freq: float = 0.08,
        tr: float = 2.0
    ):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.tr = tr
        self.fs = 1.0 / tr  # Sampling frequency
        
    def calculate_alff(
        self,
        timeseries: np.ndarray,
        detrend: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula ALFF (Amplitude of Low-Frequency Fluctuation).
        
        ALFF = amplitude média das flutuações em low-frequency band.
        
        Parameters
        ----------
        timeseries : np.ndarray
            Séries temporais [timepoints, ROIs]
        detrend : bool
            Se True, remove tendência linear
            
        Returns
        -------
        alff : np.ndarray
            ALFF para cada ROI
        power_spectrum : np.ndarray
            Espectro de potência [freqs, ROIs]
        """
        n_timepoints, n_rois = timeseries.shape
        
        alff = np.zeros(n_rois)
        
        # FFT para cada ROI
        freqs = np.fft.rfftfreq(n_timepoints, d=self.tr)
        power_spectrum = np.zeros((len(freqs), n_rois))
        
        for roi in range(n_rois):
            ts = timeseries[:, roi].copy()
            
            # Detrend
            if detrend:
                ts = signal.detrend(ts)
            
            # FFT
            fft = np.fft.rfft(ts)
            power = np.abs(fft) ** 2
            
            power_spectrum[:, roi] = power
            
            # ALFF = raiz da soma dos quadrados em low-freq band
            freq_mask = (freqs >= self.low_freq) & (freqs <= self.high_freq)
            alff[roi] = np.sqrt(np.mean(power[freq_mask]))
        
        return alff, power_spectrum
    
    def calculate_falff(
        self,
        timeseries: np.ndarray
    ) -> np.ndarray:
        """
        Calcula fALFF (fractional ALFF).
        
        fALFF = ALFF / amplitude total (normalizado)
        
        Parameters
        ----------
        timeseries : np.ndarray
            Séries temporais [timepoints, ROIs]
            
        Returns
        -------
        falff : np.ndarray
            fALFF para cada ROI
        """
        alff, power_spectrum = self.calculate_alff(timeseries)
        
        # Amplitude total do espectro
        total_amplitude = np.sqrt(np.mean(power_spectrum, axis=0))
        
        # fALFF = ALFF / total
        falff = alff / (total_amplitude + 1e-10)
        
        return falff
    
    def calculate_reho(
        self,
        timeseries: np.ndarray,
        connectivity_matrix: np.ndarray,
        n_neighbors: int = 27
    ) -> np.ndarray:
        """
        Calcula ReHo (Regional Homogeneity).
        
        ReHo = concordância de Kendall entre série temporal de um ROI
        e seus vizinhos.
        
        Parameters
        ----------
        timeseries : np.ndarray
            Séries temporais [timepoints, ROIs]
        connectivity_matrix : np.ndarray
            Matriz de conectividade estrutural
        n_neighbors : int
            Número de vizinhos a considerar
            
        Returns
        -------
        reho : np.ndarray
            ReHo para cada ROI
        """
        from scipy.stats import kendalltau
        
        n_rois = timeseries.shape[1]
        reho = np.zeros(n_rois)
        
        for roi in range(n_rois):
            # Encontra vizinhos (ROIs mais conectados)
            connections = connectivity_matrix[roi, :]
            neighbor_indices = np.argsort(connections)[-n_neighbors:]
            
            # Séries temporais do ROI e vizinhos
            roi_ts = timeseries[:, roi]
            neighbor_ts = timeseries[:, neighbor_indices]
            
            # Calcula concordância de Kendall
            concordances = []
            for neighbor_idx in range(neighbor_ts.shape[1]):
                tau, _ = kendalltau(roi_ts, neighbor_ts[:, neighbor_idx])
                concordances.append(tau)
            
            # ReHo = concordância média
            reho[roi] = np.mean(concordances)
        
        return reho
    
    def network_based_statistics(
        self,
        connectivity_covid: np.ndarray,
        connectivity_control: np.ndarray,
        threshold: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Network-Based Statistics (NBS) para comparar COVID vs Control.
        
        Identifica componentes conectados de edges alterados.
        
        Parameters
        ----------
        connectivity_covid : np.ndarray
            Conectividade do grupo COVID
        connectivity_control : np.ndarray
            Conectividade do grupo controle
        threshold : float
            Threshold de p-value
            
        Returns
        -------
        results : Dict
            Componentes alterados e estatísticas
        """
        from scipy import stats
        
        # Diferença de conectividade
        diff = connectivity_covid - connectivity_control
        
        # T-test para cada edge (simplificado)
        # Na prática, você faria permutation testing
        t_stats = diff / (np.std(connectivity_covid) + np.std(connectivity_control) + 1e-10)
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
        
        # Identifica edges significativos
        sig_edges = p_values < threshold
        
        # Encontra componentes conectados
        import networkx as nx
        G = nx.Graph()
        n_rois = connectivity_covid.shape[0]
        G.add_nodes_from(range(n_rois))
        
        for i in range(n_rois):
            for j in range(i+1, n_rois):
                if sig_edges[i, j]:
                    G.add_edge(i, j, weight=diff[i, j])
        
        # Componentes conectados
        components = list(nx.connected_components(G))
        
        return {
            'significant_edges': sig_edges,
            'p_values': p_values,
            'components': components,
            'n_components': len(components)
        }
    
    def identify_altered_networks(
        self,
        connectivity_covid: np.ndarray,
        connectivity_control: np.ndarray,
        network_assignments: np.ndarray,
        network_names: Optional[list] = None
    ) -> Dict[str, str]:
        """
        Identifica redes hiper ou hipo-conectadas em COVID.
        
        Parameters
        ----------
        connectivity_covid : np.ndarray
            Conectividade COVID
        connectivity_control : np.ndarray
            Conectividade controle
        network_assignments : np.ndarray
            Assignment de cada ROI a uma rede
        network_names : list, optional
            Nomes das redes
            
        Returns
        -------
        altered : Dict[str, str]
            Status de cada rede ('hyper', 'hypo', 'unchanged')
        """
        n_networks = len(np.unique(network_assignments))
        
        if network_names is None:
            network_names = [f"Network_{i}" for i in range(n_networks)]
        
        altered = {}
        
        for net_idx in range(n_networks):
            # Nós nesta rede
            net_mask = network_assignments == net_idx
            
            # Conectividade dentro da rede
            covid_within = connectivity_covid[np.ix_(net_mask, net_mask)]
            control_within = connectivity_control[np.ix_(net_mask, net_mask)]
            
            # Média de conectividade
            covid_mean = np.mean(covid_within[np.triu_indices_from(covid_within, k=1)])
            control_mean = np.mean(control_within[np.triu_indices_from(control_within, k=1)])
            
            # Diferença relativa
            rel_diff = (covid_mean - control_mean) / (control_mean + 1e-10)
            
            # Classifica
            if rel_diff > 0.1:  # >10% aumento
                status = 'hyper'
            elif rel_diff < -0.1:  # >10% redução
                status = 'hypo'
            else:
                status = 'unchanged'
            
            altered[network_names[net_idx]] = status
        
        return altered
    
    def analyze(
        self,
        timeseries: np.ndarray,
        structural_connectivity: np.ndarray,
        network_assignments: Optional[np.ndarray] = None,
        control_timeseries: Optional[np.ndarray] = None,
        control_connectivity: Optional[np.ndarray] = None
    ) -> COVIDMetrics:
        """
        Pipeline completo de análise COVID.
        
        Parameters
        ----------
        timeseries : np.ndarray
            Séries temporais do grupo COVID
        structural_connectivity : np.ndarray
            Conectividade estrutural
        network_assignments : np.ndarray, optional
            Assignments de rede
        control_* : optional
            Dados do grupo controle para comparação
            
        Returns
        -------
        metrics : COVIDMetrics
            Métricas completas
        """
        # Calcula ALFF, fALFF, ReHo
        alff = self.calculate_alff(timeseries)[0]
        falff = self.calculate_falff(timeseries)
        reho = self.calculate_reho(timeseries, structural_connectivity)
        
        # Eficiências de rede (placeholder)
        import networkx as nx
        G = nx.from_numpy_array(structural_connectivity)
        network_eff = {
            'global': nx.global_efficiency(G),
            'local': nx.local_efficiency(G)
        }
        
        # Identifica redes alteradas se houver controle
        if control_connectivity is not None and network_assignments is not None:
            fc_covid = np.corrcoef(timeseries.T)
            fc_control = np.corrcoef(control_timeseries.T)
            altered = self.identify_altered_networks(
                fc_covid, fc_control, network_assignments
            )
        else:
            altered = {}
        
        return COVIDMetrics(
            alff=alff,
            falff=falff,
            reho=reho,
            network_efficiency=network_eff,
            altered_networks=altered
        )
