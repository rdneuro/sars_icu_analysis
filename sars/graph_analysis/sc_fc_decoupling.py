"""
sars/graph_analysis/sc_fc_decoupling.py

SC-FC Decoupling Analysis using Graph Theory Metrics
Based on: Lee et al. (2017), Yang et al. (2024), Chen et al. (2019)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import stats
import bct  # Brain Connectivity Toolbox
import networkx as nx


@dataclass
class SCFCDecouplingResults:
    """Results from SC-FC decoupling analysis using graph metrics."""
    
    # Global coupling metrics
    global_coupling: float  # Correlation of SC-FC edges
    global_coupling_pvalue: float
    
    # Metric-specific couplings
    nodal_metric_couplings: Dict[str, np.ndarray]  # Per-node correlations
    global_metric_couplings: Dict[str, float]  # Global metric correlations
    
    # Efficiency ratios (FC/SC)
    efficiency_ratios: Dict[str, np.ndarray]
    
    # Nodal decoupling indices
    nodal_decoupling: np.ndarray  # 1 - coupling per node
    hub_decoupling: Optional[Dict[str, float]] = None
    
    # Statistical comparisons
    metric_differences: Optional[Dict[str, Dict]] = None
    
    # Raw metrics
    sc_metrics: Dict[str, np.ndarray] = field(default_factory=dict)
    fc_metrics: Dict[str, np.ndarray] = field(default_factory=dict)


def threshold_matrix(matrix: np.ndarray, 
                     method: str = 'proportional',
                     density: float = 0.15,
                     absolute: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Threshold connectivity matrix for graph analysis.
    
    Parameters
    ----------
    matrix : np.ndarray
        Connectivity matrix (N x N)
    method : str
        'proportional' (keep top X% edges) or 'absolute' (threshold value)
    density : float
        For proportional: fraction of edges to keep (0-1)
        For absolute: threshold value
    absolute : bool
        If True, use absolute values before thresholding
    
    Returns
    -------
    binary : np.ndarray
        Binary adjacency matrix
    weighted : np.ndarray
        Thresholded weighted matrix
    """
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    
    if absolute:
        mat_thresh = np.abs(mat)
    else:
        mat_thresh = mat.copy()
    
    if method == 'proportional':
        # Keep top density% of edges
        triu_vals = mat_thresh[np.triu_indices_from(mat_thresh, k=1)]
        threshold = np.percentile(triu_vals, 100 * (1 - density))
        binary = (mat_thresh >= threshold).astype(float)
    else:
        binary = (mat_thresh >= density).astype(float)
    
    # Symmetric
    binary = np.maximum(binary, binary.T)
    weighted = mat * binary
    
    return binary, weighted


def compute_nodal_metrics(matrix: np.ndarray, 
                          weighted: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive nodal graph metrics.
    
    Parameters
    ----------
    matrix : np.ndarray
        Connectivity matrix (binary or weighted)
    weighted : bool
        Whether to use weighted versions of metrics
    
    Returns
    -------
    metrics : Dict[str, np.ndarray]
        Dictionary of nodal metrics
    """
    n_nodes = matrix.shape[0]
    metrics = {}
    
    # Ensure non-negative for BCT
    mat = np.abs(matrix)
    
    # 1. Degree / Strength
    if weighted:
        metrics['strength'] = np.sum(mat, axis=1)
    else:
        metrics['degree'] = np.sum(mat > 0, axis=1).astype(float)
    
    # 2. Clustering coefficient
    if weighted:
        metrics['clustering'] = bct.clustering_coef_wu(mat)
    else:
        metrics['clustering'] = bct.clustering_coef_bu(mat)
    
    # 3. Betweenness centrality
    if weighted:
        # BCT uses distance matrix (inverse of weights)
        dist = 1.0 / (mat + np.finfo(float).eps)
        dist[mat == 0] = np.inf
        metrics['betweenness'] = bct.betweenness_wei(dist)
    else:
        metrics['betweenness'] = bct.betweenness_bin(mat)
    
    # Normalize betweenness
    metrics['betweenness'] = metrics['betweenness'] / ((n_nodes - 1) * (n_nodes - 2))
    
    # 4. Local efficiency
    if weighted:
        metrics['local_efficiency'] = bct.efficiency_wei(mat, local=True)
    else:
        metrics['local_efficiency'] = bct.efficiency_bin(mat, local=True)
    
    # 5. Nodal global efficiency (average inverse shortest path to all nodes)
    if weighted:
        dist = 1.0 / (mat + np.finfo(float).eps)
        dist[mat == 0] = np.inf
        np.fill_diagonal(dist, 0)
        D = bct.distance_wei(dist)[0]
    else:
        D = bct.distance_bin(mat)
    
    D_inv = 1.0 / D
    D_inv[np.isinf(D_inv)] = 0
    np.fill_diagonal(D_inv, 0)
    metrics['nodal_efficiency'] = np.sum(D_inv, axis=1) / (n_nodes - 1)
    
    # 6. Eigenvector centrality
    try:
        G = nx.from_numpy_array(mat)
        ec = nx.eigenvector_centrality_numpy(G, weight='weight')
        metrics['eigenvector_centrality'] = np.array([ec[i] for i in range(n_nodes)])
    except:
        metrics['eigenvector_centrality'] = np.zeros(n_nodes)
    
    # 7. Participation coefficient (requires modularity first)
    try:
        if weighted:
            ci, _ = bct.community_louvain(mat)
        else:
            ci, _ = bct.community_louvain(mat)
        metrics['participation'] = bct.participation_coef(mat, ci)
        metrics['within_module_degree'] = bct.module_degree_zscore(mat, ci)
        metrics['module_assignment'] = ci
    except:
        metrics['participation'] = np.zeros(n_nodes)
        metrics['within_module_degree'] = np.zeros(n_nodes)
    
    return metrics


def compute_global_metrics(matrix: np.ndarray, 
                           weighted: bool = True) -> Dict[str, float]:
    """
    Compute global graph metrics.
    
    Parameters
    ----------
    matrix : np.ndarray
        Connectivity matrix
    weighted : bool
        Whether to use weighted versions
    
    Returns
    -------
    metrics : Dict[str, float]
        Dictionary of global metrics
    """
    mat = np.abs(matrix)
    n_nodes = mat.shape[0]
    metrics = {}
    
    # Clustering coefficient
    if weighted:
        C = np.mean(bct.clustering_coef_wu(mat))
    else:
        C = np.mean(bct.clustering_coef_bu(mat))
    metrics['clustering_coefficient'] = C
    
    # Characteristic path length
    if weighted:
        dist = 1.0 / (mat + np.finfo(float).eps)
        dist[mat == 0] = np.inf
        np.fill_diagonal(dist, 0)
        D = bct.distance_wei(dist)[0]
    else:
        D = bct.distance_bin(mat)
    
    D_finite = D[np.isfinite(D) & (D > 0)]
    metrics['characteristic_path_length'] = np.mean(D_finite) if len(D_finite) > 0 else np.inf
    
    # Global efficiency
    if weighted:
        metrics['global_efficiency'] = bct.efficiency_wei(mat)
    else:
        metrics['global_efficiency'] = bct.efficiency_bin(mat)
    
    # Local efficiency (average)
    if weighted:
        metrics['local_efficiency'] = np.mean(bct.efficiency_wei(mat, local=True))
    else:
        metrics['local_efficiency'] = np.mean(bct.efficiency_bin(mat, local=True))
    
    # Modularity
    try:
        if weighted:
            _, Q = bct.community_louvain(mat)
        else:
            _, Q = bct.community_louvain(mat)
        metrics['modularity'] = Q
    except:
        metrics['modularity'] = 0.0
    
    # Density
    binary = (mat > 0).astype(float)
    n_edges = np.sum(binary[np.triu_indices(n_nodes, k=1)])
    max_edges = n_nodes * (n_nodes - 1) / 2
    metrics['density'] = n_edges / max_edges
    
    return metrics


def sc_fc_decoupling_analysis(
    sc_matrix: np.ndarray,
    fc_matrix: np.ndarray,
    density: float = 0.15,
    method: str = 'proportional',
    node_labels: Optional[List[str]] = None,
    compute_efficiency_ratio: bool = True,
    compute_hub_analysis: bool = True
) -> SCFCDecouplingResults:
    """
    Comprehensive SC-FC decoupling analysis using graph theory metrics.
    
    This function implements multiple approaches from the literature:
    1. Edge-wise SC-FC correlation (Honey et al., 2009)
    2. Nodal metric correlations (Chen et al., 2019)
    3. Efficiency ratios (Lee et al., 2017)
    4. Global metric comparisons (Yang et al., 2024)
    
    Parameters
    ----------
    sc_matrix : np.ndarray
        Structural connectivity matrix (N x N)
    fc_matrix : np.ndarray
        Functional connectivity matrix (N x N)
    density : float
        Network density for thresholding (default 0.15)
    method : str
        Thresholding method ('proportional' or 'absolute')
    node_labels : List[str], optional
        Labels for each node (e.g., from Brainnetome atlas)
    compute_efficiency_ratio : bool
        Whether to compute FC/SC efficiency ratios
    compute_hub_analysis : bool
        Whether to analyze decoupling in hub vs non-hub nodes
    
    Returns
    -------
    SCFCDecouplingResults
        Comprehensive results dataclass
    
    References
    ----------
    - Honey et al. (2009). PNAS
    - Lee et al. (2017). Human Brain Mapping
    - Chen et al. (2019). Human Brain Mapping
    - Yang et al. (2024). CNS Neuroscience & Therapeutics
    """
    
    n_nodes = sc_matrix.shape[0]
    
    # Threshold matrices
    sc_bin, sc_wei = threshold_matrix(sc_matrix, method=method, density=density)
    fc_bin, fc_wei = threshold_matrix(fc_matrix, method=method, density=density)
    
    # =========================================================================
    # 1. EDGE-WISE SC-FC COUPLING (Honey et al., 2009)
    # =========================================================================
    # Extract non-zero SC connections and correlate with FC
    sc_upper = sc_wei[np.triu_indices(n_nodes, k=1)]
    fc_upper = fc_wei[np.triu_indices(n_nodes, k=1)]
    
    # Use only where SC exists
    sc_nonzero_mask = sc_upper > 0
    if np.sum(sc_nonzero_mask) > 2:
        # Rescale SC to Gaussian distribution as per literature
        sc_values = sc_upper[sc_nonzero_mask]
        fc_values = fc_upper[sc_nonzero_mask]
        
        # Z-score normalization
        sc_z = (sc_values - np.mean(sc_values)) / (np.std(sc_values) + 1e-10)
        fc_z = (fc_values - np.mean(fc_values)) / (np.std(fc_values) + 1e-10)
        
        global_coupling, global_pvalue = stats.pearsonr(sc_z, fc_z)
    else:
        global_coupling = 0.0
        global_pvalue = 1.0
    
    # =========================================================================
    # 2. NODAL METRICS COMPUTATION
    # =========================================================================
    sc_nodal = compute_nodal_metrics(sc_wei, weighted=True)
    fc_nodal = compute_nodal_metrics(fc_wei, weighted=True)
    
    sc_global = compute_global_metrics(sc_wei, weighted=True)
    fc_global = compute_global_metrics(fc_wei, weighted=True)
    
    # =========================================================================
    # 3. METRIC-SPECIFIC COUPLING
    # =========================================================================
    # Common metrics to compare
    common_metrics = ['strength', 'clustering', 'betweenness', 
                      'local_efficiency', 'nodal_efficiency', 
                      'participation', 'eigenvector_centrality']
    
    nodal_metric_couplings = {}
    global_metric_couplings = {}
    
    for metric in common_metrics:
        # Handle alternative names
        sc_key = 'strength' if metric == 'strength' else metric
        fc_key = 'strength' if metric == 'strength' else metric
        
        if sc_key in sc_nodal and fc_key in fc_nodal:
            sc_vals = sc_nodal[sc_key]
            fc_vals = fc_nodal[fc_key]
            
            # Global correlation for this metric
            if np.std(sc_vals) > 0 and np.std(fc_vals) > 0:
                r, _ = stats.pearsonr(sc_vals, fc_vals)
                global_metric_couplings[metric] = r
            else:
                global_metric_couplings[metric] = 0.0
            
            # Per-node contribution to coupling is harder to define
            # Use local neighborhood correlation or leave as the metric diff
            nodal_metric_couplings[metric] = 1 - np.abs(
                (fc_vals - sc_vals) / (np.abs(fc_vals) + np.abs(sc_vals) + 1e-10)
            )
    
    # =========================================================================
    # 4. EFFICIENCY RATIOS (Lee et al., 2017)
    # =========================================================================
    efficiency_ratios = {}
    
    if compute_efficiency_ratio:
        # Nodal local efficiency ratio
        sc_loc_eff = sc_nodal['local_efficiency']
        fc_loc_eff = fc_nodal['local_efficiency']
        efficiency_ratios['local_efficiency'] = fc_loc_eff / (sc_loc_eff + 1e-10)
        
        # Nodal global efficiency ratio
        sc_glob_eff = sc_nodal['nodal_efficiency']
        fc_glob_eff = fc_nodal['nodal_efficiency']
        efficiency_ratios['nodal_efficiency'] = fc_glob_eff / (sc_glob_eff + 1e-10)
        
        # Strength ratio
        efficiency_ratios['strength'] = fc_nodal['strength'] / (sc_nodal['strength'] + 1e-10)
    
    # =========================================================================
    # 5. NODAL DECOUPLING INDEX
    # =========================================================================
    # Aggregate decoupling: average across metrics
    decoupling_components = []
    for metric in ['strength', 'clustering', 'local_efficiency', 'nodal_efficiency']:
        if metric in nodal_metric_couplings:
            decoupling_components.append(1 - nodal_metric_couplings[metric])
    
    if len(decoupling_components) > 0:
        nodal_decoupling = np.mean(decoupling_components, axis=0)
    else:
        nodal_decoupling = np.zeros(n_nodes)
    
    # =========================================================================
    # 6. HUB-SPECIFIC DECOUPLING (optional)
    # =========================================================================
    hub_decoupling = None
    
    if compute_hub_analysis:
        # Identify hubs based on SC (top 20% degree)
        sc_strength = sc_nodal['strength']
        hub_threshold = np.percentile(sc_strength, 80)
        is_hub = sc_strength >= hub_threshold
        
        hub_decoupling = {
            'hub_mean_decoupling': np.mean(nodal_decoupling[is_hub]),
            'nonhub_mean_decoupling': np.mean(nodal_decoupling[~is_hub]),
            'hub_fc_sc_strength_ratio': np.mean(efficiency_ratios['strength'][is_hub]) if 'strength' in efficiency_ratios else 0,
            'nonhub_fc_sc_strength_ratio': np.mean(efficiency_ratios['strength'][~is_hub]) if 'strength' in efficiency_ratios else 0,
            'n_hubs': np.sum(is_hub),
            'hub_indices': np.where(is_hub)[0]
        }
    
    # =========================================================================
    # 7. METRIC DIFFERENCES (for statistical comparison)
    # =========================================================================
    metric_differences = {}
    
    # Global metrics comparison
    for metric in ['clustering_coefficient', 'characteristic_path_length', 
                   'global_efficiency', 'local_efficiency', 'modularity', 'density']:
        if metric in sc_global and metric in fc_global:
            metric_differences[metric] = {
                'SC': sc_global[metric],
                'FC': fc_global[metric],
                'difference': fc_global[metric] - sc_global[metric],
                'ratio': fc_global[metric] / (sc_global[metric] + 1e-10)
            }
    
    # =========================================================================
    # BUILD RESULTS
    # =========================================================================
    results = SCFCDecouplingResults(
        global_coupling=global_coupling,
        global_coupling_pvalue=global_pvalue,
        nodal_metric_couplings=nodal_metric_couplings,
        global_metric_couplings=global_metric_couplings,
        efficiency_ratios=efficiency_ratios,
        nodal_decoupling=nodal_decoupling,
        hub_decoupling=hub_decoupling,
        metric_differences=metric_differences,
        sc_metrics=sc_nodal,
        fc_metrics=fc_nodal
    )
    
    return results


def create_decoupling_dataframe(
    results: SCFCDecouplingResults,
    node_labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a summary DataFrame of nodal decoupling results.
    
    Parameters
    ----------
    results : SCFCDecouplingResults
        Results from sc_fc_decoupling_analysis
    node_labels : List[str], optional
        Labels for each node (from atlas)
    
    Returns
    -------
    df : pd.DataFrame
        Summary DataFrame with columns for each metric and decoupling indices
    """
    n_nodes = len(results.nodal_decoupling)
    
    if node_labels is None:
        node_labels = [f'ROI_{i+1}' for i in range(n_nodes)]
    
    df = pd.DataFrame({
        'node_id': range(n_nodes),
        'label': node_labels,
        'decoupling_index': results.nodal_decoupling
    })
    
    # Add SC metrics
    for metric, values in results.sc_metrics.items():
        if isinstance(values, np.ndarray) and len(values) == n_nodes:
            df[f'SC_{metric}'] = values
    
    # Add FC metrics
    for metric, values in results.fc_metrics.items():
        if isinstance(values, np.ndarray) and len(values) == n_nodes:
            df[f'FC_{metric}'] = values
    
    # Add efficiency ratios
    for metric, values in results.efficiency_ratios.items():
        if isinstance(values, np.ndarray) and len(values) == n_nodes:
            df[f'ratio_{metric}'] = values
    
    # Sort by decoupling
    df = df.sort_values('decoupling_index', ascending=False)
    
    return df
