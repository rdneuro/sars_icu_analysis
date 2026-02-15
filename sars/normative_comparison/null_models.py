"""
╔══════════════════════════════════════════════════════════════════════╗
║  Normative Comparison - Null Models Module                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Modelos nulos para avaliação de métricas de grafo cerebral.         ║
║                                                                      ║
║  JUSTIFICATIVA:                                                      ║
║  Em vez de comparar com controles externos (problemático devido a    ║
║  diferenças de protocolo), comparamos cada rede observada com        ║
║  versões randomizadas dela mesma que preservam certas propriedades.  ║
║  Isso permite testar se a topologia observada é significativamente   ║
║  diferente do esperado por acaso.                                    ║
║                                                                      ║
║  MODELOS IMPLEMENTADOS:                                              ║
║  1. Erdős-Rényi (ER) - Rede aleatória com mesma densidade            ║
║  2. Configuration Model - Preserva sequência de graus                ║
║  3. Maslov-Sneppen - Rewiring preservando grau exato                 ║
║  4. Strength-preserving - Para redes ponderadas                      ║
║  5. Geometry-preserving - Preserva estrutura espacial (Rubinov)      ║
║  6. Lattice - Rede regular para comparação de small-worldness        ║
║                                                                      ║
║  MÉTRICAS NORMALIZADAS:                                              ║
║  - Small-worldness σ = (C/C_rand) / (L/L_rand)                       ║
║  - Small-worldness ω = L_rand/L - C/C_latt                           ║
║  - Normalized modularity Q_norm = (Q - Q_rand) / (Q_max - Q_rand)    ║
║  - Rich-club normalized ϕ_norm(k) = ϕ(k) / ϕ_rand(k)                 ║
║                                                                      ║
║  REFERÊNCIAS:                                                        ║
║  - Maslov & Sneppen (2002) - Specificity and stability               ║
║  - Rubinov & Sporns (2010) - Complex network measures                ║
║  - Humphries & Gurney (2008) - Small-worldness                       ║
║  - van den Heuvel et al. (2008) - Rich-club organization             ║
║  - Betzel et al. (2016) - Geometry-preserving null models            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Callable
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Tentar importar bibliotecas opcionais
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import bct
    HAS_BCT = True
except ImportError:
    HAS_BCT = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NullModelResult:
    """Resultado de análise com modelo nulo."""
    metric_name: str
    observed: float
    null_mean: float
    null_std: float
    z_score: float
    p_value: float
    null_distribution: np.ndarray
    n_surrogates: int
    null_model_type: str
    
    @property
    def significant(self) -> bool:
        return self.p_value < 0.05
    
    @property
    def interpretation(self) -> str:
        if self.z_score > 2:
            return "significantly_higher"
        elif self.z_score < -2:
            return "significantly_lower"
        else:
            return "not_significant"


# =============================================================================
# NULL MODEL GENERATORS
# =============================================================================

def generate_erdos_renyi(n_nodes: int, n_edges: int, seed: int = None) -> np.ndarray:
    """
    Gerar rede Erdős-Rényi (completamente aleatória).
    
    Preserva: número de nós e arestas
    Destrói: estrutura de graus, clustering, modularidade
    
    Parameters
    ----------
    n_nodes : int
        Número de nós
    n_edges : int
        Número de arestas
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Matriz de adjacência binária
    """
    if seed is not None:
        np.random.seed(seed)
    
    adj = np.zeros((n_nodes, n_nodes))
    
    # Todos os pares possíveis
    triu_idx = np.triu_indices(n_nodes, k=1)
    n_possible = len(triu_idx[0])
    
    # Selecionar arestas aleatoriamente
    selected = np.random.choice(n_possible, size=min(n_edges, n_possible), replace=False)
    
    for idx in selected:
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        adj[i, j] = adj[j, i] = 1
    
    return adj


def generate_configuration_model(degree_sequence: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Gerar rede via configuration model (preserva sequência de graus).
    
    Preserva: sequência de graus (aproximadamente)
    Destrói: clustering, correlações de grau
    
    Parameters
    ----------
    degree_sequence : np.ndarray
        Sequência de graus a preservar
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Matriz de adjacência binária
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_nodes = len(degree_sequence)
    
    if HAS_NETWORKX:
        # Usar NetworkX (mais robusto)
        try:
            # Garantir que soma dos graus é par
            degrees = degree_sequence.astype(int).tolist()
            if sum(degrees) % 2 != 0:
                # Ajustar menor grau
                min_idx = np.argmin(degrees)
                degrees[min_idx] += 1
            
            G = nx.configuration_model(degrees, seed=seed)
            G = nx.Graph(G)  # Remover multi-edges
            G.remove_edges_from(nx.selfloop_edges(G))  # Remover self-loops
            
            adj = nx.to_numpy_array(G)
            return adj
            
        except Exception:
            pass
    
    # Fallback: implementação manual simplificada
    adj = np.zeros((n_nodes, n_nodes))
    stubs = []
    
    for i, d in enumerate(degree_sequence.astype(int)):
        stubs.extend([i] * d)
    
    np.random.shuffle(stubs)
    
    # Formar arestas
    for i in range(0, len(stubs) - 1, 2):
        a, b = stubs[i], stubs[i + 1]
        if a != b:  # Evitar self-loops
            adj[a, b] = adj[b, a] = 1
    
    return adj


def maslov_sneppen_rewire(
    adj: np.ndarray,
    n_rewires: int = None,
    seed: int = None
) -> np.ndarray:
    """
    Rewiring de Maslov-Sneppen (preserva grau exato de cada nó).
    
    O algoritmo troca arestas (i-j, k-l) → (i-l, k-j) se não criar
    multi-edges ou self-loops.
    
    Preserva: grau exato de cada nó
    Destrói: clustering, correlações, motifs
    
    Parameters
    ----------
    adj : np.ndarray
        Matriz de adjacência original (binária)
    n_rewires : int, optional
        Número de rewires. Se None, usa 10 * n_edges
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Matriz de adjacência rewired
    """
    if seed is not None:
        np.random.seed(seed)
    
    adj_new = adj.copy()
    n_nodes = adj.shape[0]
    
    # Listar todas as arestas
    edges = list(zip(*np.where(np.triu(adj_new) > 0)))
    n_edges = len(edges)
    
    if n_edges < 2:
        return adj_new
    
    if n_rewires is None:
        n_rewires = 10 * n_edges
    
    successful = 0
    attempts = 0
    max_attempts = n_rewires * 10
    
    while successful < n_rewires and attempts < max_attempts:
        attempts += 1
        
        # Selecionar duas arestas aleatórias
        idx1, idx2 = np.random.choice(len(edges), size=2, replace=False)
        i, j = edges[idx1]
        k, l = edges[idx2]
        
        # Verificar se os nós são distintos
        if len(set([i, j, k, l])) != 4:
            continue
        
        # Tentar rewire: (i-j, k-l) → (i-l, k-j)
        # Verificar se novas arestas já existem
        if adj_new[i, l] == 0 and adj_new[k, j] == 0:
            # Remover arestas antigas
            adj_new[i, j] = adj_new[j, i] = 0
            adj_new[k, l] = adj_new[l, k] = 0
            
            # Adicionar novas arestas
            adj_new[i, l] = adj_new[l, i] = 1
            adj_new[k, j] = adj_new[j, k] = 1
            
            # Atualizar lista de arestas
            edges[idx1] = (min(i, l), max(i, l))
            edges[idx2] = (min(k, j), max(k, j))
            
            successful += 1
    
    return adj_new


def strength_preserving_rewire(
    adj_weighted: np.ndarray,
    n_rewires: int = None,
    seed: int = None
) -> np.ndarray:
    """
    Rewiring que preserva força (soma dos pesos) de cada nó.
    
    Preserva: força nodal (aproximadamente)
    Destrói: estrutura de pesos específica
    
    Parameters
    ----------
    adj_weighted : np.ndarray
        Matriz de adjacência ponderada
    n_rewires : int, optional
        Número de rewires
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Matriz de adjacência rewired
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Primeiro fazer rewiring binário
    adj_binary = (adj_weighted > 0).astype(float)
    adj_rewired_binary = maslov_sneppen_rewire(adj_binary, n_rewires, seed)
    
    # Redistribuir pesos
    # Extrair pesos originais
    triu_orig = np.triu_indices_from(adj_weighted, k=1)
    weights_orig = adj_weighted[triu_orig]
    weights_orig = weights_orig[weights_orig > 0]
    
    # Encontrar novas arestas
    triu_new = np.triu_indices_from(adj_rewired_binary, k=1)
    new_edges = adj_rewired_binary[triu_new] > 0
    
    # Permutar pesos e atribuir às novas arestas
    np.random.shuffle(weights_orig)
    
    adj_rewired = np.zeros_like(adj_weighted)
    edge_idx = 0
    
    for idx in range(len(triu_new[0])):
        if new_edges[idx]:
            i, j = triu_new[0][idx], triu_new[1][idx]
            if edge_idx < len(weights_orig):
                adj_rewired[i, j] = adj_rewired[j, i] = weights_orig[edge_idx]
                edge_idx += 1
    
    return adj_rewired


def generate_lattice(n_nodes: int, k_neighbors: int = 4) -> np.ndarray:
    """
    Gerar rede lattice (regular) para comparação de small-worldness.
    
    Em uma lattice, cada nó conecta-se aos k vizinhos mais próximos
    em um anel. Tem alto clustering mas longo path length.
    
    Parameters
    ----------
    n_nodes : int
        Número de nós
    k_neighbors : int
        Número de vizinhos de cada lado (total = 2*k)
        
    Returns
    -------
    np.ndarray
        Matriz de adjacência da lattice
    """
    adj = np.zeros((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in range(1, k_neighbors + 1):
            # Vizinho à direita
            right = (i + j) % n_nodes
            adj[i, right] = adj[right, i] = 1
            
            # Vizinho à esquerda
            left = (i - j) % n_nodes
            adj[i, left] = adj[left, i] = 1
    
    return adj


def geometry_preserving_rewire(
    adj: np.ndarray,
    coordinates: np.ndarray,
    n_rewires: int = None,
    distance_bins: int = 10,
    seed: int = None
) -> np.ndarray:
    """
    Rewiring que preserva distribuição de distâncias das conexões.
    
    Importante para redes cerebrais onde conexões curtas são mais
    prováveis que conexões longas devido a custos de wiring.
    
    Preserva: distribuição de comprimentos de conexão
    Destrói: topologia específica
    
    Parameters
    ----------
    adj : np.ndarray
        Matriz de adjacência
    coordinates : np.ndarray
        Coordenadas espaciais dos nós (n_nodes, 3)
    n_rewires : int, optional
        Número de rewires
    distance_bins : int
        Número de bins para distâncias
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Matriz de adjacência rewired
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_nodes = adj.shape[0]
    
    # Calcular matriz de distâncias
    distances = squareform(pdist(coordinates))
    
    # Binning das distâncias
    triu_idx = np.triu_indices(n_nodes, k=1)
    dist_values = distances[triu_idx]
    
    # Criar bins
    bin_edges = np.percentile(dist_values, np.linspace(0, 100, distance_bins + 1))
    
    # Atribuir cada par de nós a um bin
    pair_bins = np.digitize(dist_values, bin_edges[1:-1])
    
    # Separar arestas e não-arestas por bin
    adj_values = adj[triu_idx]
    
    edges_by_bin = {b: [] for b in range(distance_bins)}
    non_edges_by_bin = {b: [] for b in range(distance_bins)}
    
    for idx, (is_edge, bin_idx) in enumerate(zip(adj_values > 0, pair_bins)):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        if is_edge:
            edges_by_bin[bin_idx].append((i, j))
        else:
            non_edges_by_bin[bin_idx].append((i, j))
    
    # Rewiring dentro de cada bin
    adj_new = adj.copy()
    
    if n_rewires is None:
        n_rewires = int(np.sum(adj) / 2) * 5
    
    rewires_per_bin = n_rewires // distance_bins
    
    for bin_idx in range(distance_bins):
        edges = edges_by_bin[bin_idx]
        non_edges = non_edges_by_bin[bin_idx]
        
        if len(edges) == 0 or len(non_edges) == 0:
            continue
        
        for _ in range(min(rewires_per_bin, len(edges), len(non_edges))):
            # Selecionar aresta e não-aresta aleatórias
            edge_idx = np.random.randint(len(edges))
            non_edge_idx = np.random.randint(len(non_edges))
            
            i, j = edges[edge_idx]
            k, l = non_edges[non_edge_idx]
            
            # Swap
            adj_new[i, j] = adj_new[j, i] = 0
            adj_new[k, l] = adj_new[l, k] = 1
            
            # Atualizar listas
            edges[edge_idx] = (k, l)
            non_edges[non_edge_idx] = (i, j)
    
    return adj_new


# =============================================================================
# METRIC CALCULATIONS
# =============================================================================

def compute_clustering_coefficient(adj: np.ndarray) -> float:
    """Calcular coeficiente de clustering médio."""
    n_nodes = adj.shape[0]
    clustering = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        neighbors = np.where(adj[i] > 0)[0]
        k = len(neighbors)
        
        if k >= 2:
            subgraph = adj[np.ix_(neighbors, neighbors)]
            triangles = np.sum(subgraph) / 2
            clustering[i] = 2 * triangles / (k * (k - 1))
    
    return np.mean(clustering)


def compute_characteristic_path_length(adj: np.ndarray) -> float:
    """Calcular characteristic path length médio."""
    n_nodes = adj.shape[0]
    
    # Floyd-Warshall para distâncias
    dist = np.where(adj > 0, 1, np.inf)
    np.fill_diagonal(dist, 0)
    
    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    
    # Média excluindo infinitos e diagonal
    mask = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
    finite_dist = dist[mask]
    finite_dist = finite_dist[np.isfinite(finite_dist)]
    
    if len(finite_dist) == 0:
        return np.inf
    
    return np.mean(finite_dist)


def compute_global_efficiency(adj: np.ndarray) -> float:
    """Calcular eficiência global."""
    n_nodes = adj.shape[0]
    
    # Distâncias
    dist = np.where(adj > 0, 1, np.inf)
    np.fill_diagonal(dist, 0)
    
    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    
    # Eficiência = média de 1/d
    inv_dist = np.where(dist > 0, 1.0 / dist, 0)
    np.fill_diagonal(inv_dist, 0)
    
    return np.sum(inv_dist) / (n_nodes * (n_nodes - 1))


def compute_modularity(adj: np.ndarray, gamma: float = 1.0) -> Tuple[float, np.ndarray]:
    """
    Calcular modularidade usando algoritmo de Louvain.
    
    Returns
    -------
    Tuple[float, np.ndarray]
        (modularity Q, community assignments)
    """
    if HAS_BCT:
        try:
            ci, Q = bct.community_louvain(adj, gamma=gamma)
            return Q, ci
        except:
            pass
    
    if HAS_NETWORKX:
        try:
            G = nx.from_numpy_array(adj)
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(G)
            
            # Calcular Q
            m = np.sum(adj) / 2
            Q = 0
            
            ci = np.zeros(adj.shape[0], dtype=int)
            for c_idx, comm in enumerate(communities):
                for node in comm:
                    ci[node] = c_idx
            
            for i in range(adj.shape[0]):
                for j in range(adj.shape[0]):
                    if ci[i] == ci[j]:
                        ki = np.sum(adj[i])
                        kj = np.sum(adj[j])
                        Q += adj[i, j] - (ki * kj) / (2 * m)
            
            Q /= (2 * m)
            return Q, ci
        except:
            pass
    
    # Fallback simples
    return 0.0, np.zeros(adj.shape[0])


def compute_rich_club_coefficient(adj: np.ndarray, k: int) -> float:
    """
    Calcular rich-club coefficient para grau k.
    
    ϕ(k) = 2 * E_k / (N_k * (N_k - 1))
    
    onde E_k é o número de arestas entre nós com grau > k,
    e N_k é o número de tais nós.
    """
    degree = np.sum(adj > 0, axis=1)
    
    # Nós com grau > k
    rich_nodes = np.where(degree > k)[0]
    N_k = len(rich_nodes)
    
    if N_k < 2:
        return 0.0
    
    # Arestas entre rich nodes
    subgraph = adj[np.ix_(rich_nodes, rich_nodes)]
    E_k = np.sum(subgraph) / 2
    
    # Rich club coefficient
    phi = 2 * E_k / (N_k * (N_k - 1))
    
    return phi


# =============================================================================
# NULL MODEL ANALYSIS
# =============================================================================

def analyze_with_null_model(
    adj: np.ndarray,
    metric_func: Callable,
    metric_name: str,
    null_model: str = 'maslov_sneppen',
    n_surrogates: int = 1000,
    coordinates: np.ndarray = None,
    seed: int = 42,
    n_jobs: int = 1
) -> NullModelResult:
    """
    Analisar uma métrica comparando com modelo nulo.
    
    Parameters
    ----------
    adj : np.ndarray
        Matriz de adjacência observada
    metric_func : Callable
        Função que calcula a métrica (recebe adj, retorna float)
    metric_name : str
        Nome da métrica
    null_model : str
        Tipo de modelo nulo: 'erdos_renyi', 'configuration', 
        'maslov_sneppen', 'strength_preserving', 'geometry_preserving'
    n_surrogates : int
        Número de redes surrogate
    coordinates : np.ndarray, optional
        Coordenadas para geometry-preserving (n_nodes, 3)
    seed : int
        Random seed
    n_jobs : int
        Número de jobs paralelos (1 = serial)
        
    Returns
    -------
    NullModelResult
        Resultado da análise
    """
    np.random.seed(seed)
    
    # Calcular métrica observada
    observed = metric_func(adj)
    
    # Propriedades da rede
    n_nodes = adj.shape[0]
    n_edges = int(np.sum(adj > 0) / 2)
    degree_seq = np.sum(adj > 0, axis=1)
    
    # Selecionar gerador de modelo nulo
    def generate_surrogate(seed_i):
        if null_model == 'erdos_renyi':
            return generate_erdos_renyi(n_nodes, n_edges, seed=seed_i)
        
        elif null_model == 'configuration':
            return generate_configuration_model(degree_seq, seed=seed_i)
        
        elif null_model == 'maslov_sneppen':
            return maslov_sneppen_rewire(adj, seed=seed_i)
        
        elif null_model == 'strength_preserving':
            return strength_preserving_rewire(adj, seed=seed_i)
        
        elif null_model == 'geometry_preserving':
            if coordinates is None:
                raise ValueError("coordinates required for geometry_preserving")
            return geometry_preserving_rewire(adj, coordinates, seed=seed_i)
        
        else:
            raise ValueError(f"Unknown null model: {null_model}")
    
    # Gerar distribuição nula
    null_distribution = np.zeros(n_surrogates)
    
    if n_jobs == 1:
        # Serial
        for i in range(n_surrogates):
            surrogate = generate_surrogate(seed + i)
            null_distribution[i] = metric_func(surrogate)
    else:
        # Paralelo (cuidado com overhead)
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    lambda s: metric_func(generate_surrogate(s)), 
                    seed + i
                ): i for i in range(n_surrogates)
            }
            for future in as_completed(futures):
                i = futures[future]
                null_distribution[i] = future.result()
    
    # Estatísticas
    null_mean = np.mean(null_distribution)
    null_std = np.std(null_distribution)
    
    # Z-score
    if null_std > 0:
        z_score = (observed - null_mean) / null_std
    else:
        z_score = 0.0
    
    # P-value (two-tailed)
    p_value = np.mean(np.abs(null_distribution - null_mean) >= np.abs(observed - null_mean))
    
    return NullModelResult(
        metric_name=metric_name,
        observed=observed,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
        p_value=p_value,
        null_distribution=null_distribution,
        n_surrogates=n_surrogates,
        null_model_type=null_model
    )


# =============================================================================
# SMALL-WORLDNESS
# =============================================================================

def compute_small_worldness(
    adj: np.ndarray,
    n_surrogates: int = 100,
    seed: int = 42
) -> Dict[str, float]:
    """
    Calcular índices de small-worldness (σ e ω).
    
    σ (Humphries & Gurney, 2008):
        σ = (C/C_rand) / (L/L_rand)
        σ > 1 indica small-world
    
    ω (Telesford et al., 2011):
        ω = L_rand/L - C/C_latt
        ω ≈ 0 indica small-world
        ω > 0 indica mais aleatório
        ω < 0 indica mais regular
    
    Parameters
    ----------
    adj : np.ndarray
        Matriz de adjacência binária
    n_surrogates : int
        Número de redes aleatórias para média
    seed : int
        Random seed
        
    Returns
    -------
    Dict[str, float]
        Dicionário com métricas de small-worldness
    """
    np.random.seed(seed)
    
    # Métricas observadas
    C_obs = compute_clustering_coefficient(adj)
    L_obs = compute_characteristic_path_length(adj)
    
    # Métricas de redes aleatórias (Maslov-Sneppen)
    C_rand_list = []
    L_rand_list = []
    
    for i in range(n_surrogates):
        adj_rand = maslov_sneppen_rewire(adj, seed=seed + i)
        C_rand_list.append(compute_clustering_coefficient(adj_rand))
        L_rand_list.append(compute_characteristic_path_length(adj_rand))
    
    C_rand = np.mean(C_rand_list)
    L_rand = np.mean(L_rand_list)
    
    # Métricas de lattice
    n_nodes = adj.shape[0]
    mean_degree = np.mean(np.sum(adj > 0, axis=1))
    k_neighbors = max(1, int(mean_degree / 2))
    
    adj_latt = generate_lattice(n_nodes, k_neighbors)
    C_latt = compute_clustering_coefficient(adj_latt)
    
    # Calcular índices
    # σ = (C/C_rand) / (L/L_rand)
    if C_rand > 0 and L_rand > 0 and L_obs > 0:
        sigma = (C_obs / C_rand) / (L_obs / L_rand)
    else:
        sigma = np.nan
    
    # ω = L_rand/L - C/C_latt
    if L_obs > 0 and C_latt > 0:
        omega = (L_rand / L_obs) - (C_obs / C_latt)
    else:
        omega = np.nan
    
    return {
        'C_observed': C_obs,
        'C_random': C_rand,
        'C_lattice': C_latt,
        'L_observed': L_obs,
        'L_random': L_rand,
        'sigma': sigma,
        'omega': omega,
        'is_small_world': sigma > 1.0 if not np.isnan(sigma) else False
    }


def compute_normalized_modularity(
    adj: np.ndarray,
    n_surrogates: int = 100,
    seed: int = 42
) -> Dict[str, float]:
    """
    Calcular modularidade normalizada.
    
    Q_norm = (Q_obs - Q_rand) / (Q_max - Q_rand)
    
    onde Q_max é aproximado pelo limite teórico.
    
    Parameters
    ----------
    adj : np.ndarray
        Matriz de adjacência
    n_surrogates : int
        Número de surrogates
    seed : int
        Random seed
        
    Returns
    -------
    Dict[str, float]
        Métricas de modularidade
    """
    np.random.seed(seed)
    
    # Modularidade observada
    Q_obs, ci = compute_modularity(adj)
    n_communities = len(np.unique(ci))
    
    # Modularidade de redes aleatórias
    Q_rand_list = []
    
    for i in range(n_surrogates):
        adj_rand = maslov_sneppen_rewire(adj, seed=seed + i)
        Q_rand, _ = compute_modularity(adj_rand)
        Q_rand_list.append(Q_rand)
    
    Q_rand = np.mean(Q_rand_list)
    Q_rand_std = np.std(Q_rand_list)
    
    # Q_max teórico (aproximação)
    Q_max = 1.0 - 1.0 / n_communities if n_communities > 1 else 0.0
    
    # Normalizar
    if Q_max > Q_rand:
        Q_norm = (Q_obs - Q_rand) / (Q_max - Q_rand)
    else:
        Q_norm = Q_obs
    
    # Z-score
    if Q_rand_std > 0:
        z_score = (Q_obs - Q_rand) / Q_rand_std
    else:
        z_score = 0.0
    
    return {
        'Q_observed': Q_obs,
        'Q_random_mean': Q_rand,
        'Q_random_std': Q_rand_std,
        'Q_normalized': Q_norm,
        'z_score': z_score,
        'n_communities': n_communities
    }


def compute_normalized_rich_club(
    adj: np.ndarray,
    k_range: np.ndarray = None,
    n_surrogates: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Calcular rich-club coefficient normalizado para range de k.
    
    ϕ_norm(k) = ϕ(k) / ϕ_rand(k)
    
    ϕ_norm > 1 indica rich-club organization
    
    Parameters
    ----------
    adj : np.ndarray
        Matriz de adjacência
    k_range : np.ndarray, optional
        Range de valores de k para testar
    n_surrogates : int
        Número de surrogates
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Rich-club coefficients para cada k
    """
    np.random.seed(seed)
    
    degree = np.sum(adj > 0, axis=1)
    
    if k_range is None:
        k_min = int(np.percentile(degree, 50))
        k_max = int(np.percentile(degree, 95))
        k_range = np.arange(k_min, k_max + 1)
    
    results = []
    
    for k in k_range:
        # ϕ observado
        phi_obs = compute_rich_club_coefficient(adj, k)
        
        if phi_obs == 0:
            continue
        
        # ϕ de redes aleatórias
        phi_rand_list = []
        for i in range(n_surrogates):
            adj_rand = maslov_sneppen_rewire(adj, seed=seed + i)
            phi_rand_list.append(compute_rich_club_coefficient(adj_rand, k))
        
        phi_rand_mean = np.mean(phi_rand_list)
        phi_rand_std = np.std(phi_rand_list)
        
        # Normalizar
        if phi_rand_mean > 0:
            phi_norm = phi_obs / phi_rand_mean
        else:
            phi_norm = phi_obs
        
        # P-value
        p_value = np.mean(np.array(phi_rand_list) >= phi_obs)
        
        results.append({
            'k': k,
            'phi_observed': phi_obs,
            'phi_random_mean': phi_rand_mean,
            'phi_random_std': phi_rand_std,
            'phi_normalized': phi_norm,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
    
    return pd.DataFrame(results)


# =============================================================================
# COMPREHENSIVE ANALYSIS
# =============================================================================

def analyze_network_topology(
    adj: np.ndarray,
    metrics: List[str] = None,
    null_model: str = 'maslov_sneppen',
    n_surrogates: int = 100,
    coordinates: np.ndarray = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Análise completa de topologia de rede com modelos nulos.
    
    Parameters
    ----------
    adj : np.ndarray
        Matriz de adjacência
    metrics : List[str], optional
        Métricas para analisar. Se None, usa todas disponíveis.
    null_model : str
        Tipo de modelo nulo
    n_surrogates : int
        Número de surrogates
    coordinates : np.ndarray, optional
        Coordenadas para geometry-preserving
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Resultados para todas as métricas
    """
    if metrics is None:
        metrics = ['clustering', 'path_length', 'efficiency', 'modularity']
    
    metric_funcs = {
        'clustering': compute_clustering_coefficient,
        'path_length': compute_characteristic_path_length,
        'efficiency': compute_global_efficiency,
        'modularity': lambda x: compute_modularity(x)[0],
    }
    
    results = []
    
    for metric_name in metrics:
        if metric_name not in metric_funcs:
            warnings.warn(f"Unknown metric: {metric_name}")
            continue
        
        print(f"  Analyzing {metric_name}...")
        
        result = analyze_with_null_model(
            adj=adj,
            metric_func=metric_funcs[metric_name],
            metric_name=metric_name,
            null_model=null_model,
            n_surrogates=n_surrogates,
            coordinates=coordinates,
            seed=seed
        )
        
        results.append({
            'metric': metric_name,
            'observed': result.observed,
            'null_mean': result.null_mean,
            'null_std': result.null_std,
            'z_score': result.z_score,
            'p_value': result.p_value,
            'interpretation': result.interpretation,
            'null_model': result.null_model_type
        })
    
    return pd.DataFrame(results)


def analyze_all_subjects(
    subjects: List[str],
    load_matrix_func: Callable,
    metrics: List[str] = None,
    null_model: str = 'maslov_sneppen',
    n_surrogates: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Analisar todos os sujeitos com modelos nulos.
    
    Parameters
    ----------
    subjects : List[str]
        Lista de subject IDs
    load_matrix_func : Callable
        Função que carrega matriz dado subject_id
    metrics : List[str], optional
        Métricas para analisar
    null_model : str
        Tipo de modelo nulo
    n_surrogates : int
        Número de surrogates
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Resultados para todos os sujeitos
    """
    all_results = []
    
    for i, subject_id in enumerate(subjects):
        print(f"\n[{i+1}/{len(subjects)}] {subject_id}")
        
        try:
            adj = load_matrix_func(subject_id)
            
            # Binarizar se necessário
            if adj.max() > 1:
                adj = (adj > 0).astype(float)
            
            results = analyze_network_topology(
                adj=adj,
                metrics=metrics,
                null_model=null_model,
                n_surrogates=n_surrogates,
                seed=seed + i
            )
            
            results['subject_id'] = subject_id
            all_results.append(results)
            
        except Exception as e:
            warnings.warn(f"Failed for {subject_id}: {e}")
            continue
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


# =============================================================================
# CLI / TEST
# =============================================================================

if __name__ == '__main__':
    print("Null Models Module")
    print(f"NetworkX available: {HAS_NETWORKX}")
    print(f"BCT available: {HAS_BCT}")
    
    # Teste rápido
    print("\nQuick test with random network...")
    np.random.seed(42)
    
    # Criar rede teste
    n = 50
    adj_test = np.random.random((n, n))
    adj_test = (adj_test + adj_test.T) / 2
    adj_test = (adj_test > 0.7).astype(float)
    np.fill_diagonal(adj_test, 0)
    
    print(f"Test network: {n} nodes, {int(np.sum(adj_test)/2)} edges")
    
    # Teste small-worldness
    sw = compute_small_worldness(adj_test, n_surrogates=20)
    print(f"\nSmall-worldness:")
    print(f"  σ = {sw['sigma']:.3f}")
    print(f"  ω = {sw['omega']:.3f}")
    print(f"  Is small-world: {sw['is_small_world']}")
