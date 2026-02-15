# -*- coding: utf-8 -*-
"""
sars.graph_analysis.network_metrics
==================================================

Comprehensive graph-theoretic analysis of brain connectivity networks.

Implements global and nodal network metrics following the framework of
Rubinov & Sporns (2010) NeuroImage, organized into three categories:
  - **Segregation**: clustering coefficient, transitivity, modularity
  - **Integration**: characteristic path length, global/local efficiency
  - **Influence**: degree, betweenness centrality, eigenvector centrality

Additional analyses:
  - Small-world properties (sigma, omega coefficients)
  - Rich club organization
  - SC-FC coupling (structure-function relationship)
  - Hub classification (provincial vs connector)
  - Multi-threshold robustness (AUC approach)

Supports both weighted and binary, directed and undirected networks.
Negative edge weights are handled according to best practices
(zeroed out for graph metrics; see van den Heuvel et al. 2017).

References
----------
- Rubinov & Sporns (2010). NeuroImage 52:1059-1069. Complex network measures.
- Sporns (2013). Dialogues Clin Neurosci 15(3):247-262.
- Watts & Strogatz (1998). Nature 393:440-442. Small-world networks.
- Latora & Marchiori (2001). Phys Rev Lett 87:198701. Efficiency.
- Colizza et al. (2006). Nat Phys 2:110-115. Rich club.
- van den Heuvel & Sporns (2011). J Neurosci 31:15775. Rich club connectome.
- Telesford et al. (2011). Brain Connect 1(5):367-375. Omega small-world.
- Humphries & Gurney (2008). PLoS ONE 3:e0002051. Sigma small-world.
- Faskowitz et al. (2020). Sci Rep 10:2568. SC-FC coupling.
- van den Heuvel et al. (2017). NeuroImage 152:437. Proportional thresholding.
- Guimerà & Amaral (2005). Nature 433:895-900. Module roles.
"""

import numpy as np
import networkx as nx
from scipy import stats
from typing import Optional, Dict, List, Tuple, Union
import warnings


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _prepare_matrix(
    matrix: np.ndarray,
    remove_negative: bool = True,
    remove_diagonal: bool = True,
) -> np.ndarray:
    """
    Prepare a connectivity matrix for graph analysis.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
        Input connectivity matrix.
    remove_negative : bool
        If True, set negative values to zero (recommended for graph metrics;
        van den Heuvel et al. 2017, NeuroImage).
    remove_diagonal : bool
        If True, zero out the diagonal.

    Returns
    -------
    np.ndarray: Cleaned matrix.
    """
    mat = matrix.copy().astype(float)
    if remove_diagonal:
        np.fill_diagonal(mat, 0)
    if remove_negative:
        mat[mat < 0] = 0
    return mat


def _matrix_to_graph(
    matrix: np.ndarray,
    weighted: bool = True,
) -> nx.Graph:
    """
    Convert a connectivity matrix to a NetworkX Graph.

    Parameters
    ----------
    matrix : np.ndarray
        Adjacency/weight matrix (symmetric, non-negative).
    weighted : bool
        If True, edges carry 'weight' attribute.

    Returns
    -------
    nx.Graph
    """
    mat = _prepare_matrix(matrix)
    if weighted:
        G = nx.from_numpy_array(mat)
    else:
        G = nx.from_numpy_array((mat > 0).astype(float))
    return G


def _invert_weights(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Invert connection weights for path-length-based measures.

    For weighted graph analysis, stronger connections should correspond
    to shorter paths. We use L_ij = 1/W_ij (Rubinov & Sporns, 2010).

    Parameters
    ----------
    matrix : np.ndarray
        Weight matrix (non-negative).
    epsilon : float
        Small value to prevent division by zero.

    Returns
    -------
    np.ndarray: Inverted weight (length) matrix.
    """
    mat = matrix.copy()
    mask = mat > epsilon
    length = np.zeros_like(mat)
    length[mask] = 1.0 / mat[mask]
    return length


# =============================================================================
# GLOBAL NETWORK METRICS
# =============================================================================

def compute_global_metrics(
    matrix: np.ndarray,
    weighted: bool = True,
    normalize: bool = True,
) -> Dict:
    """
    Compute global (whole-network) graph metrics.

    Computes metrics across three organizational dimensions:
      - Segregation: clustering coefficient, transitivity
      - Integration: characteristic path length, global efficiency
      - Resilience: assortativity, density

    Parameters
    ----------
    matrix : np.ndarray (N, N)
        Connectivity matrix (FC or SC).
    weighted : bool
        If True, compute weighted versions of metrics.
    normalize : bool
        If True, normalize weights to [0, 1] before computing metrics.

    Returns
    -------
    dict with keys:
        - n_nodes, n_edges, density
        - clustering_coeff (mean), transitivity
        - char_path_length, global_efficiency, local_efficiency (mean)
        - assortativity
        - strength_mean, strength_std (weighted)
        - degree_mean, degree_std (binary)

    Notes
    -----
    For weighted metrics, edge weights are interpreted as connection
    strengths (higher = stronger). For path-based metrics, weights
    are inverted (L_ij = 1/W_ij) following Rubinov & Sporns (2010).
    """
    mat = _prepare_matrix(matrix)

    if normalize and weighted:
        m = np.max(mat)
        if m > 0:
            mat = mat / m

    N = mat.shape[0]
    G = _matrix_to_graph(mat, weighted=weighted)

    results = {
        "n_nodes": N,
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
    }

    # ------------------------------------------------------------------
    # SEGREGATION
    # ------------------------------------------------------------------
    # Clustering coefficient
    if weighted:
        # Onnela et al. (2005) weighted clustering
        cc = nx.clustering(G, weight="weight")
    else:
        cc = nx.clustering(G)
    cc_values = np.array(list(cc.values()))
    results["clustering_coeff"] = float(np.mean(cc_values))
    results["clustering_coeff_std"] = float(np.std(cc_values))

    # Transitivity (global clustering)
    results["transitivity"] = float(nx.transitivity(G))

    # ------------------------------------------------------------------
    # INTEGRATION
    # ------------------------------------------------------------------
    # Global efficiency (Latora & Marchiori, 2001)
    if weighted:
        # For weighted efficiency, use inverted weights as distances
        length_mat = _invert_weights(mat)
        G_len = nx.from_numpy_array(length_mat)
        results["global_efficiency"] = float(
            nx.global_efficiency(G_len)
        )
    else:
        results["global_efficiency"] = float(nx.global_efficiency(G))

    # Local efficiency
    if weighted:
        results["local_efficiency"] = float(
            nx.local_efficiency(G_len)
        )
    else:
        results["local_efficiency"] = float(nx.local_efficiency(G))

    # Characteristic path length (only on largest connected component)
    if nx.is_connected(G):
        if weighted:
            results["char_path_length"] = float(
                nx.average_shortest_path_length(G_len, weight="weight")
            )
        else:
            results["char_path_length"] = float(
                nx.average_shortest_path_length(G)
            )
    else:
        # Use largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        if weighted:
            subG = G_len.subgraph(largest_cc).copy()
        else:
            subG = G.subgraph(largest_cc).copy()
        if len(subG) > 1:
            if weighted:
                results["char_path_length"] = float(
                    nx.average_shortest_path_length(subG, weight="weight")
                )
            else:
                results["char_path_length"] = float(
                    nx.average_shortest_path_length(subG)
                )
        else:
            results["char_path_length"] = np.inf
        results["n_components"] = nx.number_connected_components(G)
        results["largest_component_fraction"] = len(largest_cc) / N

    # ------------------------------------------------------------------
    # RESILIENCE / CENTRALITY
    # ------------------------------------------------------------------
    # Assortativity (degree correlation)
    if G.number_of_edges() > 0:
        if weighted:
            results["assortativity"] = float(
                nx.degree_assortativity_coefficient(G, weight="weight")
            )
        else:
            results["assortativity"] = float(
                nx.degree_assortativity_coefficient(G)
            )
    else:
        results["assortativity"] = np.nan

    # Degree / strength distribution
    if weighted:
        strengths = np.array([d for _, d in G.degree(weight="weight")])
        results["strength_mean"] = float(np.mean(strengths))
        results["strength_std"] = float(np.std(strengths))
    degrees = np.array([d for _, d in G.degree()])
    results["degree_mean"] = float(np.mean(degrees))
    results["degree_std"] = float(np.std(degrees))

    return results


# =============================================================================
# NODAL NETWORK METRICS
# =============================================================================

def compute_nodal_metrics(
    matrix: np.ndarray,
    weighted: bool = True,
    normalize: bool = True,
) -> Dict:
    """
    Compute node-level (regional) graph metrics.

    For each node, computes:
      - Degree / strength
      - Clustering coefficient
      - Betweenness centrality
      - Eigenvector centrality
      - Local efficiency (nodal)
      - Closeness centrality
      - PageRank

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    weighted : bool
    normalize : bool

    Returns
    -------
    dict with keys as metric names, values as np.ndarray of shape (N,).
    Also includes 'hub_score' (composite z-score of degree, betweenness,
    closeness, and eigenvector centrality; Sporns et al. 2007).
    """
    mat = _prepare_matrix(matrix)
    if normalize and weighted:
        m = np.max(mat)
        if m > 0:
            mat = mat / m

    N = mat.shape[0]
    G = _matrix_to_graph(mat, weighted=weighted)
    weight_key = "weight" if weighted else None

    results = {"n_nodes": N}

    # Degree and strength
    results["degree"] = np.array([d for _, d in G.degree()])
    if weighted:
        results["strength"] = np.array(
            [d for _, d in G.degree(weight="weight")]
        )

    # Clustering coefficient
    cc = nx.clustering(G, weight=weight_key)
    results["clustering_coeff"] = np.array([cc[i] for i in range(N)])

    # Betweenness centrality
    if weighted:
        length_mat = _invert_weights(mat)
        G_len = nx.from_numpy_array(length_mat)
        bc = nx.betweenness_centrality(G_len, weight="weight", normalized=True)
    else:
        bc = nx.betweenness_centrality(G, normalized=True)
    results["betweenness_centrality"] = np.array([bc[i] for i in range(N)])

    # Eigenvector centrality
    try:
        if weighted:
            ec = nx.eigenvector_centrality_numpy(G, weight="weight")
        else:
            ec = nx.eigenvector_centrality_numpy(G)
        results["eigenvector_centrality"] = np.array([ec[i] for i in range(N)])
    except Exception:
        results["eigenvector_centrality"] = np.full(N, np.nan)

    # Closeness centrality
    if weighted:
        close = nx.closeness_centrality(G_len, distance="weight")
    else:
        close = nx.closeness_centrality(G)
    results["closeness_centrality"] = np.array([close[i] for i in range(N)])

    # PageRank
    try:
        pr = nx.pagerank(G, weight=weight_key)
        results["pagerank"] = np.array([pr[i] for i in range(N)])
    except Exception:
        results["pagerank"] = np.full(N, np.nan)

    # Local efficiency per node
    nodal_le = _compute_nodal_local_efficiency(mat, weighted=weighted)
    results["local_efficiency"] = nodal_le

    # ------------------------------------------------------------------
    # COMPOSITE HUB SCORE
    # ------------------------------------------------------------------
    # Z-score and average across multiple centrality measures
    # (Sporns et al. 2007; van den Heuvel & Sporns 2013)
    metrics_for_hub = ["degree", "betweenness_centrality",
                       "closeness_centrality", "eigenvector_centrality"]
    if weighted:
        metrics_for_hub.insert(0, "strength")

    z_scores = []
    for m_name in metrics_for_hub:
        vals = results[m_name]
        if np.all(np.isnan(vals)):
            continue
        mu, sigma = np.nanmean(vals), np.nanstd(vals)
        if sigma > 0:
            z_scores.append((vals - mu) / sigma)

    if z_scores:
        results["hub_score"] = np.nanmean(np.column_stack(z_scores), axis=1)
    else:
        results["hub_score"] = np.zeros(N)

    return results


def _compute_nodal_local_efficiency(
    matrix: np.ndarray,
    weighted: bool = True,
) -> np.ndarray:
    """
    Compute local efficiency for each node.

    Local efficiency of node i = global efficiency of the subgraph
    formed by the neighbors of i (Latora & Marchiori, 2001).
    """
    mat = _prepare_matrix(matrix)
    N = mat.shape[0]
    le = np.zeros(N)

    for i in range(N):
        neighbors = np.where(mat[i, :] > 0)[0]
        if len(neighbors) < 2:
            le[i] = 0.0
            continue

        # Extract subgraph of neighbors
        sub = mat[np.ix_(neighbors, neighbors)]
        if weighted:
            sub_len = _invert_weights(sub)
            G_sub = nx.from_numpy_array(sub_len)
            le[i] = nx.global_efficiency(G_sub)
        else:
            G_sub = nx.from_numpy_array((sub > 0).astype(float))
            le[i] = nx.global_efficiency(G_sub)

    return le


# =============================================================================
# RICH CLUB ANALYSIS
# =============================================================================

def compute_rich_club(
    matrix: np.ndarray,
    n_rand: int = 100,
    weighted: bool = False,
    seed: int = 42,
) -> Dict:
    """
    Compute the rich club coefficient and its normalized version.

    The rich club coefficient φ(k) measures the density of connections
    among nodes with degree > k (Colizza et al., 2006). Normalization
    against degree-preserving random null models reveals whether hub
    interconnectivity exceeds chance (van den Heuvel & Sporns, 2011).

    Parameters
    ----------
    matrix : np.ndarray (N, N)
        Connectivity matrix.
    n_rand : int
        Number of random null models for normalization.
    weighted : bool
        If True, compute weighted rich club (Opsahl et al., 2008).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with:
        - k_levels : array of degree thresholds
        - phi : rich club coefficient at each k
        - phi_rand_mean : mean φ from random networks
        - phi_rand_std : std of φ from random networks
        - phi_norm : normalized rich club φ_norm(k) = φ(k) / φ_rand(k)
        - p_values : significance at each k (one-sided test)
        - significant_k : k values where p < 0.05
        - rich_club_regime : (k_min, k_max) of significant rich club

    Notes
    -----
    The binary rich club is computed using NetworkX. For normalization,
    we generate degree-preserving random networks via double-edge swaps
    (Maslov & Sneppen, 2002).
    """
    mat = _prepare_matrix(matrix)
    G = _matrix_to_graph(mat, weighted=False)

    # Compute empirical rich club
    rc = nx.rich_club_coefficient(G, normalized=False)
    if not rc:
        return {"k_levels": np.array([]), "phi": np.array([])}

    k_levels = np.array(sorted(rc.keys()))
    phi = np.array([rc[k] for k in k_levels])

    # Null model normalization
    rng = np.random.RandomState(seed)
    phi_rand_all = np.zeros((n_rand, len(k_levels)))

    for i in range(n_rand):
        G_rand = G.copy()
        # Double-edge swap to preserve degree sequence
        n_swaps = G.number_of_edges() * 10
        try:
            nx.double_edge_swap(G_rand, nswap=n_swaps, max_tries=n_swaps * 10,
                                seed=rng.randint(0, 2**31))
        except nx.NetworkXError:
            pass
        rc_rand = nx.rich_club_coefficient(G_rand, normalized=False)
        for j, k in enumerate(k_levels):
            phi_rand_all[i, j] = rc_rand.get(k, 0.0)

    phi_rand_mean = np.mean(phi_rand_all, axis=0)
    phi_rand_std = np.std(phi_rand_all, axis=0)

    # Normalized rich club
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_norm = np.where(phi_rand_mean > 0, phi / phi_rand_mean, 0.0)

    # One-sided p-values: fraction of null >= empirical
    p_values = np.array([
        np.mean(phi_rand_all[:, j] >= phi[j])
        for j in range(len(k_levels))
    ])

    # Identify significant regime
    sig_mask = p_values < 0.05
    significant_k = k_levels[sig_mask]
    if len(significant_k) > 0:
        rich_club_regime = (int(significant_k.min()), int(significant_k.max()))
    else:
        rich_club_regime = None

    return {
        "k_levels": k_levels,
        "phi": phi,
        "phi_rand_mean": phi_rand_mean,
        "phi_rand_std": phi_rand_std,
        "phi_norm": phi_norm,
        "p_values": p_values,
        "significant_k": significant_k,
        "rich_club_regime": rich_club_regime,
    }


# =============================================================================
# SMALL-WORLD ANALYSIS
# =============================================================================

def compute_small_world(
    matrix: np.ndarray,
    n_rand: int = 100,
    weighted: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Assess small-world properties using sigma and omega coefficients.

    Two complementary measures:
      - **Sigma** (Humphries & Gurney, 2008):
          σ = (C/C_rand) / (L/L_rand)
          σ > 1 indicates small-world organization.
      - **Omega** (Telesford et al., 2011):
          ω = L_rand/L - C/C_lattice
          ω ≈ 0: small-world; ω → -1: lattice-like; ω → +1: random.
          More robust than sigma across network sizes.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    n_rand : int
        Number of random reference networks.
    weighted : bool
    seed : int

    Returns
    -------
    dict with:
        - sigma, omega
        - C_empirical, L_empirical
        - C_rand_mean, L_rand_mean
        - C_lattice (for omega)
        - small_world_index (sigma)
        - interpretation : str

    Notes
    -----
    The omega coefficient is recommended by rsfmri_best_practices.pdf
    as more robust across different network sizes than sigma.
    """
    mat = _prepare_matrix(matrix)

    if weighted:
        m = np.max(mat)
        if m > 0:
            mat = mat / m

    G = _matrix_to_graph(mat, weighted=weighted)
    weight_key = "weight" if weighted else None

    if not nx.is_connected(G):
        # Use largest component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        # Relabel nodes to be contiguous
        G = nx.convert_node_labels_to_integers(G)

    N = G.number_of_nodes()

    # Empirical metrics
    C_emp = nx.average_clustering(G, weight=weight_key)

    if weighted:
        sub_mat = nx.to_numpy_array(G, weight="weight")
        length_mat = _invert_weights(sub_mat)
        G_len = nx.from_numpy_array(length_mat)
        L_emp = nx.average_shortest_path_length(G_len, weight="weight")
    else:
        L_emp = nx.average_shortest_path_length(G)

    # Random reference networks (degree-preserving rewiring)
    rng = np.random.RandomState(seed)
    C_rands = []
    L_rands = []

    for _ in range(n_rand):
        G_rand = G.copy()
        n_swaps = G.number_of_edges() * 10
        try:
            nx.double_edge_swap(G_rand, nswap=n_swaps,
                                max_tries=n_swaps * 10,
                                seed=rng.randint(0, 2**31))
        except nx.NetworkXError:
            pass

        C_rands.append(nx.average_clustering(G_rand, weight=weight_key))
        try:
            if weighted:
                sub_r = nx.to_numpy_array(G_rand, weight="weight")
                len_r = _invert_weights(sub_r)
                G_r_len = nx.from_numpy_array(len_r)
                L_rands.append(
                    nx.average_shortest_path_length(G_r_len, weight="weight")
                )
            else:
                L_rands.append(
                    nx.average_shortest_path_length(G_rand)
                )
        except nx.NetworkXError:
            L_rands.append(np.nan)

    C_rand_mean = float(np.nanmean(C_rands))
    L_rand_mean = float(np.nanmean(L_rands))

    # Sigma coefficient
    if C_rand_mean > 0 and L_rand_mean > 0:
        gamma = C_emp / C_rand_mean  # normalized clustering
        lam = L_emp / L_rand_mean    # normalized path length
        sigma = gamma / lam if lam > 0 else np.nan
    else:
        gamma = np.nan
        lam = np.nan
        sigma = np.nan

    # Omega coefficient (Telesford et al., 2011)
    # C_lattice: approximate clustering of a ring lattice with same
    # N and mean degree
    degrees = np.array([d for _, d in G.degree()])
    k_mean = np.mean(degrees)
    # For a ring lattice, C ≈ 3(k-2) / (4(k-1)) for k >= 2
    if k_mean >= 2:
        C_lattice = 3.0 * (k_mean - 2) / (4.0 * (k_mean - 1))
    else:
        C_lattice = 0.0

    if L_emp > 0 and C_lattice > 0:
        omega = (L_rand_mean / L_emp) - (C_emp / C_lattice)
    else:
        omega = np.nan

    # Interpretation
    if not np.isnan(omega):
        if abs(omega) < 0.3:
            interpretation = "small-world"
        elif omega < -0.3:
            interpretation = "lattice-like (ordered)"
        else:
            interpretation = "random-like"
    else:
        interpretation = "indeterminate"

    return {
        "sigma": float(sigma),
        "omega": float(omega),
        "gamma": float(gamma),          # C/C_rand
        "lambda_ratio": float(lam),      # L/L_rand
        "C_empirical": float(C_emp),
        "L_empirical": float(L_emp),
        "C_rand_mean": float(C_rand_mean),
        "L_rand_mean": float(L_rand_mean),
        "C_lattice": float(C_lattice),
        "n_nodes_analyzed": N,
        "interpretation": interpretation,
    }


# =============================================================================
# SC-FC COUPLING
# =============================================================================

def compute_sc_fc_coupling(
    sc_matrix: np.ndarray,
    fc_matrix: np.ndarray,
    method: str = "spearman",
    regional: bool = True,
) -> Dict:
    """
    Compute structure-function coupling between SC and FC matrices.

    The relationship between structural and functional connectivity
    is a fundamental organizing principle of the brain (Sporns, 2013).
    Empirical R values are typically 0.3-0.6 (Honey et al., 2009).

    Parameters
    ----------
    sc_matrix : np.ndarray (N, N)
        Structural connectivity matrix.
    fc_matrix : np.ndarray (N, N)
        Functional connectivity matrix.
    method : str
        Correlation method: 'pearson', 'spearman', 'kendall'.
    regional : bool
        If True, also compute node-wise SC-FC coupling
        (correlation of each node's SC and FC profiles).

    Returns
    -------
    dict with:
        - global_r, global_p : whole-matrix correlation
        - global_r_connected, global_p_connected : correlation only
          over edges where SC > 0
        - regional_r : per-node correlation (if regional=True)
        - regional_p : per-node p-values
        - mean_regional_r, std_regional_r

    Notes
    -----
    Global coupling is computed over upper-triangle entries.
    Regional coupling for node i = corr(SC[i,:], FC[i,:]) over j≠i,
    providing a map of SC-FC correspondence across brain regions.
    See Baum et al. (2020) Proc Natl Acad Sci for regional approach.
    """
    from ..data import get_upper_triangle

    sc = _prepare_matrix(sc_matrix)
    fc = _prepare_matrix(fc_matrix, remove_negative=False)

    assert sc.shape == fc.shape, \
        f"SC {sc.shape} and FC {fc.shape} must have same dimensions"
    N = sc.shape[0]

    # Correlation function
    if method == "pearson":
        corr_func = stats.pearsonr
    elif method == "spearman":
        corr_func = stats.spearmanr
    elif method == "kendall":
        corr_func = stats.kendalltau
    else:
        raise ValueError(f"Unknown method: {method}")

    # Global SC-FC coupling (all edges)
    sc_triu = get_upper_triangle(sc, k=1)
    fc_triu = get_upper_triangle(fc, k=1)
    r_all, p_all = corr_func(sc_triu, fc_triu)

    # SC-FC coupling only where structural connection exists
    connected_mask = sc_triu > 0
    if np.sum(connected_mask) > 2:
        r_conn, p_conn = corr_func(sc_triu[connected_mask],
                                    fc_triu[connected_mask])
    else:
        r_conn, p_conn = np.nan, np.nan

    results = {
        "global_r": float(r_all),
        "global_p": float(p_all),
        "global_r_connected": float(r_conn),
        "global_p_connected": float(p_conn),
        "method": method,
        "n_edges_total": len(sc_triu),
        "n_edges_connected": int(np.sum(connected_mask)),
    }

    # Regional SC-FC coupling (per-node)
    if regional:
        regional_r = np.zeros(N)
        regional_p = np.ones(N)
        for i in range(N):
            idx = np.setdiff1d(np.arange(N), [i])
            sc_row = sc[i, idx]
            fc_row = fc[i, idx]
            if np.std(sc_row) > 0 and np.std(fc_row) > 0:
                r, p = corr_func(sc_row, fc_row)
                regional_r[i] = float(r)
                regional_p[i] = float(p)
            else:
                regional_r[i] = np.nan
                regional_p[i] = np.nan

        results["regional_r"] = regional_r
        results["regional_p"] = regional_p
        valid = ~np.isnan(regional_r)
        results["mean_regional_r"] = float(np.nanmean(regional_r))
        results["std_regional_r"] = float(np.nanstd(regional_r))
        results["median_regional_r"] = float(np.nanmedian(regional_r))

    return results


# =============================================================================
# MULTI-THRESHOLD ROBUSTNESS (AUC)
# =============================================================================

def compute_metrics_across_thresholds(
    matrix: np.ndarray,
    thresholds: Optional[List[float]] = None,
    metrics_func: str = "global",
    weighted: bool = True,
) -> Dict:
    """
    Compute graph metrics across a range of density thresholds.

    Following van den Heuvel et al. (2017, NeuroImage), reporting
    results across multiple thresholds and computing area-under-curve
    (AUC) improves robustness and avoids threshold-dependent bias.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    thresholds : list of float
        Density thresholds (fraction of edges to keep).
        Default: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30].
    metrics_func : str
        'global' or 'nodal'.
    weighted : bool

    Returns
    -------
    dict with:
        - thresholds : list of density values
        - metrics_per_threshold : list of metric dicts
        - auc : dict of AUC values for each scalar metric

    Notes
    -----
    AUC is computed using the trapezoidal rule over the threshold range,
    providing a single summary value robust to threshold choice.
    """
    from ..data import threshold_matrix

    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    mat = _prepare_matrix(matrix)
    metrics_list = []

    for thr in thresholds:
        thr_mat = threshold_matrix(mat, method="density", value=thr,
                                   absolute=True)
        if metrics_func == "global":
            m = compute_global_metrics(thr_mat, weighted=weighted)
        elif metrics_func == "nodal":
            m = compute_nodal_metrics(thr_mat, weighted=weighted)
        else:
            raise ValueError(f"Unknown metrics_func: {metrics_func}")
        metrics_list.append(m)

    # Compute AUC for scalar metrics
    auc = {}
    scalar_keys = [k for k, v in metrics_list[0].items()
                   if isinstance(v, (int, float)) and not np.isnan(v)]

    for key in scalar_keys:
        values = []
        for m in metrics_list:
            val = m.get(key, np.nan)
            if isinstance(val, (int, float)):
                values.append(val)
            else:
                values.append(np.nan)
        values = np.array(values, dtype=float)
        valid = ~np.isnan(values)
        if np.sum(valid) >= 2:
            auc[key] = float(np.trapz(values[valid],
                                       np.array(thresholds)[valid]))
        else:
            auc[key] = np.nan

    return {
        "thresholds": thresholds,
        "metrics_per_threshold": metrics_list,
        "auc": auc,
    }


# =============================================================================
# HUB CLASSIFICATION
# =============================================================================

def classify_nodes(
    matrix: np.ndarray,
    community_labels: np.ndarray,
    weighted: bool = True,
    hub_threshold: float = 1.0,
    pc_threshold: float = 0.3,
) -> Dict:
    """
    Classify nodes into roles based on within-module degree z-score
    and participation coefficient (Guimerà & Amaral, 2005).

    Node roles:
      - Hub (z > hub_threshold):
        - Connector hub: PC > pc_threshold (links multiple modules)
        - Provincial hub: PC ≤ pc_threshold (dominant within one module)
      - Non-hub (z ≤ hub_threshold):
        - Connector: PC > pc_threshold
        - Peripheral: PC ≤ pc_threshold

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    community_labels : np.ndarray (N,)
        Module assignment for each node.
    weighted : bool
    hub_threshold : float
        Within-module degree z-score threshold for hub classification.
    pc_threshold : float
        Participation coefficient threshold.

    Returns
    -------
    dict with:
        - within_module_degree_z : np.ndarray (N,)
        - participation_coeff : np.ndarray (N,)
        - roles : list of str ('connector_hub', 'provincial_hub',
                               'connector_non_hub', 'peripheral')
        - hub_mask : bool array
        - role_counts : dict
    """
    mat = _prepare_matrix(matrix)
    N = mat.shape[0]
    communities = np.asarray(community_labels)

    # Within-module degree z-score
    wmd_z = np.zeros(N)
    unique_modules = np.unique(communities)
    for mod in unique_modules:
        idx = np.where(communities == mod)[0]
        if len(idx) < 2:
            continue
        for i in idx:
            if weighted:
                ki_within = np.sum(mat[i, idx])
            else:
                ki_within = np.sum(mat[i, idx] > 0)
            # Module mean and std
            k_module = []
            for j in idx:
                if weighted:
                    k_module.append(np.sum(mat[j, idx]))
                else:
                    k_module.append(np.sum(mat[j, idx] > 0))
            k_module = np.array(k_module)
            mu = np.mean(k_module)
            sigma = np.std(k_module)
            if sigma > 0:
                wmd_z[i] = (ki_within - mu) / sigma
            else:
                wmd_z[i] = 0.0

    # Participation coefficient
    pc = np.zeros(N)
    for i in range(N):
        if weighted:
            ki_total = np.sum(mat[i, :])
        else:
            ki_total = np.sum(mat[i, :] > 0)
        if ki_total == 0:
            pc[i] = 0.0
            continue
        sum_sq = 0.0
        for mod in unique_modules:
            idx = np.where(communities == mod)[0]
            if weighted:
                ki_mod = np.sum(mat[i, idx])
            else:
                ki_mod = np.sum(mat[i, idx] > 0)
            sum_sq += (ki_mod / ki_total) ** 2
        pc[i] = 1.0 - sum_sq

    # Classification
    roles = []
    for i in range(N):
        is_hub = wmd_z[i] > hub_threshold
        is_connector = pc[i] > pc_threshold
        if is_hub and is_connector:
            roles.append("connector_hub")
        elif is_hub and not is_connector:
            roles.append("provincial_hub")
        elif not is_hub and is_connector:
            roles.append("connector_non_hub")
        else:
            roles.append("peripheral")

    hub_mask = np.array([r.endswith("hub") for r in roles])

    # Count roles
    role_counts = {}
    for r in ["connector_hub", "provincial_hub", "connector_non_hub", "peripheral"]:
        role_counts[r] = sum(1 for x in roles if x == r)

    return {
        "within_module_degree_z": wmd_z,
        "participation_coeff": pc,
        "roles": roles,
        "hub_mask": hub_mask,
        "role_counts": role_counts,
        "hub_threshold": hub_threshold,
        "pc_threshold": pc_threshold,
    }


# =============================================================================
# MASTER ANALYSIS FUNCTION
# =============================================================================

def analyze_network(
    matrix: np.ndarray,
    sc_matrix: Optional[np.ndarray] = None,
    community_labels: Optional[np.ndarray] = None,
    weighted: bool = True,
    compute_rich_club_flag: bool = True,
    compute_small_world_flag: bool = True,
    compute_sc_fc_flag: bool = True,
    compute_multi_threshold: bool = True,
    n_rand: int = 100,
    seed: int = 42,
) -> Dict:
    """
    Run a comprehensive graph analysis pipeline on a connectivity matrix.

    This master function orchestrates all network metrics and produces
    a unified results dictionary suitable for downstream analysis and
    visualization.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
        Primary connectivity matrix (typically FC).
    sc_matrix : np.ndarray, optional
        Structural connectivity matrix for SC-FC coupling.
    community_labels : np.ndarray, optional
        Pre-computed community assignments for hub classification.
    weighted : bool
    compute_rich_club_flag : bool
    compute_small_world_flag : bool
    compute_sc_fc_flag : bool
        Only if sc_matrix is provided.
    compute_multi_threshold : bool
    n_rand : int
    seed : int

    Returns
    -------
    dict with sections:
        - 'global' : global metrics dict
        - 'nodal' : nodal metrics dict
        - 'rich_club' : rich club analysis (optional)
        - 'small_world' : small-world assessment (optional)
        - 'sc_fc_coupling' : SC-FC coupling (optional)
        - 'multi_threshold' : AUC analysis (optional)
        - 'hub_classification' : node roles (optional)
    """
    results = {}

    # Global metrics
    results["global"] = compute_global_metrics(
        matrix, weighted=weighted
    )

    # Nodal metrics
    results["nodal"] = compute_nodal_metrics(
        matrix, weighted=weighted
    )

    # Rich club
    if compute_rich_club_flag:
        results["rich_club"] = compute_rich_club(
            matrix, n_rand=n_rand, seed=seed
        )

    # Small-world
    if compute_small_world_flag:
        results["small_world"] = compute_small_world(
            matrix, n_rand=n_rand, weighted=weighted, seed=seed
        )

    # SC-FC coupling
    if compute_sc_fc_flag and sc_matrix is not None:
        results["sc_fc_coupling"] = compute_sc_fc_coupling(
            sc_matrix, matrix, method="spearman", regional=True
        )

    # Multi-threshold robustness
    if compute_multi_threshold:
        results["multi_threshold"] = compute_metrics_across_thresholds(
            matrix, weighted=weighted
        )

    # Hub classification (requires community labels)
    if community_labels is not None:
        results["hub_classification"] = classify_nodes(
            matrix, community_labels, weighted=weighted
        )

    return results
