# -*- coding: utf-8 -*-
"""
sars.graph_analysis.community
============================================

Community detection and modular network analysis.

Implements multiple community detection algorithms for brain connectivity
networks, including consensus clustering to handle the stochastic nature
of modularity optimization. Provides modular metrics (participation
coefficient, within-module degree) and Network-Based Statistic (NBS)
for statistical comparison of connectivity matrices.

Algorithms
----------
- **Louvain**: Greedy modularity optimization (Blondel et al., 2008).
  Default choice for brain network analysis.
- **Leiden**: Improved community detection guaranteeing connected
  communities (Traag et al., 2019).
- **Consensus clustering**: Addresses degeneracy of modularity landscape
  by aggregating multiple partitions (Lancichinetti & Fortunato, 2012).

References
----------
- Blondel et al. (2008). J Stat Mech P10008. Louvain algorithm.
- Traag, Waltman & van Eck (2019). Sci Rep 9:5233. Leiden algorithm.
- Lancichinetti & Fortunato (2012). Sci Rep 2:336. Consensus clustering.
- Newman (2006). Phys Rev E 74:036104. Modularity.
- Guimerà & Amaral (2005). Nature 433:895-900. Module roles.
- Zalesky, Fornito & Bullmore (2010). NeuroImage 53:1197-1207. NBS.
- Rubinov & Sporns (2010). NeuroImage 52:1059-1069.
"""

import numpy as np
import networkx as nx
from scipy import stats, ndimage
from typing import Optional, Dict, List, Tuple, Union
import warnings


# =============================================================================
# LOUVAIN COMMUNITY DETECTION
# =============================================================================

def detect_communities_louvain(
    matrix: np.ndarray,
    resolution: float = 1.0,
    n_runs: int = 100,
    seed: int = 42,
    weighted: bool = True,
) -> Dict:
    """
    Detect communities using the Louvain algorithm.

    Runs multiple iterations and returns the partition with the highest
    modularity, mitigating stochastic variability.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
        Connectivity matrix (non-negative, undirected).
    resolution : float
        Resolution parameter γ for modularity. Higher values produce
        more, smaller communities. γ=1.0 is the standard Newman-Girvan
        modularity. For brain networks, γ ∈ [0.8, 1.5] is typical.
    n_runs : int
        Number of independent runs.
    seed : int
    weighted : bool

    Returns
    -------
    dict with:
        - labels : np.ndarray (N,) — best partition (0-indexed)
        - modularity : float — Q value of best partition
        - n_communities : int
        - community_sizes : list of int
        - all_modularities : list — Q for each run
        - labels_all : list of arrays — all partitions
        - resolution : float

    Notes
    -----
    Requires `python-louvain` (pip install python-louvain).
    The Louvain algorithm is the most widely used community detection
    method in neuroimaging (Rubinov & Sporns, 2010).
    """
    from .network_metrics import _prepare_matrix, _matrix_to_graph

    mat = _prepare_matrix(matrix)
    G = _matrix_to_graph(mat, weighted=weighted)
    weight_key = "weight" if weighted else None

    try:
        import community as community_louvain
    except ImportError:
        raise ImportError(
            "python-louvain is required for Louvain community detection.\n"
            "Install: pip install python-louvain"
        )

    rng = np.random.RandomState(seed)
    best_Q = -np.inf
    best_partition = None
    all_Q = []
    all_labels = []

    for i in range(n_runs):
        partition = community_louvain.best_partition(
            G, weight=weight_key, resolution=resolution,
            random_state=rng.randint(0, 2**31),
        )
        Q = community_louvain.modularity(partition, G, weight=weight_key)

        # Convert partition dict to array
        N = mat.shape[0]
        labels = np.array([partition[node] for node in range(N)])
        all_labels.append(labels)
        all_Q.append(Q)

        if Q > best_Q:
            best_Q = Q
            best_partition = labels.copy()

    # Community statistics
    unique, counts = np.unique(best_partition, return_counts=True)
    # Re-index to be 0-based contiguous
    relabel = {old: new for new, old in enumerate(unique)}
    best_partition = np.array([relabel[x] for x in best_partition])
    n_comm = len(unique)

    return {
        "labels": best_partition,
        "modularity": float(best_Q),
        "n_communities": n_comm,
        "community_sizes": sorted(counts.tolist(), reverse=True),
        "all_modularities": all_Q,
        "labels_all": all_labels,
        "resolution": resolution,
        "mean_modularity": float(np.mean(all_Q)),
        "std_modularity": float(np.std(all_Q)),
    }


# =============================================================================
# LEIDEN COMMUNITY DETECTION
# =============================================================================

def detect_communities_leiden(
    matrix: np.ndarray,
    resolution: float = 1.0,
    n_runs: int = 100,
    seed: int = 42,
    weighted: bool = True,
) -> Dict:
    """
    Detect communities using the Leiden algorithm.

    The Leiden algorithm (Traag et al., 2019) improves upon Louvain by
    guaranteeing that all detected communities are connected. This is
    particularly important for sparse brain networks.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    resolution : float
    n_runs : int
    seed : int
    weighted : bool

    Returns
    -------
    dict: Same structure as detect_communities_louvain.

    Notes
    -----
    Requires `leidenalg` and `igraph` packages.
    """
    from .network_metrics import _prepare_matrix

    mat = _prepare_matrix(matrix)
    N = mat.shape[0]

    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError(
            "leidenalg and python-igraph are required for Leiden detection.\n"
            "Install: pip install leidenalg python-igraph"
        )

    # Build igraph graph
    edges = []
    weights = []
    for i in range(N):
        for j in range(i + 1, N):
            if mat[i, j] > 0:
                edges.append((i, j))
                weights.append(float(mat[i, j]))

    G_ig = ig.Graph(n=N, edges=edges, directed=False)
    if weighted and weights:
        G_ig.es["weight"] = weights

    rng = np.random.RandomState(seed)
    best_Q = -np.inf
    best_partition = None
    all_Q = []
    all_labels = []

    for _ in range(n_runs):
        part = leidenalg.find_partition(
            G_ig,
            leidenalg.CPMVertexPartition if resolution != 1.0
            else leidenalg.ModularityVertexPartition,
            weights="weight" if weighted else None,
            seed=rng.randint(0, 2**31),
            **({"resolution_parameter": resolution}
               if resolution != 1.0 else {}),
        )

        labels = np.array(part.membership)
        Q = part.modularity
        all_labels.append(labels)
        all_Q.append(Q)

        if Q > best_Q:
            best_Q = Q
            best_partition = labels.copy()

    unique, counts = np.unique(best_partition, return_counts=True)
    relabel = {old: new for new, old in enumerate(unique)}
    best_partition = np.array([relabel[x] for x in best_partition])

    return {
        "labels": best_partition,
        "modularity": float(best_Q),
        "n_communities": len(unique),
        "community_sizes": sorted(counts.tolist(), reverse=True),
        "all_modularities": all_Q,
        "labels_all": all_labels,
        "resolution": resolution,
        "mean_modularity": float(np.mean(all_Q)),
        "std_modularity": float(np.std(all_Q)),
    }


# =============================================================================
# CONSENSUS CLUSTERING
# =============================================================================

def consensus_clustering(
    matrix: np.ndarray,
    n_partitions: int = 200,
    consensus_threshold: float = 0.5,
    max_iterations: int = 50,
    algorithm: str = "louvain",
    resolution: float = 1.0,
    seed: int = 42,
    weighted: bool = True,
) -> Dict:
    """
    Consensus clustering to address degeneracy of modularity.

    Multiple partitions are aggregated into a co-classification matrix
    D(i,j) = fraction of partitions where nodes i and j are assigned
    to the same community. This matrix is then thresholded and itself
    partitioned until convergence, yielding a stable consensus.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    n_partitions : int
        Number of initial partitions to generate.
    consensus_threshold : float
        Threshold τ for the agreement matrix. Entries D(i,j) < τ are
        set to 0. Lancichinetti & Fortunato (2012) recommend τ=0.5.
    max_iterations : int
        Maximum iterations for consensus convergence.
    algorithm : str
        'louvain' or 'leiden'.
    resolution : float
    seed : int
    weighted : bool

    Returns
    -------
    dict with:
        - labels : np.ndarray — consensus partition
        - modularity : float — Q of consensus partition
        - n_communities : int
        - community_sizes : list
        - agreement_matrix : np.ndarray (N, N) — co-classification matrix D
        - n_iterations : int — iterations until convergence
        - converged : bool

    Notes
    -----
    Consensus clustering is strongly recommended for brain network
    analysis because modularity maximization is known to produce
    many near-degenerate solutions (Good et al., 2010). The consensus
    approach (Lancichinetti & Fortunato, 2012) resolves this by
    identifying the stable core of community structure.
    """
    N = matrix.shape[0]

    # Step 1: Generate n_partitions
    if algorithm == "louvain":
        result = detect_communities_louvain(
            matrix, resolution=resolution, n_runs=n_partitions,
            seed=seed, weighted=weighted,
        )
    elif algorithm == "leiden":
        result = detect_communities_leiden(
            matrix, resolution=resolution, n_runs=n_partitions,
            seed=seed, weighted=weighted,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    all_labels = result["labels_all"]

    # Step 2: Build co-classification (agreement) matrix
    D = _build_agreement_matrix(all_labels, N)

    # Step 3: Iterative consensus
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iterations + 1):
        # Threshold the agreement matrix
        D_thr = D.copy()
        D_thr[D_thr < consensus_threshold] = 0
        np.fill_diagonal(D_thr, 0)

        # Check if D is already a perfect partition (all entries 0 or 1)
        unique_vals = np.unique(D_thr[np.triu_indices(N, k=1)])
        if len(unique_vals) <= 2 and np.all(
            (unique_vals == 0) | (np.isclose(unique_vals, 1.0))
        ):
            converged = True
            break

        # Re-partition the agreement matrix
        if algorithm == "louvain":
            sub_result = detect_communities_louvain(
                D_thr, resolution=resolution, n_runs=n_partitions,
                seed=seed + n_iter, weighted=True,
            )
        else:
            sub_result = detect_communities_leiden(
                D_thr, resolution=resolution, n_runs=n_partitions,
                seed=seed + n_iter, weighted=True,
            )
        all_labels = sub_result["labels_all"]
        D = _build_agreement_matrix(all_labels, N)

    # Extract final partition from converged D
    if converged:
        # Labels from thresholded D: connected components
        D_bin = (D_thr > 0).astype(int)
        labeled_array, n_features = ndimage.label(D_bin)
        # Use first row membership for each component
        labels = np.zeros(N, dtype=int)
        G_final = nx.from_numpy_array(D_bin)
        for comp_idx, comp in enumerate(nx.connected_components(G_final)):
            for node in comp:
                labels[node] = comp_idx
    else:
        # Use last best partition
        labels = sub_result["labels"]

    # Compute final modularity
    from .network_metrics import _prepare_matrix, _matrix_to_graph
    mat = _prepare_matrix(matrix)
    G = _matrix_to_graph(mat, weighted=weighted)
    weight_key = "weight" if weighted else None

    try:
        import community as community_louvain
        partition_dict = {i: int(labels[i]) for i in range(N)}
        Q = community_louvain.modularity(partition_dict, G, weight=weight_key)
    except ImportError:
        Q = _compute_modularity_manual(mat, labels)

    unique, counts = np.unique(labels, return_counts=True)

    return {
        "labels": labels,
        "modularity": float(Q),
        "n_communities": len(unique),
        "community_sizes": sorted(counts.tolist(), reverse=True),
        "agreement_matrix": D,
        "n_iterations": n_iter,
        "converged": converged,
    }


def _build_agreement_matrix(
    all_labels: List[np.ndarray],
    N: int,
) -> np.ndarray:
    """
    Build co-classification (agreement) matrix from multiple partitions.

    D(i,j) = fraction of partitions where i and j share the same community.
    """
    D = np.zeros((N, N))
    n_part = len(all_labels)
    for labels in all_labels:
        for mod in np.unique(labels):
            idx = np.where(labels == mod)[0]
            D[np.ix_(idx, idx)] += 1
    D /= n_part
    np.fill_diagonal(D, 0)
    return D


def _compute_modularity_manual(
    matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute Newman modularity Q manually.

    Q = (1/2m) Σ_ij [A_ij - k_i*k_j/(2m)] δ(c_i, c_j)
    """
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    m = np.sum(mat) / 2
    if m == 0:
        return 0.0
    k = np.sum(mat, axis=1)
    Q = 0.0
    for mod in np.unique(labels):
        idx = np.where(labels == mod)[0]
        for i in idx:
            for j in idx:
                Q += mat[i, j] - k[i] * k[j] / (2 * m)
    return float(Q / (2 * m))


# =============================================================================
# PARTICIPATION COEFFICIENT AND WITHIN-MODULE DEGREE
# =============================================================================

def compute_participation_coefficient(
    matrix: np.ndarray,
    community_labels: np.ndarray,
    weighted: bool = True,
) -> np.ndarray:
    """
    Compute the participation coefficient for each node.

    The participation coefficient P_i quantifies the diversity of a
    node's connections across modules (Guimerà & Amaral, 2005):

        P_i = 1 - Σ_s (k_is / k_i)²

    where k_is = number (or strength) of connections from node i to
    nodes in module s, and k_i = total degree (or strength) of node i.

    P_i ≈ 0: node connects mainly within its own module (provincial)
    P_i → 1: node distributes connections evenly across all modules (connector)

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    community_labels : np.ndarray (N,)
    weighted : bool

    Returns
    -------
    np.ndarray (N,): participation coefficient for each node.
    """
    from .network_metrics import _prepare_matrix

    mat = _prepare_matrix(matrix)
    N = mat.shape[0]
    pc = np.zeros(N)
    modules = np.unique(community_labels)

    for i in range(N):
        if weighted:
            ki = np.sum(mat[i, :])
        else:
            ki = np.sum(mat[i, :] > 0)
        if ki == 0:
            continue
        sum_sq = 0.0
        for mod in modules:
            idx = np.where(community_labels == mod)[0]
            if weighted:
                ki_s = np.sum(mat[i, idx])
            else:
                ki_s = np.sum(mat[i, idx] > 0)
            sum_sq += (ki_s / ki) ** 2
        pc[i] = 1.0 - sum_sq

    return pc


def compute_within_module_degree(
    matrix: np.ndarray,
    community_labels: np.ndarray,
    weighted: bool = True,
) -> np.ndarray:
    """
    Compute the within-module degree z-score for each node.

    The within-module degree z-score quantifies how well-connected
    a node is to other nodes in its own module, relative to other
    nodes in that module (Guimerà & Amaral, 2005):

        z_i = (k_i(s_i) - <k(s_i)>) / σ(k(s_i))

    where k_i(s_i) = intra-module connections of node i, and
    <k(s_i)>, σ(k(s_i)) = mean and std of intra-module connections
    for all nodes in module s_i.

    z_i > ~1.0 typically indicates a hub within its module.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    community_labels : np.ndarray (N,)
    weighted : bool

    Returns
    -------
    np.ndarray (N,): within-module degree z-score.
    """
    from .network_metrics import _prepare_matrix

    mat = _prepare_matrix(matrix)
    N = mat.shape[0]
    z = np.zeros(N)
    modules = np.unique(community_labels)

    for mod in modules:
        idx = np.where(community_labels == mod)[0]
        if len(idx) < 2:
            continue
        # Intra-module degree/strength for each node in module
        k_intra = np.zeros(len(idx))
        for ii, i in enumerate(idx):
            if weighted:
                k_intra[ii] = np.sum(mat[i, idx])
            else:
                k_intra[ii] = np.sum(mat[i, idx] > 0)
        mu = np.mean(k_intra)
        sigma = np.std(k_intra)
        for ii, i in enumerate(idx):
            if sigma > 0:
                z[i] = (k_intra[ii] - mu) / sigma

    return z


# =============================================================================
# NETWORK-BASED STATISTIC (NBS)
# =============================================================================

def compute_nbs(
    matrices_group1: np.ndarray,
    matrices_group2: np.ndarray,
    threshold: float = 3.0,
    n_perm: int = 5000,
    test: str = "t-test",
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """
    Network-Based Statistic (NBS) for mass-univariate testing of
    connectivity differences between two groups.

    NBS (Zalesky, Fornito & Bullmore, 2010) controls FWER at the
    network (component) level rather than the edge level, providing
    greater statistical power for detecting distributed effects.

    Algorithm:
      1. Perform edge-wise t-tests between groups.
      2. Threshold the t-statistic map at the primary threshold.
      3. Identify connected components in the supra-threshold network.
      4. Permutation testing: shuffle group labels, repeat steps 1-3,
         record the size of the largest component. Build null distribution.
      5. Compare empirical component sizes to the null to obtain p-values.

    Parameters
    ----------
    matrices_group1 : np.ndarray (n1, N, N)
        FC/SC matrices for group 1.
    matrices_group2 : np.ndarray (n2, N, N)
        FC/SC matrices for group 2.
    threshold : float
        Primary t-statistic threshold for supra-threshold network.
        Common values: 2.5-3.5 for t-tests.
    n_perm : int
        Number of permutations for FWER correction.
    test : str
        'ttest' or 't-test' for independent samples t-test.
    alpha : float
        Significance level.
    seed : int

    Returns
    -------
    dict with:
        - t_stats : np.ndarray (N, N) — edge-wise t-statistics
        - p_values_uncorrected : np.ndarray (N, N)
        - components : list of dicts, each with:
          - edges : list of (i, j) tuples
          - size : int (number of edges)
          - nodes : set of node indices
          - p_corrected : float (FWER-corrected p-value)
        - n_significant_components : int
        - null_distribution : np.ndarray of max component sizes
        - threshold : float
        - n_perm : int

    Notes
    -----
    NBS is analogous to cluster-based methods in voxel-wise analysis.
    The primary threshold choice affects sensitivity: lower thresholds
    detect more distributed (weaker) effects; higher thresholds detect
    focal (stronger) effects. Zalesky et al. (2010) recommend exploring
    multiple threshold values.
    """
    n1 = matrices_group1.shape[0]
    n2 = matrices_group2.shape[0]
    N = matrices_group1.shape[1]

    assert matrices_group1.shape[1:] == (N, N)
    assert matrices_group2.shape[1:] == (N, N)

    rng = np.random.RandomState(seed)

    # Step 1: Edge-wise statistics
    t_stats, p_uncorr = _edgewise_ttest(matrices_group1, matrices_group2)

    # Step 2-3: Supra-threshold components
    components = _find_supra_threshold_components(t_stats, threshold)

    # Step 4: Permutation testing
    all_matrices = np.concatenate([matrices_group1, matrices_group2], axis=0)
    n_total = n1 + n2
    null_max_sizes = np.zeros(n_perm)

    for perm_i in range(n_perm):
        # Shuffle group labels
        perm_idx = rng.permutation(n_total)
        perm_g1 = all_matrices[perm_idx[:n1]]
        perm_g2 = all_matrices[perm_idx[n1:]]
        t_perm, _ = _edgewise_ttest(perm_g1, perm_g2)
        perm_components = _find_supra_threshold_components(t_perm, threshold)
        if perm_components:
            null_max_sizes[perm_i] = max(c["size"] for c in perm_components)

    # Step 5: FWER-corrected p-values
    for comp in components:
        comp["p_corrected"] = float(
            np.mean(null_max_sizes >= comp["size"])
        )

    n_sig = sum(1 for c in components if c["p_corrected"] < alpha)

    return {
        "t_stats": t_stats,
        "p_values_uncorrected": p_uncorr,
        "components": components,
        "n_significant_components": n_sig,
        "null_distribution": null_max_sizes,
        "threshold": threshold,
        "n_perm": n_perm,
        "alpha": alpha,
        "n_group1": n1,
        "n_group2": n2,
    }


def _edgewise_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Edge-wise independent samples t-test.

    Parameters
    ----------
    group1 : (n1, N, N)
    group2 : (n2, N, N)

    Returns
    -------
    t_stats, p_values : np.ndarray (N, N)
    """
    N = group1.shape[1]
    t_stats = np.zeros((N, N))
    p_values = np.ones((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            x = group1[:, i, j]
            y = group2[:, i, j]
            if np.std(x) == 0 and np.std(y) == 0:
                continue
            t, p = stats.ttest_ind(x, y, equal_var=False)
            t_stats[i, j] = t
            t_stats[j, i] = t
            p_values[i, j] = p
            p_values[j, i] = p

    return t_stats, p_values


def _find_supra_threshold_components(
    t_stats: np.ndarray,
    threshold: float,
) -> List[Dict]:
    """
    Find connected components in the supra-threshold t-statistic network.
    """
    N = t_stats.shape[0]
    # Apply threshold (both tails)
    supra = np.abs(t_stats) > threshold
    np.fill_diagonal(supra, False)

    # Find connected components using NetworkX
    G = nx.from_numpy_array(supra.astype(int))
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    components = []
    for cc in nx.connected_components(G):
        subG = G.subgraph(cc)
        edges = list(subG.edges())
        components.append({
            "edges": edges,
            "size": len(edges),
            "nodes": set(cc),
            "n_nodes": len(cc),
        })

    # Sort by size (largest first)
    components.sort(key=lambda x: x["size"], reverse=True)
    return components


# =============================================================================
# MASTER COMMUNITY ANALYSIS FUNCTION
# =============================================================================

def analyze_communities(
    matrix: np.ndarray,
    algorithm: str = "louvain",
    resolution: float = 1.0,
    use_consensus: bool = True,
    n_partitions: int = 200,
    weighted: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Run comprehensive community analysis on a connectivity matrix.

    Performs community detection, computes modular metrics (participation
    coefficient, within-module degree z-score), and classifies nodes
    into functional roles.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    algorithm : str
        'louvain' or 'leiden'.
    resolution : float
    use_consensus : bool
        If True, perform consensus clustering (recommended).
    n_partitions : int
    weighted : bool
    seed : int

    Returns
    -------
    dict with:
        - community_detection : dict (labels, modularity, etc.)
        - participation_coeff : np.ndarray (N,)
        - within_module_degree_z : np.ndarray (N,)
        - node_roles : dict (from classify_nodes)
        - inter_module_connectivity : np.ndarray (M, M)
    """
    from .network_metrics import _prepare_matrix, classify_nodes

    mat = _prepare_matrix(matrix)
    N = mat.shape[0]

    # Community detection
    if use_consensus:
        comm_result = consensus_clustering(
            matrix, n_partitions=n_partitions, algorithm=algorithm,
            resolution=resolution, seed=seed, weighted=weighted,
        )
    else:
        if algorithm == "louvain":
            comm_result = detect_communities_louvain(
                matrix, resolution=resolution, n_runs=n_partitions,
                seed=seed, weighted=weighted,
            )
        elif algorithm == "leiden":
            comm_result = detect_communities_leiden(
                matrix, resolution=resolution, n_runs=n_partitions,
                seed=seed, weighted=weighted,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    labels = comm_result["labels"]

    # Modular metrics
    pc = compute_participation_coefficient(matrix, labels, weighted=weighted)
    wmd_z = compute_within_module_degree(matrix, labels, weighted=weighted)

    # Node classification
    roles = classify_nodes(matrix, labels, weighted=weighted)

    # Inter-module connectivity matrix
    n_comm = comm_result["n_communities"]
    inter_mod = np.zeros((n_comm, n_comm))
    for m1 in range(n_comm):
        idx1 = np.where(labels == m1)[0]
        for m2 in range(m1, n_comm):
            idx2 = np.where(labels == m2)[0]
            if m1 == m2:
                # Intra-module: average within-module connectivity
                sub = mat[np.ix_(idx1, idx2)]
                np.fill_diagonal(sub, 0)
                n_pairs = len(idx1) * (len(idx1) - 1)
                inter_mod[m1, m2] = np.sum(sub) / n_pairs if n_pairs > 0 else 0
            else:
                # Inter-module: average between-module connectivity
                sub = mat[np.ix_(idx1, idx2)]
                n_pairs = len(idx1) * len(idx2)
                val = np.sum(sub) / n_pairs if n_pairs > 0 else 0
                inter_mod[m1, m2] = val
                inter_mod[m2, m1] = val

    results = {
        "community_detection": comm_result,
        "participation_coeff": pc,
        "within_module_degree_z": wmd_z,
        "node_roles": roles,
        "inter_module_connectivity": inter_mod,
    }

    return results
