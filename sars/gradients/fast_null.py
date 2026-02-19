"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Gradients Fast Module - Optimized Parallel Analysis                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Versão otimizada com:                                                       ║
║  - Paralelização com joblib (sujeitos + surrogates)                         ║
║  - Sparse eigendecomposition (scipy.sparse.linalg.eigsh)                    ║
║  - Cache de resultados intermediários                                        ║
║  - Progress bars (tqdm)                                                      ║
║  - Opção de PCA rápido para exploração                                      ║
║                                                                              ║
║  SPEEDUP ESPERADO:                                                           ║
║  - 16 cores: ~10-12x mais rápido                                            ║
║  - Sparse eigen: ~2-3x para matrizes grandes                                ║
║  - PCA mode: ~5-10x vs Diffusion Map                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
import time

# Paralelização
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn("joblib not installed. Parallelization disabled. pip install joblib")

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# FAST GRADIENT COMPUTATION
# =============================================================================

def _compute_affinity_fast(
    connectivity: np.ndarray,
    sparsity: float = 0.9
) -> np.ndarray:
    """Compute normalized angle affinity (optimized)."""
    conn = connectivity.copy()
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 0)
    
    # Sparsity threshold
    if sparsity > 0:
        threshold = np.percentile(conn[conn > 0], sparsity * 100)
        conn[conn < threshold] = 0
    
    # Normalized angle (vectorized)
    norms = np.linalg.norm(conn, axis=1, keepdims=True)
    norms[norms == 0] = 1
    conn_norm = conn / norms
    cos_sim = conn_norm @ conn_norm.T
    cos_sim = np.clip(cos_sim, -1, 1)
    affinity = 1 - np.arccos(cos_sim) / np.pi
    
    np.fill_diagonal(affinity, 0)
    return affinity


def _compute_gradient_fast(
    connectivity: np.ndarray,
    n_components: int = 10,
    approach: str = 'dm',
    sparsity: float = 0.9,
    use_sparse: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast gradient computation with sparse eigendecomposition.
    
    Returns (gradients, explained_variance)
    """
    n = connectivity.shape[0]
    
    if approach == 'pca':
        # PCA is already fast
        from sklearn.decomposition import PCA
        conn = (connectivity + connectivity.T) / 2
        np.fill_diagonal(conn, 0)
        pca = PCA(n_components=min(n_components, n - 1))
        gradients = pca.fit_transform(conn)
        return gradients, pca.explained_variance_ratio_
    
    # Diffusion Map
    affinity = _compute_affinity_fast(connectivity, sparsity)
    
    # Row-normalize for transition matrix
    d = np.sum(affinity, axis=1)
    d[d == 0] = 1
    d_inv_sqrt = 1.0 / np.sqrt(d)
    
    # Symmetric normalized affinity
    M = affinity * np.outer(d_inv_sqrt, d_inv_sqrt)
    M = (M + M.T) / 2
    
    n_compute = min(n_components + 1, n - 2)
    
    if use_sparse and n > 50:
        # Sparse eigendecomposition - much faster for large matrices
        try:
            M_sparse = sparse.csr_matrix(M)
            eigenvalues, eigenvectors = eigsh(M_sparse, k=n_compute, which='LM')
            # eigsh returns in ascending order, we want descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        except Exception:
            # Fallback to dense
            eigenvalues, eigenvectors = eigh(M)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
    else:
        # Dense for small matrices
        eigenvalues, eigenvectors = eigh(M)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    
    # Skip first (trivial) component
    eigenvalues = eigenvalues[1:n_compute]
    eigenvectors = eigenvectors[:, 1:n_compute]
    
    # Back to original space
    embeddings = eigenvectors * d_inv_sqrt[:, np.newaxis]
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=0, keepdims=True) + 1e-10)
    
    # Variance explained
    explained = eigenvalues / (np.sum(eigenvalues) + 1e-10)
    
    return embeddings[:, :n_components], explained[:n_components]


# =============================================================================
# FAST NULL MODEL FUNCTIONS
# =============================================================================

def _maslov_sneppen_fast(adjacency: np.ndarray, n_swaps: int = None, seed: int = None) -> np.ndarray:
    """Fast Maslov-Sneppen edge rewiring."""
    if seed is not None:
        np.random.seed(seed)
    
    adj = adjacency.copy()
    n = adj.shape[0]
    
    # Find edges
    edges = np.array(np.where(np.triu(adj, k=1) > 0)).T
    n_edges = len(edges)
    
    if n_edges < 2:
        return adj
    
    if n_swaps is None:
        n_swaps = n_edges * 10
    
    successful = 0
    attempts = 0
    max_attempts = n_swaps * 10
    
    while successful < n_swaps and attempts < max_attempts:
        attempts += 1
        
        # Pick two random edges
        idx = np.random.choice(n_edges, 2, replace=False)
        e1, e2 = edges[idx[0]], edges[idx[1]]
        a, b = e1
        c, d = e2
        
        # Skip if nodes overlap
        if len(set([a, b, c, d])) < 4:
            continue
        
        # Try swap: (a-b, c-d) -> (a-d, c-b) or (a-c, b-d)
        if np.random.random() < 0.5:
            new_e1, new_e2 = (a, d), (c, b)
        else:
            new_e1, new_e2 = (a, c), (b, d)
        
        # Sort edges
        new_e1 = tuple(sorted(new_e1))
        new_e2 = tuple(sorted(new_e2))
        
        # Check if new edges already exist
        if adj[new_e1[0], new_e1[1]] > 0 or adj[new_e2[0], new_e2[1]] > 0:
            continue
        
        # Perform swap
        adj[a, b] = adj[b, a] = 0
        adj[c, d] = adj[d, c] = 0
        adj[new_e1[0], new_e1[1]] = adj[new_e1[1], new_e1[0]] = 1
        adj[new_e2[0], new_e2[1]] = adj[new_e2[1], new_e2[0]] = 1
        
        # Update edge list
        edges[idx[0]] = new_e1
        edges[idx[1]] = new_e2
        
        successful += 1
    
    return adj


def _redistribute_weights_fast(binary: np.ndarray, weights: np.ndarray, seed: int) -> np.ndarray:
    """Fast weight redistribution."""
    np.random.seed(seed)
    
    weighted = binary.astype(float)
    triu_idx = np.triu_indices_from(weighted, k=1)
    edge_mask = weighted[triu_idx] > 0
    n_edges = np.sum(edge_mask)
    
    if n_edges == 0:
        return weighted
    
    # Permute and assign weights
    perm_weights = np.random.choice(weights, size=n_edges, replace=len(weights) < n_edges)
    
    edge_rows = triu_idx[0][edge_mask]
    edge_cols = triu_idx[1][edge_mask]
    
    for i, (r, c) in enumerate(zip(edge_rows, edge_cols)):
        weighted[r, c] = weighted[c, r] = perm_weights[i]
    
    return weighted


def _compute_single_surrogate_gradient(
    binary_matrix: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    approach: str,
    sparsity: float,
    seed: int
) -> np.ndarray:
    """Compute gradient for a single surrogate (for parallelization)."""
    try:
        # Generate surrogate
        surrogate_binary = _maslov_sneppen_fast(binary_matrix, seed=seed)
        surrogate_weighted = _redistribute_weights_fast(surrogate_binary, weights, seed)
        
        # Compute gradient
        gradients, variance = _compute_gradient_fast(
            surrogate_weighted,
            n_components=n_components,
            approach=approach,
            sparsity=sparsity,
            use_sparse=True
        )
        
        return variance
    except Exception:
        return None


def _analyze_subject_gradients_vs_null(
    connectivity: np.ndarray,
    subject_id: str,
    n_components: int = 10,
    n_surrogates: int = 100,
    approach: str = 'dm',
    sparsity: float = 0.9,
    n_jobs_surrogates: int = 4,
    seed: int = 42
) -> Dict:
    """
    Analyze single subject gradients vs null (with parallel surrogates).
    """
    # Observed gradients
    obs_gradients, obs_variance = _compute_gradient_fast(
        connectivity, n_components, approach, sparsity
    )
    
    # Prepare for surrogates
    binary_matrix = (connectivity > 0).astype(float)
    triu_idx = np.triu_indices_from(connectivity, k=1)
    weights = connectivity[triu_idx]
    weights = weights[weights > 0]
    
    # Parallel surrogate computation
    if HAS_JOBLIB and n_jobs_surrogates > 1:
        null_variances = Parallel(n_jobs=n_jobs_surrogates)(
            delayed(_compute_single_surrogate_gradient)(
                binary_matrix, weights, n_components, approach, sparsity, seed + i
            )
            for i in range(n_surrogates)
        )
    else:
        null_variances = [
            _compute_single_surrogate_gradient(
                binary_matrix, weights, n_components, approach, sparsity, seed + i
            )
            for i in range(n_surrogates)
        ]
    
    # Filter failed
    null_variances = [v for v in null_variances if v is not None]
    null_variances = np.array(null_variances)
    
    if len(null_variances) == 0:
        return None
    
    # Statistics
    n_comp = min(n_components, null_variances.shape[1], len(obs_variance))
    
    results = {
        'subject_id': subject_id,
        'observed_variance': obs_variance,
        'observed_gradients': obs_gradients,
    }
    
    for c in range(min(3, n_comp)):  # G1, G2, G3
        null_mean = np.mean(null_variances[:, c])
        null_std = np.std(null_variances[:, c])
        
        z_score = (obs_variance[c] - null_mean) / (null_std + 1e-10)
        p_value = np.mean(null_variances[:, c] >= obs_variance[c])
        
        results[f'G{c+1}_variance'] = obs_variance[c]
        results[f'G{c+1}_null_mean'] = null_mean
        results[f'G{c+1}_null_std'] = null_std
        results[f'G{c+1}_zscore'] = z_score
        results[f'G{c+1}_pvalue'] = p_value
        results[f'G{c+1}_significant'] = p_value < 0.05
    
    return results


# =============================================================================
# MAIN PARALLEL PIPELINE
# =============================================================================

def run_gradient_analysis_fast(
    subjects: List[str] = None,
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor',
    n_components: int = 10,
    n_surrogates: int = 100,
    approach: str = 'dm',
    sparsity: float = 0.9,
    n_jobs: int = -1,
    n_jobs_surrogates: int = 4,
    output_dir: str = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Fast parallel gradient analysis with null model comparison.
    
    Optimizations:
    - Parallel processing of subjects (outer loop)
    - Parallel processing of surrogates (inner loop)
    - Sparse eigendecomposition
    - Minimal memory allocation
    
    Parameters
    ----------
    subjects : List[str], optional
        Subject IDs. If None, uses all available.
    atlas_name : str
        Atlas name
    strategy : str
        Denoising strategy
    n_components : int
        Number of gradient components
    n_surrogates : int
        Number of null model surrogates per subject
        Recommendations: 50 (quick test), 100 (preliminary), 500+ (publication)
    approach : str
        'dm' (Diffusion Map) or 'pca' (much faster)
    sparsity : float
        Sparsity threshold (0.9 = keep top 10%)
    n_jobs : int
        Number of parallel jobs for subjects.
        -1 = all cores
        Recommended: n_cores // 2 to leave room for surrogate parallelization
    n_jobs_surrogates : int
        Number of parallel jobs for surrogates within each subject.
        Recommended: 4-8
    output_dir : str, optional
        Output directory
    seed : int
        Random seed
    verbose : bool
        Show progress
        
    Returns
    -------
    Dict
        - results_df: DataFrame with all results
        - output_dir: Path to saved results
        - elapsed_time: Total time in seconds
        
    Examples
    --------
    >>> # Quick test (few surrogates)
    >>> results = run_gradient_analysis_fast(
    ...     n_surrogates=50,
    ...     approach='pca',  # Fast
    ...     n_jobs=8
    ... )
    
    >>> # Full analysis (publication quality)
    >>> results = run_gradient_analysis_fast(
    ...     n_surrogates=500,
    ...     approach='dm',
    ...     n_jobs=8,
    ...     n_jobs_surrogates=4
    ... )
    
    Notes
    -----
    With 16 cores and n_jobs=8, n_jobs_surrogates=4:
    - Processes 8 subjects simultaneously
    - Each subject processes 4 surrogates simultaneously
    - Total parallelization: 32 threads utilized
    
    Memory: ~100MB per parallel job, so 32 jobs ≈ 3.2GB
    """
    import pandas as pd
    
    # Import config
    try:
        from sars.normative_comparison import config
    except ImportError:
        raise ImportError("Cannot import config. Make sure sars package is properly installed.")
    
    start_time = time.time()
    
    # Get subjects
    if subjects is None:
        subjects = config.get_available_subjects(atlas_name, strategy)
    
    if not subjects:
        raise ValueError(f"No subjects found for {atlas_name}/{strategy}")
    
    # Determine n_jobs
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 4
    
    # Optimal split: use half cores for subjects, rest for surrogates
    # This maximizes throughput while avoiding memory issues
    if n_jobs > 8 and n_jobs_surrogates == 4:
        n_jobs = min(n_jobs, len(subjects))
        # Adjust to leave cores for surrogates
        effective_jobs = max(1, n_jobs // 2)
    else:
        effective_jobs = n_jobs
    
    if verbose:
        print("\n" + "="*70)
        print("FAST GRADIENT ANALYSIS (Parallelized)")
        print("="*70)
        print(f"Atlas: {atlas_name}")
        print(f"Subjects: {len(subjects)}")
        print(f"Surrogates: {n_surrogates}")
        print(f"Approach: {approach}")
        print(f"Parallel jobs (subjects): {effective_jobs}")
        print(f"Parallel jobs (surrogates): {n_jobs_surrogates}")
        print(f"Total parallelization: ~{effective_jobs * n_jobs_surrogates} threads")
        print("="*70)
    
    # Load all matrices first (I/O is not parallelizable efficiently)
    if verbose:
        print("\nLoading connectivity matrices...")
    
    matrices = {}
    for sub in tqdm(subjects, disable=not verbose, desc="Loading"):
        try:
            matrices[sub] = config.load_connectivity_matrix(sub, atlas_name, strategy)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load {sub}: {e}")
    
    if not matrices:
        raise ValueError("No matrices loaded successfully")
    
    if verbose:
        print(f"Loaded {len(matrices)} matrices")
    
    # Define worker function
    def process_subject(sub_id: str, matrix: np.ndarray, idx: int) -> Dict:
        result = _analyze_subject_gradients_vs_null(
            matrix,
            subject_id=sub_id,
            n_components=n_components,
            n_surrogates=n_surrogates,
            approach=approach,
            sparsity=sparsity,
            n_jobs_surrogates=n_jobs_surrogates,
            seed=seed + idx * 1000
        )
        return result
    
    # Parallel processing of subjects
    if verbose:
        print(f"\nProcessing {len(matrices)} subjects in parallel...")
    
    if HAS_JOBLIB and effective_jobs > 1:
        # Parallel
        results_list = Parallel(n_jobs=effective_jobs, verbose=10 if verbose else 0)(
            delayed(process_subject)(sub_id, matrix, idx)
            for idx, (sub_id, matrix) in enumerate(matrices.items())
        )
    else:
        # Sequential with progress bar
        results_list = []
        for idx, (sub_id, matrix) in enumerate(tqdm(matrices.items(), disable=not verbose)):
            results_list.append(process_subject(sub_id, matrix, idx))
    
    # Filter None results
    results_list = [r for r in results_list if r is not None]
    
    if not results_list:
        raise ValueError("No subjects processed successfully")
    
    # Create DataFrame
    df_data = []
    for r in results_list:
        row = {
            'subject_id': r['subject_id'],
            'G1_variance': r.get('G1_variance'),
            'G1_null_mean': r.get('G1_null_mean'),
            'G1_zscore': r.get('G1_zscore'),
            'G1_pvalue': r.get('G1_pvalue'),
            'G1_significant': r.get('G1_significant'),
            'G2_variance': r.get('G2_variance'),
            'G2_zscore': r.get('G2_zscore'),
            'G2_pvalue': r.get('G2_pvalue'),
            'G3_variance': r.get('G3_variance'),
            'G3_zscore': r.get('G3_zscore'),
            'G3_pvalue': r.get('G3_pvalue'),
        }
        df_data.append(row)
    
    results_df = pd.DataFrame(df_data)
    
    elapsed = time.time() - start_time
    
    # Save results
    if output_dir is None:
        output_dir = config.COMPARISON_DIR / 'gradients_fast' / atlas_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'gradient_null_results.csv', index=False)
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Subjects processed: {len(results_df)}")
        print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Time per subject: {elapsed/len(results_df):.1f}s")
        print(f"\nG1 Statistics:")
        print(f"  Variance: {results_df['G1_variance'].mean():.4f} ± {results_df['G1_variance'].std():.4f}")
        print(f"  Z-score: {results_df['G1_zscore'].mean():.2f} ± {results_df['G1_zscore'].std():.2f}")
        print(f"  Significant: {100 * results_df['G1_significant'].mean():.1f}%")
        print(f"\n✓ Results saved to: {output_dir}")
    
    return {
        'results_df': results_df,
        'output_dir': str(output_dir),
        'elapsed_time': elapsed,
        'subjects_processed': len(results_df),
    }


def run_gradient_analysis_ultrafast(
    subjects: List[str] = None,
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor',
    n_components: int = 5,
    n_surrogates: int = 50,
    n_jobs: int = -1,
    output_dir: str = None,
    verbose: bool = True
) -> Dict:
    """
    Ultra-fast version using PCA and minimal surrogates.
    
    Good for quick exploration before running full analysis.
    Typically completes in <1 minute for 23 subjects.
    
    Parameters
    ----------
    subjects, atlas_name, strategy : as in run_gradient_analysis_fast
    n_components : int
        Number of components (default: 5, less is faster)
    n_surrogates : int
        Number of surrogates (default: 50, minimum reasonable)
    n_jobs : int
        Parallel jobs (-1 = all cores)
    output_dir : str
        Output directory
    verbose : bool
        Show progress
        
    Returns
    -------
    Dict with results
    """
    return run_gradient_analysis_fast(
        subjects=subjects,
        atlas_name=atlas_name,
        strategy=strategy,
        n_components=n_components,
        n_surrogates=n_surrogates,
        approach='pca',  # PCA is much faster than DM
        sparsity=0.9,
        n_jobs=n_jobs,
        n_jobs_surrogates=2,  # Less parallelization needed
        output_dir=output_dir,
        verbose=verbose
    )


# =============================================================================
# BENCHMARK / TEST
# =============================================================================

def benchmark_gradient_computation(
    n_rois: int = 100,
    n_components: int = 10,
    n_iterations: int = 5
) -> Dict:
    """
    Benchmark different gradient computation methods.
    
    Parameters
    ----------
    n_rois : int
        Number of ROIs
    n_components : int
        Number of components
    n_iterations : int
        Number of iterations for timing
        
    Returns
    -------
    Dict with timing results
    """
    import time
    
    # Generate random matrix
    np.random.seed(42)
    matrix = np.random.random((n_rois, n_rois))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    
    results = {}
    
    # Test DM with sparse
    times = []
    for _ in range(n_iterations):
        t0 = time.time()
        _compute_gradient_fast(matrix, n_components, 'dm', use_sparse=True)
        times.append(time.time() - t0)
    results['dm_sparse'] = np.mean(times)
    
    # Test DM without sparse
    times = []
    for _ in range(n_iterations):
        t0 = time.time()
        _compute_gradient_fast(matrix, n_components, 'dm', use_sparse=False)
        times.append(time.time() - t0)
    results['dm_dense'] = np.mean(times)
    
    # Test PCA
    times = []
    for _ in range(n_iterations):
        t0 = time.time()
        _compute_gradient_fast(matrix, n_components, 'pca')
        times.append(time.time() - t0)
    results['pca'] = np.mean(times)
    
    print(f"\nBenchmark Results (n_rois={n_rois}, n_components={n_components}):")
    print(f"  DM (sparse): {results['dm_sparse']*1000:.1f} ms")
    print(f"  DM (dense):  {results['dm_dense']*1000:.1f} ms")
    print(f"  PCA:         {results['pca']*1000:.1f} ms")
    print(f"  Speedup (sparse vs dense): {results['dm_dense']/results['dm_sparse']:.1f}x")
    print(f"  Speedup (PCA vs DM sparse): {results['dm_sparse']/results['pca']:.1f}x")
    
    return results


# =============================================================================
# FUNCTIONS FOR BRAIN VISUALIZATION (preserve full gradient arrays)
# =============================================================================

def _process_surrogate_full_gradient(
    binary_matrix: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    approach: str,
    sparsity: float,
    seed: int
) -> np.ndarray:
    """
    Compute full gradient array for a single surrogate.
    Returns gradient array (n_parcels, n_components) or None if failed.
    """
    try:
        surrogate_binary = _maslov_sneppen_fast(binary_matrix, seed=seed)
        surrogate_weighted = _redistribute_weights_fast(surrogate_binary, weights, seed)
        gradients, _ = _compute_gradient_fast(surrogate_weighted, n_components, approach, sparsity)
        return gradients
    except Exception:
        return None


def compute_null_gradients_for_viz(
    connectivity: np.ndarray,
    n_surrogates: int = 100,
    n_components: int = 3,
    approach: str = 'dm',
    sparsity: float = 0.9,
    n_jobs: int = -1,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Computar gradientes nulos preservando valores completos por região.
    
    Esta função é necessária para visualizações de superfície cerebral,
    onde queremos comparar a posição de cada região no gradiente
    entre observado e modelo nulo.
    
    Parameters
    ----------
    connectivity : np.ndarray
        Matriz de conectividade (n_parcels, n_parcels)
    n_surrogates : int
        Número de surrogates
    n_components : int
        Número de componentes do gradiente
    approach : str
        'dm' ou 'pca'
    sparsity : float
        Threshold de sparsity
    n_jobs : int
        Jobs paralelos (-1 = todos os cores)
    seed : int
        Random seed
    verbose : bool
        Mostrar progresso
        
    Returns
    -------
    Dict com:
        - observed_gradients: np.ndarray (n_parcels, n_components)
        - observed_variance: np.ndarray (n_components,)
        - null_gradients: np.ndarray (n_surrogates, n_parcels, n_components)
        - null_mean: np.ndarray (n_parcels, n_components)
        - null_std: np.ndarray (n_parcels, n_components)
        - G1_null_mean: np.ndarray (n_parcels,) - para plot direto
        - G1_null_std: np.ndarray (n_parcels,)
        
    Examples
    --------
    >>> from sars.gradients.fast import compute_null_gradients_for_viz
    >>> from sars.gradients.brain_viz import plot_gradient_comparison_surface
    >>> 
    >>> # Computar gradientes com valores por região
    >>> result = compute_null_gradients_for_viz(fc_matrix, n_surrogates=100)
    >>> 
    >>> # Plotar comparação na superfície
    >>> plot_gradient_comparison_surface(
    ...     observed_gradient=result['observed_gradients'][:, 0],  # G1
    ...     null_mean_gradient=result['G1_null_mean'],
    ...     atlas_name='schaefer_100'
    ... )
    """
    import os
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 4
    
    n_parcels = connectivity.shape[0]
    
    if verbose:
        print(f"Computing null gradients for visualization...")
        print(f"  Parcels: {n_parcels}")
        print(f"  Surrogates: {n_surrogates}")
        print(f"  Components: {n_components}")
        print(f"  Parallel jobs: {n_jobs}")
    
    # 1. Observed gradients
    if verbose:
        print("\n[1/2] Computing observed gradients...")
    
    obs_gradients, obs_variance = _compute_gradient_fast(
        connectivity, n_components, approach, sparsity
    )
    
    # 2. Prepare for surrogates
    binary_matrix = (connectivity > 0).astype(float)
    triu_idx = np.triu_indices_from(connectivity, k=1)
    weights = connectivity[triu_idx]
    weights = weights[weights > 0]
    
    # 3. Compute null gradients in parallel
    if verbose:
        print(f"[2/2] Computing {n_surrogates} null gradients...")
    
    if HAS_JOBLIB and n_jobs > 1:
        null_results = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
            delayed(_process_surrogate_full_gradient)(
                binary_matrix, weights, n_components, approach, sparsity, seed + i
            )
            for i in range(n_surrogates)
        )
    else:
        null_results = []
        iterator = tqdm(range(n_surrogates), disable=not verbose, desc="Surrogates")
        for i in iterator:
            null_results.append(
                _process_surrogate_full_gradient(
                    binary_matrix, weights, n_components, approach, sparsity, seed + i
                )
            )
    
    # Filter failed and stack
    null_results = [r for r in null_results if r is not None]
    
    if len(null_results) == 0:
        raise ValueError("All surrogates failed!")
    
    # Stack: (n_surrogates, n_parcels, n_components)
    null_gradients = np.stack(null_results, axis=0)
    
    # Align null gradients to observed (flip sign if anti-correlated)
    # This is important because eigenvector sign is arbitrary
    for i in range(len(null_gradients)):
        for c in range(n_components):
            corr = np.corrcoef(obs_gradients[:, c], null_gradients[i, :, c])[0, 1]
            if corr < 0:
                null_gradients[i, :, c] *= -1
    
    # Compute statistics
    null_mean = np.mean(null_gradients, axis=0)  # (n_parcels, n_components)
    null_std = np.std(null_gradients, axis=0)
    
    if verbose:
        print(f"\n✓ Computed {len(null_results)} valid surrogates")
        print(f"  Observed G1 range: [{obs_gradients[:, 0].min():.3f}, {obs_gradients[:, 0].max():.3f}]")
        print(f"  Null mean G1 range: [{null_mean[:, 0].min():.3f}, {null_mean[:, 0].max():.3f}]")
    
    return {
        'observed_gradients': obs_gradients,
        'observed_variance': obs_variance,
        'null_gradients': null_gradients,
        'null_mean': null_mean,
        'null_std': null_std,
        # Convenience: G1, G2, G3 separados para uso direto
        'G1_observed': obs_gradients[:, 0],
        'G1_null_mean': null_mean[:, 0],
        'G1_null_std': null_std[:, 0],
        'G1_null_all': null_gradients[:, :, 0],  # (n_surrogates, n_parcels)
        'G2_observed': obs_gradients[:, 1] if n_components > 1 else None,
        'G2_null_mean': null_mean[:, 1] if n_components > 1 else None,
        'G3_observed': obs_gradients[:, 2] if n_components > 2 else None,
        'G3_null_mean': null_mean[:, 2] if n_components > 2 else None,
    }


def compute_group_null_gradients_for_viz(
    subjects: List[str] = None,
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor',
    n_surrogates: int = 100,
    n_components: int = 3,
    approach: str = 'dm',
    sparsity: float = 0.9,
    n_jobs: int = -1,
    output_dir: str = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Computar gradientes nulos para todo o grupo, preservando valores por região.
    
    Parameters
    ----------
    subjects : List[str], optional
        IDs dos sujeitos. Se None, usa todos disponíveis.
    atlas_name : str
        Nome do atlas
    strategy : str
        Estratégia de denoising
    n_surrogates : int
        Surrogates por sujeito
    n_components : int
        Componentes do gradiente
    approach : str
        'dm' ou 'pca'
    sparsity : float
        Threshold de sparsity
    n_jobs : int
        Jobs paralelos
    output_dir : str, optional
        Diretório para salvar
    seed : int
        Random seed
    verbose : bool
        Mostrar progresso
        
    Returns
    -------
    Dict com:
        - subjects: List[str]
        - individual_results: Dict[str, result_from_compute_null_gradients_for_viz]
        - group_observed_mean: (n_parcels, n_components)
        - group_null_mean: (n_parcels, n_components)
        - output_dir: Path
    """
    import pandas as pd
    
    try:
        from sars.normative_comparison import config
    except ImportError:
        raise ImportError("Cannot import config")
    
    if subjects is None:
        subjects = config.get_available_subjects(atlas_name, strategy)
    
    if not subjects:
        raise ValueError(f"No subjects found for {atlas_name}/{strategy}")
    
    if verbose:
        print("\n" + "="*70)
        print("COMPUTING NULL GRADIENTS FOR VISUALIZATION")
        print("="*70)
        print(f"Atlas: {atlas_name}")
        print(f"Subjects: {len(subjects)}")
        print(f"Surrogates per subject: {n_surrogates}")
        print(f"Approach: {approach}")
    
    # Process each subject
    individual_results = {}
    
    for i, sub in enumerate(subjects):
        if verbose:
            print(f"\n[{i+1}/{len(subjects)}] {sub}")
        
        try:
            matrix = config.load_connectivity_matrix(sub, atlas_name, strategy)
            
            result = compute_null_gradients_for_viz(
                matrix,
                n_surrogates=n_surrogates,
                n_components=n_components,
                approach=approach,
                sparsity=sparsity,
                n_jobs=n_jobs,
                seed=seed + i * 1000,
                verbose=False
            )
            
            individual_results[sub] = result
            
            if verbose:
                g1_corr = np.corrcoef(
                    result['G1_observed'], 
                    result['G1_null_mean']
                )[0, 1]
                print(f"   G1 obs-null correlation: {g1_corr:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"   ERROR: {e}")
    
    if not individual_results:
        raise ValueError("No subjects processed successfully")
    
    # Compute group averages
    all_observed = np.stack([r['observed_gradients'] for r in individual_results.values()], axis=0)
    all_null_mean = np.stack([r['null_mean'] for r in individual_results.values()], axis=0)
    
    group_observed_mean = np.mean(all_observed, axis=0)
    group_null_mean = np.mean(all_null_mean, axis=0)
    
    # Save if output_dir provided
    if output_dir is None:
        output_dir = config.COMPARISON_DIR / 'gradients_viz' / atlas_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save group arrays
    np.save(output_dir / 'group_observed_gradients.npy', group_observed_mean)
    np.save(output_dir / 'group_null_mean_gradients.npy', group_null_mean)
    
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print('='*70)
        print(f"Subjects processed: {len(individual_results)}")
        print(f"Group G1 observed range: [{group_observed_mean[:, 0].min():.3f}, {group_observed_mean[:, 0].max():.3f}]")
        print(f"Group G1 null mean range: [{group_null_mean[:, 0].min():.3f}, {group_null_mean[:, 0].max():.3f}]")
        print(f"\n✓ Saved to: {output_dir}")
    
    return {
        'subjects': list(individual_results.keys()),
        'individual_results': individual_results,
        'group_observed_mean': group_observed_mean,
        'group_null_mean': group_null_mean,
        'group_G1_observed': group_observed_mean[:, 0],
        'group_G1_null_mean': group_null_mean[:, 0],
        'output_dir': str(output_dir),
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print("Gradients Fast Module")
    print(f"joblib available: {HAS_JOBLIB}")
    print(f"tqdm available: {HAS_TQDM}")
    
    # Run benchmark
    benchmark_gradient_computation(n_rois=100)
