"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Gradients Fast v2 - Fixed Parallel Analysis (NO NESTED PARALLELISM)         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Versão corrigida sem nested parallelism.                                    ║
║  Usa estratégia "flat": todos os jobs (sujeito × surrogate) em uma fila.     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import time

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# CORE FAST FUNCTIONS (same as before)
# =============================================================================

def _compute_affinity_fast(conn: np.ndarray, sparsity: float = 0.9) -> np.ndarray:
    conn = conn.copy()
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 0)
    if sparsity > 0:
        threshold = np.percentile(conn[conn > 0], sparsity * 100)
        conn[conn < threshold] = 0
    norms = np.linalg.norm(conn, axis=1, keepdims=True)
    norms[norms == 0] = 1
    conn_norm = conn / norms
    cos_sim = np.clip(conn_norm @ conn_norm.T, -1, 1)
    affinity = 1 - np.arccos(cos_sim) / np.pi
    np.fill_diagonal(affinity, 0)
    return affinity


def _compute_gradient_fast(
    connectivity: np.ndarray,
    n_components: int = 10,
    approach: str = 'dm',
    sparsity: float = 0.9
) -> Tuple[np.ndarray, np.ndarray]:
    n = connectivity.shape[0]
    
    if approach == 'pca':
        from sklearn.decomposition import PCA
        conn = (connectivity + connectivity.T) / 2
        np.fill_diagonal(conn, 0)
        pca = PCA(n_components=min(n_components, n - 1))
        gradients = pca.fit_transform(conn)
        return gradients, pca.explained_variance_ratio_
    
    affinity = _compute_affinity_fast(connectivity, sparsity)
    d = np.sum(affinity, axis=1)
    d[d == 0] = 1
    d_inv_sqrt = 1.0 / np.sqrt(d)
    M = affinity * np.outer(d_inv_sqrt, d_inv_sqrt)
    M = (M + M.T) / 2
    
    n_compute = min(n_components + 1, n - 2)
    eigenvalues, eigenvectors = eigh(M)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][1:n_compute]
    eigenvectors = eigenvectors[:, idx][:, 1:n_compute]
    
    embeddings = eigenvectors * d_inv_sqrt[:, np.newaxis]
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=0, keepdims=True) + 1e-10)
    explained = eigenvalues / (np.sum(eigenvalues) + 1e-10)
    
    return embeddings[:, :n_components], explained[:n_components]


def _maslov_sneppen_fast(adjacency: np.ndarray, n_swaps: int = None, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    adj = adjacency.copy()
    edges = np.array(np.where(np.triu(adj, k=1) > 0)).T
    n_edges = len(edges)
    
    if n_edges < 2:
        return adj
    
    if n_swaps is None:
        n_swaps = n_edges * 5  # Reduced from 10
    
    successful = 0
    attempts = 0
    max_attempts = n_swaps * 5  # Reduced from 10
    
    while successful < n_swaps and attempts < max_attempts:
        attempts += 1
        idx = np.random.choice(n_edges, 2, replace=False)
        a, b = edges[idx[0]]
        c, d = edges[idx[1]]
        
        if len(set([a, b, c, d])) < 4:
            continue
        
        if np.random.random() < 0.5:
            new_e1, new_e2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        else:
            new_e1, new_e2 = tuple(sorted((a, c))), tuple(sorted((b, d)))
        
        if adj[new_e1[0], new_e1[1]] > 0 or adj[new_e2[0], new_e2[1]] > 0:
            continue
        
        adj[a, b] = adj[b, a] = 0
        adj[c, d] = adj[d, c] = 0
        adj[new_e1[0], new_e1[1]] = adj[new_e1[1], new_e1[0]] = 1
        adj[new_e2[0], new_e2[1]] = adj[new_e2[1], new_e2[0]] = 1
        edges[idx[0]] = new_e1
        edges[idx[1]] = new_e2
        successful += 1
    
    return adj


def _redistribute_weights_fast(binary: np.ndarray, weights: np.ndarray, seed: int) -> np.ndarray:
    np.random.seed(seed)
    weighted = binary.astype(float)
    triu_idx = np.triu_indices_from(weighted, k=1)
    edge_mask = weighted[triu_idx] > 0
    n_edges = np.sum(edge_mask)
    
    if n_edges == 0:
        return weighted
    
    perm_weights = np.random.choice(weights, size=n_edges, replace=len(weights) < n_edges)
    edge_rows = triu_idx[0][edge_mask]
    edge_cols = triu_idx[1][edge_mask]
    
    for i, (r, c) in enumerate(zip(edge_rows, edge_cols)):
        weighted[r, c] = weighted[c, r] = perm_weights[i]
    
    return weighted


# =============================================================================
# SINGLE JOB FUNCTION (for flat parallelization)
# =============================================================================

def _process_single_surrogate(
    subject_id: str,
    surrogate_idx: int,
    binary_matrix: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    approach: str,
    sparsity: float,
    seed: int
) -> Tuple[str, int, np.ndarray]:
    """
    Process a single surrogate for a single subject.
    Returns (subject_id, surrogate_idx, variance_array)
    """
    try:
        surrogate_binary = _maslov_sneppen_fast(binary_matrix, seed=seed)
        surrogate_weighted = _redistribute_weights_fast(surrogate_binary, weights, seed)
        _, variance = _compute_gradient_fast(surrogate_weighted, n_components, approach, sparsity)
        return (subject_id, surrogate_idx, variance)
    except Exception as e:
        return (subject_id, surrogate_idx, None)


# =============================================================================
# MAIN FUNCTION - FLAT PARALLELIZATION
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
    output_dir: str = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Fast gradient analysis with FLAT parallelization (no nesting).
    
    Creates a single job queue with all (subject × surrogate) combinations
    and processes them in parallel. Much more efficient than nested loops.
    
    Parameters
    ----------
    subjects : List[str]
        Subject IDs. If None, uses all available.
    atlas_name : str
        Atlas name
    strategy : str  
        Denoising strategy
    n_components : int
        Number of gradient components
    n_surrogates : int
        Surrogates per subject (50=quick, 100=standard, 500+=publication)
    approach : str
        'dm' or 'pca' (pca is ~5x faster)
    sparsity : float
        Sparsity threshold
    n_jobs : int
        Parallel jobs (-1 = all cores)
    output_dir : str
        Output directory
    seed : int
        Random seed
    verbose : bool
        Show progress
        
    Returns
    -------
    Dict with results_df, output_dir, elapsed_time
    """
    import pandas as pd
    
    try:
        from sars.normative_comparison import config
    except ImportError:
        raise ImportError("Cannot import config")
    
    start_time = time.time()
    
    if subjects is None:
        subjects = config.get_available_subjects(atlas_name, strategy)
    
    if not subjects:
        raise ValueError(f"No subjects found for {atlas_name}/{strategy}")
    
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 4
    
    total_jobs = len(subjects) * n_surrogates
    
    if verbose:
        print("\n" + "="*70)
        print("FAST GRADIENT ANALYSIS v2 (Flat Parallelization)")
        print("="*70)
        print(f"Atlas: {atlas_name}")
        print(f"Subjects: {len(subjects)}")
        print(f"Surrogates per subject: {n_surrogates}")
        print(f"Total jobs: {total_jobs}")
        print(f"Approach: {approach}")
        print(f"Parallel workers: {n_jobs}")
        print("="*70)
    
    # Load matrices
    if verbose:
        print("\n[1/3] Loading connectivity matrices...")
    
    matrices = {}
    for sub in tqdm(subjects, disable=not verbose, desc="Loading"):
        try:
            matrices[sub] = config.load_connectivity_matrix(sub, atlas_name, strategy)
        except Exception as e:
            if verbose:
                print(f"  Warning: {sub}: {e}")
    
    if not matrices:
        raise ValueError("No matrices loaded")
    
    if verbose:
        print(f"Loaded {len(matrices)} matrices")
    
    # Compute observed gradients
    if verbose:
        print("\n[2/3] Computing observed gradients...")
    
    observed = {}
    for sub_id, matrix in tqdm(matrices.items(), disable=not verbose, desc="Observed"):
        try:
            grads, var = _compute_gradient_fast(matrix, n_components, approach, sparsity)
            observed[sub_id] = {'gradients': grads, 'variance': var}
        except Exception as e:
            if verbose:
                print(f"  Warning: {sub_id}: {e}")
    
    if verbose:
        print(f"Computed gradients for {len(observed)} subjects")
    
    # Prepare job list
    if verbose:
        print("\n[3/3] Running null model analysis...")
        print(f"      {total_jobs} surrogate jobs across {n_jobs} workers")
    
    job_list = []
    for sub_id, matrix in matrices.items():
        binary = (matrix > 0).astype(float)
        triu_idx = np.triu_indices_from(matrix, k=1)
        weights = matrix[triu_idx]
        weights = weights[weights > 0]
        
        for surr_idx in range(n_surrogates):
            job_list.append({
                'subject_id': sub_id,
                'surrogate_idx': surr_idx,
                'binary_matrix': binary,
                'weights': weights,
                'n_components': n_components,
                'approach': approach,
                'sparsity': sparsity,
                'seed': seed + hash(sub_id) % 10000 + surr_idx
            })
    
    # Execute in parallel (FLAT - no nesting!)
    if HAS_JOBLIB and n_jobs > 1:
        results_raw = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0, batch_size='auto')(
            delayed(_process_single_surrogate)(
                j['subject_id'], j['surrogate_idx'], j['binary_matrix'],
                j['weights'], j['n_components'], j['approach'], j['sparsity'], j['seed']
            )
            for j in job_list
        )
    else:
        results_raw = []
        for j in tqdm(job_list, disable=not verbose, desc="Sequential"):
            results_raw.append(_process_single_surrogate(
                j['subject_id'], j['surrogate_idx'], j['binary_matrix'],
                j['weights'], j['n_components'], j['approach'], j['sparsity'], j['seed']
            ))
    
    # Aggregate results by subject
    null_variances = {sub: [] for sub in matrices.keys()}
    for sub_id, surr_idx, var in results_raw:
        if var is not None:
            null_variances[sub_id].append(var)
    
    # Compute statistics
    results_data = []
    for sub_id in observed.keys():
        obs_var = observed[sub_id]['variance']
        null_vars = np.array(null_variances.get(sub_id, []))
        
        if len(null_vars) == 0:
            continue
        
        row = {'subject_id': sub_id}
        
        for c in range(min(3, len(obs_var), null_vars.shape[1])):
            null_mean = np.mean(null_vars[:, c])
            null_std = np.std(null_vars[:, c])
            z_score = (obs_var[c] - null_mean) / (null_std + 1e-10)
            p_value = np.mean(null_vars[:, c] >= obs_var[c])
            
            row[f'G{c+1}_variance'] = obs_var[c]
            row[f'G{c+1}_null_mean'] = null_mean
            row[f'G{c+1}_null_std'] = null_std
            row[f'G{c+1}_zscore'] = z_score
            row[f'G{c+1}_pvalue'] = p_value
            row[f'G{c+1}_significant'] = p_value < 0.05
        
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    elapsed = time.time() - start_time
    
    # Save
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
        print(f"Subjects: {len(results_df)}")
        print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Speed: {total_jobs/elapsed:.1f} jobs/sec")
        
        if 'G1_variance' in results_df.columns:
            print(f"\nG1 Statistics:")
            print(f"  Variance: {results_df['G1_variance'].mean():.4f} ± {results_df['G1_variance'].std():.4f}")
            print(f"  Z-score:  {results_df['G1_zscore'].mean():.2f} ± {results_df['G1_zscore'].std():.2f}")
            print(f"  Significant: {100 * results_df['G1_significant'].mean():.1f}%")
        
        print(f"\n✓ Saved: {output_dir}")
    
    return {
        'results_df': results_df,
        'observed': observed,
        'output_dir': str(output_dir),
        'elapsed_time': elapsed,
    }


def run_quick_test(atlas_name: str = 'schaefer_100', strategy: str = 'acompcor'):
    """Quick test with minimal settings."""
    return run_gradient_analysis_fast(
        atlas_name=atlas_name,
        strategy=strategy,
        n_components=5,
        n_surrogates=20,
        approach='pca',
        n_jobs=-1,
        verbose=True
    )


if __name__ == '__main__':
    print("Fast Gradient Analysis v2")
    print(f"joblib: {HAS_JOBLIB}")
    print(f"tqdm: {HAS_TQDM}")
