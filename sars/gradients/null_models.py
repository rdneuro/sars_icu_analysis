"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Gradient Analysis with Null Models                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Este módulo permite avaliar se a organização hierárquica do cérebro,        ║
║  capturada pelos gradientes de conectividade, é significativamente           ║
║  diferente do esperado em redes randomizadas equivalentes.                   ║
║                                                                              ║
║  CONCEITO:                                                                   ║
║  Gradientes de conectividade (Margulies et al., 2016) representam os         ║
║  eixos principais de variação da conectividade funcional. O primeiro         ║
║  gradiente (G1) tipicamente captura o eixo sensorimotor ↔ associativo.       ║
║                                                                              ║
║  A comparação com modelos nulos permite testar:                              ║
║  - A variância explicada pelos gradientes é maior que o esperado?            ║
║  - A organização espacial dos gradientes é não-aleatória?                    ║
║  - A hierarquia cortical é biologicamente significativa?                     ║
║                                                                              ║
║  MÉTRICAS DE COMPARAÇÃO:                                                     ║
║  1. Explained variance ratio - quanto de variância cada gradiente captura    ║
║  2. Gradient dispersion - spread das regiões ao longo do gradiente           ║
║  3. Procrustes alignment - similaridade de forma entre gradientes            ║
║  4. Spearman correlation - correlação de ranking entre gradientes            ║
║  5. Gradient range - amplitude do gradiente (max - min)                      ║
║                                                                              ║
║  REFERÊNCIAS:                                                                ║
║  - Margulies et al. (2016) - Principal gradient of cortical organization    ║
║  - Vos de Wael et al. (2020) - BrainSpace toolbox                            ║
║  - Hong et al. (2019) - Gradient null models                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
import warnings

# Tentar importar BrainSpace
try:
    from brainspace.gradient import GradientMaps
    from brainspace.gradient.alignment import procrustes_alignment
    HAS_BRAINSPACE = True
except ImportError:
    HAS_BRAINSPACE = False
    warnings.warn(
        "BrainSpace not installed. Install with: pip install brainspace\n"
        "Some functions will use fallback implementations."
    )

# Importar funções de modelos nulos
try:
    from sars.normative_comparison import null_models as nm
    from sars.normative_comparison import config
except ImportError:
    try:
        # Fallback
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'normative_comparison'))
        import null_models as nm
        import config
    except ImportError:
        nm = None
        config = None


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GradientResult:
    """Resultado de uma análise de gradiente."""
    gradients: np.ndarray          # Shape: (n_rois, n_components)
    lambdas: np.ndarray            # Eigenvalues
    explained_variance: np.ndarray # Variance explained per component
    n_components: int
    approach: str                  # 'dm', 'pca', 'le'
    kernel: str                    # 'normalized_angle', 'cosine', etc.


@dataclass  
class GradientNullResult:
    """Resultado da comparação gradiente vs modelo nulo."""
    # Gradiente observado
    observed_gradients: np.ndarray
    observed_variance: np.ndarray
    
    # Distribuição nula
    null_variance_distribution: np.ndarray  # Shape: (n_surrogates, n_components)
    null_dispersion_distribution: np.ndarray
    null_range_distribution: np.ndarray
    
    # Estatísticas
    variance_zscore: np.ndarray      # Z-score por componente
    variance_pvalue: np.ndarray      # P-value por componente
    dispersion_zscore: np.ndarray
    dispersion_pvalue: np.ndarray
    
    # Procrustes (se alinhado)
    procrustes_distances: np.ndarray  # Distância a cada surrogate
    
    # Metadata
    n_surrogates: int
    null_model_type: str
    

# =============================================================================
# GRADIENT COMPUTATION
# =============================================================================

def compute_gradients(
    connectivity_matrix: np.ndarray,
    n_components: int = 10,
    approach: str = 'dm',
    kernel: str = 'normalized_angle',
    sparsity: float = 0.9,
    random_state: int = 42
) -> GradientResult:
    """
    Calcular gradientes de conectividade usando BrainSpace ou fallback.
    
    Os gradientes são uma decomposição da matriz de conectividade que
    identifica os eixos principais de variação. O primeiro gradiente
    tipicamente captura a hierarquia sensorimotor → associativo.
    
    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz de conectividade funcional (n_rois, n_rois)
    n_components : int
        Número de componentes (gradientes) a extrair
    approach : str
        Método de embedding:
        - 'dm': Diffusion Map (recomendado, captura estrutura não-linear)
        - 'pca': Principal Component Analysis
        - 'le': Laplacian Eigenmaps
    kernel : str
        Kernel para construção de afinidade:
        - 'normalized_angle': Ângulo normalizado (recomendado para FC)
        - 'cosine': Similaridade cosseno
        - 'pearson': Correlação de Pearson
        - 'spearman': Correlação de Spearman
        - 'gaussian': Kernel Gaussiano
    sparsity : float
        Proporção de conexões mais fracas a zerar (0-1)
        0.9 = manter apenas top 10% mais fortes
    random_state : int
        Seed para reprodutibilidade
        
    Returns
    -------
    GradientResult
        Objeto com gradientes e estatísticas
    """
    # Garantir simetria e sem self-connections
    matrix = connectivity_matrix.copy()
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    
    n_rois = matrix.shape[0]
    
    if HAS_BRAINSPACE:
        # Usar BrainSpace
        gm = GradientMaps(
            n_components=n_components,
            approach=approach,
            kernel=kernel,
            random_state=random_state
        )
        
        # Aplicar sparsity threshold se solicitado
        if sparsity > 0:
            threshold = np.percentile(matrix[matrix > 0], sparsity * 100)
            matrix_sparse = matrix.copy()
            matrix_sparse[matrix_sparse < threshold] = 0
        else:
            matrix_sparse = matrix
        
        # Fit
        gm.fit(matrix_sparse)
        
        gradients = gm.gradients_
        lambdas = gm.lambdas_
        
        # Calcular variância explicada
        if lambdas is not None:
            explained_variance = lambdas / np.sum(lambdas)
        else:
            explained_variance = np.var(gradients, axis=0)
            explained_variance = explained_variance / np.sum(explained_variance)
        
    else:
        # Fallback: usar PCA simples
        warnings.warn("Using PCA fallback. Install BrainSpace for better results.")
        
        from sklearn.decomposition import PCA
        
        # Aplicar sparsity
        if sparsity > 0:
            threshold = np.percentile(matrix[matrix > 0], sparsity * 100)
            matrix[matrix < threshold] = 0
        
        pca = PCA(n_components=n_components, random_state=random_state)
        gradients = pca.fit_transform(matrix)
        lambdas = pca.explained_variance_
        explained_variance = pca.explained_variance_ratio_
        approach = 'pca'
        kernel = 'none'
    
    return GradientResult(
        gradients=gradients,
        lambdas=lambdas,
        explained_variance=explained_variance,
        n_components=n_components,
        approach=approach,
        kernel=kernel
    )


def compute_gradient_metrics(gradients: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calcular métricas descritivas dos gradientes.
    
    Parameters
    ----------
    gradients : np.ndarray
        Array de gradientes (n_rois, n_components)
        
    Returns
    -------
    Dict com métricas por componente
    """
    n_components = gradients.shape[1]
    
    metrics = {
        # Range: amplitude do gradiente
        'range': np.ptp(gradients, axis=0),
        
        # Dispersion: desvio padrão das posições
        'dispersion': np.std(gradients, axis=0),
        
        # Variance: variância das posições
        'variance': np.var(gradients, axis=0),
        
        # Skewness: assimetria da distribuição
        'skewness': stats.skew(gradients, axis=0),
        
        # Kurtosis: curtose da distribuição
        'kurtosis': stats.kurtosis(gradients, axis=0),
        
        # Bimodality coefficient: indica separação em dois grupos
        'bimodality': _compute_bimodality(gradients),
    }
    
    return metrics


def _compute_bimodality(gradients: np.ndarray) -> np.ndarray:
    """
    Calcular coeficiente de bimodalidade.
    
    BC = (skewness² + 1) / (kurtosis + 3)
    BC > 0.555 sugere distribuição bimodal
    """
    skew = stats.skew(gradients, axis=0)
    kurt = stats.kurtosis(gradients, axis=0)
    
    bc = (skew**2 + 1) / (kurt + 3)
    return bc


# =============================================================================
# NULL MODEL GRADIENT ANALYSIS
# =============================================================================

def generate_null_gradients(
    connectivity_matrix: np.ndarray,
    n_surrogates: int = 100,
    null_model: str = 'maslov_sneppen',
    n_components: int = 10,
    approach: str = 'dm',
    kernel: str = 'normalized_angle',
    sparsity: float = 0.9,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[GradientResult], np.ndarray]:
    """
    Gerar gradientes de matrizes surrogate (modelo nulo).
    
    Para cada surrogate:
    1. Randomizar a matriz FC preservando propriedades básicas
    2. Calcular gradientes da matriz randomizada
    3. Armazenar para comparação
    
    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz de conectividade original
    n_surrogates : int
        Número de surrogates a gerar
    null_model : str
        Tipo de modelo nulo ('maslov_sneppen', 'configuration', 'erdos_renyi')
    n_components : int
        Número de componentes do gradiente
    approach : str
        Método de embedding
    kernel : str
        Kernel para afinidade
    sparsity : float
        Threshold de sparsity
    seed : int
        Random seed
    verbose : bool
        Mostrar progresso
        
    Returns
    -------
    Tuple[List[GradientResult], np.ndarray]
        Lista de resultados de gradientes e array de variâncias explicadas
    """
    np.random.seed(seed)
    
    # Binarizar matriz para gerar surrogates
    matrix_binary = (connectivity_matrix > 0).astype(float)
    
    # Extrair pesos originais para redistribuir
    triu_idx = np.triu_indices_from(connectivity_matrix, k=1)
    weights = connectivity_matrix[triu_idx]
    weights = weights[weights > 0]
    
    null_gradients = []
    null_variances = []
    
    if verbose:
        print(f"Generating {n_surrogates} null gradients...")
    
    for i in range(n_surrogates):
        if verbose and (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n_surrogates}")
        
        # Gerar matriz surrogate
        if null_model == 'maslov_sneppen':
            surrogate_binary = nm.maslov_sneppen_rewire(matrix_binary, seed=seed + i)
        elif null_model == 'configuration':
            degree_seq = np.sum(matrix_binary > 0, axis=1)
            surrogate_binary = nm.generate_configuration_model(degree_seq, seed=seed + i)
        elif null_model == 'erdos_renyi':
            n_nodes = matrix_binary.shape[0]
            n_edges = int(np.sum(matrix_binary) / 2)
            surrogate_binary = nm.generate_erdos_renyi(n_nodes, n_edges, seed=seed + i)
        else:
            raise ValueError(f"Unknown null model: {null_model}")
        
        # Redistribuir pesos aleatoriamente
        surrogate_weighted = _redistribute_weights(surrogate_binary, weights, seed + i)
        
        # Calcular gradientes do surrogate
        try:
            grad_result = compute_gradients(
                surrogate_weighted,
                n_components=n_components,
                approach=approach,
                kernel=kernel,
                sparsity=sparsity,
                random_state=seed + i
            )
            null_gradients.append(grad_result)
            null_variances.append(grad_result.explained_variance)
        except Exception as e:
            if verbose:
                warnings.warn(f"Surrogate {i} failed: {e}")
            continue
    
    null_variances = np.array(null_variances)
    
    if verbose:
        print(f"  Generated {len(null_gradients)} valid surrogates")
    
    return null_gradients, null_variances


def _redistribute_weights(
    binary_matrix: np.ndarray,
    weights: np.ndarray,
    seed: int
) -> np.ndarray:
    """
    Redistribuir pesos de forma aleatória em uma matriz binária.
    """
    np.random.seed(seed)
    
    weighted = binary_matrix.copy()
    
    # Encontrar arestas
    triu_idx = np.triu_indices_from(weighted, k=1)
    edge_mask = weighted[triu_idx] > 0
    n_edges = np.sum(edge_mask)
    
    # Permutar pesos
    permuted_weights = np.random.permutation(weights)[:n_edges]
    
    # Atribuir pesos
    edge_indices = np.where(edge_mask)[0]
    for idx, w in zip(edge_indices, permuted_weights):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        weighted[i, j] = weighted[j, i] = w
    
    return weighted


def compare_gradients_to_null(
    observed_result: GradientResult,
    null_gradients: List[GradientResult],
    null_variances: np.ndarray,
    align_gradients: bool = True
) -> GradientNullResult:
    """
    Comparar gradientes observados com distribuição nula.
    
    Parameters
    ----------
    observed_result : GradientResult
        Gradientes observados
    null_gradients : List[GradientResult]
        Lista de gradientes nulos
    null_variances : np.ndarray
        Array de variâncias explicadas dos nulos
    align_gradients : bool
        Se True, alinhar gradientes nulos ao observado antes de comparar
        
    Returns
    -------
    GradientNullResult
        Resultado completo da comparação
    """
    n_surrogates = len(null_gradients)
    n_components = observed_result.n_components
    
    # Métricas observadas
    obs_metrics = compute_gradient_metrics(observed_result.gradients)
    obs_variance = observed_result.explained_variance
    
    # Coletar métricas dos surrogates
    null_dispersion = np.zeros((n_surrogates, n_components))
    null_range = np.zeros((n_surrogates, n_components))
    procrustes_distances = np.zeros(n_surrogates)
    
    for i, null_grad in enumerate(null_gradients):
        null_metrics = compute_gradient_metrics(null_grad.gradients)
        null_dispersion[i] = null_metrics['dispersion']
        null_range[i] = null_metrics['range']
        
        # Procrustes distance
        if align_gradients:
            try:
                # Alinhar e calcular distância
                _, _, disparity = procrustes(
                    observed_result.gradients[:, :3],  # Usar top 3 componentes
                    null_grad.gradients[:, :3]
                )
                procrustes_distances[i] = disparity
            except:
                procrustes_distances[i] = np.nan
    
    # Calcular z-scores e p-values para variância explicada
    variance_zscore = np.zeros(n_components)
    variance_pvalue = np.zeros(n_components)
    
    for c in range(min(n_components, null_variances.shape[1])):
        null_mean = np.mean(null_variances[:, c])
        null_std = np.std(null_variances[:, c])
        
        if null_std > 0:
            variance_zscore[c] = (obs_variance[c] - null_mean) / null_std
        else:
            variance_zscore[c] = 0
        
        # P-value (one-tailed: observado > nulo)
        variance_pvalue[c] = np.mean(null_variances[:, c] >= obs_variance[c])
    
    # Z-scores para dispersion
    dispersion_zscore = np.zeros(n_components)
    dispersion_pvalue = np.zeros(n_components)
    
    for c in range(n_components):
        null_mean = np.mean(null_dispersion[:, c])
        null_std = np.std(null_dispersion[:, c])
        
        if null_std > 0:
            dispersion_zscore[c] = (obs_metrics['dispersion'][c] - null_mean) / null_std
        
        dispersion_pvalue[c] = np.mean(null_dispersion[:, c] >= obs_metrics['dispersion'][c])
    
    return GradientNullResult(
        observed_gradients=observed_result.gradients,
        observed_variance=obs_variance,
        null_variance_distribution=null_variances,
        null_dispersion_distribution=null_dispersion,
        null_range_distribution=null_range,
        variance_zscore=variance_zscore,
        variance_pvalue=variance_pvalue,
        dispersion_zscore=dispersion_zscore,
        dispersion_pvalue=dispersion_pvalue,
        procrustes_distances=procrustes_distances,
        n_surrogates=n_surrogates,
        null_model_type='maslov_sneppen'
    )


# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTION
# =============================================================================

def analyze_gradients_vs_null(
    connectivity_matrix: np.ndarray,
    subject_id: str = None,
    n_components: int = 10,
    n_surrogates: int = 100,
    null_model: str = 'maslov_sneppen',
    approach: str = 'dm',
    kernel: str = 'normalized_angle',
    sparsity: float = 0.9,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Análise completa de gradientes vs modelo nulo para um sujeito.
    
    Esta é a função principal que:
    1. Calcula gradientes observados
    2. Gera gradientes de surrogates
    3. Compara estatisticamente
    4. Retorna resultado estruturado
    
    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz de conectividade funcional
    subject_id : str, optional
        ID do sujeito
    n_components : int
        Número de componentes do gradiente
    n_surrogates : int
        Número de surrogates
    null_model : str
        Tipo de modelo nulo
    approach : str
        Método de embedding ('dm', 'pca', 'le')
    kernel : str
        Kernel para afinidade
    sparsity : float
        Threshold de sparsity (0.9 = top 10%)
    seed : int
        Random seed
    verbose : bool
        Mostrar progresso
        
    Returns
    -------
    Dict
        Dicionário com todos os resultados
    """
    if verbose:
        print(f"\n{'='*60}")
        if subject_id:
            print(f"GRADIENT NULL MODEL ANALYSIS: {subject_id}")
        else:
            print("GRADIENT NULL MODEL ANALYSIS")
        print('='*60)
        print(f"Method: {approach}, Kernel: {kernel}")
        print(f"Components: {n_components}, Surrogates: {n_surrogates}")
    
    # 1. Gradientes observados
    if verbose:
        print("\n1. Computing observed gradients...")
    
    observed = compute_gradients(
        connectivity_matrix,
        n_components=n_components,
        approach=approach,
        kernel=kernel,
        sparsity=sparsity,
        random_state=seed
    )
    
    obs_metrics = compute_gradient_metrics(observed.gradients)
    
    if verbose:
        print(f"   G1 variance: {observed.explained_variance[0]:.3f}")
        print(f"   G2 variance: {observed.explained_variance[1]:.3f}")
        print(f"   Total (G1-G3): {np.sum(observed.explained_variance[:3]):.3f}")
    
    # 2. Gradientes nulos
    if verbose:
        print("\n2. Generating null gradients...")
    
    null_grads, null_vars = generate_null_gradients(
        connectivity_matrix,
        n_surrogates=n_surrogates,
        null_model=null_model,
        n_components=n_components,
        approach=approach,
        kernel=kernel,
        sparsity=sparsity,
        seed=seed,
        verbose=verbose
    )
    
    # 3. Comparação
    if verbose:
        print("\n3. Comparing observed vs null...")
    
    comparison = compare_gradients_to_null(
        observed, null_grads, null_vars, align_gradients=True
    )
    
    # 4. Resultados
    results = {
        'subject_id': subject_id,
        'n_rois': connectivity_matrix.shape[0],
        'n_components': n_components,
        'n_surrogates': n_surrogates,
        'null_model': null_model,
        
        # Gradientes observados
        'observed_gradients': observed.gradients,
        'observed_variance': observed.explained_variance,
        'observed_metrics': obs_metrics,
        
        # Comparação
        'comparison': comparison,
        
        # Estatísticas resumidas
        'G1_variance_observed': observed.explained_variance[0],
        'G1_variance_null_mean': np.mean(null_vars[:, 0]),
        'G1_variance_null_std': np.std(null_vars[:, 0]),
        'G1_variance_zscore': comparison.variance_zscore[0],
        'G1_variance_pvalue': comparison.variance_pvalue[0],
        
        'G2_variance_observed': observed.explained_variance[1],
        'G2_variance_null_mean': np.mean(null_vars[:, 1]),
        'G2_variance_zscore': comparison.variance_zscore[1],
        'G2_variance_pvalue': comparison.variance_pvalue[1],
        
        'G1_dispersion_zscore': comparison.dispersion_zscore[0],
        'G2_dispersion_zscore': comparison.dispersion_zscore[1],
        
        'mean_procrustes_distance': np.nanmean(comparison.procrustes_distances),
    }
    
    if verbose:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"\nGradient 1 (primary hierarchy):")
        print(f"  Variance: {results['G1_variance_observed']:.4f}")
        print(f"  Null mean: {results['G1_variance_null_mean']:.4f} ± {results['G1_variance_null_std']:.4f}")
        print(f"  z-score: {results['G1_variance_zscore']:.2f}")
        print(f"  p-value: {results['G1_variance_pvalue']:.4f}")
        
        sig = "✓ SIGNIFICANT" if results['G1_variance_pvalue'] < 0.05 else "not significant"
        print(f"  → {sig}")
        
        print(f"\nGradient 2:")
        print(f"  Variance: {results['G2_variance_observed']:.4f}")
        print(f"  z-score: {results['G2_variance_zscore']:.2f}")
        
        print(f"\nProcrustes distance (shape similarity):")
        print(f"  Mean: {results['mean_procrustes_distance']:.4f}")
    
    return results


def run_gradient_null_analysis(
    subjects: List[str] = None,
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor',
    n_components: int = 10,
    n_surrogates: int = 100,
    null_model: str = 'maslov_sneppen',
    output_dir: str = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Rodar análise de gradientes vs nulo para múltiplos sujeitos.
    
    Parameters
    ----------
    subjects : List[str], optional
        Lista de subject IDs. Se None, usa todos disponíveis.
    atlas_name : str
        Nome do atlas
    strategy : str
        Estratégia de denoising
    n_components : int
        Número de componentes
    n_surrogates : int
        Número de surrogates por sujeito
    null_model : str
        Tipo de modelo nulo
    output_dir : str, optional
        Diretório de output
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Resultados de todos os sujeitos
    """
    if subjects is None:
        subjects = config.get_available_subjects(atlas_name, strategy)
    
    if not subjects:
        raise ValueError(f"No subjects found for {atlas_name}/{strategy}")
    
    print(f"\n{'='*70}")
    print("GRADIENT NULL MODEL ANALYSIS - COHORT")
    print('='*70)
    print(f"Atlas: {atlas_name}")
    print(f"Subjects: {len(subjects)}")
    print(f"Surrogates per subject: {n_surrogates}")
    print(f"Null model: {null_model}")
    
    all_results = []
    
    for i, sub in enumerate(subjects):
        print(f"\n[{i+1}/{len(subjects)}] {sub}")
        
        try:
            # Carregar matriz
            matrix = config.load_connectivity_matrix(sub, atlas_name, strategy)
            
            # Análise
            results = analyze_gradients_vs_null(
                matrix,
                subject_id=sub,
                n_components=n_components,
                n_surrogates=n_surrogates,
                null_model=null_model,
                seed=seed + i,
                verbose=False
            )
            
            # Extrair estatísticas para DataFrame
            row = {
                'subject_id': sub,
                'G1_variance': results['G1_variance_observed'],
                'G1_variance_null_mean': results['G1_variance_null_mean'],
                'G1_variance_zscore': results['G1_variance_zscore'],
                'G1_variance_pvalue': results['G1_variance_pvalue'],
                'G2_variance': results['G2_variance_observed'],
                'G2_variance_zscore': results['G2_variance_zscore'],
                'G2_variance_pvalue': results['G2_variance_pvalue'],
                'G1_dispersion_zscore': results['G1_dispersion_zscore'],
                'G2_dispersion_zscore': results['G2_dispersion_zscore'],
                'procrustes_distance': results['mean_procrustes_distance'],
                'G1_significant': results['G1_variance_pvalue'] < 0.05,
            }
            
            all_results.append(row)
            
            sig = "✓" if row['G1_significant'] else "×"
            print(f"   G1: var={row['G1_variance']:.3f}, z={row['G1_variance_zscore']:.2f} [{sig}]")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            continue
    
    results_df = pd.DataFrame(all_results)
    
    # Salvar
    if output_dir is None:
        output_dir = config.COMPARISON_DIR / 'gradients_null' / atlas_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'gradient_null_results.csv', index=False)
    
    # Sumário
    print(f"\n{'='*70}")
    print("COHORT SUMMARY")
    print('='*70)
    print(f"Subjects analyzed: {len(results_df)}")
    print(f"\nGradient 1 variance:")
    print(f"  Mean: {results_df['G1_variance'].mean():.4f} ± {results_df['G1_variance'].std():.4f}")
    print(f"  Mean z-score: {results_df['G1_variance_zscore'].mean():.2f}")
    print(f"  % significant: {100 * results_df['G1_significant'].mean():.1f}%")
    print(f"\n✓ Saved to: {output_dir}")
    
    return results_df


# =============================================================================
# CLI / TEST
# =============================================================================

if __name__ == '__main__':
    print("Gradient Null Models Module")
    print(f"BrainSpace available: {HAS_BRAINSPACE}")
    
    # Quick test with random matrix
    print("\nQuick test with random connectivity matrix...")
    np.random.seed(42)
    
    n = 100
    test_matrix = np.random.random((n, n))
    test_matrix = (test_matrix + test_matrix.T) / 2
    np.fill_diagonal(test_matrix, 0)
    
    results = analyze_gradients_vs_null(
        test_matrix,
        subject_id='test',
        n_components=5,
        n_surrogates=20,  # Poucos para teste rápido
        verbose=True
    )
    
    print("\n✓ Test completed successfully!")
