"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SARS Library - Gradients Module                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Análise de gradientes de conectividade cerebral.                            ║
║                                                                              ║
║  CONCEITO:                                                                   ║
║  Gradientes de conectividade descrevem a organização macroscópica            ║
║  do córtex ao longo de eixos contínuos (Margulies et al., 2016).            ║
║  O primeiro gradiente (G1) tipicamente captura a hierarquia                  ║
║  sensorimotor → associativo.                                                 ║
║                                                                              ║
║  MÓDULOS:                                                                    ║
║  - core: Computação de gradientes (DM, PCA, LE, UMAP)                       ║
║  - viz: Visualizações                                                        ║
║  - null_models: Comparação com modelos nulos                                 ║
║  - null_viz: Visualizações dos modelos nulos                                 ║
║                                                                              ║
║  USO RÁPIDO:                                                                 ║
║    from sars.gradients import compute_gradients, quick_gradients             ║
║    result = compute_gradients(fc_matrix)                                     ║
║    g1 = result.G1  # Primeiro gradiente                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

__version__ = '0.1.0'
__author__ = 'Velho Mago'

# =============================================================================
# IMPORTS FROM SUBMODULES
# =============================================================================

# Core functions
from .core import (
    # Main function
    compute_gradients,
    
    # Convenience functions
    quick_gradients,
    compare_methods,
    
    # Group analysis
    compute_group_gradients,
    
    # Alignment
    align_gradients_procrustes,
    align_gradients_joint,
    
    # Affinity/kernel
    compute_affinity,
    
    # Low-level methods
    compute_diffusion_map,
    compute_laplacian_eigenmaps,
    compute_pca_embedding,
    
    # Statistics
    compute_gradient_stats,
    compute_gradient_correlation,
    
    # Data classes
    GradientResult,
    GroupGradientResult,
)

# =============================================================================
# HIGH-LEVEL API
# =============================================================================

def compute_subject_gradients(
    subject_id: str,
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor',
    n_components: int = 10,
    approach: str = 'dm',
    kernel: str = 'normalized_angle',
    sparsity: float = 0.9,
    **kwargs
) -> 'GradientResult':
    """
    Computar gradientes para um sujeito específico.
    
    Esta função carrega automaticamente a matriz de conectividade
    do sujeito e computa os gradientes.
    
    Parameters
    ----------
    subject_id : str
        ID do sujeito (e.g., 'sub-01')
    atlas_name : str
        Nome do atlas
    strategy : str
        Estratégia de denoising
    n_components : int
        Número de componentes
    approach : str
        Método: 'dm' (Diffusion Map), 'pca', 'le', 'umap'
    kernel : str
        Kernel: 'normalized_angle', 'cosine', 'pearson', 'gaussian'
    sparsity : float
        Proporção de conexões fracas a zerar (0.9 = manter top 10%)
    **kwargs
        Parâmetros adicionais para o método
        
    Returns
    -------
    GradientResult
        Objeto com gradientes e estatísticas
        
    Examples
    --------
    >>> from sars.gradients import compute_subject_gradients
    >>> result = compute_subject_gradients('sub-01')
    >>> print(f"G1 variance: {result.explained_variance[0]:.3f}")
    >>> g1 = result.G1  # Primeiro gradiente
    """
    from sars.normative_comparison import config
    
    # Carregar matriz
    matrix = config.load_connectivity_matrix(subject_id, atlas_name, strategy)
    
    # Computar gradientes
    result = compute_gradients(
        matrix,
        n_components=n_components,
        approach=approach,
        kernel=kernel,
        sparsity=sparsity,
        subject_id=subject_id,
        **kwargs
    )
    
    return result


def compute_cohort_gradients(
    subjects: list = None,
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor',
    n_components: int = 10,
    approach: str = 'dm',
    kernel: str = 'normalized_angle',
    sparsity: float = 0.9,
    align: bool = True,
    verbose: bool = True,
    **kwargs
) -> 'GroupGradientResult':
    """
    Computar e alinhar gradientes para múltiplos sujeitos.
    
    Parameters
    ----------
    subjects : list, optional
        Lista de subject IDs. Se None, usa todos disponíveis.
    atlas_name : str
        Nome do atlas
    strategy : str
        Estratégia de denoising
    n_components : int
        Número de componentes
    approach : str
        Método de embedding
    kernel : str
        Kernel para afinidade
    sparsity : float
        Threshold de sparsity
    align : bool
        Se True, alinha gradientes entre sujeitos (Procrustes)
    verbose : bool
        Mostrar progresso
    **kwargs
        Parâmetros adicionais
        
    Returns
    -------
    GroupGradientResult
        Resultado com gradientes individuais, alinhados e estatísticas
        
    Examples
    --------
    >>> from sars.gradients import compute_cohort_gradients
    >>> result = compute_cohort_gradients(atlas_name='schaefer_100')
    >>> print(f"Subjects: {result.n_subjects}")
    >>> group_mean = result.group_mean
    """
    from sars.normative_comparison import config
    
    if subjects is None:
        subjects = config.get_available_subjects(atlas_name, strategy)
    
    if not subjects:
        raise ValueError(f"Nenhum sujeito encontrado para {atlas_name}/{strategy}")
    
    # Carregar matrizes
    connectivity_matrices = {}
    for sub in subjects:
        try:
            connectivity_matrices[sub] = config.load_connectivity_matrix(
                sub, atlas_name, strategy
            )
        except Exception as e:
            if verbose:
                print(f"Warning: Could not load {sub}: {e}")
    
    if not connectivity_matrices:
        raise ValueError("Nenhuma matriz carregada com sucesso")
    
    # Computar gradientes de grupo
    result = compute_group_gradients(
        connectivity_matrices,
        n_components=n_components,
        approach=approach,
        kernel=kernel,
        sparsity=sparsity,
        align=align,
        verbose=verbose,
        **kwargs
    )
    
    return result


def run_gradient_analysis(
    subjects: list = None,
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor',
    n_components: int = 10,
    approach: str = 'dm',
    kernel: str = 'normalized_angle',
    sparsity: float = 0.9,
    align: bool = True,
    compare_to_null: bool = False,
    n_surrogates: int = 100,
    output_dir: str = None,
    generate_figures: bool = True,
    verbose: bool = True
) -> dict:
    """
    Pipeline completo de análise de gradientes.
    
    Esta é a função principal que:
    1. Computa gradientes para todos os sujeitos
    2. Alinha entre sujeitos
    3. Opcionalmente compara com modelos nulos
    4. Gera visualizações
    5. Salva resultados
    
    Parameters
    ----------
    subjects : list, optional
        Lista de subject IDs
    atlas_name : str
        Nome do atlas
    strategy : str
        Estratégia de denoising
    n_components : int
        Número de componentes
    approach : str
        Método de embedding
    kernel : str
        Kernel para afinidade
    sparsity : float
        Threshold de sparsity
    align : bool
        Se True, alinha gradientes entre sujeitos
    compare_to_null : bool
        Se True, compara com modelos nulos
    n_surrogates : int
        Número de surrogates (se compare_to_null=True)
    output_dir : str, optional
        Diretório de output
    generate_figures : bool
        Se True, gera visualizações
    verbose : bool
        Mostrar progresso
        
    Returns
    -------
    dict
        - group_result: GroupGradientResult
        - null_results: DataFrame (se compare_to_null=True)
        - figures: Dict[str, Path] (se generate_figures=True)
        - output_dir: Path
        
    Examples
    --------
    >>> from sars.gradients import run_gradient_analysis
    >>> results = run_gradient_analysis(
    ...     atlas_name='schaefer_100',
    ...     compare_to_null=True,
    ...     n_surrogates=100
    ... )
    """
    from pathlib import Path
    import pandas as pd
    from sars.normative_comparison import config
    
    if verbose:
        print("\n" + "="*70)
        print("GRADIENT ANALYSIS PIPELINE")
        print("="*70)
        print(f"Atlas: {atlas_name}")
        print(f"Method: {approach}, Kernel: {kernel}")
        print(f"Components: {n_components}")
        print(f"Align: {align}")
        print(f"Compare to null: {compare_to_null}")
    
    # Output directory
    if output_dir is None:
        output_dir = config.COMPARISON_DIR / 'gradients' / atlas_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Compute cohort gradients
    if verbose:
        print("\n1. Computing gradients...")
    
    group_result = compute_cohort_gradients(
        subjects=subjects,
        atlas_name=atlas_name,
        strategy=strategy,
        n_components=n_components,
        approach=approach,
        kernel=kernel,
        sparsity=sparsity,
        align=align,
        verbose=verbose
    )
    
    if verbose:
        print(f"   ✓ {group_result.n_subjects} subjects")
    
    # Save basic results
    results_data = []
    for sub_id, result in group_result.individual_gradients.items():
        row = {
            'subject_id': sub_id,
            'G1_variance': result.explained_variance[0],
            'G2_variance': result.explained_variance[1] if len(result.explained_variance) > 1 else None,
            'G3_variance': result.explained_variance[2] if len(result.explained_variance) > 2 else None,
            'total_variance_G1G3': sum(result.explained_variance[:3]),
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(output_dir / 'gradient_results.csv', index=False)
    
    # 2. Compare to null if requested
    null_results = None
    if compare_to_null:
        if verbose:
            print("\n2. Comparing to null models...")
        
        from . import null_models as gnm
        
        null_data = []
        for sub_id, result in group_result.individual_gradients.items():
            if verbose:
                print(f"   {sub_id}...", end=" ")
            
            try:
                matrix = config.load_connectivity_matrix(sub_id, atlas_name, strategy)
                null_result = gnm.analyze_gradients_vs_null(
                    matrix,
                    subject_id=sub_id,
                    n_components=n_components,
                    n_surrogates=n_surrogates,
                    approach=approach,
                    kernel=kernel,
                    sparsity=sparsity,
                    verbose=False
                )
                
                null_data.append({
                    'subject_id': sub_id,
                    'G1_variance': null_result['G1_variance_observed'],
                    'G1_null_mean': null_result['G1_variance_null_mean'],
                    'G1_zscore': null_result['G1_variance_zscore'],
                    'G1_pvalue': null_result['G1_variance_pvalue'],
                    'G1_significant': null_result['G1_variance_pvalue'] < 0.05,
                })
                
                sig = "✓" if null_result['G1_variance_pvalue'] < 0.05 else "×"
                if verbose:
                    print(f"z={null_result['G1_variance_zscore']:.2f} [{sig}]")
                    
            except Exception as e:
                if verbose:
                    print(f"ERROR: {e}")
        
        null_results = pd.DataFrame(null_data)
        null_results.to_csv(output_dir / 'gradient_null_results.csv', index=False)
        
        if verbose:
            pct_sig = 100 * null_results['G1_significant'].mean()
            print(f"   ✓ {pct_sig:.1f}% significant G1")
    
    # 3. Generate figures
    figures = {}
    if generate_figures:
        if verbose:
            print("\n3. Generating figures...")
        
        from . import viz
        
        fig_dir = output_dir / 'figures'
        fig_dir.mkdir(exist_ok=True)
        
        # Group gradients plot
        try:
            fig = viz.plot_group_gradients(
                group_result.aligned_gradients,
                group_result.group_mean
            )
            path = fig_dir / 'group_gradients.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
            figures['group_gradients'] = path
            if verbose:
                print(f"   ✓ group_gradients.png")
        except Exception as e:
            if verbose:
                print(f"   ✗ group_gradients: {e}")
        
        # Scree plot (use first subject as example)
        first_result = list(group_result.individual_gradients.values())[0]
        try:
            fig = viz.plot_scree(first_result.explained_variance)
            path = fig_dir / 'scree_plot.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            figures['scree_plot'] = path
            if verbose:
                print(f"   ✓ scree_plot.png")
        except Exception as e:
            if verbose:
                print(f"   ✗ scree_plot: {e}")
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Subjects: {group_result.n_subjects}")
        print(f"G1 variance: {results_df['G1_variance'].mean():.4f} ± {results_df['G1_variance'].std():.4f}")
        if null_results is not None:
            print(f"G1 significant: {100 * null_results['G1_significant'].mean():.1f}%")
        print(f"\n✓ Results saved to: {output_dir}")
    
    return {
        'group_result': group_result,
        'results_df': results_df,
        'null_results': null_results,
        'figures': figures,
        'output_dir': str(output_dir)
    }


# =============================================================================
# LAZY MODULE LOADING
# =============================================================================

__all__ = [
    # High-level API
    'run_gradient_analysis',
    'compute_subject_gradients',
    'compute_cohort_gradients',
    
    # Core functions
    'compute_gradients',
    'quick_gradients',
    'compare_methods',
    'compute_group_gradients',
    
    # Alignment
    'align_gradients_procrustes',
    'align_gradients_joint',
    
    # Low-level
    'compute_affinity',
    'compute_diffusion_map',
    'compute_laplacian_eigenmaps',
    'compute_pca_embedding',
    
    # Statistics
    'compute_gradient_stats',
    'compute_gradient_correlation',
    
    # Data classes
    'GradientResult',
    'GroupGradientResult',
    
    # Submodules
    'core',
    'viz',
    'null_models',
    'null_viz',
]


def __getattr__(name):
    """Lazy loading of submodules."""
    import importlib
    
    submodules = {'core', 'viz', 'null_models', 'null_viz'}
    
    if name in submodules:
        return importlib.import_module(f'.{name}', package=__name__)
    
    raise AttributeError(f"module 'gradients' has no attribute '{name}'")
