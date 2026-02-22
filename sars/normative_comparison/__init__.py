"""
Normative Comparison - Análise com modelos nulos.
"""

__all__ = ['run_null_model_analysis', 'plot_null_model_results', 'config', 'null_models', 'null_models_viz']


def run_null_model_analysis(
    covid_subjects: list = None,
    atlas_name: str = 'schaefer_100',
    strategy: str = 'acompcor',
    null_model: str = 'maslov_sneppen',
    n_surrogates: int = 1000,
    metrics: list = None,
    threshold_percentile: float = 85,
    output_dir: str = None
):
    """
    Análise com modelos nulos.
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from . import config
    from . import null_models as nm
    
    # Sujeitos
    if covid_subjects is None:
        covid_subjects = config.get_available_subjects(atlas_name, strategy)
    
    if not covid_subjects:
        raise ValueError(f"Nenhum sujeito encontrado para {atlas_name}/{strategy}")
    
    if metrics is None:
        metrics = ['clustering', 'efficiency']
    
    print(f"\n{'='*60}")
    print("NULL MODEL ANALYSIS")
    print('='*60)
    print(f"Atlas: {atlas_name} ({config.get_atlas_n_rois(atlas_name)} ROIs)")
    print(f"Strategy: {strategy}")
    print(f"Subjects: {len(covid_subjects)}")
    print(f"Null model: {null_model}")
    print(f"Surrogates: {n_surrogates}")
    
    all_results = []
    
    for i, sub in enumerate(covid_subjects):
        print(f"\n[{i+1}/{len(covid_subjects)}] {sub}", end=" ")
        
        try:
            adj = config.load_adjacency_matrix(sub, atlas_name, strategy, threshold_percentile)
            
            if adj.max() > 1:
                adj = (adj > 0).astype(float)
            
            n_nodes = adj.shape[0]
            n_edges = int(np.sum(adj) / 2)
            
            sw = nm.compute_small_worldness(adj, n_surrogates=min(100, n_surrogates))
            mod = nm.compute_normalized_modularity(adj, n_surrogates=min(100, n_surrogates))
            basic = nm.analyze_network_topology(adj, metrics=metrics, null_model=null_model, n_surrogates=n_surrogates)
            
            result = {
                'subject_id': sub,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': 2 * n_edges / (n_nodes * (n_nodes - 1)),
                'C_observed': sw['C_observed'],
                'C_random': sw['C_random'],
                'L_observed': sw['L_observed'],
                'L_random': sw['L_random'],
                'sigma': sw['sigma'],
                'omega': sw['omega'],
                'is_small_world': sw['is_small_world'],
                'Q_observed': mod['Q_observed'],
                'Q_random_mean': mod['Q_random_mean'],
                'Q_normalized': mod['Q_normalized'],
                'Q_zscore': mod['z_score'],
                'n_communities': mod['n_communities'],
            }
            
            for _, row in basic.iterrows():
                result[f"{row['metric']}_observed"] = row['observed']
                result[f"{row['metric']}_zscore"] = row['z_score']
                result[f"{row['metric']}_pvalue"] = row['p_value']
            
            all_results.append(result)
            
            flag = "SW" if sw['is_small_world'] else "--"
            print(f"σ={sw['sigma']:.2f} Q={mod['Q_observed']:.3f} [{flag}]")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    if not all_results:
        raise ValueError("Nenhum sujeito processado!")
    
    results_df = pd.DataFrame(all_results)
    
    cohort_summary = {
        'n_subjects': len(results_df),
        'atlas': atlas_name,
        'strategy': strategy,
        'sigma_mean': results_df['sigma'].mean(),
        'sigma_std': results_df['sigma'].std(),
        'omega_mean': results_df['omega'].mean(),
        'omega_std': results_df['omega'].std(),
        'pct_small_world': 100 * results_df['is_small_world'].mean(),
        'Q_mean': results_df['Q_observed'].mean(),
        'Q_std': results_df['Q_observed'].std(),
    }
    
    if output_dir is None:
        output_dir = config.COMPARISON_DIR / 'null_models' / atlas_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / 'individual_results.csv', index=False)
    pd.DataFrame([cohort_summary]).to_csv(output_dir / 'cohort_summary.csv', index=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Subjects: {cohort_summary['n_subjects']}")
    print(f"σ = {cohort_summary['sigma_mean']:.3f} ± {cohort_summary['sigma_std']:.3f}")
    print(f"ω = {cohort_summary['omega_mean']:.3f} ± {cohort_summary['omega_std']:.3f}")
    print(f"Small-world: {cohort_summary['pct_small_world']:.1f}%")
    print(f"Q = {cohort_summary['Q_mean']:.3f} ± {cohort_summary['Q_std']:.3f}")
    print(f"\n✓ Saved to: {output_dir}")
    
    return {
        'individual_results': results_df,
        'cohort_summary': cohort_summary,
        'output_dir': str(output_dir)
    }


def plot_null_model_results(
    results: dict,
    output_dir: str = None
) -> dict:
    """
    Gerar visualizações dos resultados da análise com modelos nulos.
    
    Parameters
    ----------
    results : dict
        Output de run_null_model_analysis()
    output_dir : str, optional
        Diretório para salvar. Se None, usa o mesmo da análise.
        
    Returns
    -------
    dict
        Paths das figuras geradas
    """
    from pathlib import Path
    from . import null_models_viz as viz
    
    results_df = results['individual_results']
    atlas = results['cohort_summary'].get('atlas', 'unknown')
    
    if output_dir is None:
        output_dir = Path(results['output_dir']) / 'figures'
    else:
        output_dir = Path(output_dir)
    
    return viz.generate_full_report(results_df, output_dir, atlas)


def __getattr__(name):
    import importlib
    if name in ('config', 'null_models', 'null_models_viz'):
        return importlib.import_module(f'.{name}', package=__name__)
    raise AttributeError(f"module has no attribute '{name}'")
