"""
╔══════════════════════════════════════════════════════════════════════╗
║  EXEMPLO: Pipeline de Comparação Normativa                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Este script demonstra como usar o subpacote normative_comparison    ║
║  para comparar métricas topológicas de pacientes COVID com           ║
║  controles saudáveis usando harmonização ComBat no nível de métricas.║
║                                                                      ║
║  WORKFLOW:                                                           ║
║  1. Carregar métricas de grafo dos pacientes COVID                   ║
║  2. Obter métricas de controles (NKI-Rockland ou simulados)          ║
║  3. Aplicar ComBat harmonization no nível das métricas               ║
║  4. Comparar estatisticamente (permutações, effect sizes)            ║
║  5. Gerar visualizações publication-quality                          ║
║                                                                      ║
║  USO:                                                                ║
║    python example_normative_comparison.py                            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adicionar path se necessário
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# IMPORTS
# =============================================================================

from normative_comparison import config
from normative_comparison import control_data
from normative_comparison import harmonization
from normative_comparison import statistics
from normative_comparison import comparison_viz


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Atlas para análise (escolher um)
ATLAS_NAME = 'schaefer_100'  # 'synthseg_86', 'schaefer_100', 'aal3_170', 'brainnetome_246'

# Estratégia de denoising
DENOISING_STRATEGY = 'acompcor'

# Configurações estatísticas
N_PERMUTATIONS = 5000
ALPHA = 0.05


# =============================================================================
# PIPELINE COMPLETO
# =============================================================================

def run_full_pipeline():
    """
    Executar pipeline completo de comparação normativa.
    """
    print("\n" + "="*70)
    print("NORMATIVE COMPARISON PIPELINE")
    print("Comparing COVID patients vs Healthy Controls using Graph Metrics")
    print("="*70)
    
    # =========================================================================
    # STEP 1: Carregar métricas COVID
    # =========================================================================
    print("\n" + "-"*50)
    print("STEP 1: Loading COVID patient metrics")
    print("-"*50)
    
    try:
        covid_df = config.load_covid_global_metrics(
            subjects=None,  # Todos os disponíveis
            atlas_name=ATLAS_NAME,
            strategy=DENOISING_STRATEGY
        )
        print(f"✓ Loaded {len(covid_df)} COVID subjects")
        print(f"  Metrics available: {[c for c in covid_df.columns if c not in ['subject_id', 'group', 'site']][:5]}...")
        
    except Exception as e:
        print(f"✗ Error loading COVID metrics: {e}")
        print("\nUsing simulated COVID data for demonstration...")
        
        # Simular dados COVID para demonstração
        np.random.seed(42)
        n_covid = 23
        covid_df = pd.DataFrame({
            'subject_id': [f'sub-{str(i).zfill(2)}' for i in range(1, n_covid + 1)],
            'group': ['covid'] * n_covid,
            'site': ['local'] * n_covid,
            'modularity': np.random.normal(0.42, 0.08, n_covid),  # Ligeiramente menor que controles
            'global_efficiency': np.random.normal(0.48, 0.06, n_covid),
            'local_efficiency': np.random.normal(0.68, 0.05, n_covid),
            'mean_clustering': np.random.normal(0.44, 0.07, n_covid),
            'mean_degree': np.random.normal(14.5, 2.0, n_covid),
            'mean_strength': np.random.normal(8.0, 1.5, n_covid),
            'mean_betweenness': np.random.normal(0.022, 0.005, n_covid),
            'assortativity': np.random.normal(0.12, 0.10, n_covid),
            'transitivity': np.random.normal(0.42, 0.08, n_covid),
        })
    
    print(f"\nCOVID data summary:")
    print(covid_df.describe().round(3))
    
    # =========================================================================
    # STEP 2: Obter métricas de controles
    # =========================================================================
    print("\n" + "-"*50)
    print("STEP 2: Loading/generating control metrics")
    print("-"*50)
    
    control_df = control_data.get_control_metrics(
        source='simulated',  # 'nki_rockland', 'hcp_1200', 'precomputed', 'simulated'
        atlas_name=ATLAS_NAME,
        n_subjects=50,  # Número de controles
        age_range=(20, 60)
    )
    
    print(f"\nControl data summary:")
    print(control_df.describe().round(3))
    
    # =========================================================================
    # STEP 3: Diagnóstico de batch effects
    # =========================================================================
    print("\n" + "-"*50)
    print("STEP 3: Diagnosing batch effects")
    print("-"*50)
    
    batch_diagnosis = harmonization.diagnose_batch_effects(covid_df, control_df)
    print("\nBatch effect diagnosis:")
    print(batch_diagnosis[['metric', 'covid_mean', 'control_mean', 'variance_ratio', 'cohens_d']].round(3))
    
    # =========================================================================
    # STEP 4: Harmonização ComBat
    # =========================================================================
    print("\n" + "-"*50)
    print("STEP 4: ComBat harmonization (metric-level)")
    print("-"*50)
    
    covid_harm, control_harm = harmonization.combat_harmonize_metrics(
        covid_df, 
        control_df,
        metrics_to_harmonize=config.ROBUST_METRICS,
        parametric=True,
        eb=True
    )
    
    # Comparar antes/depois
    comparison = harmonization.compare_before_after(
        covid_df, covid_harm, 
        control_df, control_harm
    )
    print("\nBefore vs After harmonization:")
    print(comparison[['metric', 'diff_original', 'diff_harmonized', 'diff_preserved']].round(3))
    
    # =========================================================================
    # STEP 5: Comparação estatística
    # =========================================================================
    print("\n" + "-"*50)
    print("STEP 5: Statistical comparison (permutation tests)")
    print("-"*50)
    
    stats_results = statistics.compare_groups(
        covid_harm, 
        control_harm,
        metrics=config.ROBUST_METRICS,
        n_permutations=N_PERMUTATIONS,
        alpha=ALPHA,
        correction_method='fdr_bh',
        effect_size_method='hedges_g'
    )
    
    # Tabela resumo para publicação
    summary_table = statistics.generate_summary_table(stats_results, format_type='publication')
    print("\n" + "="*70)
    print("RESULTS SUMMARY (Publication Format)")
    print("="*70)
    print(summary_table.to_string(index=False))
    
    # =========================================================================
    # STEP 6: Z-scores normativos
    # =========================================================================
    print("\n" + "-"*50)
    print("STEP 6: Computing normative z-scores")
    print("-"*50)
    
    z_scores = statistics.compute_z_scores(covid_harm, control_harm)
    print(f"\nZ-scores computed for {len(z_scores)} subjects")
    
    extreme_subjects = statistics.identify_extreme_subjects(z_scores, threshold=2.0)
    if len(extreme_subjects) > 0:
        print(f"\nSubjects with extreme values (|z| > 2):")
        print(extreme_subjects[['subject_id', 'n_extreme', 'composite_zscore']].head(10))
    else:
        print("\nNo subjects with extreme values detected")
    
    # =========================================================================
    # STEP 7: Visualizações
    # =========================================================================
    print("\n" + "-"*50)
    print("STEP 7: Generating visualizations")
    print("-"*50)
    
    output_dir = Path(config.COMPARISON_DIR) / ATLAS_NAME
    
    viz_paths = comparison_viz.generate_comparison_report(
        covid_harm, 
        control_harm,
        stats_results,
        atlas_name=ATLAS_NAME,
        output_dir=output_dir
    )
    
    print("\nGenerated files:")
    for name, path in viz_paths.items():
        print(f"  {name}: {path}")
    
    # =========================================================================
    # CONCLUSÃO
    # =========================================================================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    # Sumário de achados significativos
    sig_metrics = stats_results[stats_results['significant_perm']]
    
    if len(sig_metrics) > 0:
        print(f"\n🔬 SIGNIFICANT FINDINGS ({len(sig_metrics)} metrics):")
        for _, row in sig_metrics.iterrows():
            direction = "INCREASED" if row['mean_diff'] > 0 else "DECREASED"
            print(f"\n  • {row['metric'].replace('_', ' ').title()}: {direction} in COVID")
            print(f"    Effect size: g = {row['hedges_g']:.3f} ({row['effect_interpretation']})")
            print(f"    95% CI: [{row['effect_ci_lower']:.3f}, {row['effect_ci_upper']:.3f}]")
            print(f"    p-value (FDR): {row['p_perm_corrected']:.4f}")
    else:
        print("\n⚠️ No significant differences found after FDR correction.")
        print("   This could indicate:")
        print("   - No true effect")
        print("   - Insufficient statistical power (n=23 is modest)")
        print("   - Effect heterogeneity across patients")
    
    return {
        'covid_harmonized': covid_harm,
        'control_harmonized': control_harm,
        'statistics': stats_results,
        'z_scores': z_scores,
        'visualizations': viz_paths
    }


# =============================================================================
# QUICK COMPARISON (simplified version)
# =============================================================================

def quick_comparison(atlas_name='schaefer_100'):
    """
    Versão simplificada para comparação rápida.
    """
    from normative_comparison import run_normative_comparison
    
    results = run_normative_comparison(
        covid_subjects=None,  # Todos
        atlas_name=atlas_name,
        control_source='simulated',  # Usar 'nki_rockland' quando disponível
        harmonize=True,
        n_permutations=1000,  # Menos permutações para teste rápido
        output_dir=None
    )
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Normative comparison pipeline')
    parser.add_argument('--atlas', type=str, default='schaefer_100',
                       choices=['synthseg_86', 'schaefer_100', 'aal3_170', 'brainnetome_246'],
                       help='Atlas to use')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick comparison (fewer permutations)')
    
    args = parser.parse_args()
    
    if args.quick:
        results = quick_comparison(args.atlas)
    else:
        results = run_full_pipeline()
