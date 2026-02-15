"""
╔══════════════════════════════════════════════════════════════════════╗
║  EXEMPLO: Análise com Modelos Nulos                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Este script demonstra como usar modelos nulos para avaliar          ║
║  propriedades topológicas de redes cerebrais de pacientes COVID.     ║
║                                                                      ║
║  VANTAGENS SOBRE COMPARAÇÃO COM CONTROLES:                           ║
║  1. Cada paciente é comparado consigo mesmo (rede randomizada)       ║
║  2. Elimina completamente problemas de harmonização cross-site       ║
║  3. Metodologicamente mais sólido e defensável                       ║
║  4. Permite testar se propriedades são "não-aleatórias"              ║
║                                                                      ║
║  MÉTRICAS ANALISADAS:                                                ║
║  - Small-worldness (σ, ω): Rede é small-world?                       ║
║  - Modularity normalizada: Organização modular significativa?        ║
║  - Rich-club normalizado: Hubs densamente conectados?                ║
║  - Clustering, efficiency, path length vs random                     ║
║                                                                      ║
║  USO:                                                                ║
║    python example_null_models.py                                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Adicionar path se necessário
sys.path.insert(0, str(Path(__file__).parent.parent))

from normative_comparison import null_models
from normative_comparison import config

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

ATLAS_NAME = 'schaefer_100'
DENOISING_STRATEGY = 'acompcor'
N_SURROGATES = 100  # Usar 1000 para análise final
NULL_MODEL = 'maslov_sneppen'  # Preserva grau de cada nó


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def load_adjacency_matrix(subject_id: str, atlas_name: str = ATLAS_NAME) -> np.ndarray:
    """
    Carregar matriz de adjacência de um sujeito.
    
    Ajustar paths conforme sua estrutura de diretórios.
    """
    sub_label = subject_id.replace('sub-', '')
    
    # Tentar carregar matriz thresholded
    adj_file = (config.METRICS_DIR / 'graph' / atlas_name / DENOISING_STRATEGY / 
                f'sub-{sub_label}' / 'adjacency_matrix.npy')
    
    if adj_file.exists():
        return np.load(adj_file)
    
    # Fallback: carregar connectivity e aplicar threshold
    conn_file = (config.CONNECT_DIR / atlas_name / DENOISING_STRATEGY /
                 f'sub-{sub_label}' / 'connectivity_correlation_fisherz.npy')
    
    if conn_file.exists():
        matrix = np.load(conn_file)
        # Threshold proporcional (15%)
        triu_idx = np.triu_indices_from(matrix, k=1)
        values = matrix[triu_idx]
        threshold = np.percentile(values, 85)  # top 15%
        adj = (matrix >= threshold).astype(float)
        np.fill_diagonal(adj, 0)
        return adj
    
    raise FileNotFoundError(f"Matrix not found for {subject_id}")


def simulate_covid_network(n_nodes: int = 100, seed: int = 42) -> np.ndarray:
    """
    Simular uma rede cerebral para demonstração.
    
    Cria uma rede com propriedades small-world típicas de redes cerebrais.
    """
    np.random.seed(seed)
    
    # Começar com lattice (alta clustering)
    adj = null_models.generate_lattice(n_nodes, k_neighbors=5)
    
    # Rewiring parcial (Watts-Strogatz style)
    p_rewire = 0.1
    triu_idx = np.triu_indices(n_nodes, k=1)
    
    for idx in range(len(triu_idx[0])):
        if adj[triu_idx[0][idx], triu_idx[1][idx]] > 0:
            if np.random.random() < p_rewire:
                i = triu_idx[0][idx]
                j = triu_idx[1][idx]
                
                # Escolher novo alvo
                new_j = np.random.randint(n_nodes)
                while new_j == i or adj[i, new_j] > 0:
                    new_j = np.random.randint(n_nodes)
                
                # Rewire
                adj[i, j] = adj[j, i] = 0
                adj[i, new_j] = adj[new_j, i] = 1
    
    return adj


# =============================================================================
# ANÁLISE INDIVIDUAL
# =============================================================================

def analyze_single_subject(
    subject_id: str,
    adj: np.ndarray = None,
    n_surrogates: int = N_SURROGATES,
    verbose: bool = True
) -> dict:
    """
    Análise completa de um sujeito com modelos nulos.
    
    Parameters
    ----------
    subject_id : str
        ID do sujeito
    adj : np.ndarray, optional
        Matriz de adjacência. Se None, carrega do disco.
    n_surrogates : int
        Número de redes surrogate
    verbose : bool
        Imprimir progresso
        
    Returns
    -------
    dict
        Resultados completos da análise
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"NULL MODEL ANALYSIS: {subject_id}")
        print('='*60)
    
    # Carregar matriz
    if adj is None:
        try:
            adj = load_adjacency_matrix(subject_id)
        except FileNotFoundError:
            print(f"Using simulated network for {subject_id}")
            adj = simulate_covid_network(100, seed=hash(subject_id) % 10000)
    
    n_nodes = adj.shape[0]
    n_edges = int(np.sum(adj) / 2)
    density = 2 * n_edges / (n_nodes * (n_nodes - 1))
    
    if verbose:
        print(f"\nNetwork: {n_nodes} nodes, {n_edges} edges, density={density:.3f}")
    
    results = {
        'subject_id': subject_id,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density
    }
    
    # =========================================================================
    # 1. SMALL-WORLDNESS
    # =========================================================================
    if verbose:
        print(f"\n1. Computing small-worldness (n={n_surrogates} surrogates)...")
    
    sw = null_models.compute_small_worldness(adj, n_surrogates=n_surrogates)
    
    results.update({
        'C_observed': sw['C_observed'],
        'C_random': sw['C_random'],
        'C_lattice': sw['C_lattice'],
        'L_observed': sw['L_observed'],
        'L_random': sw['L_random'],
        'sigma': sw['sigma'],
        'omega': sw['omega'],
        'is_small_world': sw['is_small_world']
    })
    
    if verbose:
        print(f"   Clustering: observed={sw['C_observed']:.3f}, random={sw['C_random']:.3f}")
        print(f"   Path length: observed={sw['L_observed']:.3f}, random={sw['L_random']:.3f}")
        print(f"   σ = {sw['sigma']:.3f} (>1 = small-world)")
        print(f"   ω = {sw['omega']:.3f} (≈0 = small-world)")
        print(f"   → {'✓ SMALL-WORLD' if sw['is_small_world'] else '✗ NOT small-world'}")
    
    # =========================================================================
    # 2. MODULARITY NORMALIZADA
    # =========================================================================
    if verbose:
        print(f"\n2. Computing normalized modularity...")
    
    mod = null_models.compute_normalized_modularity(adj, n_surrogates=n_surrogates)
    
    results.update({
        'Q_observed': mod['Q_observed'],
        'Q_random_mean': mod['Q_random_mean'],
        'Q_random_std': mod['Q_random_std'],
        'Q_normalized': mod['Q_normalized'],
        'Q_zscore': mod['z_score'],
        'n_communities': mod['n_communities']
    })
    
    if verbose:
        print(f"   Q observed: {mod['Q_observed']:.3f}")
        print(f"   Q random: {mod['Q_random_mean']:.3f} ± {mod['Q_random_std']:.3f}")
        print(f"   Q normalized: {mod['Q_normalized']:.3f}")
        print(f"   z-score: {mod['z_score']:.2f}")
        print(f"   Communities: {mod['n_communities']}")
        sig = "✓ SIGNIFICANT" if mod['z_score'] > 2 else "~ not significant"
        print(f"   → {sig}")
    
    # =========================================================================
    # 3. RICH-CLUB ORGANIZATION
    # =========================================================================
    if verbose:
        print(f"\n3. Computing rich-club coefficients...")
    
    try:
        rc = null_models.compute_normalized_rich_club(adj, n_surrogates=min(50, n_surrogates))
        
        if len(rc) > 0:
            # Encontrar k com maior rich-club normalizado
            max_rc_idx = rc['phi_normalized'].idxmax()
            max_rc = rc.loc[max_rc_idx]
            
            results.update({
                'rc_max_k': int(max_rc['k']),
                'rc_phi_observed': max_rc['phi_observed'],
                'rc_phi_normalized': max_rc['phi_normalized'],
                'rc_significant': max_rc['significant'],
                'n_significant_k': rc['significant'].sum()
            })
            
            if verbose:
                print(f"   Max ϕ_norm at k={int(max_rc['k'])}: {max_rc['phi_normalized']:.3f}")
                print(f"   Significant k values: {rc['significant'].sum()}/{len(rc)}")
                sig = "✓ RICH-CLUB" if max_rc['phi_normalized'] > 1.1 else "~ weak/no rich-club"
                print(f"   → {sig}")
        else:
            if verbose:
                print("   (insufficient high-degree nodes)")
                
    except Exception as e:
        if verbose:
            print(f"   Error: {e}")
    
    # =========================================================================
    # 4. ANÁLISE DE MÉTRICAS BÁSICAS VS NULL
    # =========================================================================
    if verbose:
        print(f"\n4. Analyzing basic metrics vs null model...")
    
    basic_results = null_models.analyze_network_topology(
        adj=adj,
        metrics=['clustering', 'efficiency'],
        null_model=NULL_MODEL,
        n_surrogates=n_surrogates
    )
    
    for _, row in basic_results.iterrows():
        results[f"{row['metric']}_observed"] = row['observed']
        results[f"{row['metric']}_null_mean"] = row['null_mean']
        results[f"{row['metric']}_zscore"] = row['z_score']
        results[f"{row['metric']}_pvalue"] = row['p_value']
        
        if verbose:
            sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
            print(f"   {row['metric']}: z={row['z_score']:.2f}, p={row['p_value']:.4f} {sig}")
    
    return results


# =============================================================================
# ANÁLISE DE GRUPO
# =============================================================================

def analyze_covid_cohort(
    subjects: list = None,
    n_surrogates: int = N_SURROGATES,
    use_simulated: bool = True
) -> pd.DataFrame:
    """
    Analisar coorte completa de pacientes COVID.
    
    Parameters
    ----------
    subjects : list, optional
        Lista de subject IDs. Se None, usa todos disponíveis.
    n_surrogates : int
        Número de surrogates por sujeito
    use_simulated : bool
        Se True, usa dados simulados quando reais não disponíveis
        
    Returns
    -------
    pd.DataFrame
        Resultados para todos os sujeitos
    """
    if subjects is None:
        subjects = config.COVID_SUBJECTS
    
    print("\n" + "="*70)
    print("COVID COHORT ANALYSIS WITH NULL MODELS")
    print("="*70)
    print(f"Subjects: {len(subjects)}")
    print(f"Surrogates per subject: {n_surrogates}")
    print(f"Null model: {NULL_MODEL}")
    
    all_results = []
    
    for i, subject_id in enumerate(subjects):
        print(f"\n[{i+1}/{len(subjects)}] {subject_id}")
        
        try:
            # Tentar carregar dados reais
            adj = load_adjacency_matrix(subject_id)
        except:
            if use_simulated:
                print(f"  Using simulated network")
                adj = simulate_covid_network(100, seed=i)
            else:
                print(f"  Skipping (no data)")
                continue
        
        results = analyze_single_subject(
            subject_id=subject_id,
            adj=adj,
            n_surrogates=n_surrogates,
            verbose=False
        )
        
        all_results.append(results)
        
        # Progress report
        if results.get('is_small_world'):
            sw_status = "SW"
        else:
            sw_status = "--"
        
        print(f"  σ={results.get('sigma', 0):.2f}, "
              f"Q={results.get('Q_observed', 0):.3f}, "
              f"ω={results.get('omega', 0):.2f} [{sw_status}]")
    
    df = pd.DataFrame(all_results)
    
    # Summary statistics
    print("\n" + "="*70)
    print("COHORT SUMMARY")
    print("="*70)
    
    if len(df) > 0:
        print(f"\nSmall-worldness:")
        print(f"  σ mean: {df['sigma'].mean():.3f} ± {df['sigma'].std():.3f}")
        print(f"  ω mean: {df['omega'].mean():.3f} ± {df['omega'].std():.3f}")
        print(f"  Small-world subjects: {df['is_small_world'].sum()}/{len(df)}")
        
        print(f"\nModularity:")
        print(f"  Q mean: {df['Q_observed'].mean():.3f} ± {df['Q_observed'].std():.3f}")
        print(f"  Q_norm mean: {df['Q_normalized'].mean():.3f} ± {df['Q_normalized'].std():.3f}")
        
        if 'rc_phi_normalized' in df.columns:
            print(f"\nRich-club:")
            print(f"  ϕ_norm mean: {df['rc_phi_normalized'].mean():.3f} ± {df['rc_phi_normalized'].std():.3f}")
    
    return df


# =============================================================================
# VISUALIZAÇÕES
# =============================================================================

def plot_null_model_results(
    results_df: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Visualizar resultados da análise com modelos nulos.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Small-worldness sigma
    ax = axes[0, 0]
    ax.hist(results_df['sigma'], bins=15, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='SW threshold')
    ax.set_xlabel('σ (small-worldness)')
    ax.set_ylabel('Count')
    ax.set_title('Small-worldness σ\n(>1 indicates small-world)')
    ax.legend()
    
    # 2. Small-worldness omega
    ax = axes[0, 1]
    ax.hist(results_df['omega'], bins=15, color='coral', edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='SW optimum')
    ax.set_xlabel('ω (small-worldness)')
    ax.set_ylabel('Count')
    ax.set_title('Small-worldness ω\n(≈0 indicates small-world)')
    ax.legend()
    
    # 3. Modularity Q
    ax = axes[0, 2]
    ax.hist(results_df['Q_observed'], bins=15, color='seagreen', edgecolor='white', alpha=0.7)
    ax.set_xlabel('Modularity Q')
    ax.set_ylabel('Count')
    ax.set_title('Modularity Q\n(higher = more modular)')
    
    # 4. Clustering vs null
    ax = axes[1, 0]
    ax.scatter(results_df['C_random'], results_df['C_observed'], 
               c='steelblue', alpha=0.6, s=50)
    lims = [min(results_df['C_random'].min(), results_df['C_observed'].min()) * 0.9,
            max(results_df['C_random'].max(), results_df['C_observed'].max()) * 1.1]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel('C (random)')
    ax.set_ylabel('C (observed)')
    ax.set_title('Clustering: Observed vs Random\n(above line = higher than random)')
    
    # 5. Path length vs null
    ax = axes[1, 1]
    ax.scatter(results_df['L_random'], results_df['L_observed'],
               c='coral', alpha=0.6, s=50)
    lims = [min(results_df['L_random'].min(), results_df['L_observed'].min()) * 0.9,
            max(results_df['L_random'].max(), results_df['L_observed'].max()) * 1.1]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel('L (random)')
    ax.set_ylabel('L (observed)')
    ax.set_title('Path Length: Observed vs Random\n(below line = shorter than random)')
    
    # 6. Q normalized z-scores
    ax = axes[1, 2]
    colors = ['seagreen' if z > 2 else 'gray' for z in results_df['Q_zscore']]
    ax.bar(range(len(results_df)), results_df['Q_zscore'], color=colors, alpha=0.7)
    ax.axhline(y=2, color='red', linestyle='--', linewidth=1, label='z=2')
    ax.axhline(y=-2, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Subject')
    ax.set_ylabel('z-score')
    ax.set_title('Modularity z-scores vs null\n(|z|>2 = significant)')
    ax.legend()
    
    fig.suptitle('COVID Cohort: Null Model Analysis Results', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_individual_null_distribution(
    adj: np.ndarray,
    metric_name: str = 'clustering',
    n_surrogates: int = 500,
    save_path: str = None
) -> plt.Figure:
    """
    Visualizar distribuição nula para uma métrica específica.
    """
    metric_funcs = {
        'clustering': null_models.compute_clustering_coefficient,
        'efficiency': null_models.compute_global_efficiency,
        'modularity': lambda x: null_models.compute_modularity(x)[0],
    }
    
    if metric_name not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric_name}")
    
    result = null_models.analyze_with_null_model(
        adj=adj,
        metric_func=metric_funcs[metric_name],
        metric_name=metric_name,
        null_model='maslov_sneppen',
        n_surrogates=n_surrogates
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histograma da distribuição nula
    ax.hist(result.null_distribution, bins=30, density=True, 
            color='lightblue', edgecolor='white', alpha=0.7, label='Null distribution')
    
    # Linha vertical para valor observado
    ax.axvline(x=result.observed, color='red', linewidth=2, 
               label=f'Observed = {result.observed:.4f}')
    
    # Média e std da distribuição nula
    ax.axvline(x=result.null_mean, color='blue', linewidth=1, linestyle='--',
               label=f'Null mean = {result.null_mean:.4f}')
    
    # Shading para p-value
    if result.p_value < 0.05:
        ax.axvspan(result.observed, ax.get_xlim()[1], alpha=0.2, color='red')
    
    ax.set_xlabel(f'{metric_name.title()} value')
    ax.set_ylabel('Density')
    ax.set_title(f'{metric_name.title()}: Observed vs Null Distribution\n'
                f'z = {result.z_score:.2f}, p = {result.p_value:.4f}')
    ax.legend()
    
    # Anotações
    textstr = f'Observed: {result.observed:.4f}\n'
    textstr += f'Null mean: {result.null_mean:.4f}\n'
    textstr += f'Null std: {result.null_std:.4f}\n'
    textstr += f'z-score: {result.z_score:.2f}\n'
    textstr += f'p-value: {result.p_value:.4f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Pipeline principal de demonstração."""
    
    print("\n" + "="*70)
    print("NULL MODEL ANALYSIS PIPELINE")
    print("Analyzing COVID patient brain networks with surrogate comparison")
    print("="*70)
    
    # =========================================================================
    # DEMO 1: Análise de um sujeito individual
    # =========================================================================
    print("\n" + "-"*50)
    print("DEMO 1: Single subject analysis")
    print("-"*50)
    
    # Simular uma rede para demo
    adj_demo = simulate_covid_network(100, seed=42)
    
    results_single = analyze_single_subject(
        subject_id='sub-demo',
        adj=adj_demo,
        n_surrogates=100,
        verbose=True
    )
    
    # =========================================================================
    # DEMO 2: Análise de coorte
    # =========================================================================
    print("\n" + "-"*50)
    print("DEMO 2: Cohort analysis (simulated data)")
    print("-"*50)
    
    # Usar primeiros 10 sujeitos simulados para demo rápido
    subjects_demo = [f'sub-{str(i).zfill(2)}' for i in range(1, 11)]
    
    results_cohort = analyze_covid_cohort(
        subjects=subjects_demo,
        n_surrogates=50,  # Reduzido para demo
        use_simulated=True
    )
    
    # =========================================================================
    # DEMO 3: Visualizações
    # =========================================================================
    print("\n" + "-"*50)
    print("DEMO 3: Generating visualizations")
    print("-"*50)
    
    output_dir = Path(config.COMPARISON_DIR) / 'null_models'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot cohort results
    fig_cohort = plot_null_model_results(
        results_cohort,
        save_path=output_dir / 'cohort_null_model_results.png'
    )
    plt.close(fig_cohort)
    
    # Plot individual null distribution
    fig_null = plot_individual_null_distribution(
        adj_demo,
        metric_name='clustering',
        n_surrogates=200,
        save_path=output_dir / 'null_distribution_clustering.png'
    )
    plt.close(fig_null)
    
    # Salvar resultados
    results_cohort.to_csv(output_dir / 'cohort_null_model_results.csv', index=False)
    print(f"\n✓ Results saved to: {output_dir}")
    
    # =========================================================================
    # INTERPRETAÇÃO
    # =========================================================================
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    
    print("""
    SMALL-WORLDNESS:
    - σ > 1: Network has small-world properties (high clustering + short paths)
    - ω ≈ 0: Balanced between regular and random
    - ω > 0: More random-like
    - ω < 0: More lattice-like
    
    MODULARITY:
    - Q > 0.3: Generally considered modular
    - z-score > 2: Significantly more modular than random networks
    - This indicates distinct functional communities exist
    
    RICH-CLUB:
    - ϕ_norm > 1: High-degree nodes (hubs) are more connected to each other
                  than expected by chance
    - This is a hallmark of brain network organization
    
    FOR COVID PATIENTS:
    - Reduced small-worldness might indicate disrupted integration/segregation balance
    - Altered modularity could reflect changes in functional community structure
    - Changes in rich-club might suggest hub vulnerability
    """)
    
    return results_cohort


if __name__ == '__main__':
    results = main()
