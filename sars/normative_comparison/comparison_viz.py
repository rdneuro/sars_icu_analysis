"""
╔══════════════════════════════════════════════════════════════════════╗
║  Normative Comparison - Visualization Module                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Visualizações publication-quality para comparação de grupos:        ║
║  - Violin plots com estatísticas                                     ║
║  - Forest plots de effect sizes                                      ║
║  - Heatmaps de z-scores                                              ║
║  - Radar plots de perfis de rede                                     ║
║  - Summary dashboards                                                ║
║                                                                      ║
║  ESTILO:                                                             ║
║  - Paleta de cores acessível e científica                            ║
║  - Fontes e tamanhos apropriados para publicação                     ║
║  - Anotações estatísticas integradas                                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import warnings

from .config import COMPARISON_DIR, ROBUST_METRICS

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Cores científicas (colorblind-friendly)
COLORS = {
    'covid': '#E64B35',       # Vermelho
    'control': '#4DBBD5',     # Azul
    'significant': '#00A087', # Verde
    'ns': '#8C8C8C',          # Cinza
    'background': '#FAFAFA',
    'grid': '#E0E0E0',
}

# Configuração de estilo matplotlib
def set_publication_style():
    """Configurar estilo para publicação."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.facecolor': COLORS['background'],
        'axes.grid': True,
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.5,
    })


def get_significance_stars(p: float) -> str:
    """Converter p-value em stars de significância."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


# =============================================================================
# VIOLIN / BOX PLOTS
# =============================================================================

def plot_group_comparison_violin(
    covid_df: pd.DataFrame,
    control_df: pd.DataFrame,
    metrics: List[str] = None,
    stats_df: pd.DataFrame = None,
    figsize: Tuple[int, int] = None,
    save_path: str = None
) -> plt.Figure:
    """
    Violin plots comparando COVID vs controles para múltiplas métricas.
    
    Parameters
    ----------
    covid_df : pd.DataFrame
        Métricas COVID
    control_df : pd.DataFrame
        Métricas controles
    metrics : List[str], optional
        Métricas para plotar
    stats_df : pd.DataFrame, optional
        Resultado de compare_groups() para anotações
    figsize : Tuple[int, int], optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    if metrics is None:
        available = set(covid_df.columns) & set(control_df.columns)
        metrics = [m for m in ROBUST_METRICS if m in available][:8]  # Max 8
    
    n_metrics = len(metrics)
    ncols = min(4, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Preparar dados
        covid_values = covid_df[metric].dropna().values
        ctrl_values = control_df[metric].dropna().values
        
        data = pd.DataFrame({
            'Value': np.concatenate([covid_values, ctrl_values]),
            'Group': ['COVID'] * len(covid_values) + ['Control'] * len(ctrl_values)
        })
        
        # Violin plot
        parts = ax.violinplot(
            [ctrl_values, covid_values],
            positions=[1, 2],
            showmeans=True,
            showmedians=False,
            showextrema=False
        )
        
        # Colorir violins
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLORS['control'] if j == 0 else COLORS['covid'])
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('black')
        
        # Box plot interno
        bp = ax.boxplot(
            [ctrl_values, covid_values],
            positions=[1, 2],
            widths=0.15,
            patch_artist=True,
            showfliers=False
        )
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor('white')
            patch.set_alpha(0.8)
        
        # Scatter points
        np.random.seed(42)
        jitter = 0.08
        ax.scatter(
            1 + np.random.uniform(-jitter, jitter, len(ctrl_values)),
            ctrl_values,
            c=COLORS['control'], alpha=0.4, s=20, zorder=3
        )
        ax.scatter(
            2 + np.random.uniform(-jitter, jitter, len(covid_values)),
            covid_values,
            c=COLORS['covid'], alpha=0.4, s=20, zorder=3
        )
        
        # Anotações estatísticas
        if stats_df is not None:
            row = stats_df[stats_df['metric'] == metric]
            if len(row) > 0:
                row = row.iloc[0]
                p_val = row.get('p_perm_corrected', row.get('p_permutation', 1.0))
                effect = row.get('hedges_g', row.get('cohens_d', 0))
                
                stars = get_significance_stars(p_val)
                
                # Linha de significância
                y_max = max(covid_values.max(), ctrl_values.max())
                y_line = y_max * 1.05
                
                ax.plot([1, 2], [y_line, y_line], 'k-', lw=1)
                ax.text(1.5, y_line * 1.02, stars, ha='center', fontsize=11, fontweight='bold')
                
                # Effect size
                ax.text(
                    0.98, 0.02, f'g = {effect:.2f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8, style='italic'
                )
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Control', 'COVID'])
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
    
    # Esconder axes vazios
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Legenda
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['control'], alpha=0.7, label='Control'),
        mpatches.Patch(facecolor=COLORS['covid'], alpha=0.7, label='COVID'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    fig.suptitle('Graph Metrics Comparison: COVID vs Healthy Controls', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# FOREST PLOT
# =============================================================================

def plot_forest_effect_sizes(
    stats_df: pd.DataFrame,
    metrics: List[str] = None,
    figsize: Tuple[int, int] = None,
    save_path: str = None
) -> plt.Figure:
    """
    Forest plot de effect sizes com confidence intervals.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Resultado de compare_groups()
    metrics : List[str], optional
        Métricas para incluir
    figsize : Tuple[int, int], optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    if metrics is None:
        df = stats_df.copy()
    else:
        df = stats_df[stats_df['metric'].isin(metrics)].copy()
    
    # Sort by effect size
    effect_col = 'hedges_g' if 'hedges_g' in df.columns else 'cohens_d'
    df = df.sort_values(effect_col).reset_index(drop=True)
    
    n_metrics = len(df)
    if figsize is None:
        figsize = (10, max(4, n_metrics * 0.4))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_positions = range(n_metrics)
    
    for i, (idx, row) in enumerate(df.iterrows()):
        effect = row[effect_col]
        ci_low = row['effect_ci_lower']
        ci_high = row['effect_ci_upper']
        significant = row.get('significant_perm', False)
        
        color = COLORS['significant'] if significant else COLORS['ns']
        
        # Error bar
        ax.errorbar(
            effect, i,
            xerr=[[effect - ci_low], [ci_high - effect]],
            fmt='o', color=color, markersize=8, capsize=4, capthick=2, linewidth=2
        )
        
        # Annotation
        stars = get_significance_stars(row.get('p_perm_corrected', 1.0))
        ax.text(
            ci_high + 0.1, i, stars,
            va='center', fontsize=10, fontweight='bold'
        )
    
    # Linha de referência (null effect)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    # Shading para diferentes magnitudes de efeito
    ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Negligible')
    ax.axvspan(-0.5, -0.2, alpha=0.15, color='blue', label='Small')
    ax.axvspan(0.2, 0.5, alpha=0.15, color='blue')
    ax.axvspan(-0.8, -0.5, alpha=0.2, color='blue', label='Medium')
    ax.axvspan(0.5, 0.8, alpha=0.2, color='blue')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df['metric'].str.replace('_', ' ').str.title())
    ax.set_xlabel("Hedges' g (95% CI)", fontsize=11)
    ax.set_title('Effect Sizes: COVID vs Controls\n(positive = higher in COVID)', 
                fontsize=12, fontweight='bold')
    
    # Legenda
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['significant'], 
               markersize=10, label='Significant (FDR-corrected)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['ns'], 
               markersize=10, label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_xlim(ax.get_xlim()[0] - 0.3, ax.get_xlim()[1] + 0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# Z-SCORE HEATMAP
# =============================================================================

def plot_zscore_heatmap(
    zscores_df: pd.DataFrame,
    metrics: List[str] = None,
    figsize: Tuple[int, int] = None,
    save_path: str = None
) -> plt.Figure:
    """
    Heatmap de z-scores individuais (pacientes vs métricas).
    
    Parameters
    ----------
    zscores_df : pd.DataFrame
        DataFrame com z-scores (output de compute_z_scores)
    metrics : List[str], optional
        Métricas para incluir
    figsize : Tuple[int, int], optional
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    # Extrair colunas de z-score
    z_cols = [c for c in zscores_df.columns if c.endswith('_zscore') and c != 'composite_zscore']
    
    if metrics:
        z_cols = [f'{m}_zscore' for m in metrics if f'{m}_zscore' in z_cols]
    
    if not z_cols:
        raise ValueError("No z-score columns found")
    
    # Preparar matriz
    data = zscores_df[z_cols].values
    subjects = zscores_df.get('subject_id', [f'S{i}' for i in range(len(zscores_df))])
    metric_labels = [c.replace('_zscore', '').replace('_', ' ').title() for c in z_cols]
    
    n_subjects, n_metrics = data.shape
    
    if figsize is None:
        figsize = (max(8, n_metrics * 0.8), max(6, n_subjects * 0.4))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Heatmap
    vmax = max(3, np.abs(data).max())
    im = ax.imshow(
        data, cmap='RdBu_r', aspect='auto',
        vmin=-vmax, vmax=vmax
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Z-score (relative to controls)', fontsize=10)
    
    # Marcar valores extremos
    for i in range(n_subjects):
        for j in range(n_metrics):
            z = data[i, j]
            if abs(z) > 2:
                ax.text(j, i, f'{z:.1f}', ha='center', va='center', 
                       fontsize=7, fontweight='bold',
                       color='white' if abs(z) > 2.5 else 'black')
    
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_subjects))
    ax.set_yticklabels(subjects, fontsize=8)
    
    ax.set_xlabel('Graph Metric', fontsize=11)
    ax.set_ylabel('COVID Patient', fontsize=11)
    ax.set_title('Individual Z-scores: COVID Patients vs Normative Controls\n(|z| > 2 highlighted)', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# RADAR / SPIDER PLOT
# =============================================================================

def plot_network_profile_radar(
    covid_df: pd.DataFrame,
    control_df: pd.DataFrame,
    metrics: List[str] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Radar plot comparando perfis de rede entre grupos.
    
    Parameters
    ----------
    covid_df : pd.DataFrame
        Métricas COVID
    control_df : pd.DataFrame
        Métricas controles
    metrics : List[str], optional
        Métricas para incluir (idealmente 5-10)
    normalize : bool
        Se True, normaliza para comparação
    figsize : Tuple[int, int]
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    if metrics is None:
        available = set(covid_df.columns) & set(control_df.columns)
        metrics = [m for m in ROBUST_METRICS if m in available][:8]
    
    n_metrics = len(metrics)
    
    # Calcular médias
    covid_means = [covid_df[m].mean() for m in metrics]
    covid_stds = [covid_df[m].std() for m in metrics]
    ctrl_means = [control_df[m].mean() for m in metrics]
    ctrl_stds = [control_df[m].std() for m in metrics]
    
    # Normalizar se solicitado
    if normalize:
        all_values = np.array(covid_means + ctrl_means)
        min_val, max_val = all_values.min(), all_values.max()
        if max_val > min_val:
            covid_means = [(v - min_val) / (max_val - min_val) for v in covid_means]
            ctrl_means = [(v - min_val) / (max_val - min_val) for v in ctrl_means]
    
    # Ângulos
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Fechar o polígono
    
    covid_means += covid_means[:1]
    ctrl_means += ctrl_means[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Plot controles
    ax.plot(angles, ctrl_means, 'o-', linewidth=2, color=COLORS['control'], label='Control')
    ax.fill(angles, ctrl_means, alpha=0.25, color=COLORS['control'])
    
    # Plot COVID
    ax.plot(angles, covid_means, 'o-', linewidth=2, color=COLORS['covid'], label='COVID')
    ax.fill(angles, covid_means, alpha=0.25, color=COLORS['covid'])
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n').title() for m in metrics], fontsize=9)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    ax.set_title('Network Profile Comparison\n(normalized to [0,1])', 
                fontsize=12, fontweight='bold', pad=20)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================

def generate_comparison_report(
    covid_df: pd.DataFrame,
    control_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    atlas_name: str = 'unknown',
    output_dir: str = None,
    save_individual: bool = True
) -> Dict[str, str]:
    """
    Gerar relatório completo de comparação com todas as visualizações.
    
    Parameters
    ----------
    covid_df : pd.DataFrame
        Métricas COVID
    control_df : pd.DataFrame
        Métricas controles
    stats_df : pd.DataFrame
        Resultados estatísticos
    atlas_name : str
        Nome do atlas
    output_dir : str, optional
        Diretório de saída
    save_individual : bool
        Se True, salva figuras individuais
        
    Returns
    -------
    Dict[str, str]
        Dicionário com paths das figuras geradas
    """
    if output_dir is None:
        output_dir = COMPARISON_DIR / atlas_name
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    print(f"\n{'='*60}")
    print(f"GENERATING COMPARISON REPORT: {atlas_name}")
    print('='*60)
    
    # 1. Violin plots
    try:
        fig_violin = plot_group_comparison_violin(
            covid_df, control_df, 
            stats_df=stats_df,
            save_path=output_dir / 'violin_comparison.png' if save_individual else None
        )
        paths['violin'] = str(output_dir / 'violin_comparison.png')
        plt.close(fig_violin)
    except Exception as e:
        warnings.warn(f"Failed to create violin plot: {e}")
    
    # 2. Forest plot
    try:
        fig_forest = plot_forest_effect_sizes(
            stats_df,
            save_path=output_dir / 'forest_effect_sizes.png' if save_individual else None
        )
        paths['forest'] = str(output_dir / 'forest_effect_sizes.png')
        plt.close(fig_forest)
    except Exception as e:
        warnings.warn(f"Failed to create forest plot: {e}")
    
    # 3. Radar plot
    try:
        fig_radar = plot_network_profile_radar(
            covid_df, control_df,
            save_path=output_dir / 'radar_profile.png' if save_individual else None
        )
        paths['radar'] = str(output_dir / 'radar_profile.png')
        plt.close(fig_radar)
    except Exception as e:
        warnings.warn(f"Failed to create radar plot: {e}")
    
    # 4. Z-scores heatmap (se disponível)
    try:
        from .statistics import compute_z_scores
        zscores = compute_z_scores(covid_df, control_df)
        fig_zscore = plot_zscore_heatmap(
            zscores,
            save_path=output_dir / 'zscore_heatmap.png' if save_individual else None
        )
        paths['zscore_heatmap'] = str(output_dir / 'zscore_heatmap.png')
        plt.close(fig_zscore)
    except Exception as e:
        warnings.warn(f"Failed to create z-score heatmap: {e}")
    
    # 5. Save statistics table
    try:
        from .statistics import generate_summary_table
        summary_table = generate_summary_table(stats_df, format_type='publication')
        summary_table.to_csv(output_dir / 'statistics_summary.csv', index=False)
        paths['stats_table'] = str(output_dir / 'statistics_summary.csv')
        
        # Also save full stats
        stats_df.to_csv(output_dir / 'statistics_full.csv', index=False)
        paths['stats_full'] = str(output_dir / 'statistics_full.csv')
    except Exception as e:
        warnings.warn(f"Failed to save statistics tables: {e}")
    
    print(f"\n✓ Report saved to: {output_dir}")
    print(f"  Generated {len(paths)} files")
    
    return paths


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print("Comparison Visualization Module")
    set_publication_style()
    print("✓ Publication style configured")
