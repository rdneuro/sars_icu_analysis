"""
╔══════════════════════════════════════════════════════════════════════╗
║  Null Models Visualization Module                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Visualizações para análise de modelos nulos em redes cerebrais.     ║
║                                                                      ║
║  FUNÇÕES PRINCIPAIS:                                                 ║
║  - plot_null_distribution: Distribuição nula com valor observado     ║
║  - plot_subject_summary: Painel resumo de um sujeito                 ║
║  - plot_cohort_smallworld: σ e ω do grupo                            ║
║  - plot_cohort_metrics: Z-scores de todas as métricas                ║
║  - plot_metric_comparison: Comparação observado vs nulo              ║
║  - generate_full_report: Relatório completo com todas as figuras     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Cores para publicação (colorblind-friendly)
COLORS = {
    'observed': '#E64B35',      # Vermelho (valor observado)
    'null': '#4DBBD5',          # Azul claro (distribuição nula)
    'null_fill': '#4DBBD520',   # Azul com transparência
    'significant': '#00A087',   # Verde (significativo)
    'not_sig': '#8C8C8C',       # Cinza (não significativo)
    'small_world': '#3C5488',   # Azul escuro
    'random': '#F39B7F',        # Laranja
    'lattice': '#91D1C2',       # Verde água
}

# Configuração de estilo para publicação
def set_publication_style():
    """Configurar estilo matplotlib para publicação."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
    })


# =============================================================================
# SINGLE METRIC VISUALIZATION
# =============================================================================

def plot_null_distribution(
    observed: float,
    null_distribution: np.ndarray,
    metric_name: str,
    subject_id: str = None,
    ax: plt.Axes = None,
    show_stats: bool = True,
    figsize: Tuple[float, float] = (8, 5)
) -> plt.Figure:
    """
    Plotar distribuição nula com valor observado marcado.
    
    Esta é a visualização fundamental: mostra o histograma dos valores
    obtidos nas redes surrogate (a "hipótese nula") e onde o valor
    observado se posiciona nessa distribuição.
    
    Parameters
    ----------
    observed : float
        Valor observado na rede real
    null_distribution : np.ndarray
        Array com valores das redes surrogate
    metric_name : str
        Nome da métrica (para título)
    subject_id : str, optional
        ID do sujeito (para título)
    ax : plt.Axes, optional
        Axes existente. Se None, cria nova figura.
    show_stats : bool
        Se True, mostra box com estatísticas
    figsize : tuple
        Tamanho da figura
        
    Returns
    -------
    plt.Figure
        Figura matplotlib
    """
    set_publication_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Estatísticas
    null_mean = np.mean(null_distribution)
    null_std = np.std(null_distribution)
    z_score = (observed - null_mean) / null_std if null_std > 0 else 0
    
    # P-value (two-tailed)
    p_value = np.mean(np.abs(null_distribution - null_mean) >= np.abs(observed - null_mean))
    
    # Histograma da distribuição nula
    n_bins = min(50, len(null_distribution) // 20)
    n, bins, patches = ax.hist(
        null_distribution, 
        bins=n_bins, 
        density=True,
        color=COLORS['null'],
        edgecolor='white',
        alpha=0.7,
        label='Null distribution'
    )
    
    # Linha vertical para média nula
    ax.axvline(
        x=null_mean, 
        color=COLORS['null'], 
        linestyle='--', 
        linewidth=1.5,
        alpha=0.8,
        label=f'Null mean = {null_mean:.4f}'
    )
    
    # Linha vertical para valor observado (mais proeminente)
    ax.axvline(
        x=observed, 
        color=COLORS['observed'], 
        linewidth=2.5,
        label=f'Observed = {observed:.4f}'
    )
    
    # Shading para região extrema (p-value visual)
    if z_score > 0:
        # Valor observado maior que média - shade à direita
        ax.axvspan(observed, ax.get_xlim()[1], alpha=0.15, color=COLORS['observed'])
    else:
        # Valor observado menor que média - shade à esquerda
        ax.axvspan(ax.get_xlim()[0], observed, alpha=0.15, color=COLORS['observed'])
    
    # Significância
    is_significant = p_value < 0.05
    sig_color = COLORS['significant'] if is_significant else COLORS['not_sig']
    sig_text = "SIGNIFICANT" if is_significant else "not significant"
    
    # Labels
    ax.set_xlabel(f'{metric_name.replace("_", " ").title()}')
    ax.set_ylabel('Density')
    
    # Título
    title = f'{metric_name.replace("_", " ").title()}: Observed vs Null Distribution'
    if subject_id:
        title = f'{subject_id} - {title}'
    ax.set_title(title)
    
    # Legenda
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Box com estatísticas
    if show_stats:
        stats_text = (
            f'z = {z_score:.2f}\n'
            f'p = {p_value:.4f}\n'
            f'{sig_text}'
        )
        
        props = dict(
            boxstyle='round,pad=0.5', 
            facecolor='white', 
            edgecolor=sig_color,
            linewidth=2,
            alpha=0.9
        )
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props,
            color=sig_color,
            fontweight='bold'
        )
    
    plt.tight_layout()
    return fig


def plot_multiple_null_distributions(
    results_dict: Dict[str, Dict],
    subject_id: str = None,
    figsize: Tuple[float, float] = (14, 10),
    save_path: str = None
) -> plt.Figure:
    """
    Plotar múltiplas distribuições nulas em um painel.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dicionário com {metric_name: {'observed': float, 'null_distribution': array}}
    subject_id : str, optional
        ID do sujeito
    figsize : tuple
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    n_metrics = len(results_dict)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    for idx, (metric_name, data) in enumerate(results_dict.items()):
        plot_null_distribution(
            observed=data['observed'],
            null_distribution=data['null_distribution'],
            metric_name=metric_name,
            ax=axes[idx],
            show_stats=True
        )
    
    # Esconder axes vazios
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    # Título geral
    suptitle = 'Null Model Analysis'
    if subject_id:
        suptitle = f'{subject_id} - {suptitle}'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# SUBJECT SUMMARY VISUALIZATION
# =============================================================================

def plot_subject_summary(
    subject_id: str,
    results: Dict,
    figsize: Tuple[float, float] = (16, 12),
    save_path: str = None
) -> plt.Figure:
    """
    Painel resumo completo para um sujeito.
    
    Inclui:
    - Small-worldness (σ e ω)
    - Clustering observado vs random
    - Path length observado vs random
    - Modularity z-score
    - Radar plot do perfil de rede
    
    Parameters
    ----------
    subject_id : str
        ID do sujeito
    results : Dict
        Dicionário com resultados do sujeito (output de analyze_single_subject)
    figsize : tuple
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Row 1: Small-worldness indicators
    # =========================================================================
    
    # 1A: Sigma gauge
    ax1 = fig.add_subplot(gs[0, 0])
    sigma = results.get('sigma', 1.0)
    _plot_gauge(ax1, sigma, 'σ (Small-worldness)', 
                vmin=0, vmax=3, threshold=1.0,
                cmap_name='RdYlGn')
    
    # 1B: Omega gauge  
    ax2 = fig.add_subplot(gs[0, 1])
    omega = results.get('omega', 0.0)
    _plot_gauge(ax2, omega, 'ω (Lattice ↔ Random)',
                vmin=-1, vmax=1, threshold=0.0,
                cmap_name='RdYlGn_r', center_is_good=True)
    
    # 1C: C observed vs random bar
    ax3 = fig.add_subplot(gs[0, 2])
    c_obs = results.get('C_observed', 0)
    c_rand = results.get('C_random', 0)
    _plot_comparison_bar(ax3, c_obs, c_rand, 'Clustering Coefficient',
                         labels=['Observed', 'Random'])
    
    # 1D: L observed vs random bar
    ax4 = fig.add_subplot(gs[0, 3])
    l_obs = results.get('L_observed', 0)
    l_rand = results.get('L_random', 0)
    _plot_comparison_bar(ax4, l_obs, l_rand, 'Path Length',
                         labels=['Observed', 'Random'],
                         lower_is_better=True)
    
    # =========================================================================
    # Row 2: Modularity and network properties
    # =========================================================================
    
    # 2A-B: Modularity analysis (wider)
    ax5 = fig.add_subplot(gs[1, 0:2])
    q_obs = results.get('Q_observed', 0)
    q_rand = results.get('Q_random_mean', 0)
    q_zscore = results.get('Q_zscore', 0)
    n_comm = results.get('n_communities', 0)
    
    bars = ax5.bar(['Observed\nModularity', 'Random\nModularity'], 
                   [q_obs, q_rand],
                   color=[COLORS['observed'], COLORS['null']],
                   edgecolor='black', linewidth=1)
    ax5.set_ylabel('Modularity (Q)')
    ax5.set_title(f'Modularity: z = {q_zscore:.2f}, {n_comm} communities')
    
    # Adicionar valor nas barras
    for bar, val in zip(bars, [q_obs, q_rand]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2C-D: Network metrics radar (wider)
    ax6 = fig.add_subplot(gs[1, 2:4], projection='polar')
    _plot_network_radar(ax6, results)
    
    # =========================================================================
    # Row 3: Z-scores summary and interpretation
    # =========================================================================
    
    # 3A-C: Z-scores bar plot
    ax7 = fig.add_subplot(gs[2, 0:3])
    _plot_zscore_bars(ax7, results)
    
    # 3D: Text interpretation
    ax8 = fig.add_subplot(gs[2, 3])
    _plot_interpretation_text(ax8, results)
    
    # Main title
    is_sw = results.get('is_small_world', False)
    sw_status = "✓ SMALL-WORLD" if is_sw else "✗ Not Small-World"
    fig.suptitle(
        f'{subject_id} - Network Topology Analysis\n{sw_status}',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def _plot_gauge(ax, value, title, vmin=0, vmax=2, threshold=1.0, 
                cmap_name='RdYlGn', center_is_good=False):
    """Plot gauge/speedometer style indicator."""
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Arco de fundo
    theta = np.linspace(0, np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Colorir o arco
    cmap = plt.cm.get_cmap(cmap_name)
    for i in range(len(theta)-1):
        t = i / (len(theta)-1)
        color = cmap(t)
        ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=15, solid_capstyle='butt')
    
    # Ponteiro
    normalized = (value - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)
    angle = np.pi * (1 - normalized)
    
    ax.annotate('', xy=(0.7*np.cos(angle), 0.7*np.sin(angle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Valor
    ax.text(0, -0.1, f'{value:.2f}', ha='center', va='top', 
            fontsize=16, fontweight='bold')
    
    # Título
    ax.set_title(title, fontsize=10, pad=5)
    
    # Labels min/max
    ax.text(-1.3, 0, f'{vmin}', ha='center', va='center', fontsize=8)
    ax.text(1.3, 0, f'{vmax}', ha='center', va='center', fontsize=8)


def _plot_comparison_bar(ax, observed, null, title, labels=['Obs', 'Null'],
                         lower_is_better=False):
    """Plot comparison bar chart."""
    colors = [COLORS['observed'], COLORS['null']]
    bars = ax.bar(labels, [observed, null], color=colors, edgecolor='black')
    
    # Indicar qual é melhor
    if lower_is_better:
        better = 'Observed' if observed < null else 'Random'
    else:
        better = 'Observed' if observed > null else 'Random'
    
    ax.set_title(f'{title}\n({better} is better)', fontsize=10)
    ax.set_ylabel('Value')
    
    # Valores nas barras
    for bar, val in zip(bars, [observed, null]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)


def _plot_network_radar(ax, results):
    """Plot radar chart of network properties."""
    # Métricas normalizadas (observado / random)
    metrics = {
        'Clustering': results.get('C_observed', 0) / max(results.get('C_random', 1), 0.001),
        'Efficiency': results.get('efficiency_observed', 0.5) / 0.5,  # normalizar por típico
        'Modularity': results.get('Q_observed', 0) / max(results.get('Q_random_mean', 0.3), 0.001),
        'Integration': 1 / max(results.get('L_observed', 2) / max(results.get('L_random', 2), 0.001), 0.001),
    }
    
    # Apenas métricas com valores válidos
    metrics = {k: v for k, v in metrics.items() if np.isfinite(v)}
    
    if len(metrics) < 3:
        ax.text(0.5, 0.5, 'Insufficient data\nfor radar plot',
                ha='center', va='center', transform=ax.transAxes)
        return
    
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    # Fechar o polígono
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color=COLORS['observed'])
    ax.fill(angles, values, alpha=0.25, color=COLORS['observed'])
    
    # Linha de referência (1 = igual ao random)
    ref_values = [1] * len(labels) + [1]
    ax.plot(angles, ref_values, '--', linewidth=1, color=COLORS['null'], alpha=0.7)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('Network Profile\n(vs Random)', fontsize=10, pad=10)


def _plot_zscore_bars(ax, results):
    """Plot z-scores as horizontal bars."""
    zscores = {}
    
    # Coletar z-scores disponíveis
    for key, val in results.items():
        if 'zscore' in key.lower() or key.endswith('_zscore'):
            metric = key.replace('_zscore', '').replace('zscore', 'modularity')
            if np.isfinite(val):
                zscores[metric] = val
    
    if not zscores:
        ax.text(0.5, 0.5, 'No z-scores available', ha='center', va='center')
        return
    
    # Ordenar por magnitude
    sorted_items = sorted(zscores.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [item[0].replace('_', ' ').title() for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Cores baseadas em significância
    colors = [COLORS['significant'] if abs(v) > 2 else COLORS['not_sig'] for v in values]
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Z-score')
    ax.set_title('Metrics vs Null Model (|z| > 2 is significant)')
    
    # Linhas de referência
    ax.axvline(x=2, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=-2, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Valores nas barras
    for bar, val in zip(bars, values):
        x_pos = val + 0.1 if val >= 0 else val - 0.1
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', ha=ha, va='center', fontsize=9)


def _plot_interpretation_text(ax, results):
    """Plot text interpretation of results."""
    ax.axis('off')
    
    sigma = results.get('sigma', 0)
    omega = results.get('omega', 0)
    q_zscore = results.get('Q_zscore', 0)
    is_sw = results.get('is_small_world', False)
    
    lines = ["INTERPRETATION", "=" * 20, ""]
    
    # Small-worldness
    if is_sw:
        lines.append("✓ Network is SMALL-WORLD")
        lines.append(f"  (σ = {sigma:.2f} > 1)")
    else:
        lines.append("✗ Network is NOT small-world")
        lines.append(f"  (σ = {sigma:.2f} < 1)")
    
    lines.append("")
    
    # Omega interpretation
    if abs(omega) < 0.3:
        lines.append("• Balanced organization")
    elif omega > 0.3:
        lines.append("• More random-like")
    else:
        lines.append("• More lattice-like")
    
    lines.append("")
    
    # Modularity
    if q_zscore > 2:
        lines.append("✓ Significant modularity")
        lines.append(f"  (z = {q_zscore:.2f})")
    else:
        lines.append("• Modularity not significant")
    
    text = '\n'.join(lines)
    ax.text(0.1, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


# =============================================================================
# COHORT VISUALIZATION
# =============================================================================

def plot_cohort_smallworld(
    results_df: pd.DataFrame,
    figsize: Tuple[float, float] = (14, 5),
    save_path: str = None
) -> plt.Figure:
    """
    Visualizar small-worldness do grupo.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame com resultados individuais (output de run_null_model_analysis)
    figsize : tuple
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # =========================================================================
    # Panel A: Sigma distribution
    # =========================================================================
    ax = axes[0]
    
    sigma_vals = results_df['sigma'].dropna()
    
    ax.hist(sigma_vals, bins=15, color=COLORS['small_world'], 
            edgecolor='white', alpha=0.7)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, 
               label='SW threshold (σ=1)')
    ax.axvline(x=sigma_vals.mean(), color='black', linestyle='-', linewidth=2,
               label=f'Mean = {sigma_vals.mean():.2f}')
    
    ax.set_xlabel('σ (Small-worldness)')
    ax.set_ylabel('Count')
    ax.set_title('A. Small-worldness σ\n(>1 indicates small-world)')
    ax.legend(loc='upper right')
    
    # Percentual small-world
    pct_sw = 100 * (sigma_vals > 1).mean()
    ax.text(0.95, 0.95, f'{pct_sw:.0f}% SW', transform=ax.transAxes,
            ha='right', va='top', fontsize=12, fontweight='bold',
            color=COLORS['significant'] if pct_sw > 50 else COLORS['not_sig'])
    
    # =========================================================================
    # Panel B: Omega distribution
    # =========================================================================
    ax = axes[1]
    
    omega_vals = results_df['omega'].dropna()
    
    ax.hist(omega_vals, bins=15, color=COLORS['random'], 
            edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2,
               label='SW optimum (ω=0)')
    ax.axvline(x=omega_vals.mean(), color='black', linestyle='-', linewidth=2,
               label=f'Mean = {omega_vals.mean():.2f}')
    
    ax.set_xlabel('ω (Lattice ↔ Random)')
    ax.set_ylabel('Count')
    ax.set_title('B. Small-worldness ω\n(≈0 indicates small-world)')
    ax.legend(loc='upper right')
    
    # Anotação
    if omega_vals.mean() > 0.3:
        tendency = "→ Random"
    elif omega_vals.mean() < -0.3:
        tendency = "← Lattice"
    else:
        tendency = "≈ Balanced"
    ax.text(0.95, 0.95, tendency, transform=ax.transAxes,
            ha='right', va='top', fontsize=11, fontweight='bold')
    
    # =========================================================================
    # Panel C: Sigma vs Omega scatter
    # =========================================================================
    ax = axes[2]
    
    # Colorir por small-world status
    is_sw = results_df['is_small_world'].values
    colors = [COLORS['significant'] if sw else COLORS['not_sig'] for sw in is_sw]
    
    ax.scatter(results_df['omega'], results_df['sigma'], 
               c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Linhas de referência
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5)
    
    # Região small-world (shading)
    ax.axhspan(1, ax.get_ylim()[1], alpha=0.1, color=COLORS['significant'])
    
    ax.set_xlabel('ω')
    ax.set_ylabel('σ')
    ax.set_title('C. σ vs ω Space\n(green = small-world)')
    
    # Legenda
    sw_patch = mpatches.Patch(color=COLORS['significant'], label='Small-world')
    nsw_patch = mpatches.Patch(color=COLORS['not_sig'], label='Not small-world')
    ax.legend(handles=[sw_patch, nsw_patch], loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_cohort_metrics(
    results_df: pd.DataFrame,
    figsize: Tuple[float, float] = (12, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Visualizar z-scores de todas as métricas do grupo.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame com resultados individuais
    figsize : tuple
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    # Identificar colunas de z-score
    zscore_cols = [c for c in results_df.columns if 'zscore' in c.lower()]
    
    if not zscore_cols:
        warnings.warn("No z-score columns found in results_df")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # =========================================================================
    # Panel A: Heatmap de z-scores por sujeito
    # =========================================================================
    ax = axes[0]
    
    zscore_data = results_df[zscore_cols].copy()
    zscore_data.index = results_df['subject_id']
    zscore_data.columns = [c.replace('_zscore', '').replace('zscore', 'Q') for c in zscore_cols]
    
    # Plot heatmap
    im = ax.imshow(zscore_data.values, cmap='RdBu_r', aspect='auto',
                   vmin=-4, vmax=4)
    
    ax.set_xticks(range(len(zscore_data.columns)))
    ax.set_xticklabels(zscore_data.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(zscore_data.index)))
    ax.set_yticklabels(zscore_data.index, fontsize=8)
    ax.set_title('A. Z-scores by Subject')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Z-score')
    
    # =========================================================================
    # Panel B: Violin plot de z-scores
    # =========================================================================
    ax = axes[1]
    
    zscore_melted = zscore_data.melt(var_name='Metric', value_name='Z-score')
    
    parts = ax.violinplot(
        [zscore_data[col].dropna().values for col in zscore_data.columns],
        positions=range(len(zscore_data.columns)),
        showmeans=True, showmedians=True
    )
    
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xticks(range(len(zscore_data.columns)))
    ax.set_xticklabels(zscore_data.columns, rotation=45, ha='right')
    ax.set_ylabel('Z-score')
    ax.set_title('B. Z-score Distribution by Metric')
    
    # =========================================================================
    # Panel C: Percentage significant per metric
    # =========================================================================
    ax = axes[2]
    
    pct_sig = [(np.abs(zscore_data[col]) > 2).mean() * 100 for col in zscore_data.columns]
    colors = [COLORS['significant'] if p > 50 else COLORS['not_sig'] for p in pct_sig]
    
    bars = ax.bar(range(len(zscore_data.columns)), pct_sig, color=colors, edgecolor='black')
    
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(zscore_data.columns)))
    ax.set_xticklabels(zscore_data.columns, rotation=45, ha='right')
    ax.set_ylabel('% Subjects with |z| > 2')
    ax.set_ylim(0, 100)
    ax.set_title('C. Percentage Significant per Metric')
    
    # Valores nas barras
    for bar, p in zip(bars, pct_sig):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{p:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # =========================================================================
    # Panel D: Mean z-scores with CI
    # =========================================================================
    ax = axes[3]
    
    means = zscore_data.mean()
    stds = zscore_data.std()
    n = len(zscore_data)
    sems = stds / np.sqrt(n)
    ci95 = 1.96 * sems
    
    x = range(len(means))
    colors = [COLORS['significant'] if abs(m) > 2 else COLORS['not_sig'] for m in means]
    
    ax.bar(x, means, yerr=ci95, color=colors, edgecolor='black',
           capsize=3, alpha=0.8)
    
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=45, ha='right')
    ax.set_ylabel('Mean Z-score (±95% CI)')
    ax.set_title('D. Group Mean Z-scores')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str = 'clustering',
    figsize: Tuple[float, float] = (10, 6),
    save_path: str = None
) -> plt.Figure:
    """
    Comparar valores observados vs nulos para uma métrica específica.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame com resultados
    metric : str
        Nome da métrica ('clustering', 'efficiency', etc.)
    figsize : tuple
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    obs_col = f'{metric}_observed'
    null_col = f'{metric}_null_mean'
    
    if obs_col not in results_df.columns:
        warnings.warn(f"Column {obs_col} not found")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Panel A: Scatter observed vs null
    ax = axes[0]
    
    obs = results_df[obs_col]
    null = results_df.get(null_col, obs)  # Fallback se não tiver null
    
    ax.scatter(null, obs, c=COLORS['observed'], s=60, alpha=0.7, 
               edgecolors='black', linewidth=0.5)
    
    # Linha de igualdade
    lims = [min(null.min(), obs.min()) * 0.9, max(null.max(), obs.max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Observed = Null')
    
    ax.set_xlabel(f'{metric.title()} (Null)')
    ax.set_ylabel(f'{metric.title()} (Observed)')
    ax.set_title(f'A. {metric.title()}: Observed vs Null Mean')
    ax.legend()
    
    # Panel B: Distribution of differences
    ax = axes[1]
    
    diff = obs - null
    
    ax.hist(diff, bins=15, color=COLORS['observed'], edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='No difference')
    ax.axvline(x=diff.mean(), color='red', linestyle='-', linewidth=2,
               label=f'Mean diff = {diff.mean():.4f}')
    
    ax.set_xlabel(f'Δ{metric.title()} (Observed - Null)')
    ax.set_ylabel('Count')
    ax.set_title(f'B. Distribution of Differences')
    ax.legend()
    
    # T-test
    from scipy import stats
    t_stat, p_val = stats.ttest_1samp(diff.dropna(), 0)
    ax.text(0.95, 0.95, f't = {t_stat:.2f}\np = {p_val:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# FULL REPORT GENERATION
# =============================================================================

def generate_full_report(
    results_df: pd.DataFrame,
    output_dir: Union[str, Path],
    atlas_name: str = 'unknown'
) -> Dict[str, Path]:
    """
    Gerar relatório completo com todas as visualizações.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame com resultados individuais
    output_dir : str or Path
        Diretório para salvar as figuras
    atlas_name : str
        Nome do atlas (para títulos)
        
    Returns
    -------
    Dict[str, Path]
        Dicionário com paths das figuras geradas
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"GENERATING VISUALIZATION REPORT")
    print(f"Atlas: {atlas_name}")
    print(f"Output: {output_dir}")
    print('='*60)
    
    saved_files = {}
    
    # 1. Small-worldness do grupo
    print("\n1. Plotting cohort small-worldness...")
    fig = plot_cohort_smallworld(results_df)
    path = output_dir / 'cohort_smallworld.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    saved_files['cohort_smallworld'] = path
    print(f"   ✓ {path}")
    
    # 2. Métricas do grupo
    print("\n2. Plotting cohort metrics...")
    fig = plot_cohort_metrics(results_df)
    if fig:
        path = output_dir / 'cohort_metrics.png'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved_files['cohort_metrics'] = path
        print(f"   ✓ {path}")
    
    # 3. Comparações de métricas
    for metric in ['clustering', 'efficiency']:
        obs_col = f'{metric}_observed'
        if obs_col in results_df.columns:
            print(f"\n3. Plotting {metric} comparison...")
            fig = plot_metric_comparison(results_df, metric)
            if fig:
                path = output_dir / f'comparison_{metric}.png'
                fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                saved_files[f'comparison_{metric}'] = path
                print(f"   ✓ {path}")
    
    # 4. Sumários individuais (opcional, pode ser demorado)
    individual_dir = output_dir / 'individual'
    individual_dir.mkdir(exist_ok=True)
    
    print(f"\n4. Plotting individual summaries...")
    for idx, row in results_df.iterrows():
        subject_id = row['subject_id']
        results_dict = row.to_dict()
        
        try:
            fig = plot_subject_summary(subject_id, results_dict)
            path = individual_dir / f'{subject_id}_summary.png'
            fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            saved_files[f'individual_{subject_id}'] = path
            print(f"   ✓ {subject_id}")
        except Exception as e:
            print(f"   ✗ {subject_id}: {e}")
    
    print(f"\n{'='*60}")
    print(f"REPORT COMPLETE: {len(saved_files)} figures generated")
    print('='*60)
    
    return saved_files


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print("Null Models Visualization Module")
    print("Use: from sars.normative_comparison.null_models_viz import *")
