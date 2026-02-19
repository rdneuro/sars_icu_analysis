"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Gradient Null Models - Visualization Module                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Visualizações para análise de gradientes vs modelos nulos.                  ║
║                                                                              ║
║  FIGURAS PRINCIPAIS:                                                         ║
║  1. Distribuição nula da variância explicada com valor observado             ║
║  2. Gradient scatter plots (G1 vs G2) observado e surrogates                 ║
║  3. Ranking de regiões ao longo do gradiente                                 ║
║  4. Cohort summary com z-scores                                              ║
║  5. Procrustes alignment visualization                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Cores
COLORS = {
    'observed': '#E64B35',
    'null': '#4DBBD5',
    'null_fill': '#4DBBD530',
    'significant': '#00A087',
    'not_sig': '#8C8C8C',
    'G1': '#3C5488',
    'G2': '#F39B7F',
    'G3': '#91D1C2',
}

# Colormap para gradientes (sensorimotor → associativo)
GRADIENT_CMAP = LinearSegmentedColormap.from_list(
    'gradient', ['#3288BD', '#66C2A5', '#ABDDA4', '#E6F598', 
                 '#FEE08B', '#FDAE61', '#F46D43', '#D53E4F']
)


def set_style():
    """Configurar estilo matplotlib."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


# =============================================================================
# SINGLE SUBJECT VISUALIZATIONS
# =============================================================================

def plot_variance_null_distribution(
    observed_variance: np.ndarray,
    null_variance_distribution: np.ndarray,
    n_components_to_show: int = 5,
    subject_id: str = None,
    figsize: Tuple[float, float] = (14, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Plotar distribuição nula da variância explicada por componente.
    
    Para cada gradiente (G1, G2, ...), mostra o histograma dos valores
    de variância explicada obtidos nos surrogates, com o valor observado
    marcado como linha vertical.
    
    Parameters
    ----------
    observed_variance : np.ndarray
        Variância explicada observada por componente
    null_variance_distribution : np.ndarray
        Array (n_surrogates, n_components) com variâncias nulas
    n_components_to_show : int
        Quantos componentes mostrar
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
    set_style()
    
    n_components = min(n_components_to_show, len(observed_variance), 
                       null_variance_distribution.shape[1])
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for c in range(n_components):
        ax = axes[c]
        
        obs_val = observed_variance[c]
        null_vals = null_variance_distribution[:, c]
        
        # Estatísticas
        null_mean = np.mean(null_vals)
        null_std = np.std(null_vals)
        z_score = (obs_val - null_mean) / null_std if null_std > 0 else 0
        p_value = np.mean(null_vals >= obs_val)
        
        # Histograma
        ax.hist(null_vals, bins=25, density=True, color=COLORS['null'],
                edgecolor='white', alpha=0.7, label='Null')
        
        # Valor observado
        ax.axvline(x=obs_val, color=COLORS['observed'], linewidth=2.5,
                   label=f'Observed = {obs_val:.4f}')
        
        # Média nula
        ax.axvline(x=null_mean, color=COLORS['null'], linewidth=1.5,
                   linestyle='--', alpha=0.8, label=f'Null mean = {null_mean:.4f}')
        
        # Shading
        if obs_val > null_mean:
            ax.axvspan(obs_val, ax.get_xlim()[1], alpha=0.15, color=COLORS['observed'])
        
        ax.set_xlabel('Explained Variance')
        ax.set_ylabel('Density')
        ax.set_title(f'G{c+1}')
        
        if c == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        # Stats box
        sig = p_value < 0.05
        sig_color = COLORS['significant'] if sig else COLORS['not_sig']
        stats_text = f'z = {z_score:.2f}\np = {p_value:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', 
                         edgecolor=sig_color, linewidth=2, alpha=0.9),
                color=sig_color, fontweight='bold')
    
    # Esconder axes extras
    for idx in range(n_components, len(axes)):
        axes[idx].set_visible(False)
    
    # Título
    title = 'Gradient Variance: Observed vs Null Distribution'
    if subject_id:
        title = f'{subject_id} - {title}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_gradient_scatter(
    observed_gradients: np.ndarray,
    null_gradients_list: List[np.ndarray] = None,
    n_surrogates_to_show: int = 5,
    roi_labels: List[str] = None,
    subject_id: str = None,
    figsize: Tuple[float, float] = (12, 10),
    save_path: str = None
) -> plt.Figure:
    """
    Scatter plot de G1 vs G2 comparando observado com surrogates.
    
    Parameters
    ----------
    observed_gradients : np.ndarray
        Gradientes observados (n_rois, n_components)
    null_gradients_list : List[np.ndarray], optional
        Lista de gradientes nulos para comparação
    n_surrogates_to_show : int
        Quantos surrogates mostrar no plot
    roi_labels : List[str], optional
        Labels das ROIs
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
    set_style()
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Painel A: G1 vs G2 observado
    ax = axes[0, 0]
    scatter = ax.scatter(observed_gradients[:, 0], observed_gradients[:, 1],
                         c=observed_gradients[:, 0], cmap=GRADIENT_CMAP,
                         s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
    ax.set_xlabel('Gradient 1 (G1)')
    ax.set_ylabel('Gradient 2 (G2)')
    ax.set_title('A. Observed Gradients')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Painel B: G1 vs G3 observado
    ax = axes[0, 1]
    if observed_gradients.shape[1] > 2:
        ax.scatter(observed_gradients[:, 0], observed_gradients[:, 2],
                   c=observed_gradients[:, 0], cmap=GRADIENT_CMAP,
                   s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
        ax.set_xlabel('Gradient 1 (G1)')
        ax.set_ylabel('Gradient 3 (G3)')
        ax.set_title('B. G1 vs G3')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    else:
        ax.set_visible(False)
    
    # Painel C: G2 vs G3 observado
    ax = axes[0, 2]
    if observed_gradients.shape[1] > 2:
        ax.scatter(observed_gradients[:, 1], observed_gradients[:, 2],
                   c=observed_gradients[:, 0], cmap=GRADIENT_CMAP,
                   s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
        ax.set_xlabel('Gradient 2 (G2)')
        ax.set_ylabel('Gradient 3 (G3)')
        ax.set_title('C. G2 vs G3')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    else:
        ax.set_visible(False)
    
    # Painéis D-F: Surrogates
    if null_gradients_list is not None and len(null_gradients_list) > 0:
        for idx, (ax, null_grad) in enumerate(zip(
            axes[1, :], null_gradients_list[:min(3, n_surrogates_to_show)]
        )):
            ax.scatter(null_grad[:, 0], null_grad[:, 1],
                       c=COLORS['null'], s=20, alpha=0.5,
                       edgecolors='none')
            ax.set_xlabel('G1')
            ax.set_ylabel('G2')
            ax.set_title(f'D. Surrogate {idx+1}')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    else:
        for ax in axes[1, :]:
            ax.set_visible(False)
    
    # Colorbar
    cbar = fig.colorbar(scatter, ax=axes[0, :], shrink=0.6, aspect=30, pad=0.02)
    cbar.set_label('G1 Value (Sensorimotor → Associative)')
    
    # Título
    title = 'Gradient Space: Observed vs Null'
    if subject_id:
        title = f'{subject_id} - {title}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_gradient_ranking(
    observed_gradients: np.ndarray,
    roi_labels: List[str] = None,
    n_top: int = 10,
    subject_id: str = None,
    figsize: Tuple[float, float] = (14, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Visualizar ranking das regiões ao longo dos gradientes.
    
    Mostra as regiões nos extremos de cada gradiente (top e bottom).
    Para G1, isso tipicamente significa sensorimotor vs associativo.
    
    Parameters
    ----------
    observed_gradients : np.ndarray
        Gradientes observados
    roi_labels : List[str], optional
        Labels das ROIs. Se None, usa índices.
    n_top : int
        Quantas regiões mostrar em cada extremo
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
    set_style()
    
    n_rois = observed_gradients.shape[0]
    
    if roi_labels is None:
        roi_labels = [f'ROI_{i}' for i in range(n_rois)]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for c, ax in enumerate(axes):
        if c >= observed_gradients.shape[1]:
            ax.set_visible(False)
            continue
        
        gradient = observed_gradients[:, c]
        
        # Ordenar por valor do gradiente
        sorted_idx = np.argsort(gradient)
        
        # Top (mais positivos) e bottom (mais negativos)
        bottom_idx = sorted_idx[:n_top]
        top_idx = sorted_idx[-n_top:][::-1]
        
        # Criar barras horizontais
        y_pos = np.arange(n_top * 2)
        
        # Bottom (azul) e top (vermelho)
        values = np.concatenate([gradient[bottom_idx], gradient[top_idx]])
        labels = [roi_labels[i] for i in bottom_idx] + [roi_labels[i] for i in top_idx]
        colors = [COLORS['G1']] * n_top + [COLORS['G2']] * n_top
        
        ax.barh(y_pos, values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel(f'G{c+1} Value')
        ax.set_title(f'Gradient {c+1} Ranking')
        
        # Anotações
        ax.text(0.02, 0.02, 'Bottom (e.g., Sensorimotor)', transform=ax.transAxes,
                fontsize=8, color=COLORS['G1'], fontweight='bold')
        ax.text(0.98, 0.02, 'Top (e.g., Associative)', transform=ax.transAxes,
                fontsize=8, color=COLORS['G2'], fontweight='bold', ha='right')
    
    # Título
    title = 'Gradient Ranking: Regions at Extremes'
    if subject_id:
        title = f'{subject_id} - {title}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_subject_gradient_summary(
    results: Dict,
    figsize: Tuple[float, float] = (16, 12),
    save_path: str = None
) -> plt.Figure:
    """
    Painel resumo completo da análise de gradientes para um sujeito.
    
    Parameters
    ----------
    results : Dict
        Output de analyze_gradients_vs_null()
    figsize : tuple
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_style()
    
    subject_id = results.get('subject_id', 'Unknown')
    comparison = results['comparison']
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Row 1: Gradient scatter and variance explained
    # =========================================================================
    
    # A: G1 vs G2 scatter
    ax1 = fig.add_subplot(gs[0, 0:2])
    gradients = results['observed_gradients']
    scatter = ax1.scatter(gradients[:, 0], gradients[:, 1],
                          c=gradients[:, 0], cmap=GRADIENT_CMAP,
                          s=50, alpha=0.8, edgecolors='black', linewidth=0.3)
    ax1.set_xlabel('Gradient 1')
    ax1.set_ylabel('Gradient 2')
    ax1.set_title('A. Gradient Space')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.colorbar(scatter, ax=ax1, shrink=0.8, label='G1')
    
    # B: Variance explained bar plot
    ax2 = fig.add_subplot(gs[0, 2:4])
    n_comp = min(5, len(results['observed_variance']))
    x = np.arange(n_comp)
    
    obs_var = results['observed_variance'][:n_comp]
    null_var_mean = np.mean(comparison.null_variance_distribution[:, :n_comp], axis=0)
    null_var_std = np.std(comparison.null_variance_distribution[:, :n_comp], axis=0)
    
    width = 0.35
    ax2.bar(x - width/2, obs_var, width, color=COLORS['observed'], 
            label='Observed', edgecolor='black')
    ax2.bar(x + width/2, null_var_mean, width, color=COLORS['null'],
            label='Null (mean)', edgecolor='black', yerr=null_var_std, capsize=3)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'G{i+1}' for i in range(n_comp)])
    ax2.set_ylabel('Explained Variance')
    ax2.set_title('B. Variance: Observed vs Null')
    ax2.legend()
    
    # =========================================================================
    # Row 2: Null distributions for G1 and G2
    # =========================================================================
    
    # C: G1 null distribution
    ax3 = fig.add_subplot(gs[1, 0:2])
    null_g1 = comparison.null_variance_distribution[:, 0]
    obs_g1 = results['G1_variance_observed']
    
    ax3.hist(null_g1, bins=25, density=True, color=COLORS['null'],
             edgecolor='white', alpha=0.7)
    ax3.axvline(x=obs_g1, color=COLORS['observed'], linewidth=2.5,
                label=f'Observed = {obs_g1:.4f}')
    ax3.axvline(x=np.mean(null_g1), color=COLORS['null'], linewidth=1.5,
                linestyle='--', label=f'Null mean')
    
    z1 = results['G1_variance_zscore']
    p1 = results['G1_variance_pvalue']
    sig1 = COLORS['significant'] if p1 < 0.05 else COLORS['not_sig']
    ax3.text(0.02, 0.98, f'z = {z1:.2f}\np = {p1:.4f}', transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=sig1, linewidth=2),
             color=sig1, fontweight='bold')
    
    ax3.set_xlabel('G1 Variance')
    ax3.set_ylabel('Density')
    ax3.set_title('C. G1 Null Distribution')
    ax3.legend()
    
    # D: G2 null distribution
    ax4 = fig.add_subplot(gs[1, 2:4])
    null_g2 = comparison.null_variance_distribution[:, 1]
    obs_g2 = results['G2_variance_observed']
    
    ax4.hist(null_g2, bins=25, density=True, color=COLORS['null'],
             edgecolor='white', alpha=0.7)
    ax4.axvline(x=obs_g2, color=COLORS['observed'], linewidth=2.5,
                label=f'Observed = {obs_g2:.4f}')
    ax4.axvline(x=np.mean(null_g2), color=COLORS['null'], linewidth=1.5,
                linestyle='--', label='Null mean')
    
    z2 = results['G2_variance_zscore']
    p2 = results['G2_variance_pvalue']
    sig2 = COLORS['significant'] if p2 < 0.05 else COLORS['not_sig']
    ax4.text(0.02, 0.98, f'z = {z2:.2f}\np = {p2:.4f}', transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=sig2, linewidth=2),
             color=sig2, fontweight='bold')
    
    ax4.set_xlabel('G2 Variance')
    ax4.set_ylabel('Density')
    ax4.set_title('D. G2 Null Distribution')
    ax4.legend()
    
    # =========================================================================
    # Row 3: Z-scores and interpretation
    # =========================================================================
    
    # E: Z-scores bar plot
    ax5 = fig.add_subplot(gs[2, 0:2])
    
    metrics = ['G1 Variance', 'G2 Variance', 'G1 Dispersion', 'G2 Dispersion']
    zscores = [
        results['G1_variance_zscore'],
        results['G2_variance_zscore'],
        results['G1_dispersion_zscore'],
        results['G2_dispersion_zscore']
    ]
    
    colors = [COLORS['significant'] if abs(z) > 2 else COLORS['not_sig'] for z in zscores]
    
    bars = ax5.barh(range(len(metrics)), zscores, color=colors, edgecolor='black')
    ax5.set_yticks(range(len(metrics)))
    ax5.set_yticklabels(metrics)
    ax5.axvline(x=2, color='red', linestyle='--', alpha=0.5)
    ax5.axvline(x=-2, color='red', linestyle='--', alpha=0.5)
    ax5.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Z-score')
    ax5.set_title('E. Z-scores vs Null Model')
    
    # F: Interpretation text
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax6.axis('off')
    
    g1_sig = results['G1_variance_pvalue'] < 0.05
    g2_sig = results['G2_variance_pvalue'] < 0.05
    
    interpretation = [
        "INTERPRETATION",
        "=" * 40,
        "",
        f"Gradient 1 (primary axis):",
        f"  Variance: {results['G1_variance_observed']:.4f}",
        f"  vs Null: {results['G1_variance_null_mean']:.4f}",
        f"  z = {results['G1_variance_zscore']:.2f}, p = {results['G1_variance_pvalue']:.4f}",
        f"  {'✓ SIGNIFICANT' if g1_sig else '× not significant'}",
        "",
        f"Gradient 2:",
        f"  Variance: {results['G2_variance_observed']:.4f}",
        f"  z = {results['G2_variance_zscore']:.2f}",
        f"  {'✓ SIGNIFICANT' if g2_sig else '× not significant'}",
        "",
        f"Procrustes distance: {results['mean_procrustes_distance']:.4f}",
        "",
        "CONCLUSION:",
    ]
    
    if g1_sig:
        interpretation.append("  The cortical hierarchy captured by G1")
        interpretation.append("  is significantly more organized than")
        interpretation.append("  expected in random networks.")
    else:
        interpretation.append("  G1 organization is not significantly")
        interpretation.append("  different from null expectation.")
    
    text = '\n'.join(interpretation)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Título principal
    sig_status = "✓ Significant G1" if g1_sig else "× G1 not significant"
    fig.suptitle(f'{subject_id} - Gradient Null Model Analysis\n{sig_status}',
                 fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# COHORT VISUALIZATIONS
# =============================================================================

def plot_cohort_gradient_summary(
    results_df: pd.DataFrame,
    figsize: Tuple[float, float] = (14, 10),
    save_path: str = None
) -> plt.Figure:
    """
    Visualizar resultados de gradientes do grupo.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame com resultados de run_gradient_null_analysis()
    figsize : tuple
        Tamanho da figura
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_style()
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # A: G1 variance distribution
    ax = axes[0, 0]
    ax.hist(results_df['G1_variance'], bins=15, color=COLORS['G1'],
            edgecolor='white', alpha=0.7)
    ax.axvline(x=results_df['G1_variance'].mean(), color='black', linewidth=2,
               label=f"Mean = {results_df['G1_variance'].mean():.4f}")
    ax.set_xlabel('G1 Explained Variance')
    ax.set_ylabel('Count')
    ax.set_title('A. G1 Variance Distribution')
    ax.legend()
    
    # B: G1 z-score distribution
    ax = axes[0, 1]
    zscores = results_df['G1_variance_zscore']
    colors = [COLORS['significant'] if z > 2 else COLORS['not_sig'] for z in zscores]
    ax.hist(zscores, bins=15, color=COLORS['G1'], edgecolor='white', alpha=0.7)
    ax.axvline(x=2, color='red', linestyle='--', linewidth=2, label='z = 2')
    ax.axvline(x=zscores.mean(), color='black', linewidth=2,
               label=f"Mean = {zscores.mean():.2f}")
    ax.set_xlabel('G1 Z-score')
    ax.set_ylabel('Count')
    ax.set_title('B. G1 Z-scores')
    ax.legend()
    
    # C: % significant
    ax = axes[0, 2]
    pct_sig = 100 * results_df['G1_significant'].mean()
    ax.bar(['Significant', 'Not Sig.'], 
           [pct_sig, 100 - pct_sig],
           color=[COLORS['significant'], COLORS['not_sig']],
           edgecolor='black')
    ax.set_ylabel('% Subjects')
    ax.set_title(f'C. G1 Significance ({pct_sig:.1f}% sig)')
    ax.set_ylim(0, 100)
    
    # D: G1 vs G2 variance
    ax = axes[1, 0]
    ax.scatter(results_df['G1_variance'], results_df['G2_variance'],
               c=results_df['G1_variance_zscore'], cmap='RdYlGn',
               s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('G1 Variance')
    ax.set_ylabel('G2 Variance')
    ax.set_title('D. G1 vs G2 Variance')
    plt.colorbar(ax.collections[0], ax=ax, label='G1 z-score')
    
    # E: Observed vs Null mean
    ax = axes[1, 1]
    ax.scatter(results_df['G1_variance_null_mean'], results_df['G1_variance'],
               c=COLORS['observed'], s=60, alpha=0.7, edgecolors='black')
    lims = [min(results_df['G1_variance_null_mean'].min(), results_df['G1_variance'].min()) * 0.9,
            max(results_df['G1_variance_null_mean'].max(), results_df['G1_variance'].max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('G1 Null Mean')
    ax.set_ylabel('G1 Observed')
    ax.set_title('E. Observed vs Null')
    
    # F: Z-scores per subject (sorted)
    ax = axes[1, 2]
    sorted_df = results_df.sort_values('G1_variance_zscore', ascending=True)
    colors = [COLORS['significant'] if z > 2 else COLORS['not_sig'] 
              for z in sorted_df['G1_variance_zscore']]
    ax.barh(range(len(sorted_df)), sorted_df['G1_variance_zscore'], color=colors)
    ax.axvline(x=2, color='red', linestyle='--', alpha=0.5)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['subject_id'], fontsize=7)
    ax.set_xlabel('G1 Z-score')
    ax.set_title('F. Individual Z-scores')
    
    fig.suptitle('Gradient Null Model Analysis - Cohort Summary',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_gradient_report(
    results_df: pd.DataFrame,
    individual_results: Dict[str, Dict] = None,
    output_dir: Union[str, Path] = None,
    atlas_name: str = 'unknown'
) -> Dict[str, Path]:
    """
    Gerar relatório completo de visualizações de gradientes.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame com resultados do grupo
    individual_results : Dict[str, Dict], optional
        Dicionário {subject_id: results} para plots individuais
    output_dir : str or Path
        Diretório de saída
    atlas_name : str
        Nome do atlas
        
    Returns
    -------
    Dict[str, Path]
        Paths das figuras geradas
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING GRADIENT VISUALIZATION REPORT")
    print(f"Output: {output_dir}")
    print('='*60)
    
    saved_files = {}
    
    # 1. Cohort summary
    print("\n1. Plotting cohort summary...")
    fig = plot_cohort_gradient_summary(results_df)
    path = output_dir / 'cohort_gradient_summary.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    saved_files['cohort_summary'] = path
    print(f"   ✓ {path}")
    
    # 2. Individual summaries
    if individual_results:
        individual_dir = output_dir / 'individual'
        individual_dir.mkdir(exist_ok=True)
        
        print(f"\n2. Plotting individual summaries...")
        for subject_id, results in individual_results.items():
            try:
                fig = plot_subject_gradient_summary(results)
                path = individual_dir / f'{subject_id}_gradient_summary.png'
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
    print("Gradient Null Models - Visualization Module")
    print("Use: from sars.normative_comparison.gradients_null_viz import *")
