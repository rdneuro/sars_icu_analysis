"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Gradients Visualization Module                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Visualizações para gradientes de conectividade.                             ║
║                                                                              ║
║  FIGURAS DISPONÍVEIS:                                                        ║
║  - Scatter plots de gradientes (G1 vs G2, G1 vs G3, etc.)                   ║
║  - Distribuição dos valores ao longo de cada gradiente                      ║
║  - Ranking de regiões nos extremos                                           ║
║  - Variância explicada (scree plot)                                          ║
║  - Comparação entre métodos                                                  ║
║  - Visualização de grupo (alinhados)                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

# Tentar importar para brain surface plots
try:
    from nilearn import plotting as ni_plotting
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False

# =============================================================================
# STYLE AND COLORS
# =============================================================================

# Colormap para gradiente sensorimotor → associativo
GRADIENT_CMAP = LinearSegmentedColormap.from_list(
    'sensorimotor_associative',
    ['#3288BD', '#66C2A5', '#ABDDA4', '#E6F598', 
     '#FEE08B', '#FDAE61', '#F46D43', '#D53E4F']
)

# Cores por componente
COMPONENT_COLORS = {
    'G1': '#3C5488',
    'G2': '#E64B35', 
    'G3': '#00A087',
    'G4': '#F39B7F',
    'G5': '#8491B4',
}

# Cores gerais
COLORS = {
    'primary': '#3C5488',
    'secondary': '#E64B35',
    'tertiary': '#00A087',
    'observed': '#E64B35',
    'null': '#4DBBD5',
    'significant': '#00A087',
    'not_sig': '#8C8C8C',
}


def set_publication_style():
    """Configurar estilo matplotlib para publicação."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })


# =============================================================================
# BASIC GRADIENT PLOTS
# =============================================================================

def plot_gradient_scatter(
    gradients: np.ndarray,
    components: Tuple[int, int] = (1, 2),
    color_by: Union[np.ndarray, str] = 'G1',
    roi_labels: List[str] = None,
    highlight_rois: List[int] = None,
    title: str = None,
    ax: plt.Axes = None,
    figsize: Tuple[float, float] = (8, 8),
    cmap: str = None,
    show_colorbar: bool = True,
    save_path: str = None
) -> plt.Figure:
    """
    Scatter plot de dois gradientes.
    
    Parameters
    ----------
    gradients : np.ndarray
        Array de gradientes (n_rois, n_components)
    components : Tuple[int, int]
        Quais componentes plotar (1-indexed). Default: (1, 2) = G1 vs G2
    color_by : np.ndarray or str
        Array de valores para colorir, ou 'G1', 'G2', etc.
    roi_labels : List[str], optional
        Labels das ROIs (para hover/anotações)
    highlight_rois : List[int], optional
        Índices de ROIs para destacar
    title : str, optional
        Título do plot
    ax : plt.Axes, optional
        Axes existente
    figsize : tuple
        Tamanho da figura
    cmap : str, optional
        Colormap
    show_colorbar : bool
        Mostrar colorbar
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Extrair componentes (1-indexed para interface)
    c1, c2 = components[0] - 1, components[1] - 1
    x = gradients[:, c1]
    y = gradients[:, c2]
    
    # Determinar cores
    if isinstance(color_by, str):
        if color_by.upper().startswith('G'):
            c_idx = int(color_by[1:]) - 1
            colors = gradients[:, c_idx]
        else:
            colors = gradients[:, 0]
    else:
        colors = color_by
    
    # Colormap
    if cmap is None:
        cmap = GRADIENT_CMAP
    
    # Scatter principal
    scatter = ax.scatter(
        x, y,
        c=colors,
        cmap=cmap,
        s=50,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.3
    )
    
    # Destacar ROIs específicas
    if highlight_rois is not None:
        ax.scatter(
            x[highlight_rois], y[highlight_rois],
            s=100,
            facecolors='none',
            edgecolors='red',
            linewidth=2,
            label='Highlighted'
        )
        
        if roi_labels is not None:
            for idx in highlight_rois:
                ax.annotate(
                    roi_labels[idx],
                    (x[idx], y[idx]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )
    
    # Linhas de referência
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Labels
    ax.set_xlabel(f'Gradient {components[0]} (G{components[0]})')
    ax.set_ylabel(f'Gradient {components[1]} (G{components[1]})')
    
    if title:
        ax.set_title(title)
    
    # Colorbar
    if show_colorbar:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        if isinstance(color_by, str):
            cbar.set_label(color_by)
        else:
            cbar.set_label('Value')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_gradient_3d(
    gradients: np.ndarray,
    components: Tuple[int, int, int] = (1, 2, 3),
    color_by: Union[np.ndarray, str] = 'G1',
    roi_labels: List[str] = None,
    title: str = None,
    figsize: Tuple[float, float] = (10, 8),
    elev: float = 20,
    azim: float = 45,
    save_path: str = None
) -> plt.Figure:
    """
    Plot 3D de três gradientes.
    
    Parameters
    ----------
    gradients : np.ndarray
        Array de gradientes
    components : Tuple[int, int, int]
        Quais componentes plotar (1-indexed)
    color_by : np.ndarray or str
        Valores para colorir
    roi_labels : List[str], optional
        Labels das ROIs
    title : str, optional
        Título
    figsize : tuple
        Tamanho da figura
    elev, azim : float
        Ângulos de visualização
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extrair componentes
    c1, c2, c3 = [c - 1 for c in components]
    x = gradients[:, c1]
    y = gradients[:, c2]
    z = gradients[:, c3]
    
    # Cores
    if isinstance(color_by, str) and color_by.upper().startswith('G'):
        c_idx = int(color_by[1:]) - 1
        colors = gradients[:, c_idx]
    else:
        colors = color_by if isinstance(color_by, np.ndarray) else gradients[:, 0]
    
    # Scatter
    scatter = ax.scatter(
        x, y, z,
        c=colors,
        cmap=GRADIENT_CMAP,
        s=30,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.2
    )
    
    # Labels
    ax.set_xlabel(f'G{components[0]}')
    ax.set_ylabel(f'G{components[1]}')
    ax.set_zlabel(f'G{components[2]}')
    
    if title:
        ax.set_title(title)
    
    # Ângulo de visualização
    ax.view_init(elev=elev, azim=azim)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('G1 Value')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_gradient_distribution(
    gradients: np.ndarray,
    n_components: int = 5,
    roi_labels: List[str] = None,
    figsize: Tuple[float, float] = (14, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Plotar distribuição dos valores ao longo de cada gradiente.
    
    Inclui histograma e violin plot para cada componente.
    
    Parameters
    ----------
    gradients : np.ndarray
        Array de gradientes
    n_components : int
        Quantos componentes mostrar
    roi_labels : List[str], optional
        Labels das ROIs
    figsize : tuple
        Tamanho
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    n_comp = min(n_components, gradients.shape[1])
    
    fig, axes = plt.subplots(2, n_comp, figsize=figsize)
    
    for c in range(n_comp):
        values = gradients[:, c]
        color = COMPONENT_COLORS.get(f'G{c+1}', COLORS['primary'])
        
        # Histograma (row 1)
        ax = axes[0, c]
        ax.hist(values, bins=25, color=color, edgecolor='white', alpha=0.7)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'G{c+1} Value')
        ax.set_ylabel('Count')
        ax.set_title(f'Gradient {c+1}')
        
        # Stats
        ax.text(0.95, 0.95, 
                f'μ={np.mean(values):.2f}\nσ={np.std(values):.2f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Violin (row 2)
        ax = axes[1, c]
        parts = ax.violinplot(values, positions=[0], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_ylabel(f'G{c+1} Value')
        ax.set_xticks([])
    
    fig.suptitle('Gradient Value Distributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_scree(
    explained_variance: np.ndarray,
    n_components: int = 10,
    cumulative: bool = True,
    title: str = None,
    figsize: Tuple[float, float] = (10, 6),
    save_path: str = None
) -> plt.Figure:
    """
    Scree plot mostrando variância explicada por componente.
    
    Parameters
    ----------
    explained_variance : np.ndarray
        Proporção de variância explicada por componente
    n_components : int
        Quantos componentes mostrar
    cumulative : bool
        Se True, mostra também linha cumulativa
    title : str, optional
        Título
    figsize : tuple
        Tamanho
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    n_comp = min(n_components, len(explained_variance))
    x = np.arange(1, n_comp + 1)
    var = explained_variance[:n_comp]
    
    # Barras de variância individual
    bars = ax.bar(x, var * 100, color=COLORS['primary'], 
                  edgecolor='black', alpha=0.7, label='Individual')
    
    # Valores nas barras
    for bar, v in zip(bars, var):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Linha cumulativa
    if cumulative:
        cumvar = np.cumsum(var) * 100
        ax2 = ax.twinx()
        ax2.plot(x, cumvar, 'o-', color=COLORS['secondary'], 
                 linewidth=2, markersize=8, label='Cumulative')
        ax2.set_ylabel('Cumulative Variance (%)', color=COLORS['secondary'])
        ax2.tick_params(axis='y', labelcolor=COLORS['secondary'])
        ax2.set_ylim(0, 105)
        
        # Linhas de referência
        for threshold in [50, 80, 90]:
            ax2.axhline(y=threshold, color='gray', linestyle=':', alpha=0.5)
            ax2.text(n_comp + 0.2, threshold, f'{threshold}%', 
                    fontsize=8, va='center', alpha=0.7)
    
    ax.set_xlabel('Gradient Component')
    ax.set_ylabel('Explained Variance (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{i}' for i in x])
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Gradient Explained Variance (Scree Plot)')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_gradient_ranking(
    gradients: np.ndarray,
    roi_labels: List[str],
    component: int = 1,
    n_top: int = 15,
    title: str = None,
    figsize: Tuple[float, float] = (12, 10),
    save_path: str = None
) -> plt.Figure:
    """
    Plotar ranking de regiões ao longo de um gradiente.
    
    Mostra as regiões nos extremos do gradiente (bottom = sensorimotor,
    top = associativo, para G1).
    
    Parameters
    ----------
    gradients : np.ndarray
        Array de gradientes
    roi_labels : List[str]
        Labels das ROIs
    component : int
        Qual componente (1-indexed)
    n_top : int
        Quantas regiões mostrar em cada extremo
    title : str, optional
        Título
    figsize : tuple
        Tamanho
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    gradient = gradients[:, component - 1]
    sorted_idx = np.argsort(gradient)
    
    # Extremos
    bottom_idx = sorted_idx[:n_top]
    top_idx = sorted_idx[-n_top:][::-1]
    
    # Preparar dados
    all_idx = np.concatenate([bottom_idx, top_idx])
    values = gradient[all_idx]
    labels = [roi_labels[i] for i in all_idx]
    
    # Cores
    colors = [COMPONENT_COLORS['G1']] * n_top + [COMPONENT_COLORS['G2']] * n_top
    
    # Plot
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    ax.set_xlabel(f'G{component} Value')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Gradient {component} Ranking (Bottom & Top {n_top})')
    
    # Anotações
    ax.text(0.02, 0.02, 'Bottom (e.g., Sensorimotor)', transform=ax.transAxes,
            fontsize=10, color=COMPONENT_COLORS['G1'], fontweight='bold')
    ax.text(0.98, 0.98, 'Top (e.g., Associative)', transform=ax.transAxes,
            fontsize=10, color=COMPONENT_COLORS['G2'], fontweight='bold',
            ha='right', va='top')
    
    # Linha separadora
    ax.axhline(y=n_top - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# COMPARISON PLOTS
# =============================================================================

def plot_method_comparison(
    results_dict: Dict,
    n_components: int = 3,
    figsize: Tuple[float, float] = (16, 10),
    save_path: str = None
) -> plt.Figure:
    """
    Comparar gradientes de diferentes métodos.
    
    Parameters
    ----------
    results_dict : Dict
        Dicionário {method_name: GradientResult}
    n_components : int
        Quantos componentes comparar
    figsize : tuple
        Tamanho
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    methods = list(results_dict.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(n_components, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    if n_components == 1:
        axes = axes.reshape(1, -1)
    
    for col, method in enumerate(methods):
        result = results_dict[method]
        gradients = result.gradients if hasattr(result, 'gradients') else result
        
        for row in range(min(n_components, gradients.shape[1])):
            ax = axes[row, col]
            
            values = gradients[:, row]
            
            # Histograma
            ax.hist(values, bins=25, color=COMPONENT_COLORS.get(f'G{row+1}', 'blue'),
                    edgecolor='white', alpha=0.7)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            if row == 0:
                ax.set_title(f'{method.upper()}')
            if col == 0:
                ax.set_ylabel(f'G{row+1}')
            
            # Variância explicada
            if hasattr(result, 'explained_variance'):
                var = result.explained_variance[row]
                ax.text(0.95, 0.95, f'Var: {var*100:.1f}%',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=8, bbox=dict(boxstyle='round', facecolor='white'))
    
    fig.suptitle('Gradient Method Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_gradient_correlation_matrix(
    gradients1: np.ndarray,
    gradients2: np.ndarray,
    labels: Tuple[str, str] = ('Method 1', 'Method 2'),
    n_components: int = 5,
    figsize: Tuple[float, float] = (8, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Plotar matriz de correlação entre gradientes de dois métodos/sujeitos.
    
    Parameters
    ----------
    gradients1, gradients2 : np.ndarray
        Gradientes a comparar
    labels : Tuple[str, str]
        Labels para cada conjunto
    n_components : int
        Quantos componentes
    figsize : tuple
        Tamanho
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    from scipy import stats
    
    n_comp = min(n_components, gradients1.shape[1], gradients2.shape[1])
    
    # Calcular correlações
    corr_matrix = np.zeros((n_comp, n_comp))
    for i in range(n_comp):
        for j in range(n_comp):
            corr_matrix[i, j], _ = stats.spearmanr(
                gradients1[:, i], gradients2[:, j]
            )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(np.abs(corr_matrix), cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
    
    # Labels
    ticks = range(n_comp)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'G{i+1}' for i in ticks])
    ax.set_yticklabels([f'G{i+1}' for i in ticks])
    ax.set_xlabel(labels[1])
    ax.set_ylabel(labels[0])
    
    # Valores nas células
    for i in range(n_comp):
        for j in range(n_comp):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    color=color, fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('|Spearman ρ|')
    
    ax.set_title(f'Gradient Correlation: {labels[0]} vs {labels[1]}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# GROUP PLOTS
# =============================================================================

def plot_group_gradients(
    aligned_gradients: Dict[str, np.ndarray],
    group_mean: np.ndarray,
    components: Tuple[int, int] = (1, 2),
    show_individuals: bool = True,
    figsize: Tuple[float, float] = (12, 10),
    save_path: str = None
) -> plt.Figure:
    """
    Plotar gradientes do grupo (alinhados).
    
    Parameters
    ----------
    aligned_gradients : Dict[str, np.ndarray]
        Dicionário {subject_id: gradients}
    group_mean : np.ndarray
        Média do grupo
    components : Tuple[int, int]
        Quais componentes plotar
    show_individuals : bool
        Se True, mostra pontos individuais transparentes
    figsize : tuple
        Tamanho
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    c1, c2 = components[0] - 1, components[1] - 1
    
    # Panel A: Todos os sujeitos sobrepostos
    ax = axes[0]
    
    for subject_id, grads in aligned_gradients.items():
        ax.scatter(grads[:, c1], grads[:, c2], 
                   s=10, alpha=0.1, color=COLORS['primary'])
    
    # Média
    scatter = ax.scatter(group_mean[:, c1], group_mean[:, c2],
                         c=group_mean[:, c1], cmap=GRADIENT_CMAP,
                         s=50, alpha=1, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'G{components[0]}')
    ax.set_ylabel(f'G{components[1]}')
    ax.set_title('A. All Subjects (aligned) + Group Mean')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # Panel B: Variabilidade
    ax = axes[1]
    
    # Calcular std por ROI
    all_grads = np.stack(list(aligned_gradients.values()), axis=0)
    std_per_roi = np.std(all_grads, axis=0)
    
    # Colorir por variabilidade
    variability = np.sqrt(std_per_roi[:, c1]**2 + std_per_roi[:, c2]**2)
    
    scatter = ax.scatter(group_mean[:, c1], group_mean[:, c2],
                         c=variability, cmap='YlOrRd',
                         s=50, alpha=0.8, edgecolors='black', linewidth=0.3)
    
    ax.set_xlabel(f'G{components[0]}')
    ax.set_ylabel(f'G{components[1]}')
    ax.set_title('B. Group Mean (colored by variability)')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Inter-subject variability (std)')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    fig.suptitle(f'Group Gradient Analysis (n={len(aligned_gradients)})',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_gradient_variability(
    aligned_gradients: Dict[str, np.ndarray],
    roi_labels: List[str],
    component: int = 1,
    n_top: int = 20,
    figsize: Tuple[float, float] = (14, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Plotar variabilidade inter-sujeito ao longo de um gradiente.
    
    Parameters
    ----------
    aligned_gradients : Dict[str, np.ndarray]
        Gradientes alinhados por sujeito
    roi_labels : List[str]
        Labels das ROIs
    component : int
        Qual componente (1-indexed)
    n_top : int
        Quantas regiões com maior variabilidade mostrar
    figsize : tuple
        Tamanho
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    c = component - 1
    
    # Calcular estatísticas
    all_grads = np.stack(list(aligned_gradients.values()), axis=0)  # (n_subjects, n_rois, n_comp)
    mean_per_roi = np.mean(all_grads[:, :, c], axis=0)
    std_per_roi = np.std(all_grads[:, :, c], axis=0)
    
    # Panel A: Mean vs Std scatter
    ax = axes[0]
    ax.scatter(mean_per_roi, std_per_roi, 
               c=mean_per_roi, cmap=GRADIENT_CMAP,
               s=50, alpha=0.7, edgecolors='black', linewidth=0.3)
    ax.set_xlabel(f'Mean G{component}')
    ax.set_ylabel(f'Std G{component}')
    ax.set_title('A. Mean vs Variability per ROI')
    
    # Panel B: Regions with highest variability
    ax = axes[1]
    
    top_var_idx = np.argsort(std_per_roi)[-n_top:][::-1]
    top_labels = [roi_labels[i] for i in top_var_idx]
    top_std = std_per_roi[top_var_idx]
    top_mean = mean_per_roi[top_var_idx]
    
    # Colorir por posição no gradiente
    colors = plt.cm.get_cmap(GRADIENT_CMAP)(
        (top_mean - top_mean.min()) / (top_mean.max() - top_mean.min() + 1e-10)
    )
    
    y_pos = np.arange(len(top_labels))
    ax.barh(y_pos, top_std, color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels, fontsize=9)
    ax.set_xlabel(f'G{component} Std')
    ax.set_title(f'B. Top {n_top} Variable Regions')
    
    fig.suptitle(f'Gradient {component} Inter-Subject Variability',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# SUMMARY PLOTS
# =============================================================================

def plot_gradient_summary(
    gradients: np.ndarray,
    explained_variance: np.ndarray,
    roi_labels: List[str] = None,
    subject_id: str = None,
    figsize: Tuple[float, float] = (16, 12),
    save_path: str = None
) -> plt.Figure:
    """
    Painel resumo completo dos gradientes de um sujeito.
    
    Parameters
    ----------
    gradients : np.ndarray
        Array de gradientes
    explained_variance : np.ndarray
        Variância explicada
    roi_labels : List[str], optional
        Labels das ROIs
    subject_id : str, optional
        ID do sujeito
    figsize : tuple
        Tamanho
    save_path : str, optional
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    set_publication_style()
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # A: G1 vs G2 scatter
    ax = fig.add_subplot(gs[0, 0])
    scatter = ax.scatter(gradients[:, 0], gradients[:, 1],
                         c=gradients[:, 0], cmap=GRADIENT_CMAP,
                         s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
    ax.set_xlabel('G1')
    ax.set_ylabel('G2')
    ax.set_title('A. Gradient Space (G1 vs G2)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    plt.colorbar(scatter, ax=ax, shrink=0.8, label='G1')
    
    # B: G1 vs G3
    if gradients.shape[1] > 2:
        ax = fig.add_subplot(gs[0, 1])
        ax.scatter(gradients[:, 0], gradients[:, 2],
                   c=gradients[:, 0], cmap=GRADIENT_CMAP,
                   s=40, alpha=0.8, edgecolors='black', linewidth=0.3)
        ax.set_xlabel('G1')
        ax.set_ylabel('G3')
        ax.set_title('B. Gradient Space (G1 vs G3)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # C: Scree plot
    ax = fig.add_subplot(gs[0, 2])
    n_comp = min(10, len(explained_variance))
    x = np.arange(1, n_comp + 1)
    ax.bar(x, explained_variance[:n_comp] * 100, 
           color=COLORS['primary'], edgecolor='black', alpha=0.7)
    ax.plot(x, np.cumsum(explained_variance[:n_comp]) * 100, 
            'o-', color=COLORS['secondary'], linewidth=2)
    ax.set_xlabel('Component')
    ax.set_ylabel('Variance (%)')
    ax.set_title('C. Explained Variance')
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{i}' for i in x], fontsize=8)
    
    # D-E-F: Distributions
    for c, ax_pos in enumerate([gs[1, 0], gs[1, 1], gs[1, 2]]):
        if c >= gradients.shape[1]:
            continue
        ax = fig.add_subplot(ax_pos)
        
        values = gradients[:, c]
        color = COMPONENT_COLORS.get(f'G{c+1}', COLORS['primary'])
        
        ax.hist(values, bins=25, color=color, edgecolor='white', alpha=0.7)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(values), color='black', linestyle='-', 
                   label=f'Mean={np.mean(values):.2f}')
        ax.set_xlabel(f'G{c+1} Value')
        ax.set_ylabel('Count')
        ax.set_title(f'D. G{c+1} Distribution (var={explained_variance[c]*100:.1f}%)')
        ax.legend(fontsize=8)
    
    # Título
    title = 'Gradient Analysis Summary'
    if subject_id:
        title = f'{subject_id} - {title}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print("Gradients Visualization Module")
    print("Use: from sars.gradients import viz")
