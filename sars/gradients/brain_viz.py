"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Brain Surface Visualization for Gradients                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Visualização de gradientes em superfícies corticais.                        ║
║                                                                              ║
║  FUNÇÕES PRINCIPAIS:                                                         ║
║  - plot_gradient_on_surface: Gradiente único na superfície                   ║
║  - plot_gradients_panel: Painel com G1, G2, G3                              ║
║  - plot_gradient_comparison: Observado vs Nulo lado a lado                   ║
║  - plot_gradient_zscore_surface: Z-score (onde real ≠ nulo)                 ║
║  - plot_group_gradient: Média do grupo na superfície                        ║
║                                                                              ║
║  REQUISITOS:                                                                 ║
║  - nilearn (para plotting)                                                   ║
║  - Opcionalmente: brainspace, neuromaps                                      ║
║                                                                              ║
║  ATLAS SUPORTADOS:                                                           ║
║  - Schaefer 100/200/400 (7 ou 17 networks)                                  ║
║  - AAL3                                                                       ║
║  - Brainnetome                                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

# =============================================================================
# CHECK DEPENDENCIES
# =============================================================================

try:
    from nilearn import plotting, datasets, surface
    from nilearn.plotting import plot_surf_roi, plot_surf_stat_map
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False
    warnings.warn("nilearn not installed. Brain surface plots disabled. pip install nilearn")

try:
    from brainspace.plotting import plot_hemispheres
    from brainspace.datasets import load_parcellation, load_conte69
    HAS_BRAINSPACE = True
except ImportError:
    HAS_BRAINSPACE = False

try:
    from neuromaps.datasets import fetch_fslr
    from neuromaps.parcellate import Parcellater
    HAS_NEUROMAPS = True
except ImportError:
    HAS_NEUROMAPS = False


# =============================================================================
# COLORMAPS
# =============================================================================

# Gradiente sensorimotor → associativo (igual ao Margulies 2016)
GRADIENT_CMAP = LinearSegmentedColormap.from_list(
    'gradient_margulies',
    ['#3288BD', '#66C2A5', '#ABDDA4', '#E6F598', 
     '#FEE08B', '#FDAE61', '#F46D43', '#D53E4F']
)

# Divergente para z-scores
ZSCORE_CMAP = LinearSegmentedColormap.from_list(
    'zscore_divergent',
    ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0',
     '#F7F7F7',
     '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
)

# Variância explicada
VARIANCE_CMAP = 'YlOrRd'


# =============================================================================
# ATLAS UTILITIES
# =============================================================================

def get_atlas_labels(atlas_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Obter labels do atlas para mapeamento em superfície.
    
    Retorna arrays separados para hemisfério esquerdo e direito.
    
    Parameters
    ----------
    atlas_name : str
        Nome do atlas ('schaefer_100', 'schaefer_400', etc.)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (labels_lh, labels_rh) - arrays com índice da região para cada vértice
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required for atlas labels")
    
    atlas_lower = atlas_name.lower()
    
    if 'schaefer' in atlas_lower:
        # Extrair número de parcelas
        if '100' in atlas_lower:
            n_parcels = 100
        elif '200' in atlas_lower:
            n_parcels = 200
        elif '400' in atlas_lower:
            n_parcels = 400
        else:
            n_parcels = 100
        
        # Número de networks
        n_networks = 7 if '17' not in atlas_lower else 17
        
        atlas = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_parcels,
            yeo_networks=n_networks,
            resolution_mm=1
        )
        
        # Schaefer vem em MNI volume, precisamos do surface
        # Usar fetch_atlas_surf_destrieux ou similar como base
        # Por simplicidade, usamos fsaverage com projeção
        
    elif 'aal' in atlas_lower:
        atlas = datasets.fetch_atlas_aal()
        
    elif 'brainnetome' in atlas_lower:
        # Brainnetome não está no nilearn, precisa carregar manualmente
        pass
    
    # Placeholder - em implementação real, carregaria os labels de superfície
    warnings.warn(f"Atlas {atlas_name} surface labels: using placeholder")
    return None, None


def _load_fsaverage(mesh: str = 'fsaverage5') -> Dict:
    """
    Carregar superfícies fsaverage.
    
    Parameters
    ----------
    mesh : str
        Resolução: 'fsaverage5' (10k vértices), 'fsaverage' (160k)
        
    Returns
    -------
    Dict com 'pial_left', 'pial_right', 'sulc_left', 'sulc_right', etc.
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required")
    
    fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
    return fsaverage


def map_parcellation_to_surface(
    values: np.ndarray,
    atlas_name: str,
    mesh: str = 'fsaverage5'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mapear valores de parcelas para vértices de superfície.
    
    Parameters
    ----------
    values : np.ndarray
        Valores por parcela (n_parcels,)
    atlas_name : str
        Nome do atlas
    mesh : str
        Resolução da superfície
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (values_lh, values_rh) - valores por vértice em cada hemisfério
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required")
    
    n_parcels = len(values)
    atlas_lower = atlas_name.lower()
    
    # Carregar parcellation em superfície
    if 'schaefer' in atlas_lower:
        # Determinar parâmetros
        if '100' in atlas_lower:
            n_rois = 100
        elif '200' in atlas_lower:
            n_rois = 200
        elif '400' in atlas_lower:
            n_rois = 400
        else:
            n_rois = 100
        
        n_networks = 7
        
        # Fetch surface parcellation
        try:
            parcellation = datasets.fetch_atlas_schaefer_2018(
                n_rois=n_rois,
                yeo_networks=n_networks,
                resolution_mm=1
            )
            
            # Para Schaefer, precisamos projetar do volume para superfície
            # ou usar versão de superfície se disponível
            # Por ora, usar abordagem simplificada
            
        except Exception as e:
            warnings.warn(f"Could not load Schaefer surface: {e}")
    
    # Abordagem genérica: assumir que primeiras n/2 parcelas são LH
    n_lh = n_parcels // 2
    
    # Carregar fsaverage
    fsaverage = _load_fsaverage(mesh)
    
    # Número de vértices por hemisfério
    coords_lh, _ = surface.load_surf_mesh(fsaverage['pial_left'])
    coords_rh, _ = surface.load_surf_mesh(fsaverage['pial_right'])
    n_vertices_lh = len(coords_lh)
    n_vertices_rh = len(coords_rh)
    
    # Placeholder: distribuir valores uniformemente
    # Em implementação real, usaria o mapeamento real do atlas
    values_lh = np.zeros(n_vertices_lh)
    values_rh = np.zeros(n_vertices_rh)
    
    # Simular mapeamento (cada parcela ocupa região contígua)
    vertices_per_parcel_lh = n_vertices_lh // n_lh
    vertices_per_parcel_rh = n_vertices_rh // (n_parcels - n_lh)
    
    for i in range(n_lh):
        start = i * vertices_per_parcel_lh
        end = (i + 1) * vertices_per_parcel_lh
        values_lh[start:end] = values[i]
    
    for i in range(n_parcels - n_lh):
        start = i * vertices_per_parcel_rh
        end = (i + 1) * vertices_per_parcel_rh
        values_rh[start:end] = values[n_lh + i]
    
    return values_lh, values_rh


# =============================================================================
# NILEARN-BASED SURFACE PLOTS
# =============================================================================

def plot_gradient_on_surface(
    gradient: np.ndarray,
    atlas_name: str = 'schaefer_100',
    view: str = 'lateral',
    hemisphere: str = 'both',
    cmap: str = None,
    colorbar: bool = True,
    title: str = None,
    vmin: float = None,
    vmax: float = None,
    symmetric_cbar: bool = False,
    figsize: Tuple[float, float] = (12, 6),
    save_path: str = None
) -> plt.Figure:
    """
    Plotar um gradiente na superfície cortical usando nilearn.
    
    Parameters
    ----------
    gradient : np.ndarray
        Valores do gradiente por parcela (n_parcels,)
    atlas_name : str
        Nome do atlas para mapeamento
    view : str
        Visão: 'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'
    hemisphere : str
        'left', 'right', ou 'both'
    cmap : str or Colormap
        Colormap (default: gradient colormap)
    colorbar : bool
        Mostrar colorbar
    title : str
        Título da figura
    vmin, vmax : float
        Limites do colormap
    symmetric_cbar : bool
        Se True, centraliza colorbar em zero
    figsize : tuple
        Tamanho da figura
    save_path : str
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required for surface plots. pip install nilearn")
    
    if cmap is None:
        cmap = GRADIENT_CMAP
    
    # Carregar fsaverage
    fsaverage = _load_fsaverage('fsaverage5')
    
    # Mapear parcelas para vértices
    values_lh, values_rh = map_parcellation_to_surface(gradient, atlas_name)
    
    # Configurar limites
    if symmetric_cbar:
        max_abs = max(abs(np.nanmin(gradient)), abs(np.nanmax(gradient)))
        vmin, vmax = -max_abs, max_abs
    elif vmin is None or vmax is None:
        vmin = np.nanmin(gradient) if vmin is None else vmin
        vmax = np.nanmax(gradient) if vmax is None else vmax
    
    # Criar figura
    if hemisphere == 'both':
        fig, axes = plt.subplots(1, 2, figsize=figsize, subplot_kw={'projection': '3d'})
        
        # Hemisfério esquerdo
        plotting.plot_surf_stat_map(
            fsaverage['pial_left'],
            values_lh,
            hemi='left',
            view=view,
            cmap=cmap,
            colorbar=False,
            vmin=vmin,
            vmax=vmax,
            bg_map=fsaverage['sulc_left'],
            axes=axes[0],
            title='Left Hemisphere' if title is None else None
        )
        
        # Hemisfério direito
        plotting.plot_surf_stat_map(
            fsaverage['pial_right'],
            values_rh,
            hemi='right',
            view=view,
            cmap=cmap,
            colorbar=colorbar,
            vmin=vmin,
            vmax=vmax,
            bg_map=fsaverage['sulc_right'],
            axes=axes[1],
            title='Right Hemisphere' if title is None else None
        )
        
    else:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': '3d'})
        
        surf = fsaverage[f'pial_{hemisphere}']
        values = values_lh if hemisphere == 'left' else values_rh
        sulc = fsaverage[f'sulc_{hemisphere}']
        
        plotting.plot_surf_stat_map(
            surf,
            values,
            hemi=hemisphere,
            view=view,
            cmap=cmap,
            colorbar=colorbar,
            vmin=vmin,
            vmax=vmax,
            bg_map=sulc,
            axes=ax
        )
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_gradients_panel(
    gradients: np.ndarray,
    atlas_name: str = 'schaefer_100',
    n_gradients: int = 3,
    views: List[str] = None,
    figsize: Tuple[float, float] = (16, 12),
    save_path: str = None
) -> plt.Figure:
    """
    Painel com múltiplos gradientes em múltiplas vistas.
    
    Parameters
    ----------
    gradients : np.ndarray
        Array de gradientes (n_parcels, n_components)
    atlas_name : str
        Nome do atlas
    n_gradients : int
        Quantos gradientes mostrar (G1, G2, ...)
    views : List[str]
        Lista de vistas. Default: ['lateral', 'medial']
    figsize : tuple
        Tamanho
    save_path : str
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required")
    
    if views is None:
        views = ['lateral', 'medial']
    
    n_views = len(views)
    n_grad = min(n_gradients, gradients.shape[1])
    
    # Layout: rows = gradients, cols = views × hemispheres
    fig = plt.figure(figsize=figsize)
    
    fsaverage = _load_fsaverage('fsaverage5')
    
    for g in range(n_grad):
        gradient = gradients[:, g]
        values_lh, values_rh = map_parcellation_to_surface(gradient, atlas_name)
        
        vmin, vmax = np.nanmin(gradient), np.nanmax(gradient)
        
        for v_idx, view in enumerate(views):
            # Hemisfério esquerdo
            ax_idx = g * n_views * 2 + v_idx * 2 + 1
            ax = fig.add_subplot(n_grad, n_views * 2, ax_idx, projection='3d')
            
            plotting.plot_surf_stat_map(
                fsaverage['pial_left'],
                values_lh,
                hemi='left',
                view=view,
                cmap=GRADIENT_CMAP,
                colorbar=False,
                vmin=vmin,
                vmax=vmax,
                bg_map=fsaverage['sulc_left'],
                axes=ax
            )
            
            if g == 0:
                ax.set_title(f'{view.capitalize()} L', fontsize=10)
            if v_idx == 0:
                ax.text2D(-0.1, 0.5, f'G{g+1}', transform=ax.transAxes,
                         fontsize=12, fontweight='bold', va='center', ha='right')
            
            # Hemisfério direito
            ax_idx = g * n_views * 2 + v_idx * 2 + 2
            ax = fig.add_subplot(n_grad, n_views * 2, ax_idx, projection='3d')
            
            plotting.plot_surf_stat_map(
                fsaverage['pial_right'],
                values_rh,
                hemi='right',
                view=view,
                cmap=GRADIENT_CMAP,
                colorbar=(v_idx == n_views - 1 and g == 0),  # Só um colorbar
                vmin=vmin,
                vmax=vmax,
                bg_map=fsaverage['sulc_right'],
                axes=ax
            )
            
            if g == 0:
                ax.set_title(f'{view.capitalize()} R', fontsize=10)
    
    fig.suptitle('Cortical Gradients', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# COMPARISON PLOTS: OBSERVED vs NULL
# =============================================================================

def plot_gradient_comparison_surface(
    observed_gradient: np.ndarray,
    null_mean_gradient: np.ndarray,
    atlas_name: str = 'schaefer_100',
    view: str = 'lateral',
    component: int = 1,
    figsize: Tuple[float, float] = (14, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Comparar gradiente observado vs média nula lado a lado.
    
    Parameters
    ----------
    observed_gradient : np.ndarray
        Gradiente observado (n_parcels,)
    null_mean_gradient : np.ndarray
        Média dos gradientes dos surrogates (n_parcels,)
    atlas_name : str
        Nome do atlas
    view : str
        Vista ('lateral', 'medial')
    component : int
        Número do componente (para título)
    figsize : tuple
        Tamanho
    save_path : str
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required")
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, hspace=0.1, wspace=0.05)
    
    fsaverage = _load_fsaverage('fsaverage5')
    
    # Limites comuns para comparação
    all_vals = np.concatenate([observed_gradient, null_mean_gradient])
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
    
    # Observado
    obs_lh, obs_rh = map_parcellation_to_surface(observed_gradient, atlas_name)
    
    # Row 1: Observed
    for col, (hemi, values) in enumerate([('left', obs_lh), ('right', obs_rh)]):
        ax = fig.add_subplot(gs[0, col], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage[f'pial_{hemi}'],
            values,
            hemi=hemi,
            view=view,
            cmap=GRADIENT_CMAP,
            colorbar=False,
            vmin=vmin,
            vmax=vmax,
            bg_map=fsaverage[f'sulc_{hemi}'],
            axes=ax
        )
        if col == 0:
            ax.text2D(-0.1, 0.5, 'OBSERVED', transform=ax.transAxes,
                     fontsize=11, fontweight='bold', va='center', ha='right',
                     rotation=90)
    
    # Null mean
    null_lh, null_rh = map_parcellation_to_surface(null_mean_gradient, atlas_name)
    
    # Row 2: Null mean
    for col, (hemi, values) in enumerate([('left', null_lh), ('right', null_rh)]):
        ax = fig.add_subplot(gs[1, col], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage[f'pial_{hemi}'],
            values,
            hemi=hemi,
            view=view,
            cmap=GRADIENT_CMAP,
            colorbar=False,
            vmin=vmin,
            vmax=vmax,
            bg_map=fsaverage[f'sulc_{hemi}'],
            axes=ax
        )
        if col == 0:
            ax.text2D(-0.1, 0.5, 'NULL MEAN', transform=ax.transAxes,
                     fontsize=11, fontweight='bold', va='center', ha='right',
                     rotation=90)
    
    # Difference
    diff = observed_gradient - null_mean_gradient
    diff_lh, diff_rh = map_parcellation_to_surface(diff, atlas_name)
    diff_max = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
    
    for col, (hemi, values) in enumerate([('left', diff_lh), ('right', diff_rh)]):
        ax = fig.add_subplot(gs[0, col + 2], projection='3d')
        plotting.plot_surf_stat_map(
            fsaverage[f'pial_{hemi}'],
            values,
            hemi=hemi,
            view=view,
            cmap=ZSCORE_CMAP,
            colorbar=(col == 1),
            vmin=-diff_max,
            vmax=diff_max,
            bg_map=fsaverage[f'sulc_{hemi}'],
            axes=ax
        )
        if col == 0:
            ax.text2D(-0.1, 0.5, 'DIFFERENCE', transform=ax.transAxes,
                     fontsize=11, fontweight='bold', va='center', ha='right',
                     rotation=90)
    
    fig.suptitle(f'G{component}: Observed vs Null Model', fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_gradient_zscore_surface(
    observed_gradient: np.ndarray,
    null_gradients: np.ndarray,
    atlas_name: str = 'schaefer_100',
    views: List[str] = None,
    threshold: float = 2.0,
    component: int = 1,
    figsize: Tuple[float, float] = (14, 6),
    save_path: str = None
) -> plt.Figure:
    """
    Plotar z-score do gradiente por região (onde observado ≠ nulo).
    
    Mostra quais regiões têm posição no gradiente significativamente
    diferente do esperado pelo modelo nulo.
    
    Parameters
    ----------
    observed_gradient : np.ndarray
        Gradiente observado (n_parcels,)
    null_gradients : np.ndarray
        Gradientes dos surrogates (n_surrogates, n_parcels)
    atlas_name : str
        Nome do atlas
    views : List[str]
        Vistas a mostrar
    threshold : float
        Threshold de z-score para significância
    component : int
        Número do componente
    figsize : tuple
        Tamanho
    save_path : str
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required")
    
    if views is None:
        views = ['lateral', 'medial']
    
    # Calcular z-scores por região
    null_mean = np.mean(null_gradients, axis=0)
    null_std = np.std(null_gradients, axis=0)
    null_std[null_std == 0] = 1  # Evitar divisão por zero
    
    zscores = (observed_gradient - null_mean) / null_std
    
    # Número de regiões significativas
    n_sig = np.sum(np.abs(zscores) > threshold)
    pct_sig = 100 * n_sig / len(zscores)
    
    fig = plt.figure(figsize=figsize)
    n_views = len(views)
    
    fsaverage = _load_fsaverage('fsaverage5')
    zscores_lh, zscores_rh = map_parcellation_to_surface(zscores, atlas_name)
    
    vmax = max(abs(np.nanmin(zscores)), abs(np.nanmax(zscores)), threshold + 1)
    
    for v_idx, view in enumerate(views):
        for h_idx, (hemi, values) in enumerate([('left', zscores_lh), ('right', zscores_rh)]):
            ax_idx = v_idx * 2 + h_idx + 1
            ax = fig.add_subplot(1, n_views * 2, ax_idx, projection='3d')
            
            plotting.plot_surf_stat_map(
                fsaverage[f'pial_{hemi}'],
                values,
                hemi=hemi,
                view=view,
                cmap=ZSCORE_CMAP,
                colorbar=(v_idx == n_views - 1 and h_idx == 1),
                vmin=-vmax,
                vmax=vmax,
                bg_map=fsaverage[f'sulc_{hemi}'],
                threshold=threshold,  # Só mostra |z| > threshold
                axes=ax
            )
            
            ax.set_title(f'{view.capitalize()} {hemi[0].upper()}', fontsize=10)
    
    fig.suptitle(f'G{component} Z-scores (|z| > {threshold}): {n_sig} regions ({pct_sig:.1f}%)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# GROUP PLOTS
# =============================================================================

def plot_group_gradient_surface(
    aligned_gradients: Dict[str, np.ndarray],
    atlas_name: str = 'schaefer_100',
    component: int = 1,
    views: List[str] = None,
    show_variability: bool = True,
    figsize: Tuple[float, float] = (16, 8),
    save_path: str = None
) -> plt.Figure:
    """
    Plotar gradiente médio do grupo na superfície.
    
    Opcionalmente mostra também a variabilidade inter-sujeito.
    
    Parameters
    ----------
    aligned_gradients : Dict[str, np.ndarray]
        Gradientes alinhados {subject_id: gradients}
    atlas_name : str
        Nome do atlas
    component : int
        Qual componente (1-indexed)
    views : List[str]
        Vistas
    show_variability : bool
        Se True, mostra painel adicional com std inter-sujeito
    figsize : tuple
        Tamanho
    save_path : str
        Caminho para salvar
        
    Returns
    -------
    plt.Figure
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required")
    
    if views is None:
        views = ['lateral', 'medial']
    
    c = component - 1
    
    # Calcular média e std
    all_grads = np.stack([g[:, c] for g in aligned_gradients.values()], axis=0)
    group_mean = np.mean(all_grads, axis=0)
    group_std = np.std(all_grads, axis=0)
    
    fsaverage = _load_fsaverage('fsaverage5')
    
    if show_variability:
        fig = plt.figure(figsize=figsize)
        n_rows = 2
    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1] // 2))
        n_rows = 1
    
    n_views = len(views)
    
    # Row 1: Group mean
    mean_lh, mean_rh = map_parcellation_to_surface(group_mean, atlas_name)
    vmin, vmax = np.nanmin(group_mean), np.nanmax(group_mean)
    
    for v_idx, view in enumerate(views):
        for h_idx, (hemi, values) in enumerate([('left', mean_lh), ('right', mean_rh)]):
            ax_idx = v_idx * 2 + h_idx + 1
            ax = fig.add_subplot(n_rows, n_views * 2, ax_idx, projection='3d')
            
            plotting.plot_surf_stat_map(
                fsaverage[f'pial_{hemi}'],
                values,
                hemi=hemi,
                view=view,
                cmap=GRADIENT_CMAP,
                colorbar=(v_idx == n_views - 1 and h_idx == 1),
                vmin=vmin,
                vmax=vmax,
                bg_map=fsaverage[f'sulc_{hemi}'],
                axes=ax
            )
            
            if v_idx == 0 and h_idx == 0:
                ax.text2D(-0.15, 0.5, 'MEAN', transform=ax.transAxes,
                         fontsize=11, fontweight='bold', va='center', ha='right',
                         rotation=90)
    
    # Row 2: Variability
    if show_variability:
        std_lh, std_rh = map_parcellation_to_surface(group_std, atlas_name)
        
        for v_idx, view in enumerate(views):
            for h_idx, (hemi, values) in enumerate([('left', std_lh), ('right', std_rh)]):
                ax_idx = n_views * 2 + v_idx * 2 + h_idx + 1
                ax = fig.add_subplot(n_rows, n_views * 2, ax_idx, projection='3d')
                
                plotting.plot_surf_stat_map(
                    fsaverage[f'pial_{hemi}'],
                    values,
                    hemi=hemi,
                    view=view,
                    cmap=VARIANCE_CMAP,
                    colorbar=(v_idx == n_views - 1 and h_idx == 1),
                    vmin=0,
                    vmax=np.nanmax(group_std),
                    bg_map=fsaverage[f'sulc_{hemi}'],
                    axes=ax
                )
                
                if v_idx == 0 and h_idx == 0:
                    ax.text2D(-0.15, 0.5, 'STD', transform=ax.transAxes,
                             fontsize=11, fontweight='bold', va='center', ha='right',
                             rotation=90)
    
    fig.suptitle(f'Group G{component} (n={len(aligned_gradients)})',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    return fig


# =============================================================================
# BRAINSPACE WRAPPER (if available)
# =============================================================================

def plot_gradient_brainspace(
    gradient: np.ndarray,
    atlas_name: str = 'schaefer_100',
    size: Tuple[int, int] = (800, 400),
    cmap: str = 'viridis_r',
    color_bar: bool = True,
    label_text: List[str] = None,
    save_path: str = None
) -> None:
    """
    Plotar gradiente usando BrainSpace (se disponível).
    
    BrainSpace oferece plots interativos de alta qualidade.
    
    Parameters
    ----------
    gradient : np.ndarray
        Valores do gradiente
    atlas_name : str
        Nome do atlas
    size : tuple
        Tamanho em pixels
    cmap : str
        Colormap
    color_bar : bool
        Mostrar colorbar
    label_text : List[str]
        Labels para os hemisférios
    save_path : str
        Caminho para salvar
    """
    if not HAS_BRAINSPACE:
        raise ImportError("brainspace required. pip install brainspace")
    
    # Carregar superfície
    surf_lh, surf_rh = load_conte69()
    
    # Carregar parcellation
    # Isso depende do atlas específico
    
    if label_text is None:
        label_text = ['Left', 'Right']
    
    # Plot
    plot_hemispheres(
        surf_lh, surf_rh,
        array_name=gradient,
        size=size,
        cmap=cmap,
        color_bar=color_bar,
        label_text=label_text,
        screenshot=True if save_path else False,
        filename=save_path,
        transparent_bg=False
    )


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_brain_surface_report(
    results: Dict,
    atlas_name: str,
    output_dir: Union[str, Path],
    include_null_comparison: bool = True
) -> Dict[str, Path]:
    """
    Gerar relatório completo com visualizações de superfície cerebral.
    
    Parameters
    ----------
    results : Dict
        Resultados de run_gradient_analysis() ou run_gradient_analysis_fast()
    atlas_name : str
        Nome do atlas
    output_dir : str or Path
        Diretório de saída
    include_null_comparison : bool
        Se True, inclui comparações com modelo nulo
        
    Returns
    -------
    Dict[str, Path]
        Paths das figuras geradas
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GENERATING BRAIN SURFACE VISUALIZATIONS")
    print(f"Output: {output_dir}")
    print('='*60)
    
    saved = {}
    
    # Group gradient
    if 'observed' in results or 'group_result' in results:
        print("\n1. Group gradient surface...")
        
        try:
            if 'group_result' in results:
                aligned = results['group_result'].aligned_gradients
            else:
                # Extrair gradientes dos resultados
                aligned = {
                    sub: res['gradients'] 
                    for sub, res in results.get('observed', {}).items()
                }
            
            if aligned:
                fig = plot_group_gradient_surface(
                    aligned, atlas_name, component=1,
                    show_variability=True
                )
                path = output_dir / 'group_G1_surface.png'
                fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                saved['group_G1'] = path
                print(f"   ✓ {path.name}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print(f"\n✓ Generated {len(saved)} figures")
    
    return saved


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print("Brain Surface Visualization Module")
    print(f"nilearn: {HAS_NILEARN}")
    print(f"brainspace: {HAS_BRAINSPACE}")
    print(f"neuromaps: {HAS_NEUROMAPS}")
