"""
Visualization Tools
===================

Ferramentas de visualização para redes cerebrais e métricas.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from pathlib import Path


def plot_connectivity_matrix(
    matrix: np.ndarray,
    title: str = "Connectivity Matrix",
    cmap: str = 'RdBu_r',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    roi_labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None
):
    """
    Plota matriz de conectividade.
    
    Parameters
    ----------
    matrix : np.ndarray
        Matriz de conectividade
    title : str
        Título do plot
    cmap : str
        Colormap
    vmin, vmax : float, optional
        Limites de cor
    roi_labels : List[str], optional
        Labels dos ROIs
    save_path : Path, optional
        Caminho para salvar figura
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('ROI', fontsize=12)
    ax.set_ylabel('ROI', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Connectivity Strength', fontsize=11)
    
    # Labels se fornecidos
    if roi_labels is not None:
        ax.set_xticks(range(len(roi_labels)))
        ax.set_yticks(range(len(roi_labels)))
        ax.set_xticklabels(roi_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(roi_labels, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_degree_distribution(
    degrees: np.ndarray,
    title: str = "Degree Distribution",
    log_scale: bool = True,
    save_path: Optional[Path] = None
):
    """
    Plota distribuição de degrees.
    
    Parameters
    ----------
    degrees : np.ndarray
        Degrees dos nós
    title : str
        Título
    log_scale : bool
        Se True, usa escala log
    save_path : Path, optional
        Caminho para salvar
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(degrees, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_brain_network_3d(
    connectivity_matrix: np.ndarray,
    node_positions: np.ndarray,
    threshold: float = 0.5,
    node_size: Optional[np.ndarray] = None,
    edge_width_scale: float = 2.0,
    title: str = "Brain Network",
    save_path: Optional[Path] = None
):
    """
    Plota rede cerebral em 3D.
    
    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Matriz de conectividade
    node_positions : np.ndarray
        Posições 3D dos nós [n_nodes, 3]
    threshold : float
        Threshold para mostrar edges
    node_size : np.ndarray, optional
        Tamanho dos nós (ex: degree)
    edge_width_scale : float
        Escala da largura dos edges
    title : str
        Título
    save_path : Path, optional
        Caminho para salvar
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Filtra edges por threshold
    edges_to_plot = np.where(connectivity_matrix > threshold)
    
    # Plota edges
    for i, j in zip(edges_to_plot[0], edges_to_plot[1]):
        if i < j:  # Evita duplicatas
            weight = connectivity_matrix[i, j]
            ax.plot(
                [node_positions[i, 0], node_positions[j, 0]],
                [node_positions[i, 1], node_positions[j, 1]],
                [node_positions[i, 2], node_positions[j, 2]],
                'gray', alpha=0.3, linewidth=weight * edge_width_scale
            )
    
    # Plota nós
    if node_size is None:
        node_size = np.ones(len(node_positions)) * 50
    
    scatter = ax.scatter(
        node_positions[:, 0],
        node_positions[:, 1],
        node_positions[:, 2],
        s=node_size,
        c=node_size,
        cmap='viridis',
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )
    
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(scatter, ax=ax, label='Node Importance', shrink=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_criticality_metrics(
    avalanche_results,
    save_path: Optional[Path] = None
):
    """
    Plota métricas de criticalidade.
    
    Parameters
    ----------
    avalanche_results : AvalancheResults
        Resultados da análise de avalanches
    save_path : Path, optional
        Caminho para salvar
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribuição de tamanhos
    axes[0, 0].hist(avalanche_results.sizes, bins=40, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Avalanche Size', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title(f'Size Distribution (α={avalanche_results.size_exponent:.2f})', fontsize=12)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Distribuição de durações
    axes[0, 1].hist(avalanche_results.durations, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Avalanche Duration', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title(f'Duration Distribution (β={avalanche_results.duration_exponent:.2f})', fontsize=12)
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Shape collapse
    axes[1, 0].scatter(avalanche_results.durations, avalanche_results.sizes, alpha=0.5)
    axes[1, 0].set_xlabel('Duration', fontsize=11)
    axes[1, 0].set_ylabel('Size', fontsize=11)
    axes[1, 0].set_title(f'Shape Collapse (κ={avalanche_results.kappa:.2f})', fontsize=12)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Métricas summary
    metrics_text = f"""
    Branching Ratio: {avalanche_results.branching_ratio:.3f}
    Size Exponent: {avalanche_results.size_exponent:.3f}
    Duration Exponent: {avalanche_results.duration_exponent:.3f}
    Shape Collapse κ: {avalanche_results.kappa:.3f}
    
    N Avalanches: {len(avalanche_results.sizes)}
    
    Critical if:
    • Branching ≈ 1.0
    • Size exp ≈ -1.5
    • Duration exp ≈ -2.0
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    axes[1, 1].axis('off')
    
    plt.suptitle('Criticality Analysis - Neuronal Avalanches', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_multilayer_comparison(
    sc_matrix: np.ndarray,
    fc_matrix: np.ndarray,
    sc_fc_coupling: np.ndarray,
    save_path: Optional[Path] = None
):
    """
    Plota comparação multilayer (SC vs FC).
    
    Parameters
    ----------
    sc_matrix : np.ndarray
        Conectividade estrutural
    fc_matrix : np.ndarray
        Conectividade funcional
    sc_fc_coupling : np.ndarray
        Acoplamento SC-FC por nó
    save_path : Path, optional
        Caminho para salvar
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # SC matrix
    im1 = axes[0, 0].imshow(sc_matrix, cmap='Reds', aspect='auto')
    axes[0, 0].set_title('Structural Connectivity', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('ROI')
    axes[0, 0].set_ylabel('ROI')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # FC matrix
    im2 = axes[0, 1].imshow(fc_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[0, 1].set_title('Functional Connectivity', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('ROI')
    axes[0, 1].set_ylabel('ROI')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # SC-FC coupling
    axes[1, 0].plot(sc_fc_coupling, linewidth=2, color='purple')
    axes[1, 0].set_xlabel('ROI Index', fontsize=11)
    axes[1, 0].set_ylabel('SC-FC Coupling', fontsize=11)
    axes[1, 0].set_title('Structure-Function Coupling per Node', fontsize=12, fontweight='bold')
    axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].grid(alpha=0.3)
    
    # Scatter SC vs FC
    triu_idx = np.triu_indices(sc_matrix.shape[0], k=1)
    sc_values = sc_matrix[triu_idx]
    fc_values = fc_matrix[triu_idx]
    
    axes[1, 1].scatter(sc_values, fc_values, alpha=0.3, s=5)
    axes[1, 1].set_xlabel('Structural Connectivity', fontsize=11)
    axes[1, 1].set_ylabel('Functional Connectivity', fontsize=11)
    axes[1, 1].set_title('SC-FC Relationship', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    # Correlation
    from scipy.stats import pearsonr
    corr, pval = pearsonr(sc_values, fc_values)
    axes[1, 1].text(0.05, 0.95, f'r = {corr:.3f}\np = {pval:.2e}',
                    transform=axes[1, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Multilayer Network Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_covid_metrics(
    covid_metrics,
    roi_labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None
):
    """
    Plota métricas COVID-19.
    
    Parameters
    ----------
    covid_metrics : COVIDMetrics
        Resultados da análise COVID
    roi_labels : List[str], optional
        Labels dos ROIs
    save_path : Path, optional
        Caminho para salvar
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ALFF
    axes[0, 0].plot(covid_metrics.alff, linewidth=2, color='blue', marker='o', markersize=2)
    axes[0, 0].set_xlabel('ROI Index', fontsize=11)
    axes[0, 0].set_ylabel('ALFF', fontsize=11)
    axes[0, 0].set_title(f'ALFF (mean={np.mean(covid_metrics.alff):.3f})', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # fALFF
    axes[0, 1].plot(covid_metrics.falff, linewidth=2, color='green', marker='o', markersize=2)
    axes[0, 1].set_xlabel('ROI Index', fontsize=11)
    axes[0, 1].set_ylabel('fALFF', fontsize=11)
    axes[0, 1].set_title(f'fALFF (mean={np.mean(covid_metrics.falff):.3f})', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # ReHo
    axes[1, 0].plot(covid_metrics.reho, linewidth=2, color='red', marker='o', markersize=2)
    axes[1, 0].set_xlabel('ROI Index', fontsize=11)
    axes[1, 0].set_ylabel('ReHo', fontsize=11)
    axes[1, 0].set_title(f'ReHo (mean={np.mean(covid_metrics.reho):.3f})', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Altered networks
    if covid_metrics.altered_networks:
        networks = list(covid_metrics.altered_networks.keys())
        statuses = list(covid_metrics.altered_networks.values())
        
        colors = ['red' if s == 'hypo' else 'blue' if s == 'hyper' else 'gray' for s in statuses]
        
        axes[1, 1].barh(networks, [1 if s == 'hyper' else -1 if s == 'hypo' else 0 for s in statuses],
                        color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Connectivity Change', fontsize=11)
        axes[1, 1].set_title('Altered Networks', fontsize=12, fontweight='bold')
        axes[1, 1].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].grid(alpha=0.3, axis='x')
    else:
        axes[1, 1].text(0.5, 0.5, 'No control data\nfor comparison',
                        ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')
    
    plt.suptitle('COVID-19 Neuroimaging Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_analysis_dashboard(
    results_dict: Dict,
    output_dir: Path
):
    """
    Cria dashboard completo de análise.
    
    Parameters
    ----------
    results_dict : Dict
        Dicionário com todos os resultados
    output_dir : Path
        Diretório para salvar figuras
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Gerando dashboard de visualizações...")
    
    # Plota cada tipo de resultado
    if 'avalanche_results' in results_dict:
        plot_criticality_metrics(
            results_dict['avalanche_results'],
            save_path=output_dir / 'criticality_metrics.png'
        )
        print("  ✓ criticality_metrics.png")
    
    if 'sc_matrix' in results_dict and 'fc_matrix' in results_dict:
        plot_multilayer_comparison(
            results_dict['sc_matrix'],
            results_dict['fc_matrix'],
            results_dict.get('sc_fc_coupling', np.zeros(100)),
            save_path=output_dir / 'multilayer_comparison.png'
        )
        print("  ✓ multilayer_comparison.png")
    
    if 'covid_metrics' in results_dict:
        plot_covid_metrics(
            results_dict['covid_metrics'],
            save_path=output_dir / 'covid_metrics.png'
        )
        print("  ✓ covid_metrics.png")
    
    print(f"\nDashboard salvo em: {output_dir}")
