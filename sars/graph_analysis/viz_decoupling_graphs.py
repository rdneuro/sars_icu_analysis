"""
Visualization functions for SC-FC decoupling analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List, Tuple


def plot_sc_fc_decoupling_summary(
    results,
    node_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
):
    """
    Create comprehensive visualization of SC-FC decoupling results.
    
    Parameters
    ----------
    results : SCFCDecouplingResults
        Results from sc_fc_decoupling_analysis
    node_labels : List[str], optional
        Node labels from atlas
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Define grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # =========================================================================
    # 1. Global SC-FC Coupling Bar Plot
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    metrics = list(results.global_metric_couplings.keys())
    coupling_vals = [results.global_metric_couplings[m] for m in metrics]
    
    colors = plt.cm.RdYlGn(np.array(coupling_vals) * 0.5 + 0.5)
    bars = ax1.barh(metrics, coupling_vals, color=colors, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('SC-FC Correlation (r)', fontsize=10)
    ax1.set_title(f'Metric-wise SC-FC Coupling\nGlobal r = {results.global_coupling:.3f}', 
                  fontsize=11, fontweight='bold')
    ax1.set_xlim(-1, 1)
    
    # =========================================================================
    # 2. Efficiency Ratio Distribution
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    if 'local_efficiency' in results.efficiency_ratios:
        ratio_data = results.efficiency_ratios['local_efficiency']
        ratio_data_clean = ratio_data[np.isfinite(ratio_data)]
        
        ax2.hist(ratio_data_clean, bins=30, color='steelblue', 
                 edgecolor='white', alpha=0.7)
        ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, label='No decoupling')
        ax2.axvline(x=np.median(ratio_data_clean), color='orange', 
                    linestyle='-', linewidth=2, label=f'Median = {np.median(ratio_data_clean):.2f}')
        ax2.set_xlabel('FC/SC Local Efficiency Ratio', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Local Efficiency Ratio Distribution', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
    
    # =========================================================================
    # 3. Nodal Decoupling Heatmap
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    n_nodes = len(results.nodal_decoupling)
    side = int(np.ceil(np.sqrt(n_nodes)))
    
    # Pad to make square
    decoupling_padded = np.full(side * side, np.nan)
    decoupling_padded[:n_nodes] = results.nodal_decoupling
    decoupling_2d = decoupling_padded.reshape(side, side)
    
    im = ax3.imshow(decoupling_2d, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax3.set_title('Nodal Decoupling Index', fontsize=11, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label='Decoupling')
    
    # =========================================================================
    # 4. SC vs FC Strength Scatter
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    if 'strength' in results.sc_metrics and 'strength' in results.fc_metrics:
        sc_str = results.sc_metrics['strength']
        fc_str = results.fc_metrics['strength']
        
        scatter = ax4.scatter(sc_str, fc_str, c=results.nodal_decoupling,
                             cmap='RdYlGn_r', alpha=0.7, edgecolors='gray', linewidth=0.5)
        
        # Regression line
        z = np.polyfit(sc_str, fc_str, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sc_str.min(), sc_str.max(), 100)
        ax4.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)
        
        ax4.set_xlabel('SC Strength', fontsize=10)
        ax4.set_ylabel('FC Strength', fontsize=10)
        ax4.set_title('SC vs FC Nodal Strength', fontsize=11, fontweight='bold')
        plt.colorbar(scatter, ax=ax4, label='Decoupling')
    
    # =========================================================================
    # 5. Hub vs Non-Hub Decoupling
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    if results.hub_decoupling is not None:
        categories = ['Hubs', 'Non-Hubs']
        decoupling_means = [
            results.hub_decoupling['hub_mean_decoupling'],
            results.hub_decoupling['nonhub_mean_decoupling']
        ]
        
        bars = ax5.bar(categories, decoupling_means, 
                       color=['indianred', 'steelblue'], edgecolor='black')
        ax5.set_ylabel('Mean Decoupling Index', fontsize=10)
        ax5.set_title(f"Hub vs Non-Hub Decoupling\n(n_hubs = {results.hub_decoupling['n_hubs']})",
                      fontsize=11, fontweight='bold')
        ax5.set_ylim(0, max(decoupling_means) * 1.2)
        
        # Add values on bars
        for bar, val in zip(bars, decoupling_means):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # =========================================================================
    # 6. Global Metric Comparison
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    if results.metric_differences:
        metrics_to_plot = ['global_efficiency', 'local_efficiency', 
                          'clustering_coefficient', 'modularity']
        metrics_available = [m for m in metrics_to_plot if m in results.metric_differences]
        
        x = np.arange(len(metrics_available))
        width = 0.35
        
        sc_vals = [results.metric_differences[m]['SC'] for m in metrics_available]
        fc_vals = [results.metric_differences[m]['FC'] for m in metrics_available]
        
        ax6.bar(x - width/2, sc_vals, width, label='SC', color='steelblue', edgecolor='black')
        ax6.bar(x + width/2, fc_vals, width, label='FC', color='coral', edgecolor='black')
        
        ax6.set_xticks(x)
        ax6.set_xticklabels([m.replace('_', '\n') for m in metrics_available], fontsize=9)
        ax6.set_ylabel('Value', fontsize=10)
        ax6.set_title('Global Metric Comparison', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=9)
    
    # =========================================================================
    # 7. Top Decoupled Regions (if labels available)
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, :2])
    
    sorted_idx = np.argsort(results.nodal_decoupling)[::-1]
    top_n = min(20, len(sorted_idx))
    
    top_indices = sorted_idx[:top_n]
    top_decoupling = results.nodal_decoupling[top_indices]
    
    if node_labels is not None:
        top_labels = [node_labels[i] if i < len(node_labels) else f'ROI_{i}' 
                      for i in top_indices]
    else:
        top_labels = [f'ROI_{i+1}' for i in top_indices]
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, top_n))
    ax7.barh(range(top_n), top_decoupling[::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)
    ax7.set_yticks(range(top_n))
    ax7.set_yticklabels(top_labels[::-1], fontsize=8)
    ax7.set_xlabel('Decoupling Index', fontsize=10)
    ax7.set_title('Top 20 Decoupled Regions', fontsize=11, fontweight='bold')
    ax7.set_xlim(0, max(top_decoupling) * 1.1)
    
    # =========================================================================
    # 8. Text Summary
    # =========================================================================
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary_text = f"""
    SC-FC DECOUPLING SUMMARY
    ══════════════════════════════
    
    Global Coupling:
      r = {results.global_coupling:.4f}
      p = {results.global_coupling_pvalue:.2e}
    
    Metric Couplings:
    """
    
    for metric, r in results.global_metric_couplings.items():
        summary_text += f"\n      {metric}: r = {r:.3f}"
    
    if results.hub_decoupling:
        summary_text += f"""
    
    Hub Analysis:
      N hubs = {results.hub_decoupling['n_hubs']}
      Hub decoupling = {results.hub_decoupling['hub_mean_decoupling']:.3f}
      Non-hub decoupling = {results.hub_decoupling['nonhub_mean_decoupling']:.3f}
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Structure-Function Decoupling Analysis', fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    plt.tight_layout()
    return fig
