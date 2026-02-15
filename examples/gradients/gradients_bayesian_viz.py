"""
Publication-quality figures for COVID-19 ICU Gradient Analysis
─────────────────────────────────────────────────────────────
Figure 1: Forest plot — Network-level Bayesian effects (HDI 94%)
Figure 2: Brain surface — Beta values projected on cortex
Figure 3: Combined panel figure (Figure 1 + Figure 2)
"""

# SSL bypass for nilearn downloads
import requests
import urllib3
urllib3.disable_warnings()
old_init = requests.Session.__init__
def _patched_init(self, *args, **kwargs):
    old_init(self, *args, **kwargs)
    self.verify = False
requests.Session.__init__ = _patched_init

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patheffects as pe
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_surf_fsaverage
from nilearn import plotting, surface
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURAÇÃO VISUAL (PAPER-READY)
# =====================================================================
# Paleta sofisticada — azul-coral com tons neutros
COLORS = {
    'bg':           '#FAFBFD',
    'text':         '#1A1A2E',
    'text_light':   '#6B7280',
    'grid':         '#E5E7EB',
    'accent_pos':   '#E85D4A',   # coral — efeito positivo
    'accent_neg':   '#3B82C4',   # azul steel — efeito negativo  
    'accent_null':  '#9CA3AF',   # cinza — não credível
    'ci_pos':       '#FECACA',   # fundo CI positivo
    'ci_neg':       '#BFDBFE',   # fundo CI negativo
    'zero_line':    '#374151',
}

# Cores por rede Yeo 7 (padrão da literatura)
YEO_COLORS = {
    'Vis':          '#781286',
    'SomMot':       '#4682B4',
    'DorsAttn':     '#00760E',
    'SalVentAttn':  '#C43AFA',
    'Limbic':       '#DCF8A4',
    'Cont':         '#E69422',
    'Default':      '#CD3E4E',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#D1D5DB',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# =====================================================================
# DADOS DOS RESULTADOS
# =====================================================================
networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
network_labels_full = [
    'Visual', 'Somatomotor', 'Dorsal\nAttention', 'Salience/\nVent. Attn',
    'Limbic', 'Frontoparietal\nControl', 'Default\nMode'
]

means  = np.array([0.540, -0.075,  0.093, -0.184, -0.023, -0.038,  0.069])
hdi_lo = np.array([0.431, -0.196, -0.069, -0.355, -0.338, -0.237, -0.067])
hdi_hi = np.array([0.640,  0.055,  0.256, -0.020,  0.282,  0.165,  0.195])
sds    = np.array([0.056,  0.065,  0.087,  0.087,  0.165,  0.104,  0.070])

credible = (hdi_lo > 0) | (hdi_hi < 0)

# =====================================================================
# FIGURA 1: FOREST PLOT
# =====================================================================
def make_forest_plot(save_path):
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    n = len(networks)
    y_pos = np.arange(n)[::-1]  # de baixo pra cima

    # Fundo alternado sutil
    for i, yp in enumerate(y_pos):
        if i % 2 == 0:
            ax.axhspan(yp - 0.4, yp + 0.4, color='#F3F4F6', alpha=0.5, zorder=0)

    # Linha de zero (referência)
    ax.axvline(x=0, color=COLORS['zero_line'], linewidth=1.2,
               linestyle='-', alpha=0.6, zorder=1)
    ax.text(0.005, y_pos[0] + 0.65, 'null', fontsize=7, color=COLORS['text_light'],
            ha='center', style='italic')

    # Plotar cada rede
    for i, (net, label, m, lo, hi, cred) in enumerate(
        zip(networks, network_labels_full, means, hdi_lo, hdi_hi, credible)
    ):
        yp = y_pos[i]
        yeo_color = YEO_COLORS[net]

        if cred:
            bar_color = COLORS['accent_pos'] if m > 0 else COLORS['accent_neg']
            ci_color = COLORS['ci_pos'] if m > 0 else COLORS['ci_neg']
            dot_size = 110
            alpha_bar = 0.95
            alpha_ci = 0.35
            lw = 2.5
        else:
            bar_color = COLORS['accent_null']
            ci_color = '#F3F4F6'
            dot_size = 70
            alpha_bar = 0.5
            alpha_ci = 0.2
            lw = 1.5

        # HDI bar (fundo colorido)
        fancy_box = FancyBboxPatch(
            (lo, yp - 0.15), hi - lo, 0.30,
            boxstyle="round,pad=0.02",
            facecolor=ci_color, edgecolor='none',
            alpha=alpha_ci, zorder=2
        )
        ax.add_patch(fancy_box)

        # HDI line
        ax.hlines(yp, lo, hi, color=bar_color, linewidth=lw,
                  alpha=alpha_bar, zorder=3, capstyle='round')

        # Ponto central
        ax.scatter(m, yp, s=dot_size, color=bar_color, edgecolors='white',
                   linewidths=1.2, zorder=4, alpha=alpha_bar)

        # Bolinha da cor Yeo como badge no label
        ax.scatter(-0.62, yp, s=100, color=yeo_color, edgecolors='white',
                   linewidths=0.8, zorder=5, marker='o')

        # Label da rede
        ax.text(-0.57, yp, label, fontsize=9, va='center', ha='left',
                color=COLORS['text'], fontweight='bold' if cred else 'normal',
                zorder=5)

        # Valor numérico à direita
        star = ' ✱' if cred else ''
        ax.text(hi + 0.035, yp, f'β = {m:+.3f}{star}',
                fontsize=8, va='center', ha='left',
                color=bar_color if cred else COLORS['text_light'],
                fontweight='bold' if cred else 'normal')

    # Eixos
    ax.set_xlim(-0.75, 0.85)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_yticks([])
    ax.set_xlabel('Posterior mean β  (94% HDI)', fontsize=10,
                  color=COLORS['text'], labelpad=10)

    ax.tick_params(axis='x', colors=COLORS['text_light'], labelsize=9)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Título
    ax.text(-0.75, n + 0.1,
            'Network-Specific Structure–Function Coupling\nFollowing ICU Admission for COVID-19',
            fontsize=12, fontweight='bold', color=COLORS['text'],
            va='bottom', ha='left', linespacing=1.4)

    # Legenda
    ax.text(-0.75, -1.05,
            '✱ Credible effect: 94% HDI excludes zero  |  '
            'Positive β: gradient deviation → decoupling  |  '
            'Negative β: gradient deviation → re-coupling',
            fontsize=6.5, color=COLORS['text_light'], va='top', ha='left',
            style='italic')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"✓ Forest plot saved: {save_path}")


# =====================================================================
# FIGURA 2: BRAIN SURFACE MAPS
# =====================================================================
def make_brain_surface(save_path):
    """Project network-level betas onto cortical surface."""
    # Fetch atlas and surface
    atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
    fsaverage = fetch_surf_fsaverage()

    parcel_names = [
        label.decode() if isinstance(label, bytes) else label
        for label in atlas.labels
    ]
    # Skip first label (background = 0)
    parcel_names = [p for p in parcel_names if '7Networks' in p]

    # Map each parcel to its network beta
    network_map = {
        'Vis': 0.540, 'SomMot': -0.075, 'DorsAttn': 0.093,
        'SalVentAttn': -0.184, 'Limbic': -0.023, 'Cont': -0.038,
        'Default': 0.069
    }

    parcel_betas = np.zeros(100)
    for i, name in enumerate(parcel_names):
        for net_name, beta_val in network_map.items():
            if net_name in name:
                parcel_betas[i] = beta_val
                break

    # Map parcels to surface vertices
    atlas_img = atlas.maps
    
    # Create volumetric beta map
    import nibabel as nib
    atlas_nib = nib.load(atlas_img)
    atlas_data = atlas_nib.get_fdata()
    beta_vol = np.zeros_like(atlas_data, dtype=float)
    for i in range(100):
        beta_vol[atlas_data == (i + 1)] = parcel_betas[i]

    beta_img = nib.Nifti1Image(beta_vol, atlas_nib.affine, atlas_nib.header)

    # Project to surface
    texture_lh = surface.vol_to_surf(beta_img, fsaverage.pial_left,
                                      interpolation='nearest_most_frequent', radius=3)
    texture_rh = surface.vol_to_surf(beta_img, fsaverage.pial_right,
                                      interpolation='nearest_most_frequent', radius=3)

    # Custom diverging colormap (blue → white → coral)
    colors_cmap = ['#1E5A8C', '#3B82C4', '#89B8E0', '#D4E5F5',
                   '#FAFBFD',
                   '#FDE5E0', '#F5A898', '#E85D4A', '#B83A2E']
    cmap = LinearSegmentedColormap.from_list('blue_coral', colors_cmap, N=256)

    vmax = 0.60
    vmin = -0.40

    # Create figure with 4 views
    fig, axes = plt.subplots(2, 2, figsize=(10, 7),
                              subplot_kw={'projection': '3d'},
                              facecolor=COLORS['bg'])
    fig.subplots_adjust(wspace=-0.02, hspace=0.02)

    views = [
        (axes[0, 0], fsaverage.infl_left,  texture_lh, 'lateral',  'Left Lateral'),
        (axes[0, 1], fsaverage.infl_right, texture_rh, 'lateral',  'Right Lateral'),
        (axes[1, 0], fsaverage.infl_left,  texture_lh, 'medial',   'Left Medial'),
        (axes[1, 1], fsaverage.infl_right, texture_rh, 'medial',   'Right Medial'),
    ]

    for ax, surf_mesh, texture, view, title in views:
        plotting.plot_surf_stat_map(
            surf_mesh, texture,
            hemi='left' if 'left' in str(surf_mesh).lower() or 'Left' in title else 'right',
            view=view,
            cmap=cmap, vmax=vmax,
            symmetric_cbar=False,
            threshold=0.01,
            bg_map=fsaverage.sulc_left if 'Left' in title else fsaverage.sulc_right,
            axes=ax,
            colorbar=False,
            bg_on_data=True,
        )
        ax.set_title(title, fontsize=9, color=COLORS['text'], pad=-5, fontweight='medium')

    # Title
    fig.suptitle(
        'Network-Level β Coefficients Projected on Cortical Surface',
        fontsize=13, fontweight='bold', color=COLORS['text'],
        y=0.98
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.30, 0.02, 0.40, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('β (Structure–Function Coupling)', fontsize=9,
                   color=COLORS['text'], labelpad=5)
    cbar.ax.tick_params(labelsize=8, colors=COLORS['text_light'])
    cbar.outline.set_linewidth(0.5)

    fig.savefig(save_path, dpi=300, facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"✓ Brain surface saved: {save_path}")


# =====================================================================
# FIGURA 3: COMBINED PANEL
# =====================================================================
def make_combined_figure(forest_path, brain_path, save_path):
    """Combine forest + brain into a single 2-panel figure."""
    from PIL import Image

    forest_img = Image.open(forest_path)
    brain_img = Image.open(brain_path)

    # Target widths (same)
    target_w = max(forest_img.width, brain_img.width)

    # Scale both to same width
    f_ratio = target_w / forest_img.width
    b_ratio = target_w / brain_img.width
    forest_resized = forest_img.resize(
        (target_w, int(forest_img.height * f_ratio)), Image.LANCZOS
    )
    brain_resized = brain_img.resize(
        (target_w, int(brain_img.height * b_ratio)), Image.LANCZOS
    )

    # Vertical stack with panel labels
    total_h = forest_resized.height + brain_resized.height + 40
    combined = Image.new('RGB', (target_w, total_h), color=(250, 251, 253))

    combined.paste(forest_resized, (0, 0))
    combined.paste(brain_resized, (0, forest_resized.height + 40))

    # Add panel labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)
    except:
        font = ImageFont.load_default()

    label_color = (26, 26, 46)
    draw.text((25, 10), "A", fill=label_color, font=font)
    draw.text((25, forest_resized.height + 45), "B", fill=label_color, font=font)

    combined.save(save_path, dpi=(300, 300))
    print(f"✓ Combined figure saved: {save_path}")


# =====================================================================
# FIGURA 4: RADAR/SPIDER PLOT (bonus)
# =====================================================================
def make_radar_plot(save_path):
    """Radar plot showing effect magnitude and direction per network."""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True),
                            facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    n = len(networks)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    values = means.tolist() + [means[0]]
    hi_vals = hdi_hi.tolist() + [hdi_hi[0]]
    lo_vals = hdi_lo.tolist() + [hdi_lo[0]]

    # Fill HDI region
    ax.fill_between(angles, lo_vals, hi_vals, alpha=0.15, color='#6366F1')

    # Main line
    ax.plot(angles, values, 'o-', linewidth=2.2, color='#6366F1',
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            markeredgecolor='#6366F1', zorder=5)

    # Highlight credible networks
    for i, (net, cred) in enumerate(zip(networks, credible)):
        if cred:
            color = COLORS['accent_pos'] if means[i] > 0 else COLORS['accent_neg']
            ax.plot(angles[i], values[i], 'o', markersize=12,
                    markerfacecolor=color, markeredgecolor='white',
                    markeredgewidth=2, zorder=6)

    # Zero circle
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_circle, [0] * 100, '--', color=COLORS['zero_line'],
            linewidth=0.8, alpha=0.5)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(network_labels_full, fontsize=8.5, color=COLORS['text'],
                        fontweight='medium')

    # Radial ticks
    ax.set_yticks([-0.3, 0, 0.3, 0.6])
    ax.set_yticklabels(['-0.3', '0', '0.3', '0.6'],
                        fontsize=7, color=COLORS['text_light'])
    ax.set_ylim(-0.5, 0.75)

    # Grid styling
    ax.xaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.yaxis.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.spines['polar'].set_color(COLORS['grid'])

    ax.set_title(
        'Network-Specific Effects: Gradient Deviation → S-F Coupling',
        fontsize=11, fontweight='bold', color=COLORS['text'],
        pad=25
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['accent_pos'],
               markersize=10, markeredgecolor='white', markeredgewidth=1.5,
               label='Credible positive (decoupling)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['accent_neg'],
               markersize=10, markeredgecolor='white', markeredgewidth=1.5,
               label='Credible negative (re-coupling)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#6366F1',
               markersize=8, markeredgecolor='white', markeredgewidth=1.5,
               label='Non-credible'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              bbox_to_anchor=(1.35, -0.05), fontsize=7.5,
              frameon=True, fancybox=True,
              edgecolor=COLORS['grid'], facecolor=COLORS['bg'])

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"✓ Radar plot saved: {save_path}")


# =====================================================================
# EXECUTAR
# =====================================================================
if __name__ == '__main__':
    import os
    out = '/mnt/nvme1n1p1/sars_cov_2_project/figs'
    os.makedirs(out, exist_ok=True)

    print("Generating publication figures...\n")

    make_forest_plot(f'{out}/fig1_forest_plot.png')
    make_brain_surface(f'{out}/fig2_brain_surface.png')
    make_radar_plot(f'{out}/fig4_radar_plot.png')
    make_combined_figure(
        f'{out}/fig1_forest_plot.png',
        f'{out}/fig2_brain_surface.png',
        f'{out}/fig3_combined_panel.png'
    )

    print("\n✓ All figures generated!")
