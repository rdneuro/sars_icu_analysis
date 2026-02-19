#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  CONN2RES — BRAIN SURFACE PLOTS (nilearn 0.10.4 compatible)
================================================================================

  Compatível com: nilearn 0.10.4 + Python 3.8.12
  display_mode:   'ortho', 'tiled', 'mosaic', 'x', 'y', 'z', 'xz', 'yz'
                  ⚠️ 'lyrz' NÃO existe no 0.10.4 (requer 0.11+)

  Execução: Rodar APÓS o tutorial v2 (precisa das variáveis em memória)
            OU como standalone (carrega os dados do zero).

  Output:  /mnt/nvme1n1p1/sars_cov_2_project/figs/conn2res_v2/
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from nilearn import plotting as ni_plot
from nilearn import datasets as ni_data
from nilearn import image as ni_image

# %% ========================================================================
#  CONFIG
# ===========================================================================

DATA_ROOT  = "/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs"
INFO_DIR   = os.path.join(DATA_ROOT, "info")
OUTPUT_DIR = "/mnt/nvme1n1p1/sars_cov_2_project/figs/conn2res_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATLAS   = "schaefer100"
N_NODES = 100
N_TRS   = 390
SEED    = 42

ALL_SUBS = [f"sub-{n:02d}" for n in range(1, 25)]
ALL_SUBS.remove("sub-21")

# Yeo7 palette
YEO7_DISPLAY = {
    'Cont': 'FP', 'Default': 'DMN', 'DorsAttn': 'DA',
    'Limbic': 'Lim', 'SalVentAttn': 'VA', 'SomMot': 'SM', 'Vis': 'Vis'
}
YEO7_COLORS = {
    'FP': '#E69422', 'DMN': '#CD3E4E', 'DA': '#00760E',
    'Lim': '#DCF8A4', 'VA': '#C43AFA', 'SM': '#4682B4', 'Vis': '#781286',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


# %% ========================================================================
#  LOAD DATA (standalone — se as variáveis não estiverem em memória)
# ===========================================================================

def load_atlas_info():
    roi_labels = np.loadtxt(os.path.join(INFO_DIR, f"labels_{ATLAS}.txt"), dtype=str)
    nets_raw = np.loadtxt(os.path.join(INFO_DIR, "networks_schaefer100.txt"), dtype=str)
    le = LabelEncoder()
    net_int = le.fit_transform(nets_raw)
    int2short = {}
    for i, name in enumerate(le.classes_):
        int2short[i] = YEO7_DISPLAY.get(name, name[:3])
    return roi_labels, net_int, int2short


def load_subject(sub, sc_treat='raw'):
    base = os.path.join(DATA_ROOT, sub, ATLAS)
    SC = np.load(os.path.join(base, "dmri", "connectivity_sift2.npy"))
    FC = np.load(os.path.join(base, "fmri", "connectivity_correlation.npy"))
    TS = np.load(os.path.join(base, "fmri", "timeseries.npy"))
    np.fill_diagonal(SC, 0); np.fill_diagonal(FC, 0)
    SC = (SC + SC.T) / 2; FC = (FC + FC.T) / 2
    for mtx in [SC, FC]:
        if np.any(~np.isfinite(mtx)):
            mtx[:] = SimpleImputer(strategy='median').fit_transform(mtx)
    if sc_treat == 'log':
        SC = np.log1p(SC)
    return SC, FC, TS


def compute_coupling_single_simple(SC, FC, labels, alpha=1.0, n_runs=10,
                                    washout=200, noise_std=0.1, seed=42,
                                    int2short=None):
    """Simplified coupling computation returning key metrics."""
    from conn2res.connectivity import Conn
    from conn2res.reservoir import EchoStateNetwork

    rng_loc = np.random.default_rng(seed)
    conn = Conn(w=SC.copy())
    conn.scale_and_normalize()
    n_act = conn.n_nodes
    idx_act = np.where(conn.idx_node)[0]
    FC_act = FC[np.ix_(idx_act, idx_act)]
    lab_act = labels[idx_act]

    w_in = np.eye(n_act)
    FC_rc_accum = np.zeros((n_act, n_act))
    for _ in range(n_runs):
        ext = rng_loc.standard_normal((N_TRS, n_act)) * noise_std
        esn = EchoStateNetwork(w=alpha * conn.w, activation_function='tanh')
        rs = esn.simulate(ext_input=ext, w_in=w_in, output_nodes=None)
        FC_rc_accum += np.corrcoef(rs[washout:].T)
    FC_pred = FC_rc_accum / n_runs

    # Nodal metrics
    nodal_mae = np.array([np.mean(np.abs(FC_act[i, :] - FC_pred[i, :]))
                          for i in range(n_act)])
    nodal_corr = np.array([
        stats.pearsonr(FC_pred[i, :], FC_act[i, :])[0]
        if np.std(FC_pred[i, :]) > 1e-10 else 0.0
        for i in range(n_act)
    ])

    # Intra-network
    short = int2short or {}
    intra = {}
    for mod in np.unique(lab_act):
        nodes = np.where(lab_act == mod)[0]
        if len(nodes) < 3: continue
        triu_sub = np.triu_indices(len(nodes), k=1)
        if len(triu_sub[0]) > 2:
            r, _ = stats.pearsonr(FC_pred[np.ix_(nodes, nodes)][triu_sub],
                                  FC_act[np.ix_(nodes, nodes)][triu_sub])
            intra[short.get(mod, f"M{mod}")] = r

    return {
        'FC_pred': FC_pred, 'FC_emp': FC_act,
        'active_labels': lab_act, 'idx_active': idx_act,
        'nodal_mae': nodal_mae, 'nodal_corr': nodal_corr,
        'intra': intra,
    }


# --- Load everything ---
print("Loading data...")
ROI_LABELS, YEO7_INT, INT2SHORT = load_atlas_info()
YEO7_NAMES = [INT2SHORT[i] for i in range(7)]
YEO7_COLORS_ORD = [YEO7_COLORS[INT2SHORT[i]] for i in range(7)]

# Fetch atlas and template
print("Fetching Schaefer-100 atlas and MNI template...")
schaefer = ni_data.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
atlas_img = schaefer['maps']
atlas_data = ni_image.get_data(atlas_img)
mni_bg = ni_data.load_mni152_template(resolution=2)
coords = ni_plot.find_parcellation_cut_coords(atlas_img)


def roi_to_volume(values):
    """Map (N_NODES,) vector → NIfTI volume using Schaefer parcellation."""
    vol = np.zeros_like(atlas_data, dtype=float)
    for ri in range(min(len(values), N_NODES)):
        vol[atlas_data == (ri + 1)] = values[ri]
    return ni_image.new_img_like(atlas_img, vol)


def savefig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'))
    plt.close(fig)
    print(f"  ✅ {name}")


# %% ========================================================================
#  COMPUTE GROUP METRICS (or load from existing .npy if available)
# ===========================================================================

npy_mae  = os.path.join(OUTPUT_DIR, 'group_nodal_mae.npy')
npy_corr = os.path.join(OUTPUT_DIR, 'group_nodal_corr.npy')

if os.path.exists(npy_mae) and os.path.exists(npy_corr):
    print("Loading pre-computed group nodal metrics...")
    group_nodal_mae  = np.load(npy_mae)
    group_nodal_corr = np.load(npy_corr)
else:
    print("Computing group coupling (this takes a few minutes)...")
    group_nodal_mae  = np.zeros((len(ALL_SUBS), N_NODES))
    group_nodal_corr = np.zeros((len(ALL_SUBS), N_NODES))

    for i, sub in enumerate(ALL_SUBS):
        try:
            sc_s, fc_s, _ = load_subject(sub, sc_treat='raw')
            res = compute_coupling_single_simple(
                sc_s, fc_s, YEO7_INT, seed=SEED + i, int2short=INT2SHORT
            )
            idx = res['idx_active']
            group_nodal_mae[i, idx]  = res['nodal_mae']
            group_nodal_corr[i, idx] = res['nodal_corr']
            print(f"    {sub} done")
        except Exception as e:
            print(f"    {sub} ⚠️ {e}")

    np.save(npy_mae, group_nodal_mae)
    np.save(npy_corr, group_nodal_corr)

# Group statistics
mean_nodal_mae  = np.mean(group_nodal_mae, axis=0)
std_nodal_mae   = np.std(group_nodal_mae, axis=0)
mean_nodal_corr = np.mean(group_nodal_corr, axis=0)
std_nodal_corr  = np.std(group_nodal_corr, axis=0)
cv_nodal_mae    = std_nodal_mae / np.maximum(mean_nodal_mae, 1e-8)

# Nodal t-test against zero
nodal_t = np.zeros(N_NODES)
nodal_p = np.ones(N_NODES)
for node in range(N_NODES):
    vals = group_nodal_corr[:, node]
    vals = vals[vals != 0]
    if len(vals) > 3:
        nodal_t[node], nodal_p[node] = stats.ttest_1samp(vals, 0)

# FDR correction
try:
    from statsmodels.stats.multitest import multipletests
    reject, nodal_p_fdr, _, _ = multipletests(nodal_p, alpha=0.05, method='fdr_bh')
except ImportError:
    # Manual BH procedure if statsmodels not available
    print("  ⚠️ statsmodels not installed — using manual BH correction")
    sorted_p = np.sort(nodal_p)
    n_tests = len(nodal_p)
    thresholds = 0.05 * np.arange(1, n_tests + 1) / n_tests
    try:
        max_idx = np.max(np.where(sorted_p <= thresholds)[0])
        bh_threshold = sorted_p[max_idx]
    except ValueError:
        bh_threshold = 0.0
    reject = nodal_p <= bh_threshold
    nodal_p_fdr = nodal_p  # approximate

n_sig = np.sum(reject)
print(f"  FDR-significant nodes: {n_sig} / {N_NODES}")

# Group-mean SC for connectome plot
print("Computing group-mean SC...")
sc_group = np.zeros((N_NODES, N_NODES))
for sub in ALL_SUBS:
    try:
        sc_s, _, _ = load_subject(sub, sc_treat='raw')
        sc_group += sc_s
    except:
        pass
sc_group /= len(ALL_SUBS)
np.fill_diagonal(sc_group, 0)

# Compute example subject coupling for FC_pred/FC_emp display
print("Computing example subject coupling...")
sc_ex, fc_ex, _ = load_subject('sub-01', sc_treat='raw')
res_ex = compute_coupling_single_simple(
    sc_ex, fc_ex, YEO7_INT, seed=SEED, int2short=INT2SHORT
)

# Network-level coupling for the Yeo7 coloring brain plot
# Each node gets its network's group-mean intra-network coupling
intra_group_mean = {}
for ni in range(7):
    net = INT2SHORT[ni]
    # Average nodal coupling across nodes in this network
    mask = YEO7_INT == ni
    intra_group_mean[net] = np.mean(mean_nodal_corr[mask])

network_coupling_per_node = np.zeros(N_NODES)
for ri in range(N_NODES):
    net = INT2SHORT[YEO7_INT[ri]]
    network_coupling_per_node[ri] = intra_group_mean.get(net, 0)


# %% ========================================================================
#  BRAIN PLOT 1 — GROUP MEAN NODAL SC-FC COUPLING (ortho)
# ===========================================================================
#  Mostra: para cada nó, a correlação média (Pearson) entre sua linha na
#  FC_predicted e sua linha na FC_empirical, promediado sobre os 23 sujeitos.
#  Nós "quentes" = alta correspondência estrutura→função.
#  Nós "frios"   = desacoplamento (a SC não prediz bem a FC).

print("\n" + "=" * 60)
print("  BRAIN PLOTS (nilearn 0.10.4 compatible)")
print("=" * 60)

fig = plt.figure(figsize=(10, 3.5))
ni_plot.plot_stat_map(
    roi_to_volume(mean_nodal_corr),
    bg_img=mni_bg,
    display_mode='ortho',
    cmap='RdYlBu_r',
    colorbar=True,
    black_bg=False,
    title=None,
    figure=fig,
    threshold=0.001,
    vmax=np.percentile(mean_nodal_corr[mean_nodal_corr > 0], 95),
    annotate=True,
    draw_cross=True,
)
savefig(fig, 'B01_brain_mean_coupling_ortho')


# %% ========================================================================
#  BRAIN PLOT 2 — GROUP MEAN NODAL DECOUPLING (MAE)
# ===========================================================================
#  MAE alto = grande discrepância entre FC predita pelo reservoir e FC empírica.
#  Regiões com alto MAE são onde a estrutura anatômica "falha" em prever a função.

fig = plt.figure(figsize=(10, 3.5))
ni_plot.plot_stat_map(
    roi_to_volume(mean_nodal_mae),
    bg_img=mni_bg,
    display_mode='ortho',
    cmap='hot',
    colorbar=True,
    black_bg=False,
    title=None,
    figure=fig,
    threshold=0.001,
    annotate=True,
    draw_cross=True,
)
savefig(fig, 'B02_brain_mean_decoupling_ortho')


# %% ========================================================================
#  BRAIN PLOT 3 — NODAL COUPLING T-STATISTIC (FDR-masked)
# ===========================================================================
#  One-sample t-test (coupling > 0) em cada nó, corrigido por FDR.
#  Nós significativos (FDR < 0.05) = coupling consistentemente > 0 no grupo.

t_masked = nodal_t.copy()
t_masked[~reject] = 0

fig = plt.figure(figsize=(10, 3.5))
ni_plot.plot_stat_map(
    roi_to_volume(t_masked),
    bg_img=mni_bg,
    display_mode='ortho',
    cmap='cold_hot',
    colorbar=True,
    black_bg=False,
    title=f'Nodal Coupling t-stat (FDR < 0.05, {n_sig}/{N_NODES} sig.)',
    figure=fig,
    threshold=0.001,
    annotate=True,
    draw_cross=True,
)
savefig(fig, 'B03_brain_tstat_fdr_ortho')


# %% ========================================================================
#  BRAIN PLOT 4 — INTER-SUBJECT VARIABILITY (CoV of decoupling)
# ===========================================================================
#  CoV alto = grande variabilidade inter-sujeito no decoupling daquele nó.
#  Clinicamente: nós onde a patologia pós-COVID é mais heterogênea entre pacientes.

fig = plt.figure(figsize=(10, 3.5))
ni_plot.plot_stat_map(
    roi_to_volume(cv_nodal_mae),
    bg_img=mni_bg,
    display_mode='ortho',
    cmap='YlOrRd',
    colorbar=True,
    black_bg=False,
    title=None,
    figure=fig,
    threshold=0.001,
    annotate=True,
    draw_cross=True,
    vmax=np.percentile(cv_nodal_mae[cv_nodal_mae > 0], 95),
)
savefig(fig, 'B04_brain_variability_ortho')


# %% ========================================================================
#  BRAIN PLOT 5 — GROUP-MEAN STRUCTURAL CONNECTOME (top 5% edges)
# ===========================================================================
#  Mostra os tratos de fibra branca mais fortes no grupo, com nós coloridos
#  pela rede Yeo7. Útil para visualizar a arquitetura estrutural de base.

node_colors = [YEO7_COLORS_ORD[YEO7_INT[i]] for i in range(N_NODES)]

fig = plt.figure(figsize=(10, 3.5))
ni_plot.plot_connectome(
    sc_group,
    coords,
    edge_threshold="95%",
    display_mode='ortho',
    title=None,
    figure=fig,
    node_size=15,
    node_color=node_colors,
    edge_cmap='YlOrRd',
    black_bg=False,
    annotate=True,
)
savefig(fig, 'B05_brain_group_connectome_ortho')


# %% ========================================================================
#  BRAIN PLOT 6 — MULTI-VIEW: Coupling em 3 fatias separadas (x, y, z)
# ===========================================================================
#  Para dar ao leitor uma visão mais detalhada do que 'ortho' permite,
#  criamos um painel 1×3 com fatias sagital, coronal e axial individuais.
#  Cada uma pode mostrar cut_coords diferentes para maximizar a informação.

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

# Sagital (x) — corte pelo lobo temporal medial
ax = axes[0]
display_x = ni_plot.plot_stat_map(
    roi_to_volume(mean_nodal_corr), bg_img=mni_bg,
    display_mode='x', cut_coords=[-40, 0, 40],
    cmap='RdYlBu_r', colorbar=False, black_bg=False,
    axes=ax, threshold=0.001,
    vmax=np.percentile(mean_nodal_corr[mean_nodal_corr > 0], 95),
)
ax.set_title('Sagittal', fontsize=9, fontweight='bold')

# Coronal (y) — corte pelo giro do cíngulo
ax = axes[1]
ni_plot.plot_stat_map(
    roi_to_volume(mean_nodal_corr), bg_img=mni_bg,
    display_mode='y', cut_coords=[-30, 0, 30],
    cmap='RdYlBu_r', colorbar=False, black_bg=False,
    axes=ax, threshold=0.001,
    vmax=np.percentile(mean_nodal_corr[mean_nodal_corr > 0], 95),
)
ax.set_title('Coronal', fontsize=9, fontweight='bold')

# Axial (z) — corte pelo nível ventricular
ax = axes[2]
disp_z = ni_plot.plot_stat_map(
    roi_to_volume(mean_nodal_corr), bg_img=mni_bg,
    display_mode='z', cut_coords=[-10, 15, 45],
    cmap='RdYlBu_r', colorbar=True, black_bg=False,
    axes=ax, threshold=0.001,
    vmax=np.percentile(mean_nodal_corr[mean_nodal_corr > 0], 95),
)
ax.set_title('Axial', fontsize=9, fontweight='bold')

fig.suptitle('Group Mean Nodal SC-FC Coupling — Multi-View',
             fontsize=10, fontweight='bold', y=1.02)
fig.tight_layout()
savefig(fig, 'B06_brain_coupling_multiview')


# %% ========================================================================
#  BRAIN PLOT 7 — MULTI-VIEW: Decoupling (MAE) em fatias axiais
# ===========================================================================
#  Série de fatias axiais para mostrar a distribuição espacial completa
#  do desacoplamento. Particularmente útil para COVID que pode afetar
#  tanto regiões corticais quanto profundas (tálamo, gânglios da base).

fig = plt.figure(figsize=(10, 3))
ni_plot.plot_stat_map(
    roi_to_volume(mean_nodal_mae), bg_img=mni_bg,
    display_mode='z',
    cut_coords=[-20, -5, 10, 25, 40, 55],
    cmap='hot', colorbar=True, black_bg=False,
    title=None,
    figure=fig, threshold=0.001,
)
savefig(fig, 'B07_brain_decoupling_axial_series')


# %% ========================================================================
#  BRAIN PLOT 8 — YEO7 NETWORK PARCELLATION (referência visual)
# ===========================================================================
#  Mapa das 7 redes de Yeo no espaço do Schaefer-100, para que o leitor
#  possa fazer referência cruzada com os mapas de coupling/decoupling.

network_vol = np.zeros_like(atlas_data, dtype=float)
for ri in range(N_NODES):
    network_vol[atlas_data == (ri + 1)] = YEO7_INT[ri] + 1
network_img = ni_image.new_img_like(atlas_img, network_vol)

fig = plt.figure(figsize=(10, 3.5))
ni_plot.plot_roi(
    network_img, bg_img=mni_bg,
    display_mode='ortho',
    title=None,
    figure=fig, black_bg=False, annotate=True,
)
savefig(fig, 'B08_brain_yeo7_parcellation')


# %% ========================================================================
#  BRAIN PLOT 9 — NETWORK-LEVEL COUPLING PAINTED ON BRAIN
# ===========================================================================
#  Cada nó recebe o valor médio de coupling da sua rede no grupo.
#  Permite visualizar o gradiente de coupling entre redes no espaço anatômico.

fig = plt.figure(figsize=(10, 3.5))
ni_plot.plot_stat_map(
    roi_to_volume(network_coupling_per_node),
    bg_img=mni_bg,
    display_mode='ortho',
    cmap='RdYlBu_r',
    colorbar=True,
    black_bg=False,
    title=None,
    figure=fig,
    threshold=0.001,
    annotate=True,
    draw_cross=True,
)
savefig(fig, 'B09_brain_network_coupling_strength')


# %% ========================================================================
#  BRAIN PLOT 10 — GLASS BRAIN: FC PREDICTED vs EMPIRICAL
# ===========================================================================
#  Mostra as top edges da FC_predicted do sujeito exemplo como um
#  "connectome" sobre o glass brain. Comparação visual direta com a SC.

# FC_predicted do example subject → full 100×100
FC_pred_full = np.zeros((N_NODES, N_NODES))
idx_ex = res_ex['idx_active']
n_ex = len(idx_ex)
for ii in range(n_ex):
    for jj in range(n_ex):
        FC_pred_full[idx_ex[ii], idx_ex[jj]] = res_ex['FC_pred'][ii, jj]
np.fill_diagonal(FC_pred_full, 0)

fig = plt.figure(figsize=(10, 4))
gs10 = gridspec.GridSpec(1, 2, wspace=0.1)

# FC empirical
ax1 = fig.add_subplot(gs10[0, 0])
ni_plot.plot_connectome(
    fc_ex, coords, edge_threshold="95%",
    display_mode='z', cut_coords=[0],
    title=None,
    node_size=10, node_color=node_colors,
    edge_cmap='RdBu_r', black_bg=False,
    axes=ax1,
)

# FC predicted
ax2 = fig.add_subplot(gs10[0, 1])
ni_plot.plot_connectome(
    FC_pred_full, coords, edge_threshold="95%",
    display_mode='z', cut_coords=[0],
    title=None,
    node_size=10, node_color=node_colors,
    edge_cmap='RdBu_r', black_bg=False,
    axes=ax2,
)

fig.suptitle('FC Empirical vs RC-Predicted (sub-01)',
             fontsize=10, fontweight='bold', y=1.02)
savefig(fig, 'B10_brain_fc_emp_vs_pred')


# %% ========================================================================
#  BRAIN PLOT 11 — STD MAP (INTER-SUBJECT HETEROGENEITY IN COUPLING)
# ===========================================================================
#  Desvio-padrão inter-sujeito do coupling nodal.
#  Regiões com alta variância = onde os pacientes diferem mais.

fig = plt.figure(figsize=(10, 3.5))
ni_plot.plot_stat_map(
    roi_to_volume(std_nodal_corr),
    bg_img=mni_bg,
    display_mode='ortho',
    cmap='inferno',
    colorbar=True,
    black_bg=False,
    title=None,
    figure=fig,
    threshold=0.001,
    annotate=True,
)
savefig(fig, 'B11_brain_coupling_std')


# %% ========================================================================
#  COMPOSITE BRAIN FIGURE (4-panel para o paper principal)
# ===========================================================================
#  Um painel composto com os 4 mapas mais informativos:
#  a) Mean coupling, b) Mean decoupling, c) t-stat FDR, d) Variability
#  Cada um em ortho, formando uma grade 2×2.

fig = plt.figure(figsize=(10, 7))
gs_comp = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.2)

# a: Mean coupling
ax = fig.add_subplot(gs_comp[0, 0])
ni_plot.plot_stat_map(
    roi_to_volume(mean_nodal_corr), bg_img=mni_bg,
    display_mode='ortho', cmap='RdYlBu_r', colorbar=True,
    black_bg=False, axes=ax, threshold=0.001,
    vmax=np.percentile(mean_nodal_corr[mean_nodal_corr > 0], 95),
    title=None, annotate=False,
)

# b: Mean decoupling
ax = fig.add_subplot(gs_comp[0, 1])
ni_plot.plot_stat_map(
    roi_to_volume(mean_nodal_mae), bg_img=mni_bg,
    display_mode='ortho', cmap='hot', colorbar=True,
    black_bg=False, axes=ax, threshold=0.001,
    title=None, annotate=False,
)

# c: t-stat FDR
ax = fig.add_subplot(gs_comp[1, 0])
ni_plot.plot_stat_map(
    roi_to_volume(t_masked), bg_img=mni_bg,
    display_mode='ortho', cmap='cold_hot', colorbar=True,
    black_bg=False, axes=ax, threshold=0.001,
    title=f'c  t-stat (FDR < 0.05, {n_sig} nodes)', annotate=False,
)

# d: Variability
ax = fig.add_subplot(gs_comp[1, 1])
ni_plot.plot_stat_map(
    roi_to_volume(cv_nodal_mae), bg_img=mni_bg,
    display_mode='ortho', cmap='YlOrRd', colorbar=True,
    black_bg=False, axes=ax, threshold=0.001,
    title=None, annotate=False,
    vmax=np.percentile(cv_nodal_mae[cv_nodal_mae > 0], 95),
)

fig.suptitle('Group-Level SC-FC Coupling — Brain Maps\n'
             'COVID-19 ICU Survivors (n=23, Schaefer-100)',
             fontsize=11, fontweight='bold', y=0.98)
savefig(fig, 'B_COMPOSITE_brain')


# %% ========================================================================
#  SUMMARY
# ===========================================================================

print("\n" + "=" * 60)
print("  BRAIN PLOTS COMPLETE!")
print("=" * 60)
print(f"\n  Output: {OUTPUT_DIR}/")

brains = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith('B')])
for f in brains:
    print(f"    {f}")

print(f"""
  ╔════════════════════════════════════════════════════════════════╗
  ║  B01: Mean nodal coupling (r)              — ortho           ║
  ║  B02: Mean nodal decoupling (MAE)          — ortho           ║
  ║  B03: Coupling t-stat (FDR-masked)         — ortho           ║
  ║  B04: Inter-subject variability (CoV)      — ortho           ║
  ║  B05: Group structural connectome          — ortho           ║
  ║  B06: Mean coupling multi-view             — x + y + z       ║
  ║  B07: Decoupling axial series              — z (6 slices)    ║
  ║  B08: Yeo7 parcellation reference          — ortho           ║
  ║  B09: Network-level coupling strength      — ortho           ║
  ║  B10: FC empirical vs RC-predicted         — z (glass brain) ║
  ║  B11: Inter-subject SD of coupling         — ortho           ║
  ║  B_COMPOSITE: 4-panel summary figure       — ortho           ║
  ║                                                              ║
  ║  All compatible with nilearn 0.10.4 / Python 3.8             ║
  ╚════════════════════════════════════════════════════════════════╝
""")
