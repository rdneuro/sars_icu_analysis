#!/usr/bin/env python3
"""
===============================================================================
SARS-CoV-2 Neuroimaging Study — GNN Multimodal Analysis
===============================================================================

Complete analysis pipeline running all 5 GNN models on the SARS cohort.
Produces publication-quality figures suitable for NeuroImage / Brain / HBM.

Models:
    1. SC → FC Prediction (GATv2)   → Learned SC-FC decoupling
    2. Graph VAE (SC + FC)          → Latent embeddings & reconstruction error
    3. Multimodal Heterogeneous GNN → SC-FC coupling weights per region
    4. Node Anomaly Detection       → Systematically atypical regions
    5. Graph-level Embedding        → Subject fingerprinting & clustering

Usage:
    python run_gnn_analysis.py

Author: SARS-1 Project / Velho Mago
Date: February 2026
===============================================================================
"""

#%% IMPORTS
import json, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
warnings.filterwarnings('ignore')

from gnn_multimodal.gnn_connectome import (
    SCFCPredictor, BrainVGAE, MultimodalHeteroGNN,
    NodeAnomalyDetector, GraphLevelEmbedder, BrainGNNPipeline,
    connectivity_to_pyg, build_multimodal_features,
)
import torch
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

#%% PUBLICATION STYLE
STYLE = {
    'figure.facecolor': 'white', 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'axes.grid': False, 'legend.frameon': False,
}
plt.rcParams.update(STYLE)

NETWORK_COLORS = {
    'Vis': '#781286', 'SomMot': '#4682B4', 'DorsAttn': '#00760E',
    'SalVentAttn': '#C43AFA', 'Limbic': '#DCF8A4',
    'Cont': '#E69422', 'Default': '#CD3E4E',
}
CMAP_DIV = LinearSegmentedColormap.from_list(
    'rdbu', ['#2166AC','#67A9CF','#D1E5F0','#F7F7F7',
             '#FDDBC7','#EF8A62','#B2182B'], N=256)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    'seq', ['#F7FBFF','#DEEBF7','#9ECAE1','#3182BD','#08306B'], N=256)
CMAP_HOT = LinearSegmentedColormap.from_list(
    'hot2', ['#FFFFCC','#FFEDA0','#FED976','#FD8D3C',
             '#E31A1C','#800026'], N=256)

#%% PATHS — ADJUST TO YOUR SYSTEM
RSFMRI_ROOT = Path("/home/rd/local/res/sars_cov_2_project_rsfmri")
DWI_ROOT    = Path("/home/rd/local/res/sars_cov_2_project")
CONNECT_DIR = RSFMRI_ROOT / "data/outputs/rsfmri/connectivity"
METRICS_DIR = RSFMRI_ROOT / "data/outputs/rsfmri/regional_metrics"
DWI_OUTPUT  = DWI_ROOT / "data/outputs"
FIGURES_DIR = RSFMRI_ROOT / "data/outputs/rsfmri/figures/gnn"
RESULTS_DIR = RSFMRI_ROOT / "data/outputs/rsfmri/gnn_results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [f"sub-{i:02d}" for i in range(1, 25) if i != 21]
ATLAS = "schaefer_100"
STRATEGY = "acompcor"

print(f"\n{'='*70}")
print(f"SARS-CoV-2 GNN MULTIMODAL ANALYSIS")
print(f"{'='*70}")
print(f"  Atlas: {ATLAS} | Strategy: {STRATEGY} | N={len(SUBJECTS)}")

#%% LOAD LABELS
def load_labels(atlas_name):
    paths = [
        RSFMRI_ROOT / "info/atlases/labels_schaefer_100_7networks.csv",
        Path("/mnt/project/labels_schaefer_100_7networks.csv"),
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            labels = df['label_roi'].tolist()
            networks = []
            for lbl in labels:
                parts = lbl.split('_')
                networks.append(parts[1] if len(parts) >= 2 and parts[0] in ('LH','RH') else 'Unknown')
            return labels, networks
    n = {'synthseg':86,'schaefer_100':100,'aal3':170,'brainnetome':246}.get(atlas_name,100)
    return [f"ROI_{i}" for i in range(n)], ['Unknown']*n

ROI_LABELS, ROI_NETWORKS = load_labels(ATLAS)
N_ROIS = len(ROI_LABELS)
print(f"  ROIs: {N_ROIS} | Networks: {sorted(set(ROI_NETWORKS))}")

#%% DATA LOADING FUNCTIONS
def load_fc_matrix(sub, atlas=ATLAS, strategy=STRATEGY):
    sub_label = sub.replace('sub-', '')
    base = CONNECT_DIR / atlas / strategy / f"sub-{sub_label}"
    for f in ['connectivity_correlation_fisherz.npy', 'connectivity_correlation.npy']:
        p = base / f
        if p.exists(): return np.load(p)
    return None

def load_sc_matrix(sub, atlas=ATLAS):
    atlas_map = {'schaefer_100':'schaefer100','aal3':'aal3','brainnetome':'brainnetome','synthseg':'synthseg'}
    d = DWI_OUTPUT / sub / "matrices" / atlas_map.get(atlas, atlas)
    for f in ['connectivity_sift2.npy','connectivity_count.npy','connectivity_sift2.csv','connectivity_count.csv']:
        p = d / f
        if p.exists():
            return np.load(p) if f.endswith('.npy') else np.loadtxt(p)
    return None

def load_regional_metrics(sub, atlas=ATLAS, strategy=STRATEGY):
    sub_label = sub.replace('sub-', '')
    base = METRICS_DIR / atlas / strategy / f"sub-{sub_label}"
    if not base.exists(): base = METRICS_DIR / f"sub-{sub_label}" / atlas
    metrics = {}
    for name in ['alff','falff','reho','bold_variance','bold_mean']:
        p = base / f'{name}.npy'
        if p.exists():
            arr = np.load(p)
            if len(arr) > 0: metrics[name] = arr
    return metrics or None

def load_graph_metrics(sub, atlas=ATLAS, strategy=STRATEGY):
    sub_label = sub.replace('sub-', '')
    base = METRICS_DIR / "graph" / atlas / strategy / f"sub-{sub_label}"
    metrics = {}
    for name, fname in [('degree_fc','degree.npy'),('strength_fc','strength.npy'),('clustering_fc','clustering.npy')]:
        p = base / fname
        if p.exists(): metrics[name] = np.load(p)
    return metrics or None

def load_dti_roi_metrics(sub, atlas=ATLAS):
    d = DWI_OUTPUT / sub / "dti"
    atlas_dwi = {'schaefer_100':'schaefer100'}.get(atlas, atlas)
    metrics = {}
    for name in ['fa','md','ad','rd']:
        for pat in [f'{name}_roi_{atlas_dwi}.npy', f'{name}_mean_{atlas_dwi}.npy', f'{name}_{atlas_dwi}.npy']:
            p = d / pat
            if p.exists():
                metrics[f'{name}_mean'] = np.load(p)
                break
    return metrics or None

#%% ASSEMBLE DATA
print(f"\n{'─'*70}\nLoading data...\n{'─'*70}")

sc_matrices, fc_matrices, node_features_list, valid_subjects = [], [], [], []

for sub in SUBJECTS:
    fc = load_fc_matrix(sub)
    sc = load_sc_matrix(sub)
    if fc is None:
        print(f"  ✗ {sub}: FC not found, skipping"); continue
    if sc is None:
        print(f"  ⚠ {sub}: SC not found, using thresholded |FC| as proxy")
        sc = np.abs(fc.copy())
        thr = np.percentile(sc[sc > 0], 75) if np.any(sc > 0) else 0.1
        sc[sc < thr] = 0

    n = min(fc.shape[0], sc.shape[0], N_ROIS)
    fc, sc = fc[:n,:n], sc[:n,:n]

    feat_fmri, feat_dmri = {}, {}
    regional = load_regional_metrics(sub)
    if regional:
        for k,v in regional.items():
            if len(v) >= n: feat_fmri[k] = v[:n]
    graph_m = load_graph_metrics(sub)
    if graph_m:
        for k,v in graph_m.items():
            if len(v) >= n: feat_fmri[k] = v[:n]
    dti_m = load_dti_roi_metrics(sub)
    if dti_m:
        for k,v in dti_m.items():
            if len(v) >= n: feat_dmri[k] = v[:n]

    fc_abs = np.abs(fc); np.fill_diagonal(fc_abs, 0)
    feat_fmri['strength_fc_abs'] = fc_abs.sum(axis=1)
    feat_fmri['eigenvec_fc'] = np.linalg.eigh(fc_abs)[1][:, -1]
    sc_clean = sc.copy(); np.fill_diagonal(sc_clean, 0)
    if sc_clean.max() > 0:
        feat_dmri['strength_sc'] = sc_clean.sum(axis=1)
        feat_dmri['degree_sc'] = (sc_clean > 0).sum(axis=1).astype(float)

    try:
        nf = build_multimodal_features(
            fmri_metrics=feat_fmri or None, dmri_metrics=feat_dmri or None, n_nodes=n)
    except ValueError:
        nf = np.column_stack([fc_abs.sum(1), sc_clean.sum(1) if sc_clean.max()>0 else np.zeros(n)])

    sc_matrices.append(sc); fc_matrices.append(fc)
    node_features_list.append(nf); valid_subjects.append(sub)
    print(f"  ✓ {sub}: SC{sc.shape} FC{fc.shape} {nf.shape[1]}feat")

N_SUBJ = len(valid_subjects)
N_NODES = fc_matrices[0].shape[0] if fc_matrices else 0
N_FEAT = node_features_list[0].shape[1] if node_features_list else 0
print(f"\n  ═══ {N_SUBJ} subjects × {N_NODES} ROIs × {N_FEAT} features ═══")

#%% RUN GNN PIPELINE
print(f"\n{'='*70}\nRUNNING GNN PIPELINE\n{'='*70}")
pipeline = BrainGNNPipeline(atlas=ATLAS, verbose=True)
results = pipeline.run_all(
    sc_matrices=sc_matrices, fc_matrices=fc_matrices,
    node_features_list=node_features_list, subject_ids=valid_subjects,
    scfc_kwargs=dict(hidden_channels=64, n_heads=4, n_layers=3, dropout=0.3, lr=1e-3, epochs=500, patience=50),
    vgae_kwargs=dict(latent_dim=32, hidden_channels=64, encoder_type='gat', epochs=300, patience=40),
    hetero_kwargs=dict(hidden_channels=32, n_heads=4, epochs=400, patience=40),
    anomaly_kwargs=dict(latent_dim=32, hidden_channels=64, epochs=200, z_threshold=2.0),
    embed_kwargs=dict(embed_dim=32, hidden_channels=64, n_heads=4, epochs=300, patience=40),
)
summary_df = BrainGNNPipeline.summarize(results, labels=ROI_LABELS[:N_NODES])
summary_df.to_csv(RESULTS_DIR / f"gnn_summary_{ATLAS}.csv")
print(f"\n✓ Summary saved: {RESULTS_DIR / f'gnn_summary_{ATLAS}.csv'}")

#%% =========================================================================
# FIGURE 1 — SC → FC PREDICTION: LEARNED DECOUPLING
# ===========================================================================
print(f"\n{'─'*70}\nGenerating Figure 1: SC→FC Decoupling\n{'─'*70}")

scfc = results.get('scfc', {})
if scfc:
    n = N_NODES; subs = list(scfc.keys())
    all_decoupling = np.stack([scfc[s].decoupling_score for s in subs])
    mean_decoupling = all_decoupling.mean(axis=0)
    global_r2s = [scfc[s].r2_global for s in subs]
    rep_idx = np.argsort(global_r2s)[len(global_r2s)//2]
    rep_sub = subs[rep_idx]; rep = scfc[rep_sub]

    fig = plt.figure(figsize=(7.2, 7.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35, left=0.08, right=0.95, top=0.93, bottom=0.07)

    # A: Predicted vs True FC
    ax = fig.add_subplot(gs[0,0])
    vmax = np.percentile(np.abs(rep.fc_true), 95)
    combined = np.triu(rep.fc_true, k=1) + np.tril(rep.fc_predicted, k=-1)
    im = ax.imshow(combined, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax, aspect='equal', interpolation='none')
    ax.set_title('A  FC True (▲) vs Predicted (▼)', fontweight='bold', fontsize=9, loc='left')
    ax.set_xlabel('ROI'); ax.set_ylabel('ROI')
    cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02); cb.set_label('Fisher z', fontsize=7); cb.ax.tick_params(labelsize=7)
    ax.plot([0,n-1],[0,n-1],'k-',lw=0.5,alpha=0.4)
    ax.text(n*0.25, n*0.75, 'Predicted', ha='center', fontsize=7, fontstyle='italic', alpha=0.5)
    ax.text(n*0.75, n*0.25, 'True', ha='center', fontsize=7, fontstyle='italic', alpha=0.5)

    # B: Nodal decoupling sorted by network
    ax = fig.add_subplot(gs[0,1])
    net_order = ['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']
    sort_idx, net_bounds = [], []
    for net in net_order:
        idxs = [i for i,nw in enumerate(ROI_NETWORKS[:n]) if nw==net]
        if idxs: net_bounds.append((len(sort_idx), net)); sort_idx.extend(idxs)
    remaining = [i for i in range(n) if i not in sort_idx]
    if remaining: net_bounds.append((len(sort_idx),'Other')); sort_idx.extend(remaining)
    ax.barh(range(n), mean_decoupling[sort_idx], color='#3182BD', edgecolor='none', height=1.0, alpha=0.85)
    for pos, name in net_bounds:
        ax.axhline(pos-0.5, color='gray', lw=0.3, ls='--', alpha=0.5)
        c = NETWORK_COLORS.get(name,'#888')
        ax.text(-0.02, pos+2, name, fontsize=6, color=c, fontweight='bold', ha='right', transform=ax.get_yaxis_transform())
    ax.set_ylim(-0.5,n-0.5); ax.invert_yaxis(); ax.set_xlabel('Decoupling Score')
    ax.set_title('B  Nodal SC-FC Decoupling (group)', fontweight='bold', fontsize=9, loc='left'); ax.set_yticks([])

    # C: Decoupling by network (violin)
    ax = fig.add_subplot(gs[1,0])
    plot_data = []
    for i in range(n):
        net = ROI_NETWORKS[i] if i < len(ROI_NETWORKS) else 'Unknown'
        if net in NETWORK_COLORS:
            for si in range(len(subs)):
                plot_data.append({'Network': net, 'Decoupling': all_decoupling[si,i]})
    df_plot = pd.DataFrame(plot_data)
    if len(df_plot) > 0:
        order = [ne for ne in net_order if ne in df_plot['Network'].unique()]
        parts = ax.violinplot([df_plot[df_plot['Network']==ne]['Decoupling'].values for ne in order],
                              positions=range(len(order)), showmeans=True, showextrema=False, showmedians=True)
        for i,pc in enumerate(parts['bodies']):
            pc.set_facecolor(NETWORK_COLORS[order[i]]); pc.set_alpha(0.7); pc.set_edgecolor('black'); pc.set_linewidth(0.5)
        parts['cmeans'].set_color('black'); parts['cmedians'].set_color('white')
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Decoupling Score')
        groups = [df_plot[df_plot['Network']==ne]['Decoupling'].values for ne in order]
        groups = [g for g in groups if len(g)>0]
        if len(groups) >= 2:
            H, p_kw = stats.kruskal(*groups)
            sig = '***' if p_kw<0.001 else '**' if p_kw<0.01 else '*' if p_kw<0.05 else 'n.s.'
            ax.text(0.98, 0.95, f'H={H:.1f}, p={p_kw:.2e} {sig}', transform=ax.transAxes, fontsize=6, ha='right', va='top', fontstyle='italic')
    ax.set_title('C  Decoupling by Network', fontweight='bold', fontsize=9, loc='left')

    # D: Training curves
    ax = fig.add_subplot(gs[1,1])
    ax.plot(rep.train_losses, color='#2166AC', lw=1.0, alpha=0.8, label='Train')
    ax.plot(rep.val_losses, color='#B2182B', lw=1.0, alpha=0.8, label='Validation')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title(f'D  Training ({rep_sub}, R²={rep.r2_global:.3f})', fontweight='bold', fontsize=9, loc='left')
    ax.legend(loc='upper right', fontsize=7); ax.set_yscale('log')
    axi = ax.inset_axes([0.55, 0.45, 0.4, 0.35])
    axi.hist(global_r2s, bins=8, color='#3182BD', edgecolor='white', alpha=0.8, linewidth=0.5)
    axi.set_xlabel('R²', fontsize=6); axi.set_ylabel('N', fontsize=6); axi.tick_params(labelsize=5)
    axi.set_title('Global R² dist.', fontsize=6); axi.spines['top'].set_visible(False); axi.spines['right'].set_visible(False)

    plt.savefig(FIGURES_DIR / "fig1_scfc_decoupling.png", dpi=300, facecolor='white'); plt.show()
    print(f"  ✓ Fig 1 saved")

#%% =========================================================================
# FIGURE 2 — GRAPH VAE: EMBEDDINGS & RECONSTRUCTION ERROR
# ===========================================================================
print(f"\n{'─'*70}\nGenerating Figure 2: VGAE Embeddings\n{'─'*70}")

vgae_sc = results.get('vgae_sc', {})
vgae_fc = results.get('vgae_fc', {})
if vgae_sc:
    n = N_NODES; subs = list(vgae_sc.keys())
    rep_sub = subs[len(subs)//2]
    emb_sc = vgae_sc[rep_sub].embeddings
    emb_fc = vgae_fc[rep_sub].embeddings if rep_sub in vgae_fc else None

    fig = plt.figure(figsize=(7.2, 7.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35, left=0.08, right=0.95, top=0.93, bottom=0.07)

    # A: Latent space PCA
    ax = fig.add_subplot(gs[0,0])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2); coords = pca.fit_transform(emb_sc)
    for net in ['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']:
        idx = [i for i,nw in enumerate(ROI_NETWORKS[:n]) if nw==net]
        if idx: ax.scatter(coords[idx,0], coords[idx,1], c=NETWORK_COLORS.get(net,'#888'),
                           s=20, alpha=0.8, edgecolors='white', linewidths=0.3, label=net, zorder=3)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('A  SC-VGAE Latent Space', fontweight='bold', fontsize=9, loc='left')
    ax.legend(fontsize=5, markerscale=0.6, ncol=2, loc='lower right', handletextpad=0.2, columnspacing=0.5)

    # B: Reconstruction error
    ax = fig.add_subplot(gs[0,1])
    all_recon = np.stack([vgae_sc[s].nodal_recon_error for s in subs])
    mean_recon = all_recon.mean(axis=0)
    colors = [NETWORK_COLORS.get(ROI_NETWORKS[i],'#888') for i in range(n)]
    ax.bar(range(n), mean_recon, color=colors, edgecolor='none', width=1.0, alpha=0.8)
    ax.set_xlabel('ROI'); ax.set_ylabel('Mean Recon. Error'); ax.set_xlim(-0.5,n-0.5)
    ax.set_title('B  SC-VGAE Reconstruction Error', fontweight='bold', fontsize=9, loc='left')
    for idx in np.argsort(mean_recon)[-8:]:
        ax.annotate(ROI_LABELS[idx].replace('LH_','L.').replace('RH_','R.')[:12],
                    (idx, mean_recon[idx]), fontsize=4, rotation=55, ha='left', va='bottom')

    # C: SC vs FC embedding
    ax = fig.add_subplot(gs[1,0])
    if emb_fc is not None:
        from scipy.spatial import procrustes
        emb_sc_n = (emb_sc - emb_sc.mean(0)) / (emb_sc.std(0)+1e-8)
        emb_fc_n = (emb_fc - emb_fc.mean(0)) / (emb_fc.std(0)+1e-8)
        try:
            _, _, disp = procrustes(emb_sc_n, emb_fc_n)
            sc_c = [NETWORK_COLORS.get(ROI_NETWORKS[i],'#888') for i in range(n)]
            ax.scatter(emb_sc_n[:,0], emb_fc_n[:,0], c=sc_c, s=15, alpha=0.7, edgecolors='white', linewidths=0.2)
            ax.plot([emb_sc_n[:,0].min(),emb_sc_n[:,0].max()],[emb_sc_n[:,0].min(),emb_sc_n[:,0].max()],'k--',lw=0.5,alpha=0.4)
            ax.set_xlabel('SC Embedding (dim 1)'); ax.set_ylabel('FC Embedding (dim 1)')
            ax.text(0.05,0.92,f'Procrustes d={disp:.3f}',transform=ax.transAxes,fontsize=7,fontstyle='italic')
        except: ax.text(0.5,0.5,'Procrustes failed',ha='center',transform=ax.transAxes)
    ax.set_title('C  SC vs FC Embedding Alignment', fontweight='bold', fontsize=9, loc='left')

    # D: AUC/AP
    ax = fig.add_subplot(gs[1,1])
    aucs_sc = [vgae_sc[s].auc for s in subs]; aps_sc = [vgae_sc[s].ap for s in subs]
    aucs_fc = [vgae_fc[s].auc for s in subs if s in vgae_fc]; aps_fc = [vgae_fc[s].ap for s in subs if s in vgae_fc]
    bp = ax.boxplot([aucs_sc, aps_sc, aucs_fc, aps_fc], positions=[0,1,2.5,3.5], widths=0.6, patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], ['#2166AC','#67A9CF','#B2182B','#EF8A62']):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    for el in ['whiskers','caps','medians']:
        for item in bp[el]: item.set_color('black'); item.set_linewidth(0.8)
    ax.set_xticks([0,1,2.5,3.5]); ax.set_xticklabels(['AUC\n(SC)','AP\n(SC)','AUC\n(FC)','AP\n(FC)'],fontsize=7)
    ax.set_ylabel('Score'); ax.axhline(0.5,color='gray',ls='--',lw=0.5,alpha=0.4)
    ax.set_title('D  Link Prediction Performance', fontweight='bold', fontsize=9, loc='left')

    plt.savefig(FIGURES_DIR / "fig2_vgae_embeddings.png", dpi=300, facecolor='white'); plt.show()
    print(f"  ✓ Fig 2 saved")

#%% =========================================================================
# FIGURE 3 — HETERO-GNN COUPLING + ANOMALY DETECTION
# ===========================================================================
print(f"\n{'─'*70}\nGenerating Figure 3: Coupling & Anomaly\n{'─'*70}")

hetero = results.get('hetero', {}); anomaly = results.get('anomaly')
n = N_NODES

fig = plt.figure(figsize=(7.2, 7.5))
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35, left=0.08, right=0.95, top=0.93, bottom=0.07)

# A: Coupling by network
ax = fig.add_subplot(gs[0,0])
if hetero:
    subs_h = list(hetero.keys())
    all_coupling = np.stack([hetero[s].coupling_weights for s in subs_h])
    mean_coupling = all_coupling.mean(axis=0)
    net_order = ['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']
    plot_data_h = []
    for i in range(n):
        net = ROI_NETWORKS[i] if i<len(ROI_NETWORKS) else 'Unknown'
        if net in NETWORK_COLORS:
            for si in range(len(subs_h)):
                plot_data_h.append({'Network':net, 'Coupling':all_coupling[si,i]})
    df_h = pd.DataFrame(plot_data_h)
    if len(df_h)>0:
        order = [ne for ne in net_order if ne in df_h['Network'].unique()]
        parts = ax.violinplot([df_h[df_h['Network']==ne]['Coupling'].values for ne in order],
                              positions=range(len(order)), showmeans=True, showextrema=False, showmedians=True)
        for i,pc in enumerate(parts['bodies']):
            pc.set_facecolor(NETWORK_COLORS[order[i]]); pc.set_alpha(0.7); pc.set_edgecolor('black'); pc.set_linewidth(0.5)
        parts['cmeans'].set_color('black'); parts['cmedians'].set_color('white')
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=45,ha='right',fontsize=7)
        ax.set_ylabel('Coupling Weight'); ax.axhline(0.5,color='gray',ls='--',lw=0.5,alpha=0.4)
ax.set_title('A  SC-FC Coupling (Hetero-GNN)', fontweight='bold', fontsize=9, loc='left')

# B: Coupling vs embedding sim
ax = fig.add_subplot(gs[0,1])
if hetero:
    all_sim = np.stack([hetero[s].embedding_similarity for s in subs_h]); mean_sim = all_sim.mean(axis=0)
    sc_c = [NETWORK_COLORS.get(ROI_NETWORKS[i],'#888') for i in range(n)]
    ax.scatter(mean_coupling, mean_sim, c=sc_c, s=18, alpha=0.75, edgecolors='white', linewidths=0.3)
    slope,intercept,r,p,_ = stats.linregress(mean_coupling, mean_sim)
    xl = np.linspace(mean_coupling.min(), mean_coupling.max(), 100)
    ax.plot(xl, slope*xl+intercept, 'k--', lw=0.8, alpha=0.6)
    ax.text(0.05,0.92,f'r={r:.3f}, p={p:.2e}',transform=ax.transAxes,fontsize=7,fontstyle='italic')
    ax.set_xlabel('Coupling Weight'); ax.set_ylabel('Embedding Cosine Similarity')
ax.set_title('B  Coupling vs Embed. Similarity', fontweight='bold', fontsize=9, loc='left')

# C: Anomaly z-scores
ax = fig.add_subplot(gs[1,0])
if anomaly is not None:
    z = anomaly.z_scores
    colors_a = ['#B2182B' if zi>anomaly.threshold else '#FD8D3C' if zi>1.5 else '#3182BD' for zi in z]
    ax.bar(range(n), z, color=colors_a, edgecolor='none', width=1.0, alpha=0.85)
    ax.axhline(anomaly.threshold, color='red', ls='--', lw=0.8, label=f'z={anomaly.threshold}')
    ax.axhline(0, color='gray', ls='-', lw=0.3); ax.set_xlabel('ROI'); ax.set_ylabel('Anomaly Z-score')
    ax.legend(fontsize=6, loc='upper right'); ax.set_xlim(-0.5,n-0.5)
    for idx in anomaly.flagged_nodes:
        if idx<n:
            ax.annotate(ROI_LABELS[idx].replace('LH_','L.').replace('RH_','R.')[:14],
                        (idx,z[idx]),fontsize=5,rotation=45,ha='left',va='bottom',color='#B2182B')
    ax.text(0.02,0.92,f'{len(anomaly.flagged_nodes)} flagged',transform=ax.transAxes,fontsize=7,fontstyle='italic',color='#B2182B')
ax.set_title('C  Node Anomaly Scores', fontweight='bold', fontsize=9, loc='left')

# D: Cross-method convergence
ax = fig.add_subplot(gs[1,1])
md = {}
if scfc:
    subs_s = list(scfc.keys()); md['SCFC\nDecoup'] = np.stack([scfc[s].decoupling_score for s in subs_s]).mean(0)
if 'vgae_sc' in results:
    subs_v = list(results['vgae_sc'].keys()); md['VGAE\nRecon'] = np.stack([results['vgae_sc'][s].nodal_recon_error for s in subs_v]).mean(0)
if hetero: md['Hetero\n1-Coup'] = 1.0 - mean_coupling
if anomaly is not None: md['Anomaly'] = anomaly.nodal_anomaly_score
if len(md) >= 2:
    names = list(md.keys()); nm = len(names)
    corr_mat = np.eye(nm)
    for i in range(nm):
        for j in range(i+1,nm):
            r,_ = stats.spearmanr(md[names[i]], md[names[j]]); corr_mat[i,j]=r; corr_mat[j,i]=r
    im = ax.imshow(corr_mat, cmap=CMAP_DIV, vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(nm)); ax.set_yticks(range(nm))
    ax.set_xticklabels(names,fontsize=6,rotation=45,ha='right'); ax.set_yticklabels(names,fontsize=6)
    for i in range(nm):
        for j in range(nm):
            c = 'white' if abs(corr_mat[i,j])>0.6 else 'black'
            ax.text(j,i,f'{corr_mat[i,j]:.2f}',ha='center',va='center',fontsize=7,color=c,fontweight='bold')
    cb = plt.colorbar(im,ax=ax,shrink=0.8,pad=0.02); cb.set_label("Spearman ρ",fontsize=7); cb.ax.tick_params(labelsize=6)
ax.set_title('D  Cross-Method Convergence', fontweight='bold', fontsize=9, loc='left')

plt.savefig(FIGURES_DIR / "fig3_hetero_anomaly.png", dpi=300, facecolor='white'); plt.show()
print(f"  ✓ Fig 3 saved")

#%% =========================================================================
# FIGURE 4 — GRAPH-LEVEL EMBEDDING: SUBJECT FINGERPRINTING
# ===========================================================================
print(f"\n{'─'*70}\nGenerating Figure 4: Graph Embedding\n{'─'*70}")

emb_result = results.get('embedding')
if emb_result is not None:
    embeddings = emb_result.subject_embeddings; sim_matrix = emb_result.similarity_matrix
    node_imp = emb_result.node_importance; clusters = emb_result.cluster_labels
    n_subj = len(valid_subjects)

    fig = plt.figure(figsize=(7.2, 7.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35, left=0.08, right=0.95, top=0.93, bottom=0.07)

    # A: Subject PCA
    ax = fig.add_subplot(gs[0,0])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2); coords = pca.fit_transform(embeddings)
    if clusters is not None:
        cc = plt.cm.Set2(np.linspace(0,1,max(clusters)+1))
        for c in np.unique(clusters):
            m = clusters==c
            ax.scatter(coords[m,0],coords[m,1],c=[cc[c]],s=40,alpha=0.8,edgecolors='black',linewidths=0.5,label=f'Cluster {c+1}',zorder=3)
    else:
        ax.scatter(coords[:,0],coords[:,1],c='#3182BD',s=40,alpha=0.8,edgecolors='black',linewidths=0.5,zorder=3)
    for i,sub in enumerate(valid_subjects):
        ax.annotate(sub.replace('sub-',''),(coords[i,0],coords[i,1]),fontsize=5,ha='center',va='bottom',xytext=(0,4),textcoords='offset points')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('A  Subject Embedding Space', fontweight='bold', fontsize=9, loc='left')
    if clusters is not None: ax.legend(fontsize=6, loc='lower right')

    # B: Dendrogram
    ax = fig.add_subplot(gs[0,1])
    dist_m = np.maximum(1-sim_matrix, 0); np.fill_diagonal(dist_m, 0)
    Z = linkage(squareform(dist_m, checks=False), method='ward')
    dn = dendrogram(Z, ax=ax, labels=[s.replace('sub-','') for s in valid_subjects],
                    leaf_rotation=90, leaf_font_size=6, color_threshold=0, above_threshold_color='#3182BD')
    ax.set_ylabel('Distance'); ax.set_title('B  Subject Dendrogram', fontweight='bold', fontsize=9, loc='left')

    # C: Node importance by network
    ax = fig.add_subplot(gs[1,0])
    net_order = ['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']
    nm, ns, vn = [], [], []
    for net in net_order:
        idx = [i for i,nw in enumerate(ROI_NETWORKS[:N_NODES]) if nw==net]
        if idx:
            vals = node_imp[idx]; nm.append(vals.mean()); ns.append(vals.std()/np.sqrt(len(idx))); vn.append(net)
    cb = [NETWORK_COLORS.get(ne,'#888') for ne in vn]
    ax.bar(range(len(vn)), nm, yerr=ns, color=cb, edgecolor='white', linewidth=0.5, alpha=0.85, capsize=2, error_kw={'linewidth':0.8})
    ax.set_xticks(range(len(vn))); ax.set_xticklabels(vn,rotation=45,ha='right',fontsize=7)
    ax.set_ylabel('Mean Attention Weight'); ax.set_title('C  Node Importance by Network', fontweight='bold', fontsize=9, loc='left')

    # D: Similarity heatmap
    ax = fig.add_subplot(gs[1,1])
    order = dn['leaves']; sim_ord = sim_matrix[np.ix_(order,order)]
    sl = [valid_subjects[i].replace('sub-','') for i in order]
    im = ax.imshow(sim_ord, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect='equal', interpolation='none')
    ax.set_xticks(range(n_subj)); ax.set_yticks(range(n_subj))
    ax.set_xticklabels(sl,fontsize=5,rotation=90); ax.set_yticklabels(sl,fontsize=5)
    ax.set_title('D  Subject Similarity Matrix', fontweight='bold', fontsize=9, loc='left')
    cb = plt.colorbar(im,ax=ax,shrink=0.8,pad=0.02); cb.set_label('Cosine Sim.',fontsize=7); cb.ax.tick_params(labelsize=6)

    plt.savefig(FIGURES_DIR / "fig4_graph_embedding.png", dpi=300, facecolor='white'); plt.show()
    print(f"  ✓ Fig 4 saved")

#%% =========================================================================
# FIGURE 5 — INTEGRATED HEATMAP (SUPPLEMENTARY)
# ===========================================================================
print(f"\n{'─'*70}\nGenerating Figure 5: Integrated Summary\n{'─'*70}")

if not summary_df.empty:
    cols = [c for c in summary_df.columns if not c.endswith('_std') and c != 'anomaly_flagged']
    if cols:
        net_order = ['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']
        sort_idx, net_bounds = [], []
        for net in net_order:
            idxs = [i for i,nw in enumerate(ROI_NETWORKS[:N_NODES]) if nw==net]
            if idxs: net_bounds.append((len(sort_idx), len(sort_idx)+len(idxs), net)); sort_idx.extend(idxs)
        remaining = [i for i in range(N_NODES) if i not in sort_idx]
        if remaining: net_bounds.append((len(sort_idx), len(sort_idx)+len(remaining), 'Other')); sort_idx.extend(remaining)
        data = summary_df.iloc[sort_idx][cols].copy()
        for col in cols:
            v = data[col].values.astype(float); mu,sd = np.nanmean(v),np.nanstd(v)
            if sd>0: data[col] = (v-mu)/sd

        fig,ax = plt.subplots(figsize=(7.2, 9))
        dc = [c.replace('scfc_','SCFC\n').replace('vgae_','VGAE\n').replace('hetero_','Hetero\n')
               .replace('anomaly_','Anom.\n').replace('attention_','Attn\n').replace('_',' ') for c in cols]
        im = ax.imshow(data.values, cmap=CMAP_DIV, aspect='auto', vmin=-2.5, vmax=2.5, interpolation='none')
        for s,e,ne in net_bounds:
            c = NETWORK_COLORS.get(ne,'#888')
            ax.add_patch(plt.Rectangle((-1.8,s-0.5),1.3,e-s,facecolor=c,edgecolor='none',alpha=0.8,clip_on=False))
            ax.text(-1.1,(s+e)/2,ne,fontsize=5,ha='center',va='center',fontweight='bold',rotation=90,clip_on=False)
            ax.axhline(s-0.5, color='white', lw=0.5)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(dc,fontsize=6,rotation=45,ha='right')
        ax.set_yticks([]); ax.set_ylabel(f'ROIs (n={N_NODES})')
        cb = plt.colorbar(im,ax=ax,shrink=0.6,pad=0.02); cb.set_label('Z-score',fontsize=8); cb.ax.tick_params(labelsize=7)
        ax.set_title('GNN-Derived Nodal Metrics — Integrated Summary', fontweight='bold', fontsize=10)
        plt.savefig(FIGURES_DIR / "fig5_integrated_summary.png", dpi=300, facecolor='white'); plt.show()
        print(f"  ✓ Fig 5 saved")

#%% SAVE NUMERICAL RESULTS
print(f"\n{'='*70}\nSAVING RESULTS\n{'='*70}")

if 'scfc' in results:
    subs_s = list(results['scfc'].keys())
    np.savez(RESULTS_DIR / f'scfc_results_{ATLAS}.npz',
             decoupling=np.stack([results['scfc'][s].decoupling_score for s in subs_s]),
             r2_nodal=np.stack([results['scfc'][s].r2_nodal for s in subs_s]),
             nodal_error=np.stack([results['scfc'][s].nodal_error for s in subs_s]))
    pd.DataFrame({'subject':subs_s, 'r2_global':[results['scfc'][s].r2_global for s in subs_s]}).to_csv(
        RESULTS_DIR / f'scfc_r2_global_{ATLAS}.csv', index=False)

if 'vgae_sc' in results:
    for sub,r in results['vgae_sc'].items():
        np.save(RESULTS_DIR / f'vgae_sc_emb_{sub}_{ATLAS}.npy', r.embeddings)

if 'hetero' in results:
    subs_h = list(results['hetero'].keys())
    np.save(RESULTS_DIR / f'hetero_coupling_{ATLAS}.npy',
            np.stack([results['hetero'][s].coupling_weights for s in subs_h]))

if 'anomaly' in results:
    a = results['anomaly']
    pd.DataFrame({'roi':ROI_LABELS[:N_NODES], 'network':ROI_NETWORKS[:N_NODES],
                  'anomaly_score':a.nodal_anomaly_score, 'z_score':a.z_scores,
                  'flagged':np.isin(np.arange(N_NODES),a.flagged_nodes).astype(int)}).to_csv(
        RESULTS_DIR / f'anomaly_results_{ATLAS}.csv', index=False)

if 'embedding' in results:
    e = results['embedding']
    df_e = pd.DataFrame(e.subject_embeddings, index=valid_subjects,
                        columns=[f'dim_{i}' for i in range(e.subject_embeddings.shape[1])])
    if e.cluster_labels is not None: df_e['cluster'] = e.cluster_labels
    df_e.to_csv(RESULTS_DIR / f'graph_embeddings_{ATLAS}.csv')

print(f"\n✓ All results saved to: {RESULTS_DIR}")

#%% FINAL REPORT
print(f"\n{'='*70}\nANALYSIS COMPLETE\n{'='*70}")
if 'scfc' in results:
    r2s = [results['scfc'][s].r2_global for s in results['scfc']]
    print(f"  SC→FC:   R² = {np.mean(r2s):.3f} ± {np.std(r2s):.3f} [{min(r2s):.3f}, {max(r2s):.3f}]")
if 'vgae_sc' in results:
    aucs = [results['vgae_sc'][s].auc for s in results['vgae_sc']]
    print(f"  SC-VGAE: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
if 'hetero' in results:
    c = np.stack([results['hetero'][s].coupling_weights for s in results['hetero']])
    print(f"  Hetero:  Coupling = {c.mean():.3f} ± {c.std():.3f}")
if 'anomaly' in results:
    a = results['anomaly']
    print(f"  Anomaly: {len(a.flagged_nodes)}/{N_NODES} flagged")
    if len(a.flagged_nodes)>0:
        print(f"           {', '.join([ROI_LABELS[i] for i in a.flagged_nodes[:10] if i<N_NODES])}")
if 'embedding' in results:
    e = results['embedding']
    if e.cluster_labels is not None:
        from collections import Counter
        for c,cnt in sorted(Counter(e.cluster_labels).items()):
            print(f"  Cluster {c+1}: {cnt} subjects")
print(f"\n  Figures: {FIGURES_DIR}")
print(f"  Data:    {RESULTS_DIR}\n{'='*70}\n")
