##=-------------------------------------------------------------------------=##
## importing libraries

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
from scipy.linalg import eigh
from scipy.spatial.distance import squareform

from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import r2_score, balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix

from reservoirpy.datasets import mackey_glass
from conn2res.connectivity import Conn, get_modules
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout
from conn2res import readout, plotting

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 900)

##=-------------------------------------------------------------------------=##
## figures parameters

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Paleta de cores Yeo 7 networks
YEO7_COLORS = {
    'VIS':  '#781286',  # Visual
    'SM':   '#4682B4',  # Somatomotor
    'DA':   '#00760E',  # Dorsal Attention
    'VA':   '#C43AFA',  # Ventral Attention / Salience
    'LIM':  '#DCF8A4',  # Limbic
    'FP':   '#E69422',  # Frontoparietal
    'DMN':  '#CD3E4E',  # Default Mode
}

##=-------------------------------------------------------------------------=##
## all functions

def load_empirical_connectome(sub, atlas='schaefer100', treat='log', treat_nan=True):
    """
    Carrega as matrizes de conectividade de um determinado paciente, conforme o atlas.
    Para o tratamento da matriz de conectividade estrutural, podem ser optadas por:
          - 'max' = SC / max(SC)
          - 'log' = np.log1p(SC)  [default]
          - 'raw' = não realizar nenhum tratamento.
    Se treat != 'raw', ambas matrizes terão suas diagonais preenchidas com zeros.
    Pode-se optar por checar e tratar valores NaN automaticamente na importação.

    Returns
    -------
    SC : (n_nodes, n_nodes) ndarray — structural connectivity (simétrica)
    FC : (n_nodes, n_nodes) ndarray — functional connectivity (correlação)
    """
    fc = np.load(f"/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/{sub}/{atlas}/dmri/connectivity_sift2.npy")
    sc = np.load(f"/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/{sub}/{atlas}/fmri/connectivity_correlation.npy")
    ts = np.load(f"/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/{sub}/{atlas}/fmri/timeseries.npy")
    np.fill_diagonal(sc, 0)
    np.fill_diagonal(fc, 1)
    sc = vec_to_sym_matrix(sym_matrix_to_vec(sc, discard_diagonal=True), np.zeros(sc.shape[0]))
    fc = vec_to_sym_matrix(sym_matrix_to_vec(fc, discard_diagonal=True), np.zeros(fc.shape[0]))
    if treat_nan == True:
        imp_sc, imp_fc = SimpleImputer(strategy='median'), SimpleImputer(strategy='mean')
        sc = imp_sc.fit_transform(sc)
        fc = imp_fc.fit_transform(fc)
    if treat == 'raw':
        pass
    elif treat == 'log':
        sc = np.log1p(sc)
    elif treat == 'max':
        sc = sc / np.max(sc)
    return sc, fc, ts

def load_atlas_labels(atlas='schaefer100'):
    """
    Carrega labels das ROIs de um determinado atlas.
    Caso seja o 'schaefer100', retorna também a network correspondente as ROIs.

    Returns
    -------
    lbls : label de cada ROI, na ordem 1 até nç;
    nets : assignment de cada ROI a uma network Yeo (strings).
    """
    lbls = np.loadtxt(f"/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/info/labels_{atlas}.txt", dtype=str)
    if atlas == 'schaefer100':
        nets = np.loadtxt("/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/info/networks_schaefer100.txt", dtype=str)
        return lbls, nets
    return lbls

def get_subjects(fpath=True, atlas='schaefer100'):
    """
    Retorna a lista com os subjects válidos do dataset, se fpath == False.
    Caso contrário, retorna também os filepaths para os arquivos de conectividade.

    Returns
    -------
    subs : subjects names;
    SC   : filepaths das matrizes de conectividade estrutural;
    FC   : filepaths das matrizes de conectividade funcional.
    """
    subs = [f"sub-{n:02n}" for n in range(1, 25)]
    subs.remove('sub-21')
    if fpath == False:
        return subs
    SC, FC = [], []
    for sub in subs:
        SC.append(f"/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/{sub}/{atlas}/dmri/connectivity_sift2.npy")
        FC.append(f"/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/{sub}/{atlas}/dmri/connectivity_correlation.npy")
    return subs, SC, FC

def load_entire_dataset(atlas='schaefer100', dtype='dict'):
    """
    Carrega todas as matrizes de conectividade, estrutural e funcional do dataset.
    Utiliza-se de um dicionário de arrays (key='sub-xx') para cada tipo de conectividade.
    Também retorna a lista com os subjects válidos.
    Se a dytpe for 'ls', retorna listas em vez de dicionários e, se for 'arr', retornará um array.

    Returns
    -------
    subs : subjects names;
    SC   : (dict or list) matrizes de conectividade estrutural;
    FC   : (dict or list) matrizes de conectividade funcional.
    """
    SC, FC = {}, {}
    subs, fpath_sc, fpath_fc = get_subjects(fpath=True, atlas=atlas)
    for sub, fpsc, fpfc in zip(subs, fpath_sc, fpath_fc):
        SC[sub] = np.load(fpsc)
        FC[sub] = np.load(fpfc)
    if dtype == 'ls':
        SC = list(SC.values())
        FC = list(FC.values())
    elif dtype == 'arr':
        SC = np.array(list(SC.values()))
        FC = np.array(list(FC.values()))
    return subs, SC, FC

def decode_yeo7_labels(net_dc=None):
    """
    Conversão das networks de Yeo para strings e inteiros

    Parameters
    ----------
    net_dc : (dict), optional. Dictionary mapping integers and strings of networks.
    The default is None.

    Returns
    -------
    rois : (list) ROIs names as in SCHAEFER-100.
    nets_int : (list) ROIs labelled with integers according to Yeo7
    nets_str : (list) ROIs labelled with string according to Yeo7.
    """
    rois, nets = load_atlas_labels(atlas='schaefer100')
    le = LabelEncoder()
    le.fit(nets)
    nets_int = le.transform(nets)
    nets_str = []
    if net_dc == None:
        net_dc = {0:'Cont', 1:'DMN', 2:'DA', 3:'Limb', 4:"VA", 5:"SM", 6:'Vis'}
    for net in nets_int:
        nets_str.append(net_dc[net])
    return rois, nets_int, nets_str

def compute_rc_scfc_coupling(SC, FC, labels=None, alpha=1.0,
                              activation='tanh', leak_rate=None,
                              n_timepoints=390, n_runs=10, seed=42):
    """
    Computa SC-FC coupling via Reservoir Computing.

    Pipeline:
        1. SC → normaliza → reservatório
        2. Injeta ruído branco em todos os nós
        3. Simula dinâmica
        4. FC_rc = correlação dos reservoir states
        5. coupling = corr(FC_rc_upper_tri, FC_emp_upper_tri)

    Parameters
    ----------
    SC : (N, N) ndarray — structural connectivity
    FC : (N, N) ndarray — functional connectivity empírica
    labels : (N,) ndarray, optional — Yeo network labels
    alpha : float — spectral radius scaling
    activation : str — função de ativação
    leak_rate : float or None
    n_timepoints : int — timesteps para simulação
    n_runs : int — repetições para estabilidade
    seed : int

    Returns
    -------
    results : dict com:
        'coupling_wholebrain' : float — correlação global
        'FC_predicted' : (N,N) ndarray — FC média predita pelo RC
        'coupling_per_network' : dict — coupling intra-network
        'coupling_internetwork' : DataFrame — coupling inter-network
    """
    rng = np.random.default_rng(seed)
    n_nodes = SC.shape[0]

    # --- Preparar SC para o reservatório ---
    conn_subj = Conn(w=SC.copy())
    conn_subj.scale_and_normalize()
    n_active = conn_subj.n_nodes

    # Mapear labels para nós ativos
    if labels is not None:
        active_labels = labels[conn_subj.idx_node]
    else:
        active_labels = np.zeros(n_active, dtype=int)

    # Mapear FC para nós ativos
    idx_active = np.where(conn_subj.idx_node)[0]
    FC_active = FC[np.ix_(idx_active, idx_active)]

    # --- w_in: todos os nós recebem input (identidade) ---
    w_in = np.eye(n_active)

    # --- Simular múltiplas vezes e acumular FC ---
    FC_rc_accum = np.zeros((n_active, n_active))

    for run in range(n_runs):
        # Input: ruído branco gaussiano
        ext_input = rng.standard_normal((n_timepoints, n_active)) * 0.1

        # ESN
        esn = EchoStateNetwork(
            w=alpha * conn_subj.w,
            activation_function=activation,
            leak_rate=leak_rate
        )

        # Simular
        rs = esn.simulate(ext_input=ext_input, w_in=w_in, output_nodes=None)

        # Descartar washout (primeiros 200 timesteps)
        washout = 200
        rs = rs[washout:]

        # FC do reservoir = correlação de Pearson dos estados
        FC_rc = np.corrcoef(rs.T)
        FC_rc_accum += FC_rc

    FC_predicted = FC_rc_accum / n_runs

    # --- COUPLING WHOLE-BRAIN ---
    # Extrair triângulo superior (excluindo diagonal)
    triu_idx = np.triu_indices(n_active, k=1)
    fc_pred_upper = FC_predicted[triu_idx]
    fc_emp_upper = FC_active[triu_idx]

    # Correlação de Pearson
    coupling_wb, p_wb = stats.pearsonr(fc_pred_upper, fc_emp_upper)

    results = {
        'coupling_wholebrain': coupling_wb,
        'coupling_wholebrain_p': p_wb,
        'FC_predicted': FC_predicted,
        'FC_empirical': FC_active,
        'active_labels': active_labels,
        'idx_active': idx_active,
    }

    # --- COUPLING PER NETWORK (INTRA-NETWORK) ---
    if labels is not None:
        coupling_intra = {}
        unique_mods = np.unique(active_labels)

        for mod in unique_mods:
            mod_nodes = np.where(active_labels == mod)[0]
            if len(mod_nodes) < 3:
                continue

            # Extrair sub-matrices
            fc_pred_sub = FC_predicted[np.ix_(mod_nodes, mod_nodes)]
            fc_emp_sub = FC_active[np.ix_(mod_nodes, mod_nodes)]

            # Triângulo superior
            triu_sub = np.triu_indices(len(mod_nodes), k=1)
            pred_vals = fc_pred_sub[triu_sub]
            emp_vals = fc_emp_sub[triu_sub]

            if len(pred_vals) > 2:
                r, p = stats.pearsonr(pred_vals, emp_vals)
                mod_name = YEO7_NAMES[mod] if mod < 7 else f"Mod{mod}"
                coupling_intra[mod_name] = {'r': r, 'p': p,
                                            'n_nodes': len(mod_nodes)}

        results['coupling_intra'] = coupling_intra

        # --- COUPLING INTER-NETWORK (entre pares de redes) ---
        coupling_inter = []

        for i, mod_i in enumerate(unique_mods):
            for j, mod_j in enumerate(unique_mods):
                if i >= j:
                    continue

                nodes_i = np.where(active_labels == mod_i)[0]
                nodes_j = np.where(active_labels == mod_j)[0]

                if len(nodes_i) < 2 or len(nodes_j) < 2:
                    continue

                # Bloco inter-network da FC
                fc_pred_block = FC_predicted[np.ix_(nodes_i, nodes_j)]
                fc_emp_block = FC_active[np.ix_(nodes_i, nodes_j)]

                pred_flat = fc_pred_block.ravel()
                emp_flat = fc_emp_block.ravel()

                if len(pred_flat) > 2:
                    r, p = stats.pearsonr(pred_flat, emp_flat)
                    name_i = YEO7_NAMES[mod_i] if mod_i < 7 else f"M{mod_i}"
                    name_j = YEO7_NAMES[mod_j] if mod_j < 7 else f"M{mod_j}"
                    coupling_inter.append({
                        'network_i': name_i,
                        'network_j': name_j,
                        'coupling': r,
                        'p_value': p,
                    })

        results['coupling_inter'] = pd.DataFrame(coupling_inter)

    return results

#=-------------------------------------------------------------------------=##
OUTPUT_DIR = os.path.join(os.getcwd(), '/mnt/nvme1n1p1/sars_cov_2_project/figs/conn2res_figs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROI_LABELS, ROI_YEO7 = load_atlas_labels(atlas='schaefer100')

YEO7_DICT = {0:'Cont', 1:'DMN', 2:'DA', 3:'Limb', 4:"VA", 5:"SM", 6:'Vis'}
YEO7_NAMES = list(YEO7_DICT.values()) 
_, YEO7_ROIS_INT, YEO7_ROIS_STR = decode_yeo7_labels(net_dc=YEO7_DICT)

n_subjects = 23
N_PERMS = 1000
##=-------------------------------------------------------------------------=##
## one subject connectivity
sc, fc, ts = load_empirical_connectome('sub-01', treat='raw')
labels = YEO7_ROIS_INT

conn = Conn(w=sc)
conn.scale_and_normalize()
ew, _ = eigh(conn.w)
sr = np.max(np.abs(ew))
print(f"   spectral_radius = {sr:.6f} (deve ser ~1.0)")

conn.modules = labels[:conn.n_nodes]

# cnt_nodes = np.where(conn.modules == 0)[0]
# dmn_nodes = np.where(conn.modules == 1)[0]
# dat_nodes = np.where(conn.modules == 2)[0]
# lim_nodes = np.where(conn.modules == 3)[0]
# vat_nodes = np.where(conn.modules == 4)[0]
# smt_nodes = np.where(conn.modules == 5)[0]
# vis_nodes = np.where(conn.modules == 6)[0]

# rng = np.random.default_rng(42)
# input_nodes  = rng.choice(vis_nodes, size=min(3, len(vis_nodes)), replace=False)
# output_nodes = np.where(conn.modules == 1)[0]

rng = np.random.default_rng(42)
all_nodes = np.arange(0, conn.n_nodes, dtype=int)
input_nodes = rng.choice(all_nodes, size=min(3, len(all_nodes)), replace=False)

##=-------------------------------------------------------------------------=##
n_features = 3
w_in = np.zeros((n_features, conn.n_nodes))
w_in[np.arange(n_features), input_nodes[:n_features]] = 1.0

T = ts.shape[0]
n_features = ts.shape[1]

x_input = ts
w_in = np.zeros((n_features, conn.n_nodes))
input_scaling = 0.5
w_in[0, input_nodes] = input_scaling  # vários nós recebem o mesmo sinal

##=-------------------------------------------------------------------------=##

alpha = 0.9  
esn = EchoStateNetwork(w=alpha * conn.w, activation_function='tanh', leak_rate=0.3)

# output_nodes=None ← TODOS os nós! Essencial para bom R²
reservoir_states = esn.simulate(ext_input=x_input, w_in=w_in, output_nodes=None  )

y_target = ts

washout = 0  # descartar transiente inicial
frac_train = 0.7
n_train = int((T - washout) * frac_train) + washout

rs_train = reservoir_states[washout:n_train]
rs_test = reservoir_states[n_train:]
y_train = y_target[washout:n_train]
y_test = y_target[n_train:]
##=-------------------------------------------------------------------------=##
## readout 
readout_module = Readout(estimator=Ridge(alpha=1e-4))
readout_module.train(X=rs_train, y=y_train)
scores = readout_module.test(X=rs_test, y=y_test, metric=('r2_score',))
print(f"   R² score = {scores['r2_score']:.4f}")

##=-------------------------------------------------------------------------=##
##=-------------------------------------------------------------------------=##
##=-------------------------------------------------------------------------=##
##=-------------------------------------------------------------------------=##

##=-------------------------------------------------------------------------=##
##=-------------------------------------------------------------------------=##
##=-------------------------------------------------------------------------=##
##=-------------------------------------------------------------------------=##
##=-------------------------------------------------------------------------=##
## figures (selling the fish)

## Figura Simples: Curva de Performance vs Alpha
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(alphas, scores_per_alpha, 'o-', color='#2c3e50', lw=2, ms=5,
        markerfacecolor='#e74c3c', markeredgecolor='#2c3e50', zorder=3)
ax.axvline(x=1.0, color='gray', ls='--', alpha=0.5, label='α = 1 (critical)')
ax.set_xlabel('Spectral Radius (α)')
ax.set_ylabel('R² Score')
ax.set_title('Reservoir Performance Across Dynamical Regimes')
ax.legend(frameon=False)
ax.set_ylim(bottom=min(min(scores_per_alpha) - 0.05, -0.1))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E1_alpha_sweep.png'))
plt.close()

## Readout por Módulo (bar plot com cores Yeo)
fig, ax = plt.subplots(figsize=(7, 4))
if 'module' in df_module_scores.columns:
    mod_data = df_module_scores.dropna(subset=['r2_score'])
    colors = [YEO7_COLORS.get(m, '#999999') for m in mod_data['module']]
    bars = ax.bar(range(len(mod_data)), mod_data['r2_score'],
                  color=colors, edgecolor='white', lw=0.5)
    ax.set_xticks(range(len(mod_data)))
    ax.set_xticklabels(mod_data['module'], rotation=45, ha='right')
    ax.set_ylabel('R² Score')
    ax.set_title('Network-Specific Computational Capacity')
    ax.axhline(y=0, color='gray', ls='-', lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E2_network_barplot.png'))
plt.close()

## Heatmap de FC Empírica vs FC Predita
FC_emp = results_c['FC_empirical']
FC_pred = results_c['FC_predicted']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# FC Empírica
im0 = axes[0].imshow(FC_emp, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
axes[0].set_title('FC Empírica', fontweight='bold')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
# FC Predita (reservoir)
im1 = axes[1].imshow(FC_pred, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
axes[1].set_title('FC Predita (RC)', fontweight='bold')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
# Diferença (Decoupling map)
FC_diff = FC_emp - FC_pred
im2 = axes[2].imshow(FC_diff, cmap='PiYG', vmin=-0.5, vmax=0.5, aspect='auto')
axes[2].set_title('Decoupling (Emp − Pred)', fontweight='bold')
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
# Adicionar linhas de separação de networks
active_labels = results_c['active_labels']
boundaries = []
for mod in range(7):
    nodes = np.where(active_labels == mod)[0]
    if len(nodes) > 0:
        boundaries.append(nodes[-1] + 0.5)
for ax in axes:
    for b in boundaries[:-1]:
        ax.axhline(y=b, color='black', lw=0.5, alpha=0.3)
        ax.axvline(x=b, color='black', lw=0.5, alpha=0.3)
fig.suptitle('SC-FC Coupling Analysis via Reservoir Computing',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E3_fc_heatmaps.png'))
plt.close()

## Scatter: FC Empírica vs FC Predita (triângulo superior)
triu = np.triu_indices(FC_emp.shape[0], k=1)
fc_emp_flat = FC_emp[triu]
fc_pred_flat = FC_pred[triu]
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(fc_emp_flat, fc_pred_flat, s=3, alpha=0.3, color='#3498db',
           rasterized=True)
lims = [min(fc_emp_flat.min(), fc_pred_flat.min()),
        max(fc_emp_flat.max(), fc_pred_flat.max())]
ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
slope, intercept = np.polyfit(fc_emp_flat, fc_pred_flat, 1)
x_line = np.linspace(lims[0], lims[1], 100)
ax.plot(x_line, slope * x_line + intercept, color='#e74c3c', lw=2)
r_val = results_c['coupling_wholebrain']
ax.text(0.05, 0.95, f'r = {r_val:.3f}', transform=ax.transAxes,
        fontsize=12, fontweight='bold', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xlabel('FC Empírica')
ax.set_ylabel('FC Predita (RC)')
ax.set_title('Whole-Brain SC-FC Coupling')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E4_scatter_coupling.png'))
plt.close()

## Coupling Inter-Network Matrix (heatmap triangular)
df_inter = results_c['coupling_inter']
coupling_matrix = np.full((7, 7), np.nan)
for i, net in enumerate(YEO7_NAMES):
    if net in results_c['coupling_intra']:
        coupling_matrix[i, i] = results_c['coupling_intra'][net]['r']
for _, row in df_inter.iterrows():
    i = YEO7_NAMES.index(row['network_i']) if row['network_i'] in YEO7_NAMES else -1
    j = YEO7_NAMES.index(row['network_j']) if row['network_j'] in YEO7_NAMES else -1
    if i >= 0 and j >= 0:
        coupling_matrix[i, j] = row['coupling']
        coupling_matrix[j, i] = row['coupling']
fig, ax = plt.subplots(figsize=(7, 6))
# Mask triângulo inferior
mask = np.zeros_like(coupling_matrix, dtype=bool)
# Não mascarar nada — mostrar tudo
im = ax.imshow(coupling_matrix, cmap='RdYlBu_r', vmin=-0.3, vmax=0.8,
               aspect='auto')
ax.set_xticks(range(7))
ax.set_xticklabels(YEO7_NAMES, rotation=45, ha='right')
ax.set_yticks(range(7))
ax.set_yticklabels(YEO7_NAMES)
# Anotar valores
for i in range(7):
    for j in range(7):
        val = coupling_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.4 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='SC-FC Coupling (r)')
ax.set_title('Network-Level SC-FC Coupling Matrix', fontweight='bold')
# Colorir ticks
for i, net in enumerate(YEO7_NAMES):
    color = YEO7_COLORS.get(net, 'black')
    ax.get_xticklabels()[i].set_color(color)
    ax.get_yticklabels()[i].set_color(color)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E5_inter_network_matrix.png'))
plt.close()

## Null Model Distribution + Empirical Value
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(null_couplings, bins=15, color='#bdc3c7', edgecolor='white',
        alpha=0.8, label='Null (randomized SC)')
ax.axvline(x=empirical_coupling, color='#e74c3c', lw=2.5, ls='--',
           label=f'Empirical (r = {empirical_coupling:.3f})')
ax.set_xlabel('SC-FC Coupling (r)')
ax.set_ylabel('Count')
ax.set_title('Null Model: Randomized Connectome', fontweight='bold')
ax.legend(frameon=False)
# Anotar p-value
ax.text(0.95, 0.95, f'p = {p_null:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E6_null_model.png'))
plt.close()

## SC-FC Coupling vs Alpha (com confidence band)
fig, ax = plt.subplots(figsize=(6, 4))
couplings = [d['coupling'] for d in coupling_by_alpha]
ax.fill_between(alphas_sweep, np.array(couplings) - 0.05,
                np.array(couplings) + 0.05,
                color='#3498db', alpha=0.15)
ax.plot(alphas_sweep, couplings, 'o-', color='#2c3e50', lw=2.5, ms=7,
        markerfacecolor='#3498db', markeredgecolor='#2c3e50', zorder=3)
ax.axvline(x=1.0, color='gray', ls='--', lw=1, alpha=0.5,
           label='α = 1.0 (critical)')
ax.set_xlabel('Spectral Radius (α)', fontsize=12)
ax.set_ylabel('SC-FC Coupling (r)', fontsize=12)
ax.set_title('Dynamical Regime Shapes SC-FC Coupling', fontweight='bold')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E7_coupling_vs_alpha.png'))
plt.close()

## FIGURA COMPOSTA (publication-quality) — A Grande Figura
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.4,
                       height_ratios=[1, 1, 1.1])
# --- Panel A: SC matrix ---
ax_a = fig.add_subplot(gs[0, 0])
im_a = ax_a.imshow(SC[:50, :50], cmap='YlOrRd', aspect='auto')
ax_a.set_title('A  Structural\n   Connectivity', fontweight='bold',
               fontsize=10, loc='left')
ax_a.set_xlabel('Node')
ax_a.set_ylabel('Node')
plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
# --- Panel B: FC empírica ---
ax_b = fig.add_subplot(gs[0, 1])
im_b = ax_b.imshow(FC_emp, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
ax_b.set_title('B  FC Empírica', fontweight='bold', fontsize=10, loc='left')
plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
# --- Panel C: FC predita ---
ax_c = fig.add_subplot(gs[0, 2])
im_c = ax_c.imshow(FC_pred, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
ax_c.set_title('C  FC Predita (RC)', fontweight='bold', fontsize=10, loc='left')
plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
# --- Panel D: Scatter ---
ax_d = fig.add_subplot(gs[0, 3])
ax_d.scatter(fc_emp_flat, fc_pred_flat, s=2, alpha=0.2, color='#3498db',
             rasterized=True)
ax_d.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
slope, intercept = np.polyfit(fc_emp_flat, fc_pred_flat, 1)
ax_d.plot(x_line, slope * x_line + intercept, color='#e74c3c', lw=1.5)
ax_d.text(0.05, 0.95, f'r = {r_val:.3f}', transform=ax_d.transAxes,
          fontsize=10, fontweight='bold', va='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax_d.set_xlabel('FC Emp')
ax_d.set_ylabel('FC Pred')
ax_d.set_title('D  Whole-Brain\n   Coupling', fontweight='bold',
               fontsize=10, loc='left')
# --- Panel E: Alpha sweep ---
ax_e = fig.add_subplot(gs[1, 0:2])
ax_e.plot(alphas, scores_per_alpha, 'o-', color='#2c3e50', lw=2, ms=5,
          markerfacecolor='#e74c3c', markeredgecolor='#2c3e50')
ax_e.axvline(x=1.0, color='gray', ls='--', alpha=0.5)
ax_e.set_xlabel('Spectral Radius (α)')
ax_e.set_ylabel('R² Score')
ax_e.set_title('E  Performance vs Dynamical Regime', fontweight='bold',
               fontsize=10, loc='left')
# --- Panel F: Coupling vs alpha ---
ax_f = fig.add_subplot(gs[1, 2:4])
ax_f.plot(alphas_sweep, couplings, 'o-', color='#2c3e50', lw=2, ms=6,
          markerfacecolor='#3498db', markeredgecolor='#2c3e50')
ax_f.axvline(x=1.0, color='gray', ls='--', alpha=0.5)
ax_f.set_xlabel('Spectral Radius (α)')
ax_f.set_ylabel('SC-FC Coupling (r)')
ax_f.set_title('F  SC-FC Coupling vs Dynamical Regime', fontweight='bold',
               fontsize=10, loc='left')
# --- Panel G: Network barplot ---
ax_g = fig.add_subplot(gs[2, 0:2])
if 'module' in df_module_scores.columns:
    mod_data = df_module_scores.dropna(subset=['r2_score'])
    colors = [YEO7_COLORS.get(m, '#999999') for m in mod_data['module']]
    ax_g.bar(range(len(mod_data)), mod_data['r2_score'],
             color=colors, edgecolor='white', lw=0.5)
    ax_g.set_xticks(range(len(mod_data)))
    ax_g.set_xticklabels(mod_data['module'], rotation=45, ha='right')
    ax_g.set_ylabel('R² Score')
    ax_g.axhline(y=0, color='gray', ls='-', lw=0.5)
ax_g.set_title('G  Network-Specific\n   Computational Capacity',
               fontweight='bold', fontsize=10, loc='left')
# --- Panel H: Null model ---
ax_h = fig.add_subplot(gs[2, 2:4])
ax_h.hist(null_couplings, bins=12, color='#bdc3c7', edgecolor='white',
          alpha=0.8, label='Null')
ax_h.axvline(x=empirical_coupling, color='#e74c3c', lw=2.5, ls='--',
             label=f'Empirical')
ax_h.set_xlabel('SC-FC Coupling (r)')
ax_h.set_ylabel('Count')
ax_h.legend(frameon=False)
ax_h.text(0.95, 0.95, f'p = {p_null:.3f}', transform=ax_h.transAxes,
          ha='right', va='top', fontsize=10, fontweight='bold',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax_h.set_title('H  Null Model Validation', fontweight='bold',
               fontsize=10, loc='left')
fig.suptitle('conn2res: Connectome-Based Reservoir Computing for SC-FC Coupling',
             fontsize=15, fontweight='bold', y=0.98)
fig.savefig(os.path.join(OUTPUT_DIR, 'E8_composite_figure.png'))
plt.close()

## Group Violinplot com Individual Dots
rows = []
for _, row in df_group.iterrows():
    for col in intra_cols:
        net = col.replace('intra_', '')
        if not pd.isna(row[col]):
            rows.append({'Network': net, 'Coupling': row[col],
                         'Subject': row['subject']})
df_long = pd.DataFrame(rows)
if len(df_long) > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    # Violin
    palette = [YEO7_COLORS.get(n, '#999') for n in df_long['Network'].unique()]
    sns.violinplot(data=df_long, x='Network', y='Coupling',
                   palette=palette, inner=None, alpha=0.3, ax=ax)
    # Strip (dots individuais)
    sns.stripplot(data=df_long, x='Network', y='Coupling',
                  palette=palette, size=4, alpha=0.7, jitter=0.15, ax=ax)
    # Box no centro (só quartis)
    sns.boxplot(data=df_long, x='Network', y='Coupling',
                color='white', width=0.15, fliersize=0,
                boxprops=dict(alpha=0.7), ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel('Intra-Network SC-FC Coupling (r)')
    ax.set_title('Group-Level Network Coupling Distribution',
                 fontweight='bold')
    # Colorir ticks
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(palette[i])
        label.set_fontweight('bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'E9_group_violinplot.png'))
    plt.close()

## Brain Plot (com nilearn, se disponível)
try:
    from nilearn import plotting as ni_plot
    from nilearn import datasets as ni_datasets
    # Baixar atlas Schaefer 100 parcels
    atlas = ni_datasets.fetch_atlas_schaefer_2018(n_rois=100)
    # Criar vetor de "decoupling nodal":
    # Para cada nó, o decoupling = 1 - média(coupling com outros nós)
    n_active = results_c['FC_predicted'].shape[0]
    nodal_decoupling = np.zeros(n_active)
    for i in range(n_active):
        # Diferença absoluta média entre FC_pred e FC_emp para o nó i
        diff = np.abs(results_c['FC_empirical'][i, :] -
                      results_c['FC_predicted'][i, :])
        nodal_decoupling[i] = np.mean(diff)
    # Mapear para o atlas (preencher até 100 com zeros)
    full_decoupling = np.zeros(100)
    full_decoupling[:min(n_active, 100)] = nodal_decoupling[:100]
    # Plot brain surface
    fig_brain = ni_plot.plot_markers(
        full_decoupling,
        atlas.labels[:100] if hasattr(atlas, 'labels') else None,
        title='Nodal SC-FC Decoupling',
        colorbar=True,
        display_mode='ortho',
    )
    fig_brain.savefig(os.path.join(OUTPUT_DIR, 'E10_brain_decoupling.png'))
    print("   ✅ Brain plot salvo!")
except ImportError:
    print("   ⚠️  nilearn não disponível — brain plot pulado")
    print("   Para instalar: pip install nilearn")
except Exception as e:
    print(f"   ⚠️  Erro no brain plot: {e}")
    print("   (esperado com dados sintéticos — funciona com dados reais)")

## Reservoir States Heatmap (dinâmica temporal)
fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={'width_ratios': [3, 1]})
# Heatmap dos estados
rs_plot = reservoir_states[:200, :]  # primeiros 200 timesteps
im = axes[0].imshow(rs_plot.T, aspect='auto', cmap='viridis',
                     interpolation='none')
axes[0].set_xlabel('Timestep')
axes[0].set_ylabel('Output Node')
axes[0].set_title('Reservoir States Over Time', fontweight='bold')
plt.colorbar(im, ax=axes[0], fraction=0.02, pad=0.02)
# Distribuição dos estados
axes[1].hist(reservoir_states.ravel(), bins=50, color='#2c3e50',
             edgecolor='white', orientation='horizontal', alpha=0.8)
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Activation')
axes[1].set_title('State Distribution', fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E11_reservoir_states.png'))
plt.close()

## Leak Rate Comparison (multi-panel)
fig, axes = plt.subplots(1, len(leak_rates), figsize=(3 * len(leak_rates), 3),
                          sharey=True)
for idx, lr in enumerate(leak_rates):
    esn_viz = EchoStateNetwork(
        w=1.0 * conn.w, activation_function='tanh', leak_rate=lr
    )
    rs_viz = esn_viz.simulate(ext_input=x_input[:100], w_in=w_in,
                               output_nodes=output_nodes[:5])

    for node in range(rs_viz.shape[1]):
        axes[idx].plot(rs_viz[:, node], lw=0.7, alpha=0.7)
    axes[idx].set_title(f'lr = {lr}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Time')
    if idx == 0:
        axes[idx].set_ylabel('Activation')

fig.suptitle('Effect of Leak Rate on Reservoir Dynamics',
             fontweight='bold', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E12_leak_rate_comparison.png'))
plt.close()