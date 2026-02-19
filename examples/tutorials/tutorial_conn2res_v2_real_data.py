#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  CONN2RES TUTORIAL v2 — DADOS REAIS (SARS-CoV-2 ICU Survivors)
================================================================================

  Autor:  Velho Mago — INNT/UFRJ, PhD Neurociência Translacional
  Data:   Fevereiro 2026
  Ref:    Suárez et al. (2024) Nat Commun 15:656
          https://github.com/netneurolab/conn2res

  Dados:  Schaefer-100 (Yeo 7-network), 23 sujeitos
          SC = SIFT2 tractography  (100×100)
          FC = Pearson correlation  (100×100)
          TS = rs-fMRI timeseries   (390×100)

  Seções:
    A — Pipeline Quick-Start com conectoma real
    B — Truques Avançados (alpha sweep, network readout, leak rate, SC vs rand)
    C — SC-FC Coupling via Reservoir Computing (sujeito único)
    D — Análise de Grupo + Null Models (23 sujeitos)
    E — Visualizações Publication-Quality (14 figuras inc. brain surface)

  NOTA: Execute no Spyder com working dir em:
        /mnt/nvme1n1p1/sars_cov_2_project/code/current

  ⚠️  BUGFIXES aplicados em relação ao conn2res_results.py original:
      1. SC/FC paths estavam TROCADOS em load_empirical_connectome()
      2. FC path apontava para dmri/ em get_subjects() — deveria ser fmri/
      3. w_in era sobrescrito com shapes incompatíveis
      4. metric em run_task() deve ser LIST, não tuple (bug do conn2res)
================================================================================
"""

# %% ========================================================================
#  IMPORTS
# ===========================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from scipy import stats
from scipy.linalg import eigh
from scipy.spatial.distance import squareform

from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix
from reservoirpy.datasets import mackey_glass

from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork
from conn2res.readout import Readout

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 900)


# %% ========================================================================
#  CONFIGURAÇÃO GLOBAL
# ===========================================================================

DATA_ROOT  = "/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs"
INFO_DIR   = os.path.join(DATA_ROOT, "info")
OUTPUT_DIR = "/mnt/nvme1n1p1/sars_cov_2_project/figs/conn2res_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATLAS   = "schaefer100"
N_NODES = 100
N_TRS   = 390

# Subjects válidos (sub-21 excluído)
ALL_SUBS = [f"sub-{n:02d}" for n in range(1, 25)]
ALL_SUBS.remove("sub-21")
N_SUBJECTS = len(ALL_SUBS)   # 23

SEED = 42
rng  = np.random.default_rng(SEED)


# %% ========================================================================
#  ESTILO DE FIGURAS — JOURNAL-QUALITY
# ===========================================================================
# Calibrado para: coluna = 89 mm (~3.5 in), página = 183 mm (~7.2 in).
# Type-42 embedding garante texto editável em Illustrator/Inkscape.

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
    'font.size':          8,
    'axes.labelsize':     9,
    'axes.titlesize':     10,
    'xtick.labelsize':    7.5,
    'ytick.labelsize':    7.5,
    'legend.fontsize':    7.5,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
    'axes.grid':          False,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.6,
    'xtick.major.width':  0.6,
    'ytick.major.width':  0.6,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'lines.linewidth':    1.2,
    'lines.markersize':   4,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

# --- Paleta Yeo 7 Networks ---
# ⚠️ LabelEncoder ordena ALFABETICAMENTE os nomes do arquivo networks_schaefer100.txt.
# A ordem típica é: 0=Cont, 1=Default, 2=DorsAttn, 3=Limbic, 4=SalVentAttn, 5=SomMot, 6=Vis
# Mas isso DEPENDE do conteúdo exato do arquivo. O código abaixo detecta automaticamente.

YEO7_DISPLAY = {
    'Cont': 'FP', 'Default': 'DMN', 'DorsAttn': 'DA',
    'Limbic': 'Lim', 'SalVentAttn': 'VA', 'SomMot': 'SM', 'Vis': 'Vis'
}

YEO7_COLORS = {
    'FP':  '#E69422',
    'DMN': '#CD3E4E',
    'DA':  '#00760E',
    'Lim': '#DCF8A4',
    'VA':  '#C43AFA',
    'SM':  '#4682B4',
    'Vis': '#781286',
}


# %% ========================================================================
#  FUNÇÕES DE I/O E UTILIDADES
# ===========================================================================

def load_atlas_info(atlas='schaefer100'):
    """
    Carrega labels das ROIs e Yeo7 network assignments.
    A conversão para inteiros usa LabelEncoder (ordena alfabeticamente).

    Returns
    -------
    roi_labels : (N,) str array — nomes das ROIs
    net_int : (N,) int array — inteiro 0–6 por ROI
    net_str : (N,) str array — abreviação (FP, DMN, DA, ...) por ROI
    int_to_short : dict — mapeia inteiro → abreviação
    """
    roi_labels = np.loadtxt(
        os.path.join(INFO_DIR, f"labels_{atlas}.txt"), dtype=str
    )
    nets_raw = np.loadtxt(
        os.path.join(INFO_DIR, "networks_schaefer100.txt"), dtype=str
    )
    le = LabelEncoder()
    net_int = le.fit_transform(nets_raw)

    # Mapear automaticamente: detecta o nome canônico de cada inteiro
    int_to_short = {}
    for i, class_name in enumerate(le.classes_):
        short = YEO7_DISPLAY.get(class_name, class_name[:3].upper())
        int_to_short[i] = short

    net_str = np.array([int_to_short[i] for i in net_int])
    return roi_labels, net_int, net_str, int_to_short


def load_subject_data(sub, atlas='schaefer100', sc_treat='log'):
    """
    Carrega SC (SIFT2), FC (correlação) e timeseries de um sujeito.

    ⚠️  BUGFIX: No código original, SC e FC estavam com paths TROCADOS!
         SC vem de dmri/, FC vem de fmri/.

    Parameters
    ----------
    sub : str — e.g. 'sub-01'
    sc_treat : str — 'raw', 'log' (np.log1p), ou 'max' (SC/max)

    Returns
    -------
    SC : (N, N) ndarray — structural connectivity
    FC : (N, N) ndarray — functional connectivity
    TS : (T, N) ndarray — rs-fMRI timeseries
    """
    base = os.path.join(DATA_ROOT, sub, atlas)

    # ⚠️ SC = dMRI, FC = fMRI (NÃO TROCAR!)
    SC = np.load(os.path.join(base, "dmri", "connectivity_sift2.npy"))
    FC = np.load(os.path.join(base, "fmri", "connectivity_correlation.npy"))
    TS = np.load(os.path.join(base, "fmri", "timeseries.npy"))

    # Simetrizar e zerar diagonal
    np.fill_diagonal(SC, 0)
    np.fill_diagonal(FC, 0)
    SC = (SC + SC.T) / 2
    FC = (FC + FC.T) / 2

    # Tratar NaN/Inf
    for mtx in [SC, FC]:
        if np.any(~np.isfinite(mtx)):
            imp = SimpleImputer(strategy='median')
            mtx[:] = imp.fit_transform(mtx)

    # Tratamento da SC
    if sc_treat == 'log':
        SC = np.log1p(SC)
    elif sc_treat == 'max':
        if np.max(SC) > 0:
            SC = SC / np.max(SC)

    return SC, FC, TS


def get_readout_modules(labels, n_active, idx_node=None, int_to_short=None):
    """
    Constrói dict {name: node_indices} para Readout.run_task(readout_modules=...).
    """
    if idx_node is not None:
        active_labels = labels[idx_node]
    else:
        active_labels = labels[:n_active]

    modules = {}
    for mod in np.unique(active_labels):
        name = int_to_short[mod] if int_to_short else f"M{mod}"
        modules[name] = np.where(active_labels == mod)[0]
    return modules


def compute_rc_scfc_coupling(SC, FC, labels=None, alpha=1.0,
                              activation='tanh', leak_rate=None,
                              n_timepoints=390, n_runs=10, washout=200,
                              input_noise_std=0.1, seed=42,
                              int_to_short=None):
    """
    Computa SC-FC coupling via Reservoir Computing (Suárez et al. 2024).

    Pipeline:
      1. SC → Conn() → scale_and_normalize() → SR = 1.0
      2. w = alpha × SC_norm
      3. w_in = I (identidade) — todos os nós recebem ruído independente
      4. Simula ESN com ruído branco → reservoir states
      5. FC_rc = corrcoef(states.T) → média sobre n_runs
      6. coupling = pearsonr(triu(FC_rc), triu(FC_emp))

    Returns
    -------
    results : dict com coupling, FC_predicted, FC_empirical, etc.
    """
    rng_local = np.random.default_rng(seed)

    conn_subj = Conn(w=SC.copy())
    conn_subj.scale_and_normalize()
    n_active = conn_subj.n_nodes

    idx_active = np.where(conn_subj.idx_node)[0]
    FC_active = FC[np.ix_(idx_active, idx_active)]
    active_labels = labels[idx_active] if labels is not None else None

    w_in = np.eye(n_active)
    FC_rc_accum = np.zeros((n_active, n_active))

    for run in range(n_runs):
        ext_input = rng_local.standard_normal((n_timepoints, n_active)) * input_noise_std
        esn = EchoStateNetwork(
            w=alpha * conn_subj.w,
            activation_function=activation,
            leak_rate=leak_rate
        )
        rs = esn.simulate(ext_input=ext_input, w_in=w_in, output_nodes=None)
        rs = rs[washout:]
        FC_rc = np.corrcoef(rs.T)
        FC_rc_accum += FC_rc

    FC_predicted = FC_rc_accum / n_runs

    triu_idx = np.triu_indices(n_active, k=1)
    coupling_wb, p_wb = stats.pearsonr(
        FC_predicted[triu_idx], FC_active[triu_idx]
    )

    results = {
        'coupling_wholebrain': coupling_wb,
        'coupling_wholebrain_p': p_wb,
        'FC_predicted': FC_predicted,
        'FC_empirical': FC_active,
        'active_labels': active_labels,
        'idx_active': idx_active,
        'n_active': n_active,
    }

    # Intra-network coupling
    if active_labels is not None:
        short = int_to_short or {}
        coupling_intra = {}
        for mod in np.unique(active_labels):
            nodes = np.where(active_labels == mod)[0]
            if len(nodes) < 3:
                continue
            triu_sub = np.triu_indices(len(nodes), k=1)
            if len(triu_sub[0]) > 2:
                r, p = stats.pearsonr(
                    FC_predicted[np.ix_(nodes, nodes)][triu_sub],
                    FC_active[np.ix_(nodes, nodes)][triu_sub]
                )
                name = short.get(mod, f"M{mod}")
                coupling_intra[name] = {'r': r, 'p': p, 'n_nodes': len(nodes)}
        results['coupling_intra'] = coupling_intra

        # Inter-network coupling
        coupling_inter = []
        unique_mods = np.unique(active_labels)
        for ii, mi in enumerate(unique_mods):
            for jj, mj in enumerate(unique_mods):
                if ii >= jj:
                    continue
                ni = np.where(active_labels == mi)[0]
                nj = np.where(active_labels == mj)[0]
                if len(ni) < 2 or len(nj) < 2:
                    continue
                block_p = FC_predicted[np.ix_(ni, nj)].ravel()
                block_e = FC_active[np.ix_(ni, nj)].ravel()
                if len(block_p) > 2:
                    r, p = stats.pearsonr(block_p, block_e)
                    coupling_inter.append({
                        'network_i': short.get(mi, f"M{mi}"),
                        'network_j': short.get(mj, f"M{mj}"),
                        'coupling': r, 'p_value': p
                    })
        results['coupling_inter'] = pd.DataFrame(coupling_inter)

    return results


# --- Helpers para figuras ---

def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=12):
    """Adiciona label (a, b, c...) no estilo Nature."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')


def add_network_boundaries(ax, labels, lw=0.4, color='k', alpha=0.3):
    """Linhas de separação de networks em heatmaps."""
    for mod in np.unique(labels):
        nodes = np.where(labels == mod)[0]
        if len(nodes) > 0:
            b = nodes[-1] + 0.5
            ax.axhline(y=b, color=color, lw=lw, alpha=alpha)
            ax.axvline(x=b, color=color, lw=lw, alpha=alpha)


def make_network_sidebar(ax, labels, int_to_short, orientation='vertical'):
    """Barra lateral colorida com as networks Yeo7."""
    for mod in np.unique(labels):
        nodes = np.where(labels == mod)[0]
        if len(nodes) > 0:
            name = int_to_short.get(mod, f"M{mod}")
            color = YEO7_COLORS.get(name, '#999')
            ax.axhspan(nodes[0], nodes[-1] + 1, color=color, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(labels))
    ax.invert_yaxis()
    ax.axis('off')


def density_scatter(ax, x, y, cmap='inferno', s=3, alpha=0.6):
    """Scatter com cores por densidade (KDE)."""
    from scipy.stats import gaussian_kde
    try:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        return ax.scatter(x[idx], y[idx], c=z[idx], s=s, alpha=alpha,
                          cmap=cmap, rasterized=True, edgecolors='none')
    except Exception:
        ax.scatter(x, y, s=s, alpha=0.3, color='#3498db', rasterized=True)
        return None


# %% ========================================================================
#  CARREGAR DADOS
# ===========================================================================

print("=" * 70)
print("  CONN2RES v2 — TUTORIAL COM DADOS REAIS")
print("=" * 70)

ROI_LABELS, YEO7_INT, YEO7_STR, INT2SHORT = load_atlas_info(atlas=ATLAS)
YEO7_NAMES = [INT2SHORT[i] for i in range(7)]
YEO7_COLORS_ORD = [YEO7_COLORS[INT2SHORT[i]] for i in range(7)]

print(f"\n  Atlas: {ATLAS}  |  ROIs: {len(ROI_LABELS)}  |  Subjects: {N_SUBJECTS}")
print(f"  Output: {OUTPUT_DIR}\n")
print("  Mapeamento LabelEncoder → Yeo7:")
for i in range(7):
    n = np.sum(YEO7_INT == i)
    print(f"    {i} → {INT2SHORT[i]:>4s}  ({n} ROIs)")


# %% ========================================================================
#  SEÇÃO A — PIPELINE QUICK-START
# ===========================================================================
print("\n" + "=" * 70)
print("  SEÇÃO A — PIPELINE QUICK-START")
print("=" * 70)

# A.1: Carregar conectoma real
sub_ex = 'sub-14'
SC, FC, TS = load_subject_data(sub_ex, sc_treat='log')
print(f"\n[A.1] {sub_ex}: SC {SC.shape}, FC {FC.shape}, TS {TS.shape}")
print(f"   SC range: [{SC.min():.2f}, {SC.max():.2f}]")
print(f"   FC range: [{FC.min():.3f}, {FC.max():.3f}]")

# A.2: Normalizar
conn = Conn(w=SC.copy())
ew_pre, _ = eigh(conn.w)
sr_pre = np.max(np.abs(ew_pre))
print(f"\n[A.2] SR antes: {sr_pre:.2f}")

conn.scale_and_normalize()
ew_post, _ = eigh(conn.w)
sr_post = np.max(np.abs(ew_post))
print(f"   SR depois: {sr_post:.6f}")
print(f"   n_active: {conn.n_nodes} / {N_NODES}")

# A.3: Configurar network modules
labels = YEO7_INT.copy()
conn.modules = labels[:conn.n_nodes]
n_active = conn.n_nodes
idx_active = np.where(conn.idx_node)[0]

# Timeseries dos nós ativos
TS_active = TS[:, idx_active]
T = TS_active.shape[0]

# w_in = identidade escalada — cada ROI recebe sua própria timeseries
# Calibra o scaling para que a entrada não sature a tanh
ts_std = np.std(TS_active)
input_scaling = 0.5 / max(ts_std, 1e-8)
w_in = np.eye(n_active) * input_scaling

print(f"\n[A.3] TS_active: {TS_active.shape}, std = {ts_std:.4f}")
print(f"   input_scaling = {input_scaling:.4f}")

# A.4: Simular ESN
alpha = 0.9
esn = EchoStateNetwork(w=alpha * conn.w, activation_function='tanh',
                        leak_rate=0.3)
reservoir_states = esn.simulate(ext_input=TS_active, w_in=w_in,
                                 output_nodes=None)
print(f"\n[A.4] Reservoir states: {reservoir_states.shape}")

# A.5: Readout — predição 1-step ahead
x_input = TS_active[:-1]
y_target = TS_active[1:]
rs = reservoir_states[:-1]

washout = 50
frac_train = 0.7
n_train = int((T - 1 - washout) * frac_train) + washout

rs_train, rs_test = rs[washout:n_train], rs[n_train:]
y_train, y_test = y_target[washout:n_train], y_target[n_train:]

readout_a = Readout(estimator=Ridge(alpha=1.0))
readout_a.train(X=rs_train, y=y_train)
scores_a = readout_a.test(X=rs_test, y=y_test, metric=('r2_score',))
print(f"\n[A.5] R² (1-step ahead) = {scores_a['r2_score']:.4f}")

# A.6: Benchmark Mackey-Glass
mg = mackey_glass(n_timesteps=T + 50, tau=17)
w_in_mg = np.zeros((1, n_active))
w_in_mg[0, rng.choice(n_active, 10, replace=False)] = 0.3
esn_mg = EchoStateNetwork(w=alpha * conn.w, activation_function='tanh',
                           leak_rate=0.3)
rs_mg = esn_mg.simulate(ext_input=mg[:T-1].reshape(-1, 1),
                          w_in=w_in_mg, output_nodes=None)
rdout_mg = Readout(estimator=Ridge(alpha=1e-4))
rdout_mg.train(X=rs_mg[washout:n_train], y=mg[1:T].reshape(-1,1)[washout:n_train])
sc_mg = rdout_mg.test(X=rs_mg[n_train:],
                       y=mg[1:T].reshape(-1,1)[n_train:],
                       metric=('r2_score',))
print(f"[A.6] Mackey-Glass R² = {sc_mg['r2_score']:.4f}")
print("\n✅ Seção A concluída!")


# %% ========================================================================
#  SEÇÃO B — TRUQUES AVANÇADOS
# ===========================================================================
print("\n" + "=" * 70)
print("  SEÇÃO B — TRUQUES AVANÇADOS")
print("=" * 70)

# B.1: Alpha sweep
print("\n[B.1] Alpha Sweep")
alphas = np.linspace(0.1, 2.0, 20)
scores_per_alpha = []
for a in alphas:
    esn_sw = EchoStateNetwork(w=a * conn.w, activation_function='tanh',
                               leak_rate=0.3)
    rs_sw = esn_sw.simulate(ext_input=TS_active, w_in=w_in, output_nodes=None)
    rs_sw = rs_sw[:-1]
    rdout = Readout(estimator=Ridge(alpha=1.0))
    rdout.train(X=rs_sw[washout:n_train], y=y_target[washout:n_train])
    sc_sw = rdout.test(X=rs_sw[n_train:], y=y_target[n_train:],
                        metric=('r2_score',))
    scores_per_alpha.append(sc_sw['r2_score'])

best_idx = np.argmax(scores_per_alpha)
best_alpha = alphas[best_idx]
print(f"   Best α = {best_alpha:.2f} (R² = {scores_per_alpha[best_idx]:.4f})")

# B.2: Readout por Network
print("\n[B.2] Readout por Network")
readout_modules_dict = get_readout_modules(
    labels, n_active, idx_node=conn.idx_node, int_to_short=INT2SHORT
)
rdout_mod = Readout(estimator=Ridge(alpha=1.0))
# ⚠️ metric DEVE ser LIST (bug conn2res: internamente faz ['module'] + metric)
df_module_scores = rdout_mod.run_task(
    X=rs[washout:], y=y_target[washout:],
    frac_train=0.7, metric=['r2_score'],
    readout_modules=readout_modules_dict
)
print(df_module_scores[['module', 'n_nodes', 'r2_score']].to_string(index=False))

# B.3: Leak rate
print("\n[B.3] Leak Rate Effect")
leak_rates = [0.1, 0.3, 0.5, 0.8, 1.0]
scores_per_lr = []
rs_per_lr = {}
for lr in leak_rates:
    esn_lr = EchoStateNetwork(w=alpha * conn.w, activation_function='tanh',
                               leak_rate=lr)
    rs_lr = esn_lr.simulate(ext_input=TS_active, w_in=w_in, output_nodes=None)
    rs_per_lr[lr] = rs_lr.copy()
    rs_lr = rs_lr[:-1]
    rdout = Readout(estimator=Ridge(alpha=1.0))
    rdout.train(X=rs_lr[washout:n_train], y=y_target[washout:n_train])
    sc_lr = rdout.test(X=rs_lr[n_train:], y=y_target[n_train:],
                        metric=('r2_score',))
    scores_per_lr.append(sc_lr['r2_score'])
    print(f"   lr = {lr:.1f} → R² = {sc_lr['r2_score']:.4f}")

# B.4: Real vs Randomized
print("\n[B.4] SC Real vs Randomizada")
score_real = scores_a['r2_score']
scores_random = []
for i in range(5):
    conn_rand = Conn(w=SC.copy())
    conn_rand.scale_and_normalize()
    conn_rand.randomize(swaps=10)
    idx_r = np.where(conn_rand.idx_node)[0]
    TS_r = TS[:, idx_r]
    w_in_r = np.eye(conn_rand.n_nodes) * input_scaling
    esn_r = EchoStateNetwork(w=alpha * conn_rand.w, activation_function='tanh',
                              leak_rate=0.3)
    rs_r = esn_r.simulate(ext_input=TS_r, w_in=w_in_r, output_nodes=None)
    rs_r = rs_r[:-1]
    y_r = TS_r[1:]
    n_tr = int((TS_r.shape[0] - 1 - washout) * frac_train) + washout
    rdout = Readout(estimator=Ridge(alpha=1.0))
    rdout.train(X=rs_r[washout:n_tr], y=y_r[washout:n_tr])
    sc_r = rdout.test(X=rs_r[n_tr:], y=y_r[n_tr:], metric=('r2_score',))
    scores_random.append(sc_r['r2_score'])

print(f"   Real:   R² = {score_real:.4f}")
print(f"   Random: R² = {np.mean(scores_random):.4f} ± {np.std(scores_random):.4f}")
print("\n✅ Seção B concluída!")


# %% ========================================================================
#  SEÇÃO C — SC-FC COUPLING (SUJEITO ÚNICO)
# ===========================================================================
print("\n" + "=" * 70)
print("  SEÇÃO C — SC-FC COUPLING (Sujeito Único)")
print("=" * 70)

# Para coupling, usar SC raw (sem log) — o log distorce a magnitude dos pesos
SC_raw, FC_raw, _ = load_subject_data(sub_ex, sc_treat='raw')

print(f"\n[C.1] Coupling whole-brain ({sub_ex})...")
t0 = time.time()
results_c = compute_rc_scfc_coupling(
    SC_raw, FC_raw, labels=YEO7_INT, alpha=1.0,
    n_timepoints=N_TRS, n_runs=10, washout=200, seed=SEED,
    int_to_short=INT2SHORT
)
print(f"   r = {results_c['coupling_wholebrain']:.4f} "
      f"(p = {results_c['coupling_wholebrain_p']:.2e})")
print(f"   n_active = {results_c['n_active']}, t = {time.time()-t0:.1f}s")

print(f"\n[C.2] Intra-Network Coupling:")
for net, v in results_c['coupling_intra'].items():
    sig = "***" if v['p'] < 0.001 else "**" if v['p'] < 0.01 else \
          "*" if v['p'] < 0.05 else "ns"
    print(f"   {net:>4s}: r = {v['r']:.4f}  (n = {v['n_nodes']}) {sig}")

df_inter = results_c['coupling_inter'].sort_values('coupling', ascending=False)
print(f"\n[C.3] Inter-Network Coupling (top 5):")
print(df_inter.head(5).to_string(index=False))

print(f"\n[C.4] Alpha Sweep do Coupling:")
alphas_sweep = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
coupling_by_alpha = []
for a in alphas_sweep:
    res_a = compute_rc_scfc_coupling(
        SC_raw, FC_raw, labels=YEO7_INT, alpha=a,
        n_timepoints=N_TRS, n_runs=5, washout=200, seed=SEED,
        int_to_short=INT2SHORT
    )
    coupling_by_alpha.append({
        'alpha': a, 'coupling': res_a['coupling_wholebrain']
    })
    print(f"   α = {a:.1f} → r = {res_a['coupling_wholebrain']:.4f}")

print("\n✅ Seção C concluída!")


# %% ========================================================================
#  SEÇÃO D — ANÁLISE DE GRUPO + NULL MODELS
# ===========================================================================
print("\n" + "=" * 70)
print("  SEÇÃO D — ANÁLISE DE GRUPO + NULL MODELS")
print("=" * 70)

# D.1: Group coupling
print(f"\n[D.1] Group Coupling ({N_SUBJECTS} sujeitos)")
group_results = []
t0 = time.time()
for i, sub in enumerate(ALL_SUBS):
    try:
        sc_s, fc_s, _ = load_subject_data(sub, sc_treat='raw')
        res = compute_rc_scfc_coupling(
            sc_s, fc_s, labels=YEO7_INT, alpha=1.0,
            n_timepoints=N_TRS, n_runs=10, washout=200,
            seed=SEED + i, int_to_short=INT2SHORT
        )
        row = {'subject': sub, 'coupling_wb': res['coupling_wholebrain'],
               'n_active': res['n_active']}
        if 'coupling_intra' in res:
            for net, v in res['coupling_intra'].items():
                row[f'intra_{net}'] = v['r']
        group_results.append(row)
        print(f"   {sub}: r = {res['coupling_wholebrain']:.4f}")
    except Exception as e:
        print(f"   {sub}: ⚠️ {e}")

df_group = pd.DataFrame(group_results)
print(f"\n   Total: {time.time()-t0:.0f}s")
print(f"   WB coupling: {df_group['coupling_wb'].mean():.4f} "
      f"± {df_group['coupling_wb'].std():.4f}")

intra_cols = sorted([c for c in df_group.columns if c.startswith('intra_')])
print("\n   Intra-network (mean ± std):")
for col in intra_cols:
    v = df_group[col].dropna()
    print(f"   {col.replace('intra_',''):>4s}: {v.mean():.4f} ± {v.std():.4f}")

# D.2: Null model — SC randomizada
print(f"\n[D.2] Null Model — SC Randomizada")
N_PERMS_NULL = 100  # ⚠️ para paper use 1000+

# Sujeito mediano
med_sub = df_group.iloc[
    (df_group['coupling_wb'] - df_group['coupling_wb'].median()).abs().argsort()[:1]
]['subject'].values[0]
sc_null, fc_null, _ = load_subject_data(med_sub, sc_treat='raw')

empirical_coupling = compute_rc_scfc_coupling(
    sc_null, fc_null, labels=YEO7_INT, alpha=1.0,
    n_timepoints=N_TRS, n_runs=10, washout=200, seed=SEED,
    int_to_short=INT2SHORT
)['coupling_wholebrain']

null_couplings = []
print(f"   Ref: {med_sub} (r = {empirical_coupling:.4f})")
for perm in range(N_PERMS_NULL):
    conn_n = Conn(w=sc_null.copy())
    conn_n.scale_and_normalize()
    conn_n.randomize(swaps=10)
    # Reconstruir no espaço original
    SC_rnd = np.zeros_like(sc_null)
    idx_a = np.where(conn_n.idx_node)[0]
    for ii, ri in enumerate(idx_a):
        for jj, rj in enumerate(idx_a):
            SC_rnd[ri, rj] = conn_n.w[ii, jj]
    res_n = compute_rc_scfc_coupling(
        SC_rnd, fc_null, labels=YEO7_INT, alpha=1.0,
        n_timepoints=N_TRS, n_runs=3, washout=200,
        seed=SEED + perm, int_to_short=INT2SHORT
    )
    null_couplings.append(res_n['coupling_wholebrain'])
    if (perm + 1) % 25 == 0:
        print(f"     {perm+1}/{N_PERMS_NULL}")

null_couplings = np.array(null_couplings)
p_null = np.mean(null_couplings >= empirical_coupling)
z_null = (empirical_coupling - np.mean(null_couplings)) / max(np.std(null_couplings), 1e-8)
print(f"   Null: {np.mean(null_couplings):.4f} ± {np.std(null_couplings):.4f}")
print(f"   p = {p_null:.4f}, z = {z_null:.2f}")

# D.3: Null model — label permutation
print(f"\n[D.3] Null Model — Label Permutation")
N_PERMS_LABEL = 200
active_labels_c = results_c['active_labels']
FC_pred_c = results_c['FC_predicted']
FC_emp_c = results_c['FC_empirical']

label_null_intra = {net: [] for net in YEO7_NAMES}
for _ in range(N_PERMS_LABEL):
    perm_labels = rng.permutation(active_labels_c)
    for mod in np.unique(perm_labels):
        nodes = np.where(perm_labels == mod)[0]
        if len(nodes) < 3:
            continue
        triu_sub = np.triu_indices(len(nodes), k=1)
        if len(triu_sub[0]) > 2:
            r, _ = stats.pearsonr(
                FC_pred_c[np.ix_(nodes, nodes)][triu_sub],
                FC_emp_c[np.ix_(nodes, nodes)][triu_sub]
            )
            name = INT2SHORT.get(mod, f"M{mod}")
            label_null_intra[name].append(r)

print("   emp vs null:")
for net in YEO7_NAMES:
    if net in results_c['coupling_intra']:
        emp_r = results_c['coupling_intra'][net]['r']
        nv = np.array(label_null_intra.get(net, []))
        if len(nv) > 0:
            p_l = np.mean(nv >= emp_r)
            print(f"   {net:>4s}: emp={emp_r:.4f} null={np.mean(nv):.4f}±{np.std(nv):.4f} p={p_l:.3f}")

print("\n✅ Seção D concluída!")


# %% ========================================================================
#  SEÇÃO E — VISUALIZAÇÕES PUBLICATION-QUALITY
# ===========================================================================
print("\n" + "=" * 70)
print("  SEÇÃO E — VISUALIZAÇÕES")
print("=" * 70)

FC_emp = results_c['FC_empirical']
FC_pred = results_c['FC_predicted']
FC_diff = FC_emp - FC_pred
act_labels = results_c['active_labels']
n_act = results_c['n_active']
triu = np.triu_indices(n_act, k=1)
fc_emp_flat = FC_emp[triu]
fc_pred_flat = FC_pred[triu]
r_wb = results_c['coupling_wholebrain']

# ======================================================================
#  E.01 — Performance vs Alpha
# ======================================================================
print("[E.01]")
fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.plot(alphas, scores_per_alpha, 'o-', color='#2c3e50', lw=1.5, ms=4,
        markerfacecolor='#e74c3c', markeredgecolor='#2c3e50', zorder=3)
ax.scatter([best_alpha], [scores_per_alpha[best_idx]],
           s=80, facecolors='none', edgecolors='#e74c3c', lw=1.5, zorder=4)
ax.axvline(x=1.0, color='#95a5a6', ls='--', lw=0.8)
ax.set_xlabel('Spectral radius (α)'); ax.set_ylabel('R² score')
ax.set_title('Reservoir Performance vs\nDynamical Regime', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E01_alpha_sweep.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E01_alpha_sweep.pdf')); plt.close()

# ======================================================================
#  E.02 — Network Barplot
# ======================================================================
print("[E.02]")
fig, ax = plt.subplots(figsize=(3.5, 2.8))
if 'module' in df_module_scores.columns:
    md = df_module_scores.dropna(subset=['r2_score']).sort_values('r2_score', ascending=False)
    colors = [YEO7_COLORS.get(m, '#999') for m in md['module']]
    ax.bar(range(len(md)), md['r2_score'], color=colors, edgecolor='white',
           lw=0.5, width=0.7)
    for i, (_, r) in enumerate(md.iterrows()):
        ax.text(i, r['r2_score'] + 0.003, f"{r['r2_score']:.3f}",
                ha='center', fontsize=5.5)
    ax.set_xticks(range(len(md)))
    ax.set_xticklabels(md['module'], fontweight='bold')
    for i, l in enumerate(ax.get_xticklabels()): l.set_color(colors[i])
    ax.set_ylabel('R² score'); ax.axhline(y=0, color='#bdc3c7', lw=0.5)
ax.set_title('Network-Specific Computational\nCapacity', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E02_network_barplot.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E02_network_barplot.pdf')); plt.close()

# ======================================================================
#  E.03 — FC Heatmaps (Empirical | Predicted | Decoupling)
# ======================================================================
print("[E.03]")
fig = plt.figure(figsize=(7.2, 2.6))
gs3 = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.25)
titles3 = ['FC Empirical', 'FC Predicted (RC)', 'Decoupling (Emp − Pred)']
data3 = [FC_emp, FC_pred, FC_diff]
cmaps3 = ['RdBu_r', 'RdBu_r', 'PiYG']
vmin3 = [-0.6, -0.6, -0.4]; vmax3 = [0.6, 0.6, 0.4]
for i in range(3):
    ax = fig.add_subplot(gs3[0, i])
    im = ax.imshow(data3[i], cmap=cmaps3[i], vmin=vmin3[i], vmax=vmax3[i],
                   aspect='equal', interpolation='none')
    add_network_boundaries(ax, act_labels)
    ax.set_title(titles3[i], fontsize=8)
    ax.set_xlabel('Node')
    if i == 0: ax.set_ylabel('Node')
    else: ax.set_yticklabels([])
    add_panel_label(ax, chr(97+i))
cax = fig.add_subplot(gs3[0, 3])
plt.colorbar(im, cax=cax, label='r')
fig.savefig(os.path.join(OUTPUT_DIR, 'E03_fc_heatmaps.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E03_fc_heatmaps.pdf')); plt.close()

# ======================================================================
#  E.04 — Density Scatter
# ======================================================================
print("[E.04]")
fig, ax = plt.subplots(figsize=(3.2, 3.0))
sc_d = density_scatter(ax, fc_emp_flat, fc_pred_flat, cmap='inferno', s=2)
lims = [min(fc_emp_flat.min(), fc_pred_flat.min()) - 0.05,
        max(fc_emp_flat.max(), fc_pred_flat.max()) + 0.05]
ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.4)
sl, ic = np.polyfit(fc_emp_flat, fc_pred_flat, 1)
xl = np.linspace(lims[0], lims[1], 100)
ax.plot(xl, sl * xl + ic, color='#e74c3c', lw=1.5)
ax.text(0.05, 0.95, f'r = {r_wb:.3f}\nn = {len(fc_emp_flat)}',
        transform=ax.transAxes, fontsize=8, fontweight='bold', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#bdc3c7', alpha=0.9))
ax.set_xlabel('FC empirical'); ax.set_ylabel('FC predicted (RC)')
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
if sc_d: plt.colorbar(sc_d, ax=ax, fraction=0.04, pad=0.02, label='Density')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E04_scatter_coupling.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E04_scatter_coupling.pdf')); plt.close()

# ======================================================================
#  E.05 — Inter-Network Coupling Matrix
# ======================================================================
print("[E.05]")
coup_mat = np.full((7, 7), np.nan)
for i, net in enumerate(YEO7_NAMES):
    if net in results_c['coupling_intra']:
        coup_mat[i, i] = results_c['coupling_intra'][net]['r']
for _, row in df_inter.iterrows():
    if row['network_i'] in YEO7_NAMES and row['network_j'] in YEO7_NAMES:
        ii = YEO7_NAMES.index(row['network_i'])
        jj = YEO7_NAMES.index(row['network_j'])
        coup_mat[ii, jj] = row['coupling']
        coup_mat[jj, ii] = row['coupling']

fig, ax = plt.subplots(figsize=(3.8, 3.4))
mask = np.tril(np.ones(7, dtype=bool)[:, None] * np.ones(7, dtype=bool)[None, :], k=-1)
coup_show = np.where(mask, np.nan, coup_mat)
im = ax.imshow(coup_show, cmap='RdYlBu_r', vmin=-0.2, vmax=0.8, aspect='equal')
ax.set_xticks(range(7)); ax.set_xticklabels(YEO7_NAMES, rotation=45, ha='right', fontweight='bold')
ax.set_yticks(range(7)); ax.set_yticklabels(YEO7_NAMES, fontweight='bold')
for i, n in enumerate(YEO7_NAMES):
    ax.get_xticklabels()[i].set_color(YEO7_COLORS[n])
    ax.get_yticklabels()[i].set_color(YEO7_COLORS[n])
for i in range(7):
    for j in range(i, 7):
        v = coup_mat[i, j]
        if not np.isnan(v):
            c = 'white' if v > 0.5 or v < -0.1 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                    color=c, fontweight='bold' if i == j else 'normal')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='SC-FC Coupling (r)')
ax.set_title('Network-Level SC-FC Coupling', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E05_network_matrix.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E05_network_matrix.pdf')); plt.close()

# ======================================================================
#  E.06 — Null Model Distribution
# ======================================================================
print("[E.06]")
fig, ax = plt.subplots(figsize=(3.5, 2.8))
ax.hist(null_couplings, bins=20, color='#bdc3c7', edgecolor='white', alpha=0.85)
ax.axvline(x=empirical_coupling, color='#e74c3c', lw=2, ls='--',
           label=f'Empirical (r = {empirical_coupling:.3f})')
from scipy.stats import gaussian_kde
if len(null_couplings) > 5:
    kde = gaussian_kde(null_couplings)
    xk = np.linspace(null_couplings.min()-0.05, null_couplings.max()+0.05, 200)
    bw = (null_couplings.max() - null_couplings.min()) / 20
    ax.plot(xk, kde(xk) * len(null_couplings) * bw, color='#7f8c8d', lw=1.2)
ax.set_xlabel('SC-FC Coupling (r)'); ax.set_ylabel('Count')
ax.legend(frameon=False, fontsize=7)
ax.text(0.95, 0.95, f'p = {p_null:.3f}\nz = {z_null:.2f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=8, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='#f39c12', alpha=0.9))
ax.set_title('Null Model Validation', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E06_null_model.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E06_null_model.pdf')); plt.close()

# ======================================================================
#  E.07 — Coupling vs Alpha
# ======================================================================
print("[E.07]")
fig, ax = plt.subplots(figsize=(3.5, 2.8))
apl = [d['alpha'] for d in coupling_by_alpha]
cpl = [d['coupling'] for d in coupling_by_alpha]
ax.plot(apl, cpl, 'o-', color='#2c3e50', lw=1.5, ms=5,
        markerfacecolor='#3498db', markeredgecolor='#2c3e50', zorder=3)
ax.fill_between(apl, np.array(cpl)-0.03, np.array(cpl)+0.03,
                color='#3498db', alpha=0.12)
ax.axvline(x=1.0, color='#95a5a6', ls='--', lw=0.8)
ax.set_xlabel('Spectral radius (α)'); ax.set_ylabel('SC-FC Coupling (r)')
ax.set_title('Dynamical Regime Shapes\nSC-FC Coupling', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E07_coupling_vs_alpha.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E07_coupling_vs_alpha.pdf')); plt.close()

# ======================================================================
#  E.08 — COMPOSITE FIGURE (8 panels)
# ======================================================================
print("[E.08] Composite figure")
fig = plt.figure(figsize=(7.2, 8.5))
gs8 = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.45, height_ratios=[1, 1, 1.1])

# Panel a: SC
ax_a = fig.add_subplot(gs8[0, 0])
SC_disp = SC[:n_act, :n_act] if SC.shape[0] >= n_act else SC
im_a = ax_a.imshow(np.log1p(SC_disp), cmap='YlOrRd', aspect='equal')
add_network_boundaries(ax_a, act_labels); plt.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
ax_a.set_xlabel('Node'); ax_a.set_ylabel('Node'); add_panel_label(ax_a, 'a')

# Panel b: FC emp
ax_b = fig.add_subplot(gs8[0, 1])
im_b = ax_b.imshow(FC_emp, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='equal')
add_network_boundaries(ax_b, act_labels); plt.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
ax_b.set_yticklabels([]); add_panel_label(ax_b, 'b')

# Panel c: FC pred
ax_c = fig.add_subplot(gs8[0, 2])
im_c = ax_c.imshow(FC_pred, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='equal')
add_network_boundaries(ax_c, act_labels); plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
ax_c.set_yticklabels([]); add_panel_label(ax_c, 'c')

# Panel d: Scatter
ax_d = fig.add_subplot(gs8[0, 3])
density_scatter(ax_d, fc_emp_flat, fc_pred_flat, cmap='inferno', s=1, alpha=0.5)
lims_d = [min(fc_emp_flat.min(), fc_pred_flat.min())-0.05,
          max(fc_emp_flat.max(), fc_pred_flat.max())+0.05]
ax_d.plot(lims_d, lims_d, 'k--', lw=0.6, alpha=0.4)
sl_d, ic_d = np.polyfit(fc_emp_flat, fc_pred_flat, 1)
ax_d.plot(xl, sl_d*xl+ic_d, color='#e74c3c', lw=1.2)
ax_d.text(0.05, 0.95, f'r = {r_wb:.3f}', transform=ax_d.transAxes, fontsize=7,
          fontweight='bold', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax_d.set_xlabel('FC emp'); ax_d.set_ylabel('FC pred'); ax_d.set_aspect('equal')
add_panel_label(ax_d, 'd')

# Panel e: Alpha sweep
ax_e = fig.add_subplot(gs8[1, 0:2])
ax_e.plot(alphas, scores_per_alpha, 'o-', color='#2c3e50', lw=1.5, ms=3.5,
          markerfacecolor='#e74c3c', markeredgecolor='#2c3e50')
ax_e.axvline(x=1.0, color='#95a5a6', ls='--', lw=0.8)
ax_e.set_xlabel('Spectral radius (α)'); ax_e.set_ylabel('R² score')
add_panel_label(ax_e, 'e')

# Panel f: Coupling vs alpha
ax_f = fig.add_subplot(gs8[1, 2:4])
ax_f.plot(apl, cpl, 'o-', color='#2c3e50', lw=1.5, ms=4,
          markerfacecolor='#3498db', markeredgecolor='#2c3e50')
ax_f.fill_between(apl, np.array(cpl)-0.03, np.array(cpl)+0.03, color='#3498db', alpha=0.1)
ax_f.axvline(x=1.0, color='#95a5a6', ls='--', lw=0.8)
ax_f.set_xlabel('Spectral radius (α)'); ax_f.set_ylabel('SC-FC Coupling (r)')
add_panel_label(ax_f, 'f')

# Panel g: Network barplot
ax_g = fig.add_subplot(gs8[2, 0:2])
if 'module' in df_module_scores.columns:
    md_g = df_module_scores.dropna(subset=['r2_score'])
    cl_g = [YEO7_COLORS.get(m, '#999') for m in md_g['module']]
    ax_g.bar(range(len(md_g)), md_g['r2_score'], color=cl_g, edgecolor='white', lw=0.5, width=0.7)
    ax_g.set_xticks(range(len(md_g))); ax_g.set_xticklabels(md_g['module'], fontweight='bold', fontsize=7)
    for i, l in enumerate(ax_g.get_xticklabels()): l.set_color(cl_g[i])
    ax_g.set_ylabel('R² score'); ax_g.axhline(y=0, color='#bdc3c7', lw=0.5)
add_panel_label(ax_g, 'g')

# Panel h: Null
ax_h = fig.add_subplot(gs8[2, 2:4])
ax_h.hist(null_couplings, bins=15, color='#bdc3c7', edgecolor='white', alpha=0.85, label='Null')
ax_h.axvline(x=empirical_coupling, color='#e74c3c', lw=2, ls='--', label='Empirical')
ax_h.set_xlabel('SC-FC Coupling (r)'); ax_h.set_ylabel('Count')
ax_h.legend(frameon=False, fontsize=6)
ax_h.text(0.95, 0.95, f'p = {p_null:.3f}\nz = {z_null:.2f}', transform=ax_h.transAxes,
          ha='right', va='top', fontsize=7, fontweight='bold',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
add_panel_label(ax_h, 'h')

fig.savefig(os.path.join(OUTPUT_DIR, 'E08_composite.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E08_composite.pdf')); plt.close()

# ======================================================================
#  E.09 — Group Violinplot (raincloud-style)
# ======================================================================
print("[E.09]")
rows_long = []
for _, row in df_group.iterrows():
    for col in intra_cols:
        net = col.replace('intra_', '')
        if not pd.isna(row[col]):
            rows_long.append({'Network': net, 'Coupling': row[col], 'Subject': row['subject']})
df_long = pd.DataFrame(rows_long)

if len(df_long) > 0:
    order = df_long.groupby('Network')['Coupling'].median().sort_values(ascending=False).index.tolist()
    pal = [YEO7_COLORS.get(n, '#999') for n in order]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    sns.violinplot(data=df_long, x='Network', y='Coupling', order=order,
                   palette=pal, inner=None, alpha=0.25, ax=ax, cut=0, density_norm='width')
    sns.stripplot(data=df_long, x='Network', y='Coupling', order=order,
                  palette=pal, size=3, alpha=0.7, jitter=0.12, ax=ax, zorder=3)
    sns.boxplot(data=df_long, x='Network', y='Coupling', order=order,
                color='white', width=0.12, fliersize=0,
                boxprops=dict(alpha=0.7, lw=0.8), whiskerprops=dict(lw=0.8),
                medianprops=dict(color='black', lw=1), ax=ax, zorder=2)
    ax.set_xlabel(''); ax.set_ylabel('Intra-Network SC-FC Coupling (r)')
    ax.set_title('Group-Level Network Coupling', fontsize=9)
    for i, l in enumerate(ax.get_xticklabels()):
        l.set_color(pal[i]); l.set_fontweight('bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'E09_group_violinplot.png'), dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, 'E09_group_violinplot.pdf')); plt.close()

# ======================================================================
#  E.10 — Brain Surface Plots (nilearn)
# ======================================================================
print("[E.10] Brain surface plots")
try:
    from nilearn import plotting as ni_plot
    from nilearn import datasets as ni_data
    from nilearn import image as ni_image

    schaefer = ni_data.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    atlas_img = schaefer['maps']
    atlas_data = ni_image.get_data(atlas_img)

    # E.10a — Nodal decoupling (MAE entre FC_emp e FC_pred)
    nodal_dec = np.array([np.mean(np.abs(FC_emp[i, :] - FC_pred[i, :])) for i in range(n_act)])
    full_dec = np.zeros(N_NODES)
    full_dec[results_c['idx_active']] = nodal_dec

    dec_vol = np.zeros_like(atlas_data, dtype=float)
    for ri in range(N_NODES):
        dec_vol[atlas_data == (ri + 1)] = full_dec[ri]
    dec_img = ni_image.new_img_like(atlas_img, dec_vol)

    fig1 = plt.figure(figsize=(7.2, 2.2))
    ni_plot.plot_stat_map(dec_img, bg_img=ni_data.load_mni152_template(resolution=2),
                          display_mode='ortho', cmap='hot', colorbar=True,
                          title='Nodal SC-FC Decoupling (MAE)', figure=fig1,
                          threshold=0.001, black_bg=False)
    fig1.savefig(os.path.join(OUTPUT_DIR, 'E10a_brain_decoupling.png'), dpi=300)
    fig1.savefig(os.path.join(OUTPUT_DIR, 'E10a_brain_decoupling.pdf')); plt.close(fig1)
    print("   ✅ E.10a salvo")

    # E.10b — Network coupling strength
    full_coup = np.zeros(N_NODES)
    for i, idx in enumerate(results_c['idx_active']):
        nl = act_labels[i]
        nm = INT2SHORT.get(nl, '')
        if nm in results_c['coupling_intra']:
            full_coup[idx] = results_c['coupling_intra'][nm]['r']

    coup_vol = np.zeros_like(atlas_data, dtype=float)
    for ri in range(N_NODES):
        coup_vol[atlas_data == (ri + 1)] = full_coup[ri]
    coup_img = ni_image.new_img_like(atlas_img, coup_vol)

    fig2 = plt.figure(figsize=(7.2, 2.2))
    ni_plot.plot_stat_map(coup_img, bg_img=ni_data.load_mni152_template(resolution=2),
                          display_mode='ortho', cmap='RdYlBu_r', colorbar=True,
                          title='Network SC-FC Coupling Strength', figure=fig2,
                          threshold=0.001, black_bg=False)
    fig2.savefig(os.path.join(OUTPUT_DIR, 'E10b_brain_coupling.png'), dpi=300)
    fig2.savefig(os.path.join(OUTPUT_DIR, 'E10b_brain_coupling.pdf')); plt.close(fig2)
    print("   ✅ E.10b salvo")

    # E.10c — Yeo7 parcellation
    net_vol = np.zeros_like(atlas_data, dtype=float)
    for ri in range(N_NODES):
        net_vol[atlas_data == (ri + 1)] = YEO7_INT[ri] + 1
    net_img = ni_image.new_img_like(atlas_img, net_vol)

    fig3 = plt.figure(figsize=(7.2, 2.2))
    ni_plot.plot_roi(net_img, bg_img=ni_data.load_mni152_template(resolution=2),
                     display_mode='ortho', title='Yeo 7-Network Parcellation',
                     figure=fig3, black_bg=False)
    fig3.savefig(os.path.join(OUTPUT_DIR, 'E10c_brain_yeo7.png'), dpi=300)
    fig3.savefig(os.path.join(OUTPUT_DIR, 'E10c_brain_yeo7.pdf')); plt.close(fig3)
    print("   ✅ E.10c salvo")

    # E.10d — Structural connectome (top 5% edges)
    coords = ni_plot.find_parcellation_cut_coords(atlas_img)
    node_colors = [YEO7_COLORS_ORD[YEO7_INT[i]] for i in range(N_NODES)]

    fig4 = plt.figure(figsize=(7.2, 2.5))
    ni_plot.plot_connectome(SC_raw, coords, edge_threshold="95%",
                            display_mode='ortho', title='SC (top 5% edges)',
                            figure=fig4, node_size=15, node_color=node_colors,
                            edge_cmap='YlOrRd', black_bg=False)
    fig4.savefig(os.path.join(OUTPUT_DIR, 'E10d_brain_connectome.png'), dpi=300)
    fig4.savefig(os.path.join(OUTPUT_DIR, 'E10d_brain_connectome.pdf')); plt.close(fig4)
    print("   ✅ E.10d salvo")

except ImportError:
    print("   ⚠️  nilearn não disponível. pip install nilearn")
except Exception as e:
    print(f"   ⚠️  Erro: {e}")
    import traceback; traceback.print_exc()

# ======================================================================
#  E.11 — Reservoir States Heatmap
# ======================================================================
print("[E.11]")
fig = plt.figure(figsize=(7.2, 2.8))
gs11 = gridspec.GridSpec(1, 3, width_ratios=[0.06, 3, 0.8], wspace=0.05)
ax_sb = fig.add_subplot(gs11[0, 0])
make_network_sidebar(ax_sb, labels[:n_active], INT2SHORT)
ax_ht = fig.add_subplot(gs11[0, 1])
im11 = ax_ht.imshow(reservoir_states[:200, :].T, aspect='auto', cmap='viridis', interpolation='none')
ax_ht.set_xlabel('TR'); ax_ht.set_ylabel('Node')
ax_ht.set_title('Reservoir State Dynamics', fontsize=9)
plt.colorbar(im11, ax=ax_ht, fraction=0.02, pad=0.02, label='Activation')
ax_di = fig.add_subplot(gs11[0, 2])
ax_di.hist(reservoir_states.ravel(), bins=50, color='#2c3e50', edgecolor='white',
           orientation='horizontal', alpha=0.8)
ax_di.set_xlabel('Count'); ax_di.set_yticklabels([])
fig.savefig(os.path.join(OUTPUT_DIR, 'E11_reservoir_states.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E11_reservoir_states.pdf')); plt.close()

# ======================================================================
#  E.12 — Leak Rate Comparison
# ======================================================================
print("[E.12]")
fig, axes = plt.subplots(1, len(leak_rates), figsize=(7.2, 2), sharey=True)
ex_nodes = np.linspace(0, n_active-1, 5, dtype=int)
nc = [YEO7_COLORS_ORD[labels[n] if n < len(labels) else 0] for n in ex_nodes]
for idx, lr in enumerate(leak_rates):
    rv = rs_per_lr[lr][:100, :]
    for ni, nd in enumerate(ex_nodes):
        if nd < rv.shape[1]:
            axes[idx].plot(rv[:, nd], lw=0.7, alpha=0.8, color=nc[ni])
    axes[idx].set_title(f'τ = {lr}', fontsize=8, fontweight='bold')
    axes[idx].set_xlabel('TR', fontsize=7)
    if idx == 0: axes[idx].set_ylabel('Activation')
fig.suptitle('Effect of Leak Rate on Reservoir Dynamics', fontweight='bold', fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E12_leak_rate.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E12_leak_rate.pdf')); plt.close()

# ======================================================================
#  E.13 — Group WB Distribution
# ======================================================================
print("[E.13]")
fig, ax = plt.subplots(figsize=(3, 3.5))
cv = df_group['coupling_wb'].values
ax.boxplot(cv, positions=[0], widths=0.4, patch_artist=True,
           boxprops=dict(facecolor='#3498db', alpha=0.3),
           medianprops=dict(color='#2c3e50', lw=1.5),
           whiskerprops=dict(lw=0.8), flierprops=dict(ms=3))
jit = rng.uniform(-0.1, 0.1, len(cv))
ax.scatter(jit, cv, s=20, color='#3498db', alpha=0.7, edgecolors='#2c3e50', lw=0.5, zorder=3)
ax.text(0.35, np.median(cv), f'median = {np.median(cv):.3f}\nIQR = [{np.percentile(cv,25):.3f}, {np.percentile(cv,75):.3f}]',
        fontsize=7, va='center')
ax.set_xlim(-0.5, 0.8); ax.set_xticks([]); ax.set_ylabel('Whole-Brain SC-FC Coupling (r)')
ax.set_title(f'Group Distribution (n = {len(cv)})', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E13_group_wb.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E13_group_wb.pdf')); plt.close()

# ======================================================================
#  E.14 — Network Coupling Gradient (mean ± SEM)
# ======================================================================
print("[E.14]")
fig, ax = plt.subplots(figsize=(4, 3.5))
means_net = []
for i, net in enumerate(YEO7_NAMES):
    col = f'intra_{net}'
    if col in df_group.columns:
        vals = df_group[col].dropna()
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.15, 0.15, len(vals)),
                   vals, s=15, color=YEO7_COLORS[net], alpha=0.5, edgecolors='white', lw=0.3, zorder=3)
        m, sem = vals.mean(), vals.std() / np.sqrt(len(vals))
        ax.errorbar(i, m, yerr=sem, fmt='D', ms=6, color=YEO7_COLORS[net],
                    markeredgecolor='#2c3e50', markeredgewidth=0.8, capsize=3, capthick=1, zorder=4)
        means_net.append(m)
    else:
        means_net.append(np.nan)
ax.plot(range(7), means_net, 'k--', lw=0.6, alpha=0.4, zorder=1)
ax.set_xticks(range(7)); ax.set_xticklabels(YEO7_NAMES, fontweight='bold')
for i, l in enumerate(ax.get_xticklabels()): l.set_color(YEO7_COLORS[YEO7_NAMES[i]])
ax.set_ylabel('Intra-Network SC-FC Coupling (r)')
ax.set_title('Network Coupling Gradient (mean ± SEM)', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'E14_network_gradient.png'), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, 'E14_network_gradient.pdf')); plt.close()


# %% ========================================================================
#  RESUMO
# ===========================================================================
print("\n" + "=" * 70)
print("  TUTORIAL CONN2RES v2 — COMPLETO!")
print("=" * 70)
print(f"\n  Figuras em: {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'): print(f"    {f}")
print(f"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  A: Pipeline (SC→Conn→ESN→Readout, R² com dados reais)            ║
  ║  B: Alpha sweep, network readout, leak rate, real vs random        ║
  ║  C: SC-FC coupling (whole-brain, intra, inter, alpha sweep)        ║
  ║  D: Group (n=23) + null (SC random + label permutation)            ║
  ║  E: 14 figuras (PNG+PDF) inc. 4 brain plots                       ║
  ║                                                                    ║
  ║  ⚠️  Bugfixes: SC/FC swap, fmri path, metric=list, w_in scaling   ║
  ║  Ref: Suárez et al. (2024) Nat Commun 15:656                      ║
  ╚══════════════════════════════════════════════════════════════════════╝
""")
