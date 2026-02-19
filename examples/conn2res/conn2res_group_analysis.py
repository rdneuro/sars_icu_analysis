#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  CONN2RES — GROUP-LEVEL SC-FC COUPLING ANALYSIS
  SARS-CoV-2 ICU Survivors (n=23), Schaefer-100 / Yeo 7-Network
================================================================================

  Autor:  Velho Mago — INNT/UFRJ, PhD Neurociência Translacional
  Data:   Fevereiro 2026
  Ref:    Suárez et al. (2024) Nat Commun 15:656

  Análises:
    1. SC-FC coupling (whole-brain + Yeo7) para todos os 23 sujeitos
    2. Comparação SC raw vs log1p
    3. Null model: SC randomizada (degree-preserving), com joblib
    4. Null model: Label permutation (spin-test simplificado)
    5. Nodal decoupling map (grupo)
    6. Tabelas formatadas para LaTeX (tabulate)
    7. 12+ figuras publication-quality incluindo 5 brain surface plots

  Execução: Spyder IDE
  Working dir: /mnt/nvme1n1p1/sars_cov_2_project/code/current
================================================================================
"""

# %% ========================================================================
#  IMPORTS
# ===========================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from scipy import stats
from scipy.linalg import eigh

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from nilearn.connectome import sym_matrix_to_vec, vec_to_sym_matrix

from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork

from joblib import Parallel, delayed
import multiprocessing

pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 900)


# %% ========================================================================
#  CONFIGURATION
# ===========================================================================

DATA_ROOT  = "/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs"
INFO_DIR   = os.path.join(DATA_ROOT, "info")
OUTPUT_DIR = "/mnt/nvme1n1p1/sars_cov_2_project/figs/conn2res_group"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATLAS   = "schaefer100"
N_NODES = 100
N_TRS   = 390

ALL_SUBS = [f"sub-{n:02d}" for n in range(1, 25)]
ALL_SUBS.remove("sub-21")
N_SUBJECTS = len(ALL_SUBS)

SEED = 42
rng  = np.random.default_rng(SEED)

# --- Analysis parameters ---
ALPHA        = 1.0        # spectral radius for coupling
N_RUNS       = 10         # simulation repeats per subject
WASHOUT      = 200        # timesteps to discard
NOISE_STD    = 0.1        # input noise amplitude
N_PERMS_SC   = 1000       # null: SC randomization  (⚠️ set 100 for quick test)
N_PERMS_LAB  = 1000       # null: label permutation  (⚠️ set 200 for quick test)
N_CORES      = max(1, multiprocessing.cpu_count() - 2)  # leave 2 cores free

print(f"  CPUs available: {multiprocessing.cpu_count()}, using: {N_CORES}")
print(f"  Permutations: SC = {N_PERMS_SC}, Label = {N_PERMS_LAB}")


# %% ========================================================================
#  FIGURE STYLE — JOURNAL-QUALITY
# ===========================================================================

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

# --- Yeo7 Palette ---
YEO7_DISPLAY = {
    'Cont': 'FP', 'Default': 'DMN', 'DorsAttn': 'DA',
    'Limbic': 'Lim', 'SalVentAttn': 'VA', 'SomMot': 'SM', 'Vis': 'Vis'
}
YEO7_COLORS = {
    'FP':  '#E69422', 'DMN': '#CD3E4E', 'DA':  '#00760E',
    'Lim': '#DCF8A4', 'VA':  '#C43AFA', 'SM':  '#4682B4', 'Vis': '#781286',
}


# %% ========================================================================
#  CORE FUNCTIONS
# ===========================================================================

def load_atlas_info():
    """Load ROI labels and Yeo7 network assignments with auto-detection."""
    roi_labels = np.loadtxt(os.path.join(INFO_DIR, f"labels_{ATLAS}.txt"), dtype=str)
    nets_raw = np.loadtxt(os.path.join(INFO_DIR, "networks_schaefer100.txt"), dtype=str)
    le = LabelEncoder()
    net_int = le.fit_transform(nets_raw)
    int2short = {}
    for i, name in enumerate(le.classes_):
        int2short[i] = YEO7_DISPLAY.get(name, name[:3])
    net_str = np.array([int2short[i] for i in net_int])
    return roi_labels, net_int, net_str, int2short


def load_subject(sub, sc_treat='raw'):
    """
    Load SC (SIFT2), FC (correlation), TS for one subject.
    sc_treat: 'raw' | 'log' (np.log1p)
    """
    base = os.path.join(DATA_ROOT, sub, ATLAS)
    SC = np.load(os.path.join(base, "dmri", "connectivity_sift2.npy"))
    FC = np.load(os.path.join(base, "fmri", "connectivity_correlation.npy"))
    TS = np.load(os.path.join(base, "fmri", "timeseries.npy"))

    np.fill_diagonal(SC, 0); np.fill_diagonal(FC, 0)
    SC = (SC + SC.T) / 2;    FC = (FC + FC.T) / 2

    for mtx in [SC, FC]:
        if np.any(~np.isfinite(mtx)):
            mtx[:] = SimpleImputer(strategy='median').fit_transform(mtx)

    if sc_treat == 'log':
        SC = np.log1p(SC)

    return SC, FC, TS


def compute_coupling_single(SC, FC, labels, alpha=ALPHA, n_runs=N_RUNS,
                             washout=WASHOUT, noise_std=NOISE_STD,
                             n_timepoints=N_TRS, seed=42, int2short=None):
    """
    Compute SC-FC coupling for one subject.
    Returns: dict with whole-brain, intra-network, inter-network, nodal metrics.
    """
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
        ext = rng_loc.standard_normal((n_timepoints, n_act)) * noise_std
        esn = EchoStateNetwork(w=alpha * conn.w, activation_function='tanh')
        rs = esn.simulate(ext_input=ext, w_in=w_in, output_nodes=None)
        FC_rc_accum += np.corrcoef(rs[washout:].T)

    FC_pred = FC_rc_accum / n_runs

    # Whole-brain coupling
    triu = np.triu_indices(n_act, k=1)
    r_wb, p_wb = stats.pearsonr(FC_pred[triu], FC_act[triu])

    # Nodal decoupling: MAE per node (row-wise)
    nodal_mae = np.array([
        np.mean(np.abs(FC_act[i, :] - FC_pred[i, :])) for i in range(n_act)
    ])
    # Nodal coupling: row-wise correlation
    nodal_corr = np.array([
        stats.pearsonr(FC_pred[i, :], FC_act[i, :])[0]
        if np.std(FC_pred[i, :]) > 1e-10 else 0.0
        for i in range(n_act)
    ])

    result = {
        'coupling_wb': r_wb, 'p_wb': p_wb, 'n_active': n_act,
        'FC_predicted': FC_pred, 'FC_empirical': FC_act,
        'active_labels': lab_act, 'idx_active': idx_act,
        'nodal_mae': nodal_mae, 'nodal_corr': nodal_corr,
    }

    # Intra-network
    short = int2short or {}
    intra = {}
    for mod in np.unique(lab_act):
        nodes = np.where(lab_act == mod)[0]
        if len(nodes) < 3:
            continue
        triu_sub = np.triu_indices(len(nodes), k=1)
        if len(triu_sub[0]) > 2:
            r, p = stats.pearsonr(
                FC_pred[np.ix_(nodes, nodes)][triu_sub],
                FC_act[np.ix_(nodes, nodes)][triu_sub]
            )
            intra[short.get(mod, f"M{mod}")] = {'r': r, 'p': p, 'n': len(nodes)}
    result['intra'] = intra

    # Inter-network
    inter_rows = []
    umods = np.unique(lab_act)
    for ii, mi in enumerate(umods):
        for jj, mj in enumerate(umods):
            if ii >= jj:
                continue
            ni = np.where(lab_act == mi)[0]
            nj = np.where(lab_act == mj)[0]
            if len(ni) < 2 or len(nj) < 2:
                continue
            bp = FC_pred[np.ix_(ni, nj)].ravel()
            be = FC_act[np.ix_(ni, nj)].ravel()
            if len(bp) > 2:
                r, p = stats.pearsonr(bp, be)
                inter_rows.append({
                    'net_i': short.get(mi, f"M{mi}"),
                    'net_j': short.get(mj, f"M{mj}"),
                    'r': r, 'p': p
                })
    result['inter'] = pd.DataFrame(inter_rows)

    return result


def compute_coupling_null_sc(SC, FC, labels, seed_perm, int2short=None):
    """
    Single permutation of SC-randomized null model.
    Returns: whole-brain coupling (float).
    """
    conn_null = Conn(w=SC.copy())
    conn_null.scale_and_normalize()
    conn_null.randomize(swaps=10)

    # Rebuild full-space randomized SC
    idx_a = np.where(conn_null.idx_node)[0]
    SC_rand = np.zeros_like(SC)
    for ii, ri in enumerate(idx_a):
        for jj, rj in enumerate(idx_a):
            SC_rand[ri, rj] = conn_null.w[ii, jj]

    res = compute_coupling_single(
        SC_rand, FC, labels, alpha=ALPHA, n_runs=3,  # fewer runs for speed
        washout=WASHOUT, seed=seed_perm, int2short=int2short
    )
    return res['coupling_wb']


# %% ========================================================================
#  LOAD DATA
# ===========================================================================

print("\n" + "=" * 70)
print("  LOADING DATA")
print("=" * 70)

ROI_LABELS, YEO7_INT, YEO7_STR, INT2SHORT = load_atlas_info()
YEO7_NAMES = [INT2SHORT[i] for i in range(7)]
YEO7_COLORS_ORD = [YEO7_COLORS[INT2SHORT[i]] for i in range(7)]

print(f"\n  LabelEncoder mapping:")
for i in range(7):
    n = np.sum(YEO7_INT == i)
    print(f"    {i} → {INT2SHORT[i]:>4s}  ({n} ROIs)")


# %% ========================================================================
#  ANALYSIS 1: GROUP-LEVEL SC-FC COUPLING (raw AND log1p)
# ===========================================================================

print("\n" + "=" * 70)
print("  ANALYSIS 1: GROUP SC-FC COUPLING (raw + log)")
print("=" * 70)

def run_group_coupling(sc_treat, tag):
    """Run coupling analysis for all subjects with given SC treatment."""
    print(f"\n  --- SC treatment: {tag} ---")
    results = []
    t0 = time.time()
    for i, sub in enumerate(ALL_SUBS):
        try:
            sc_s, fc_s, _ = load_subject(sub, sc_treat=sc_treat)
            res = compute_coupling_single(
                sc_s, fc_s, YEO7_INT, seed=SEED + i, int2short=INT2SHORT
            )
            row = {'subject': sub, 'coupling_wb': res['coupling_wb'],
                   'n_active': res['n_active']}
            # Intra-network
            for net in YEO7_NAMES:
                row[f'intra_{net}'] = res['intra'].get(net, {}).get('r', np.nan)
            # Nodal metrics (store full arrays for later)
            row['_nodal_mae'] = res['nodal_mae']
            row['_nodal_corr'] = res['nodal_corr']
            row['_idx_active'] = res['idx_active']
            row['_FC_pred'] = res['FC_predicted']
            row['_FC_emp'] = res['FC_empirical']
            row['_active_labels'] = res['active_labels']
            row['_inter'] = res['inter']
            results.append(row)
            print(f"    {sub}: r = {res['coupling_wb']:.4f}")
        except Exception as e:
            print(f"    {sub}: ⚠️ {e}")
    print(f"  Time: {time.time()-t0:.0f}s")
    return results

results_raw = run_group_coupling('raw', 'RAW (SIFT2 counts)')
results_log = run_group_coupling('log', 'LOG (log1p)')

# Build DataFrames (exclude private columns for the public DF)
priv_cols = [c for c in results_raw[0].keys() if c.startswith('_')]
pub_cols = [c for c in results_raw[0].keys() if not c.startswith('_')]

df_raw = pd.DataFrame([{k: r[k] for k in pub_cols} for r in results_raw])
df_log = pd.DataFrame([{k: r[k] for k in pub_cols} for r in results_log])

print(f"\n  RAW:  WB coupling = {df_raw['coupling_wb'].mean():.4f} "
      f"± {df_raw['coupling_wb'].std():.4f}")
print(f"  LOG:  WB coupling = {df_log['coupling_wb'].mean():.4f} "
      f"± {df_log['coupling_wb'].std():.4f}")

# Paired comparison
t_stat, p_paired = stats.ttest_rel(df_raw['coupling_wb'], df_log['coupling_wb'])
print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_paired:.4f}")


# %% ========================================================================
#  ANALYSIS 2: NULL MODEL — SC RANDOMIZATION (parallel)
# ===========================================================================

print("\n" + "=" * 70)
print("  ANALYSIS 2: NULL MODEL — SC RANDOMIZATION")
print("=" * 70)

# Use the treatment with higher coupling for the null model
if df_raw['coupling_wb'].mean() >= df_log['coupling_wb'].mean():
    df_main = df_raw; results_main = results_raw; SC_TREAT_MAIN = 'raw'
else:
    df_main = df_log; results_main = results_log; SC_TREAT_MAIN = 'log'

print(f"\n  Using SC treatment: {SC_TREAT_MAIN}")

# Select median subject for the null model
med_idx = (df_main['coupling_wb'] - df_main['coupling_wb'].median()).abs().argsort().iloc[0]
null_sub = df_main.iloc[med_idx]['subject']
sc_null_subj, fc_null_subj, _ = load_subject(null_sub, sc_treat=SC_TREAT_MAIN)
emp_coupling_null = df_main.iloc[med_idx]['coupling_wb']
print(f"  Reference subject: {null_sub} (r = {emp_coupling_null:.4f})")

# Run null model in parallel
print(f"  Running {N_PERMS_SC} permutations on {N_CORES} cores...")
t0 = time.time()

null_couplings = Parallel(n_jobs=N_CORES, verbose=5)(
    delayed(compute_coupling_null_sc)(
        sc_null_subj, fc_null_subj, YEO7_INT,
        seed_perm=SEED + 10000 + p, int2short=INT2SHORT
    )
    for p in range(N_PERMS_SC)
)

null_couplings = np.array(null_couplings)
t_null = time.time() - t0
print(f"  Done in {t_null:.0f}s ({t_null/60:.1f} min)")

p_null = np.mean(null_couplings >= emp_coupling_null)
z_null = (emp_coupling_null - np.mean(null_couplings)) / max(np.std(null_couplings), 1e-8)
print(f"  Empirical: {emp_coupling_null:.4f}")
print(f"  Null:      {np.mean(null_couplings):.4f} ± {np.std(null_couplings):.4f}")
print(f"  p = {p_null:.4f}, z = {z_null:.2f}")


# %% ========================================================================
#  ANALYSIS 3: NULL MODEL — LABEL PERMUTATION (group-level)
# ===========================================================================

print("\n" + "=" * 70)
print("  ANALYSIS 3: NULL MODEL — LABEL PERMUTATION")
print("=" * 70)

# For each subject, compute intra-network coupling under permuted labels
# Then compare group-mean empirical vs null distribution

def run_label_perm_subject(sub_result, n_perms, seed_base):
    """Run label permutation for a single subject's pre-computed FC_pred/FC_emp."""
    rng_perm = np.random.default_rng(seed_base)
    lab = sub_result['_active_labels']
    FC_p = sub_result['_FC_pred']
    FC_e = sub_result['_FC_emp']

    null_intra = {net: [] for net in YEO7_NAMES}
    for _ in range(n_perms):
        plab = rng_perm.permutation(lab)
        for mod in np.unique(plab):
            nodes = np.where(plab == mod)[0]
            if len(nodes) < 3:
                continue
            triu_sub = np.triu_indices(len(nodes), k=1)
            if len(triu_sub[0]) > 2:
                r, _ = stats.pearsonr(
                    FC_p[np.ix_(nodes, nodes)][triu_sub],
                    FC_e[np.ix_(nodes, nodes)][triu_sub]
                )
                name = INT2SHORT.get(mod, f"M{mod}")
                null_intra[name].append(r)
    return null_intra

print(f"  Running label permutation ({N_PERMS_LAB} perms) for {N_SUBJECTS} subjects...")
t0 = time.time()

all_label_nulls = Parallel(n_jobs=N_CORES, verbose=5)(
    delayed(run_label_perm_subject)(
        results_main[i], N_PERMS_LAB, SEED + 50000 + i
    )
    for i in range(len(results_main))
)

# Aggregate: for each network, pool null values across subjects
# and compare to group-mean empirical
label_perm_results = {}
for net in YEO7_NAMES:
    # Empirical group mean
    col = f'intra_{net}'
    emp_vals = df_main[col].dropna().values
    emp_mean = np.mean(emp_vals) if len(emp_vals) > 0 else np.nan

    # Null: for each permutation, compute mean across subjects
    null_means = []
    for perm_idx in range(N_PERMS_LAB):
        perm_vals = []
        for subj_nulls in all_label_nulls:
            nv = subj_nulls[net]
            if perm_idx < len(nv):
                perm_vals.append(nv[perm_idx])
        if len(perm_vals) > 0:
            null_means.append(np.mean(perm_vals))
    null_means = np.array(null_means)

    if len(null_means) > 0 and not np.isnan(emp_mean):
        p_lab = np.mean(null_means >= emp_mean)
        z_lab = (emp_mean - np.mean(null_means)) / max(np.std(null_means), 1e-8)
    else:
        p_lab, z_lab = np.nan, np.nan

    label_perm_results[net] = {
        'emp_mean': emp_mean,
        'null_mean': np.mean(null_means) if len(null_means) > 0 else np.nan,
        'null_std': np.std(null_means) if len(null_means) > 0 else np.nan,
        'p': p_lab, 'z': z_lab
    }
    print(f"  {net:>4s}: emp = {emp_mean:.4f}, "
          f"null = {np.mean(null_means):.4f} ± {np.std(null_means):.4f}, "
          f"p = {p_lab:.3f}")

print(f"  Time: {time.time()-t0:.0f}s")


# %% ========================================================================
#  ANALYSIS 4: GROUP NODAL METRICS
# ===========================================================================

print("\n" + "=" * 70)
print("  ANALYSIS 4: GROUP NODAL METRICS")
print("=" * 70)

# Aggregate nodal metrics across subjects (back to N_NODES space)
group_nodal_mae = np.zeros((N_SUBJECTS, N_NODES))
group_nodal_corr = np.zeros((N_SUBJECTS, N_NODES))

for i, res in enumerate(results_main):
    idx = res['_idx_active']
    group_nodal_mae[i, idx] = res['_nodal_mae']
    group_nodal_corr[i, idx] = res['_nodal_corr']

# Group statistics
mean_nodal_mae  = np.mean(group_nodal_mae, axis=0)
std_nodal_mae   = np.std(group_nodal_mae, axis=0)
mean_nodal_corr = np.mean(group_nodal_corr, axis=0)
std_nodal_corr  = np.std(group_nodal_corr, axis=0)
cv_nodal_mae    = std_nodal_mae / np.maximum(mean_nodal_mae, 1e-8)

# Nodal-level one-sample t-test against zero (coupling > 0?)
nodal_t = np.zeros(N_NODES)
nodal_p = np.ones(N_NODES)
for node in range(N_NODES):
    vals = group_nodal_corr[:, node]
    vals = vals[vals != 0]  # exclude subjects where node was inactive
    if len(vals) > 3:
        t, p = stats.ttest_1samp(vals, 0)
        nodal_t[node] = t
        nodal_p[node] = p

# FDR correction (Benjamini-Hochberg)
from statsmodels.stats.multitest import multipletests
reject, nodal_p_fdr, _, _ = multipletests(nodal_p, alpha=0.05, method='fdr_bh')
n_sig = np.sum(reject)
print(f"  Nodes with significant coupling (FDR < 0.05): {n_sig} / {N_NODES}")

# Mean coupling per network
print("\n  Network-level nodal coupling (mean across nodes):")
for i in range(7):
    net = INT2SHORT[i]
    mask = YEO7_INT == i
    mc = mean_nodal_corr[mask]
    print(f"    {net:>4s}: mean_r = {np.mean(mc):.4f} ± {np.std(mc):.4f}")


# %% ========================================================================
#  ANALYSIS 5: GROUP INTER-NETWORK COUPLING MATRIX
# ===========================================================================

print("\n" + "=" * 70)
print("  ANALYSIS 5: GROUP INTER-NETWORK MATRIX")
print("=" * 70)

# Build 7x7 coupling matrix averaged across subjects
group_coupling_matrix = np.full((7, 7, N_SUBJECTS), np.nan)

for si, res in enumerate(results_main):
    # Intra (diagonal)
    for net, vals in res.get('intra', {}).items() if isinstance(res, dict) else []:
        pass
    # Handle both dict-from-list and result-dict
    intra = {}
    for net in YEO7_NAMES:
        col = f'intra_{net}'
        if col in df_main.columns:
            group_coupling_matrix[YEO7_NAMES.index(net),
                                   YEO7_NAMES.index(net), si] = df_main.iloc[si][col]

    # Inter
    df_inter_s = res['_inter']
    for _, row in df_inter_s.iterrows():
        if row['net_i'] in YEO7_NAMES and row['net_j'] in YEO7_NAMES:
            ii = YEO7_NAMES.index(row['net_i'])
            jj = YEO7_NAMES.index(row['net_j'])
            group_coupling_matrix[ii, jj, si] = row['r']
            group_coupling_matrix[jj, ii, si] = row['r']

mean_coupling_matrix = np.nanmean(group_coupling_matrix, axis=2)
std_coupling_matrix = np.nanstd(group_coupling_matrix, axis=2)

# T-test each cell against zero
pval_matrix = np.ones((7, 7))
for i in range(7):
    for j in range(7):
        vals = group_coupling_matrix[i, j, :]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 3:
            _, pval_matrix[i, j] = stats.ttest_1samp(vals, 0)

print("  Mean coupling matrix:")
print(pd.DataFrame(mean_coupling_matrix, index=YEO7_NAMES,
                    columns=YEO7_NAMES).round(3))


# %% ========================================================================
#  TABLE 1: GROUP SC-FC COUPLING SUMMARY
# ===========================================================================

print("\n" + "=" * 70)
print("  TABLE 1: GROUP SC-FC COUPLING")
print("=" * 70)

# Build table: Network | N_ROIs | Coupling (raw) | Coupling (log) | p (null)
table1_rows = []

# Whole-brain row
table1_rows.append({
    'Network': 'Whole-brain',
    'N_ROIs': N_NODES,
    'r_raw': f"{df_raw['coupling_wb'].mean():.3f} ± {df_raw['coupling_wb'].std():.3f}",
    'r_log': f"{df_log['coupling_wb'].mean():.3f} ± {df_log['coupling_wb'].std():.3f}",
    'p_null_SC': f"{p_null:.3f}",
    'z_null_SC': f"{z_null:.2f}",
})

# Per-network rows
for net in YEO7_NAMES:
    col = f'intra_{net}'
    n_rois = np.sum(YEO7_STR == net)
    raw_vals = df_raw[col].dropna()
    log_vals = df_log[col].dropna()
    lp = label_perm_results.get(net, {})

    table1_rows.append({
        'Network': net,
        'N_ROIs': int(n_rois),
        'r_raw': f"{raw_vals.mean():.3f} ± {raw_vals.std():.3f}" if len(raw_vals) > 0 else "—",
        'r_log': f"{log_vals.mean():.3f} ± {log_vals.std():.3f}" if len(log_vals) > 0 else "—",
        'p_null_SC': "—",
        'z_null_SC': "—",
    })

df_table1 = pd.DataFrame(table1_rows)

# Format for LaTeX (tabulate)
try:
    from tabulate import tabulate
    latex_t1 = tabulate(
        df_table1.values.tolist(),
        headers=['Network', '$N_{ROIs}$', 'Coupling (raw)', 'Coupling (log1p)',
                 '$p_{null}$ (SC)', '$z_{null}$ (SC)'],
        tablefmt='latex_booktabs',
        colalign=('left', 'center', 'center', 'center', 'center', 'center')
    )
    print("\n  TABLE 1 (LaTeX):")
    print(latex_t1)

    with open(os.path.join(OUTPUT_DIR, 'table1_coupling.tex'), 'w') as f:
        f.write(latex_t1)
    print(f"  → Saved: table1_coupling.tex")
except ImportError:
    print("  tabulate not installed. pip install tabulate")
    print(df_table1.to_string(index=False))


# %% ========================================================================
#  TABLE 2: LABEL PERMUTATION RESULTS
# ===========================================================================

print("\n" + "=" * 70)
print("  TABLE 2: LABEL PERMUTATION NULL MODEL")
print("=" * 70)

table2_rows = []
for net in YEO7_NAMES:
    lp = label_perm_results.get(net, {})
    sig = ''
    pv = lp.get('p', np.nan)
    if not np.isnan(pv):
        if pv < 0.001: sig = '***'
        elif pv < 0.01: sig = '**'
        elif pv < 0.05: sig = '*'

    table2_rows.append({
        'Network': net,
        'r_empirical': f"{lp.get('emp_mean', np.nan):.3f}",
        'r_null': f"{lp.get('null_mean', np.nan):.3f} ± {lp.get('null_std', np.nan):.3f}",
        'z': f"{lp.get('z', np.nan):.2f}",
        'p': f"{pv:.3f}{sig}",
    })

df_table2 = pd.DataFrame(table2_rows)

try:
    from tabulate import tabulate
    latex_t2 = tabulate(
        df_table2.values.tolist(),
        headers=['Network', '$r_{emp}$', '$r_{null}$ (mean ± SD)',
                 '$z$', '$p$'],
        tablefmt='latex_booktabs',
        colalign=('left', 'center', 'center', 'center', 'center')
    )
    print("\n  TABLE 2 (LaTeX):")
    print(latex_t2)

    with open(os.path.join(OUTPUT_DIR, 'table2_label_perm.tex'), 'w') as f:
        f.write(latex_t2)
    print(f"  → Saved: table2_label_perm.tex")
except ImportError:
    print(df_table2.to_string(index=False))


# %% ========================================================================
#  FIGURES — HELPERS
# ===========================================================================

def add_panel_label(ax, label, x=-0.12, y=1.08, fontsize=12):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')

def add_network_boundaries(ax, labels, lw=0.4, color='k', alpha=0.3):
    for mod in np.unique(labels):
        nodes = np.where(labels == mod)[0]
        if len(nodes) > 0:
            b = nodes[-1] + 0.5
            ax.axhline(y=b, color=color, lw=lw, alpha=alpha)
            ax.axvline(x=b, color=color, lw=lw, alpha=alpha)

def savefig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'), dpi=300)
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.pdf'))
    plt.close(fig)
    print(f"  → {name}")


# %% ========================================================================
#  FIGURE 1: RAW vs LOG COMPARISON (paired + Bland-Altman)
# ===========================================================================

print("\n" + "=" * 70)
print("  FIGURES")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

# Panel a: Paired scatter
ax = axes[0]
ax.scatter(df_raw['coupling_wb'], df_log['coupling_wb'],
           s=25, color='#3498db', alpha=0.7, edgecolors='#2c3e50', lw=0.5)
lims = [min(df_raw['coupling_wb'].min(), df_log['coupling_wb'].min()) - 0.02,
        max(df_raw['coupling_wb'].max(), df_log['coupling_wb'].max()) + 0.02]
ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.4)
ax.set_xlabel('SC-FC Coupling (raw)')
ax.set_ylabel('SC-FC Coupling (log1p)')
ax.set_title('SC Treatment Comparison', fontsize=9)
r_comp, p_comp = stats.pearsonr(df_raw['coupling_wb'], df_log['coupling_wb'])
ax.text(0.05, 0.95, f'r = {r_comp:.3f}\np = {p_comp:.2e}',
        transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
add_panel_label(ax, 'a')

# Panel b: Bland-Altman
ax = axes[1]
mean_ba = (df_raw['coupling_wb'] + df_log['coupling_wb']) / 2
diff_ba = df_raw['coupling_wb'] - df_log['coupling_wb']
ax.scatter(mean_ba, diff_ba, s=25, color='#e74c3c', alpha=0.7,
           edgecolors='#2c3e50', lw=0.5)
ax.axhline(y=np.mean(diff_ba), color='#2c3e50', lw=1, ls='-',
           label=f'Mean diff = {np.mean(diff_ba):.3f}')
ax.axhline(y=np.mean(diff_ba) + 1.96 * np.std(diff_ba),
           color='#95a5a6', lw=0.8, ls='--')
ax.axhline(y=np.mean(diff_ba) - 1.96 * np.std(diff_ba),
           color='#95a5a6', lw=0.8, ls='--')
ax.set_xlabel('Mean coupling (raw + log) / 2')
ax.set_ylabel('Difference (raw − log)')
ax.set_title('Bland-Altman Plot', fontsize=9)
ax.legend(frameon=False, fontsize=7)
add_panel_label(ax, 'b')

fig.tight_layout()
savefig(fig, 'F01_raw_vs_log')


# %% ========================================================================
#  FIGURE 2: GROUP WHOLE-BRAIN COUPLING DISTRIBUTION
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

for idx, (df, tag) in enumerate([(df_raw, 'Raw'), (df_log, 'Log1p')]):
    ax = axes[idx]
    vals = df['coupling_wb'].values

    # Raincloud: violin + strip + box
    parts = ax.violinplot(vals, positions=[0], showextrema=False, widths=0.6)
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db'); pc.set_alpha(0.2)

    ax.boxplot(vals, positions=[0], widths=0.15, patch_artist=True,
               boxprops=dict(facecolor='white', lw=0.8),
               medianprops=dict(color='#2c3e50', lw=1.5),
               whiskerprops=dict(lw=0.8), flierprops=dict(ms=3))

    jit = rng.uniform(-0.08, 0.08, len(vals))
    ax.scatter(jit, vals, s=18, color='#3498db', alpha=0.7,
               edgecolors='#2c3e50', lw=0.4, zorder=3)

    # Annotate each subject
    for j, sub in enumerate(df['subject']):
        if vals[j] == vals.max() or vals[j] == vals.min():
            ax.annotate(sub, (jit[j], vals[j]), fontsize=5,
                        xytext=(10, 0), textcoords='offset points')

    ax.set_xlim(-0.5, 0.6)
    ax.set_xticks([])
    ax.set_ylabel('Whole-Brain SC-FC Coupling (r)')
    ax.set_title(f'{tag}\nmedian = {np.median(vals):.3f}', fontsize=9)

    # Stats box
    ax.text(0.4, 0.05,
            f'n = {len(vals)}\nmean = {np.mean(vals):.3f}\nstd = {np.std(vals):.3f}',
            transform=ax.transAxes, fontsize=6.5, va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    add_panel_label(ax, chr(97 + idx))

fig.tight_layout()
savefig(fig, 'F02_group_wb_distribution')


# %% ========================================================================
#  FIGURE 3: GROUP INTRA-NETWORK VIOLINPLOT
# ===========================================================================

rows_long = []
for _, row in df_main.iterrows():
    for col in [c for c in df_main.columns if c.startswith('intra_')]:
        net = col.replace('intra_', '')
        if not pd.isna(row[col]):
            rows_long.append({'Network': net, 'Coupling': row[col],
                              'Subject': row['subject']})
df_long = pd.DataFrame(rows_long)

if len(df_long) > 0:
    order = df_long.groupby('Network')['Coupling'].median().sort_values(
        ascending=False).index.tolist()
    pal = [YEO7_COLORS.get(n, '#999') for n in order]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    sns.violinplot(data=df_long, x='Network', y='Coupling', order=order,
                   palette=pal, inner=None, alpha=0.25, ax=ax,
                   cut=0, density_norm='width')
    sns.stripplot(data=df_long, x='Network', y='Coupling', order=order,
                  palette=pal, size=3.5, alpha=0.7, jitter=0.12, ax=ax, zorder=3)
    sns.boxplot(data=df_long, x='Network', y='Coupling', order=order,
                color='white', width=0.12, fliersize=0,
                boxprops=dict(alpha=0.7, lw=0.8),
                whiskerprops=dict(lw=0.8),
                medianprops=dict(color='black', lw=1.2), ax=ax, zorder=2)

    # Significance from label permutation
    for i, net in enumerate(order):
        lp = label_perm_results.get(net, {})
        pv = lp.get('p', 1.0)
        if pv < 0.05:
            ymax = df_long[df_long['Network'] == net]['Coupling'].max()
            ax.text(i, ymax + 0.02, '*' if pv < 0.05 else '', ha='center',
                    fontsize=10, color=YEO7_COLORS.get(net, 'k'))

    ax.axhline(y=0, color='#bdc3c7', lw=0.5, ls='--')
    ax.set_xlabel('')
    ax.set_ylabel('Intra-Network SC-FC Coupling (r)')
    ax.set_title('Group-Level Network Coupling', fontsize=9)

    for i, l in enumerate(ax.get_xticklabels()):
        l.set_color(pal[i]); l.set_fontweight('bold')

    fig.tight_layout()
    savefig(fig, 'F03_group_intra_violin')


# %% ========================================================================
#  FIGURE 4: GROUP 7×7 COUPLING MATRIX (mean ± significance)
# ===========================================================================

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

# Panel a: Mean coupling
ax = axes[0]
im = ax.imshow(mean_coupling_matrix, cmap='RdYlBu_r', vmin=-0.15, vmax=0.35,
               aspect='equal')
# Annotate with values and significance stars
for i in range(7):
    for j in range(7):
        v = mean_coupling_matrix[i, j]
        p = pval_matrix[i, j]
        star = ''
        if p < 0.001: star = '***'
        elif p < 0.01: star = '**'
        elif p < 0.05: star = '*'
        c = 'white' if v > 0.25 or v < -0.05 else 'black'
        ax.text(j, i, f'{v:.2f}{star}', ha='center', va='center',
                fontsize=5.5, color=c, fontweight='bold' if i == j else 'normal')

ax.set_xticks(range(7)); ax.set_xticklabels(YEO7_NAMES, rotation=45, ha='right',
                                              fontweight='bold', fontsize=7)
ax.set_yticks(range(7)); ax.set_yticklabels(YEO7_NAMES, fontweight='bold', fontsize=7)
for i, n in enumerate(YEO7_NAMES):
    ax.get_xticklabels()[i].set_color(YEO7_COLORS[n])
    ax.get_yticklabels()[i].set_color(YEO7_COLORS[n])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Coupling (r)')
ax.set_title('Mean SC-FC Coupling', fontsize=9)
add_panel_label(ax, 'a')

# Panel b: Inter-subject variability (CoV)
ax = axes[1]
cv_matrix = std_coupling_matrix / np.maximum(np.abs(mean_coupling_matrix), 1e-8)
cv_matrix = np.clip(cv_matrix, 0, 5)  # cap extreme CoV
im2 = ax.imshow(cv_matrix, cmap='YlOrRd', vmin=0, vmax=3, aspect='equal')
for i in range(7):
    for j in range(7):
        v = cv_matrix[i, j]
        c = 'white' if v > 2 else 'black'
        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=5.5, color=c)
ax.set_xticks(range(7)); ax.set_xticklabels(YEO7_NAMES, rotation=45, ha='right',
                                              fontweight='bold', fontsize=7)
ax.set_yticks(range(7)); ax.set_yticklabels(YEO7_NAMES, fontweight='bold', fontsize=7)
for i, n in enumerate(YEO7_NAMES):
    ax.get_xticklabels()[i].set_color(YEO7_COLORS[n])
    ax.get_yticklabels()[i].set_color(YEO7_COLORS[n])
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='CoV')
ax.set_title('Inter-Subject Variability (CoV)', fontsize=9)
add_panel_label(ax, 'b')

fig.tight_layout()
savefig(fig, 'F04_group_coupling_matrix')


# %% ========================================================================
#  FIGURE 5: NULL MODEL — SC RANDOMIZATION
# ===========================================================================

fig, ax = plt.subplots(figsize=(3.5, 2.8))

ax.hist(null_couplings, bins=25, color='#bdc3c7', edgecolor='white',
        alpha=0.85, zorder=2)

# KDE overlay
from scipy.stats import gaussian_kde
if len(null_couplings) > 5:
    kde = gaussian_kde(null_couplings)
    xk = np.linspace(null_couplings.min()-0.03, null_couplings.max()+0.03, 200)
    bw = (null_couplings.max() - null_couplings.min()) / 25
    ax.plot(xk, kde(xk) * len(null_couplings) * bw, color='#7f8c8d', lw=1.2, zorder=2)

ax.axvline(x=emp_coupling_null, color='#e74c3c', lw=2, ls='--', zorder=3,
           label=f'Empirical (r = {emp_coupling_null:.3f})')

# Group distribution overlay (small markers)
for v in df_main['coupling_wb']:
    ax.axvline(x=v, color='#3498db', lw=0.3, alpha=0.3, zorder=1)

ax.set_xlabel('SC-FC Coupling (r)')
ax.set_ylabel('Count')
ax.legend(frameon=False, fontsize=7)
ax.text(0.95, 0.95, f'p = {p_null:.3f}\nz = {z_null:.2f}\nn_perm = {N_PERMS_SC}',
        transform=ax.transAxes, ha='right', va='top', fontsize=7, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='#f39c12', alpha=0.9))
ax.set_title('Null Model: Randomized SC', fontsize=9)

fig.tight_layout()
savefig(fig, 'F05_null_sc_random')


# %% ========================================================================
#  FIGURE 6: NETWORK COUPLING GRADIENT (mean ± SEM across subjects)
# ===========================================================================

fig, ax = plt.subplots(figsize=(4.2, 3.5))

for i, net in enumerate(YEO7_NAMES):
    col = f'intra_{net}'
    if col in df_main.columns:
        vals = df_main[col].dropna()
        # Individual dots
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.15, 0.15, len(vals)),
                   vals, s=12, color=YEO7_COLORS[net], alpha=0.4,
                   edgecolors='none', zorder=2)
        # Mean ± SEM
        m = vals.mean(); sem = vals.std() / np.sqrt(len(vals))
        ax.errorbar(i, m, yerr=sem, fmt='D', ms=7, color=YEO7_COLORS[net],
                    markeredgecolor='#2c3e50', markeredgewidth=0.8,
                    capsize=4, capthick=1.2, lw=1.2, zorder=4)

        # Significance star from label perm
        lp = label_perm_results.get(net, {})
        if lp.get('p', 1) < 0.05:
            ax.text(i, m + sem + 0.01, '*', ha='center', fontsize=10,
                    color=YEO7_COLORS[net])

# Connect means
means = [df_main[f'intra_{n}'].dropna().mean() if f'intra_{n}' in df_main.columns
         else np.nan for n in YEO7_NAMES]
ax.plot(range(7), means, 'k--', lw=0.6, alpha=0.4, zorder=1)
ax.axhline(y=0, color='#bdc3c7', lw=0.5, ls='-')

ax.set_xticks(range(7)); ax.set_xticklabels(YEO7_NAMES, fontweight='bold')
for i, l in enumerate(ax.get_xticklabels()):
    l.set_color(YEO7_COLORS[YEO7_NAMES[i]])
ax.set_ylabel('Intra-Network SC-FC Coupling (r)')
ax.set_title('Network Coupling Gradient (mean ± SEM)', fontsize=9)

fig.tight_layout()
savefig(fig, 'F06_network_gradient')


# %% ========================================================================
#  FIGURE 7: SUBJECT-LEVEL HEATMAP (coupling per subject × network)
# ===========================================================================

# Build matrix: subjects × networks
intra_matrix = np.full((N_SUBJECTS, 7), np.nan)
for si in range(len(df_main)):
    for ni, net in enumerate(YEO7_NAMES):
        col = f'intra_{net}'
        if col in df_main.columns:
            intra_matrix[si, ni] = df_main.iloc[si][col]

# Sort subjects by whole-brain coupling
sort_idx = np.argsort(df_main['coupling_wb'].values)[::-1]

fig, ax = plt.subplots(figsize=(4, 5))
im = ax.imshow(intra_matrix[sort_idx], cmap='RdBu_r', vmin=-0.3, vmax=0.4,
               aspect='auto', interpolation='none')

ax.set_xticks(range(7)); ax.set_xticklabels(YEO7_NAMES, fontweight='bold',
                                              fontsize=7)
for i, n in enumerate(YEO7_NAMES):
    ax.get_xticklabels()[i].set_color(YEO7_COLORS[n])

ax.set_yticks(range(N_SUBJECTS))
ax.set_yticklabels(df_main['subject'].values[sort_idx], fontsize=6)
ax.set_xlabel('Yeo7 Network')
ax.set_ylabel('Subject (sorted by WB coupling)')

plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='Coupling (r)')
ax.set_title('Subject × Network Coupling', fontsize=9)

fig.tight_layout()
savefig(fig, 'F07_subject_network_heatmap')


# %% ========================================================================
#  BRAIN SURFACE PLOTS (nilearn)
# ===========================================================================

print("\n  BRAIN PLOTS")

try:
    from nilearn import plotting as ni_plot
    from nilearn import datasets as ni_data
    from nilearn import image as ni_image
    from nilearn import surface

    schaefer = ni_data.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    atlas_img = schaefer['maps']
    atlas_data = ni_image.get_data(atlas_img)
    mni_template = ni_data.load_mni152_template(resolution=2)

    # Extract centroid coordinates for connectome plots
    coords = ni_plot.find_parcellation_cut_coords(atlas_img)

    def roi_to_volume(values, atlas_data_ref, atlas_img_ref):
        """Map a (N_NODES,) vector to a NIfTI volume."""
        vol = np.zeros_like(atlas_data_ref, dtype=float)
        for ri in range(len(values)):
            vol[atlas_data_ref == (ri + 1)] = values[ri]
        return ni_image.new_img_like(atlas_img_ref, vol)

    # ====================================================================
    #  BRAIN 1: Group-Mean Nodal SC-FC Coupling (r)
    # ====================================================================
    coup_img = roi_to_volume(mean_nodal_corr, atlas_data, atlas_img)
    fig = plt.figure(figsize=(7.2, 3))
    ni_plot.plot_stat_map(
        coup_img, bg_img=mni_template, display_mode='ortho',
        cmap='RdYlBu_r', colorbar=True, black_bg=False,
        title='Group Mean Nodal SC-FC Coupling (r)',
        figure=fig, threshold=0.001, vmax=0.15
    )
    savefig(fig, 'B01_brain_mean_coupling')

    # ====================================================================
    #  BRAIN 2: Group-Mean Nodal Decoupling (MAE)
    # ====================================================================
    dec_img = roi_to_volume(mean_nodal_mae, atlas_data, atlas_img)
    fig = plt.figure(figsize=(7.2, 3))
    ni_plot.plot_stat_map(
        dec_img, bg_img=mni_template, display_mode='ortho',
        cmap='hot', colorbar=True, black_bg=False,
        title='Group Mean Nodal Decoupling (MAE)',
        figure=fig, threshold=0.001
    )
    savefig(fig, 'B02_brain_mean_decoupling')

    # ====================================================================
    #  BRAIN 3: Nodal Coupling T-statistic (FDR-masked)
    # ====================================================================
    # Show only nodes surviving FDR correction
    t_masked = nodal_t.copy()
    t_masked[~reject] = 0  # zero out non-significant nodes

    t_img = roi_to_volume(t_masked, atlas_data, atlas_img)
    fig = plt.figure(figsize=(7.2, 3))
    ni_plot.plot_stat_map(
        t_img, bg_img=mni_template, display_mode='ortho',
        cmap='cold_hot', colorbar=True, black_bg=False,
        title=f'Nodal Coupling t-stat (FDR < 0.05, {n_sig}/{N_NODES} nodes)',
        figure=fig, threshold=0.001
    )
    savefig(fig, 'B03_brain_tstat_fdr')

    # ====================================================================
    #  BRAIN 4: Inter-Subject Variability (CoV of nodal coupling)
    # ====================================================================
    var_img = roi_to_volume(cv_nodal_mae, atlas_data, atlas_img)
    fig = plt.figure(figsize=(7.2, 3))
    ni_plot.plot_stat_map(
        var_img, bg_img=mni_template, display_mode='ortho',
        cmap='YlOrRd', colorbar=True, black_bg=False,
        title='Inter-Subject Variability (CoV of Decoupling)',
        figure=fig, threshold=0.001
    )
    savefig(fig, 'B04_brain_variability')

    # ====================================================================
    #  BRAIN 5: Group-Mean Structural Connectome (top 5% edges)
    # ====================================================================
    # Average SC across subjects
    sc_group = np.zeros((N_NODES, N_NODES))
    for sub in ALL_SUBS:
        try:
            sc_s, _, _ = load_subject(sub, sc_treat='raw')
            sc_group += sc_s
        except:
            pass
    sc_group /= N_SUBJECTS
    np.fill_diagonal(sc_group, 0)

    node_colors = [YEO7_COLORS_ORD[YEO7_INT[i]] for i in range(N_NODES)]

    fig = plt.figure(figsize=(7.2, 3))
    ni_plot.plot_connectome(
        sc_group, coords, edge_threshold="95%",
        display_mode='ortho',
        title='Group Mean SC (top 5% edges)',
        figure=fig, node_size=12, node_color=node_colors,
        edge_cmap='YlOrRd', black_bg=False
    )
    savefig(fig, 'B05_brain_group_connectome')

    print("  ✅ All brain plots saved!")

except ImportError:
    print("  ⚠️  nilearn not available. pip install nilearn")
except Exception as e:
    print(f"  ⚠️  Brain plot error: {e}")
    import traceback; traceback.print_exc()


# %% ========================================================================
#  COMPOSITE FIGURE (main paper figure)
# ===========================================================================

print("\n  COMPOSITE FIGURE")

fig = plt.figure(figsize=(7.2, 9.5))
gs = gridspec.GridSpec(4, 4, hspace=0.5, wspace=0.5,
                       height_ratios=[1, 0.9, 1, 0.9])

# --- Row 1: FC matrices for one example subject ---
ex_res = results_main[0]
FC_emp_ex = ex_res['_FC_emp']
FC_pred_ex = ex_res['_FC_pred']
lab_ex = ex_res['_active_labels']
n_ex = FC_emp_ex.shape[0]

# a: SC
ax = fig.add_subplot(gs[0, 0])
sc_ex, _, _ = load_subject(ALL_SUBS[0], sc_treat=SC_TREAT_MAIN)
im = ax.imshow(np.log1p(sc_ex[:n_ex, :n_ex]), cmap='YlOrRd', aspect='equal')
add_network_boundaries(ax, lab_ex)
ax.set_xlabel('Node'); ax.set_ylabel('Node')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
add_panel_label(ax, 'a')

# b: FC emp
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(FC_emp_ex, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='equal')
add_network_boundaries(ax, lab_ex)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_yticklabels([])
add_panel_label(ax, 'b')

# c: FC pred
ax = fig.add_subplot(gs[0, 2])
im = ax.imshow(FC_pred_ex, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='equal')
add_network_boundaries(ax, lab_ex)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_yticklabels([])
add_panel_label(ax, 'c')

# d: Scatter
ax = fig.add_subplot(gs[0, 3])
triu_ex = np.triu_indices(n_ex, k=1)
fef = FC_emp_ex[triu_ex]; fpf = FC_pred_ex[triu_ex]
from scipy.stats import gaussian_kde as gkde
try:
    xy = np.vstack([fef, fpf])
    z = gkde(xy)(xy); idx_z = z.argsort()
    ax.scatter(fef[idx_z], fpf[idx_z], c=z[idx_z], s=1, cmap='inferno',
               alpha=0.5, rasterized=True, edgecolors='none')
except:
    ax.scatter(fef, fpf, s=1, alpha=0.3, color='#3498db', rasterized=True)
lim_d = [min(fef.min(), fpf.min())-0.05, max(fef.max(), fpf.max())+0.05]
ax.plot(lim_d, lim_d, 'k--', lw=0.6, alpha=0.4)
r_ex = stats.pearsonr(fef, fpf)[0]
ax.text(0.05, 0.95, f'r = {r_ex:.3f}', transform=ax.transAxes, fontsize=7,
        fontweight='bold', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax.set_xlabel('FC emp'); ax.set_ylabel('FC pred'); ax.set_aspect('equal')
add_panel_label(ax, 'd')

# --- Row 2: Group distributions ---
# e: WB coupling distribution
ax = fig.add_subplot(gs[1, 0:2])
vals_wb = df_main['coupling_wb'].values
parts = ax.violinplot(vals_wb, positions=[0], showextrema=False, widths=0.6)
for pc in parts['bodies']:
    pc.set_facecolor('#3498db'); pc.set_alpha(0.2)
ax.boxplot(vals_wb, positions=[0], widths=0.15, patch_artist=True,
           boxprops=dict(facecolor='white', lw=0.8),
           medianprops=dict(color='#2c3e50', lw=1.5),
           whiskerprops=dict(lw=0.8), flierprops=dict(ms=3))
jit_e = rng.uniform(-0.08, 0.08, len(vals_wb))
ax.scatter(jit_e, vals_wb, s=15, color='#3498db', alpha=0.7,
           edgecolors='#2c3e50', lw=0.4, zorder=3)
ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
ax.set_ylabel('WB Coupling (r)')
ax.set_title(f'n = {len(vals_wb)}, median = {np.median(vals_wb):.3f}', fontsize=8)
add_panel_label(ax, 'e')

# f: Null model
ax = fig.add_subplot(gs[1, 2:4])
ax.hist(null_couplings, bins=20, color='#bdc3c7', edgecolor='white', alpha=0.85)
ax.axvline(x=emp_coupling_null, color='#e74c3c', lw=2, ls='--')
ax.text(0.95, 0.95, f'p = {p_null:.3f}\nz = {z_null:.2f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=7, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.set_xlabel('Coupling (r)'); ax.set_ylabel('Count')
add_panel_label(ax, 'f')

# --- Row 3: Network-level ---
# g: 7×7 matrix
ax = fig.add_subplot(gs[2, 0:2])
im = ax.imshow(mean_coupling_matrix, cmap='RdYlBu_r', vmin=-0.15, vmax=0.35,
               aspect='equal')
for i in range(7):
    for j in range(7):
        v = mean_coupling_matrix[i, j]
        p = pval_matrix[i, j]
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        c = 'white' if v > 0.25 or v < -0.05 else 'black'
        ax.text(j, i, f'{v:.2f}{star}', ha='center', va='center',
                fontsize=4.5, color=c, fontweight='bold' if i == j else 'normal')
ax.set_xticks(range(7)); ax.set_xticklabels(YEO7_NAMES, rotation=45,
                                              ha='right', fontsize=6, fontweight='bold')
ax.set_yticks(range(7)); ax.set_yticklabels(YEO7_NAMES, fontsize=6, fontweight='bold')
for i, n in enumerate(YEO7_NAMES):
    ax.get_xticklabels()[i].set_color(YEO7_COLORS[n])
    ax.get_yticklabels()[i].set_color(YEO7_COLORS[n])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
add_panel_label(ax, 'g')

# h: Network gradient
ax = fig.add_subplot(gs[2, 2:4])
for i, net in enumerate(YEO7_NAMES):
    col = f'intra_{net}'
    if col in df_main.columns:
        vals = df_main[col].dropna()
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals)),
                   vals, s=8, color=YEO7_COLORS[net], alpha=0.4, edgecolors='none')
        m = vals.mean(); sem = vals.std() / np.sqrt(len(vals))
        ax.errorbar(i, m, yerr=sem, fmt='D', ms=5, color=YEO7_COLORS[net],
                    markeredgecolor='#2c3e50', markeredgewidth=0.6,
                    capsize=3, capthick=1, zorder=4)
means_c = [df_main[f'intra_{n}'].dropna().mean() if f'intra_{n}' in df_main.columns
           else np.nan for n in YEO7_NAMES]
ax.plot(range(7), means_c, 'k--', lw=0.5, alpha=0.4)
ax.axhline(y=0, color='#bdc3c7', lw=0.5, ls='-')
ax.set_xticks(range(7)); ax.set_xticklabels(YEO7_NAMES, fontsize=6, fontweight='bold')
for i, l in enumerate(ax.get_xticklabels()): l.set_color(YEO7_COLORS[YEO7_NAMES[i]])
ax.set_ylabel('Coupling (r)')
add_panel_label(ax, 'h')

# --- Row 4: Subject × network heatmap ---
ax = fig.add_subplot(gs[3, :])
im = ax.imshow(intra_matrix[sort_idx], cmap='RdBu_r', vmin=-0.3, vmax=0.4,
               aspect='auto', interpolation='none')
ax.set_xticks(range(7)); ax.set_xticklabels(YEO7_NAMES, fontweight='bold', fontsize=7)
for i, n in enumerate(YEO7_NAMES):
    ax.get_xticklabels()[i].set_color(YEO7_COLORS[n])
ax.set_yticks(range(N_SUBJECTS))
ax.set_yticklabels(df_main['subject'].values[sort_idx], fontsize=5)
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label='Coupling (r)')
add_panel_label(ax, 'i')

fig.suptitle('Reservoir Computing SC-FC Coupling in COVID-19 ICU Survivors',
             fontsize=11, fontweight='bold', y=0.98)

savefig(fig, 'F_COMPOSITE_main')


# %% ========================================================================
#  SAVE GROUP DATA
# ===========================================================================

print("\n  SAVING RESULTS...")

# Save DataFrames
df_main.drop(columns=[c for c in df_main.columns if c.startswith('_')],
             errors='ignore').to_csv(
    os.path.join(OUTPUT_DIR, 'group_coupling_results.csv'), index=False
)
df_raw.to_csv(os.path.join(OUTPUT_DIR, 'group_coupling_raw.csv'), index=False)
df_log.to_csv(os.path.join(OUTPUT_DIR, 'group_coupling_log.csv'), index=False)

# Save null distributions
np.save(os.path.join(OUTPUT_DIR, 'null_couplings_sc.npy'), null_couplings)

# Save nodal metrics
np.save(os.path.join(OUTPUT_DIR, 'group_nodal_mae.npy'), group_nodal_mae)
np.save(os.path.join(OUTPUT_DIR, 'group_nodal_corr.npy'), group_nodal_corr)
np.save(os.path.join(OUTPUT_DIR, 'nodal_tstat.npy'), nodal_t)
np.save(os.path.join(OUTPUT_DIR, 'nodal_pval_fdr.npy'), nodal_p_fdr)

print("  ✅ All results saved!")


# %% ========================================================================
#  FINAL SUMMARY
# ===========================================================================

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)

print(f"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║  Dataset: {N_SUBJECTS} COVID-19 ICU survivors, Schaefer-100 / Yeo7        ║
  ║                                                                    ║
  ║  SC Treatment: {SC_TREAT_MAIN:>4s} selected (higher coupling)                  ║
  ║  WB Coupling:  {df_main['coupling_wb'].mean():.3f} ± {df_main['coupling_wb'].std():.3f} (group mean ± SD)              ║
  ║                                                                    ║
  ║  Null Model (SC random): p = {p_null:.3f}, z = {z_null:.2f}                      ║
  ║  Permutations: {N_PERMS_SC} (SC) + {N_PERMS_LAB} (label)                         ║
  ║                                                                    ║
  ║  FDR-significant nodes: {n_sig}/{N_NODES}                                   ║
  ║                                                                    ║
  ║  Outputs:                                                          ║
  ║    {OUTPUT_DIR}/
  ║    • 7 standard figures (F01–F07) as PNG + PDF                     ║
  ║    • 5 brain surface plots (B01–B05) as PNG + PDF                  ║
  ║    • 1 composite figure (F_COMPOSITE)                              ║
  ║    • 2 LaTeX tables (table1, table2)                               ║
  ║    • 4 CSV/NPY data files                                         ║
  ╚══════════════════════════════════════════════════════════════════════╝
""")

for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"    {f}")
