import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

# ============================================================
# Imports do sars.config
# ============================================================
from sars.config import (
    ALL_SUBJECT_IDS,
    ATLASES,
    ATLAS_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    TR,
    get_timeseries_path,
    get_connectivity_path,
)

# ============================================================
# Estilo de publicação (Nature/Brain)
# ============================================================
STYLE_CONFIG = {
    'figure.facecolor': 'white',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelpad': 4,
    'axes.titlepad': 8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.grid': False,
    'legend.frameon': False,
}
plt.rcParams.update(STYLE_CONFIG)

# ============================================================
# Carregar timeseries de um sujeito
# ============================================================
atlas_name = "schaefer_100"
subject = "sub-01"

ts_path = get_timeseries_path(subject, atlas_name)
timeseries = np.load(ts_path)  # shape: (n_timepoints, n_rois)

n_timepoints, n_rois = timeseries.shape
print(f"Timeseries: {subject} | {atlas_name}")
print(f"  Shape: {n_timepoints} timepoints × {n_rois} ROIs")
print(f"  Duration: {n_timepoints * TR / 60:.1f} min")
print(f"  TR: {TR} s")


def load_all_timeseries(atlas_name="schaefer_100"):
    """Carrega timeseries de todos os sujeitos."""
    all_ts = {}
    for sub in ALL_SUBJECT_IDS:
        try:
            ts_path = get_timeseries_path(sub, atlas_name)
            ts = np.load(ts_path)
            all_ts[sub] = ts
        except FileNotFoundError:
            print(f"  [SKIP] {sub}")
    print(f"Carregadas: {len(all_ts)}/{len(ALL_SUBJECT_IDS)} timeseries")
    return all_ts

all_ts = load_all_timeseries("schaefer_100")


def detect_avalanches(timeseries, threshold_sd=2.0, min_duration=1):
    """
    Detectar avalanches neuronais em dados de fMRI.
    
    Uma avalanche é definida como uma cascata contígua de atividade supra-threshold.
    Baseado em Tagliazucchi et al. (2012) e Beggs & Plenz (2003).
    
    Parameters
    ----------
    timeseries : np.ndarray (n_timepoints, n_rois)
        Séries temporais z-scored internamente.
    threshold_sd : float
        Threshold em desvios padrão.
    min_duration : int
        Duração mínima em TRs para contar como avalanche.
    
    Returns
    -------
    dict com:
        - sizes: np.ndarray de tamanhos
        - durations: np.ndarray de durações
        - binary_activity: np.ndarray (n_timepoints, n_rois) binarizada
        - n_active_per_tr: np.ndarray (n_timepoints,) ROIs ativas por TR
        - n_avalanches: int
        - avalanches: list of dicts com detalhes de cada avalanche
    """
    n_timepoints, n_rois = timeseries.shape
    
    # Z-score por ROI (cada ROI normalizada independentemente)
    ts_z = stats.zscore(timeseries, axis=0)
    
    # Binarizar: ROI ativa se |z| > threshold
    binary = (np.abs(ts_z) > threshold_sd).astype(int)
    n_active = binary.sum(axis=1)  # ROIs ativas por TR
    
    # Detectar avalanches como segmentos contíguos de n_active > 0
    avalanches = []
    sizes = []
    durations = []
    
    in_avalanche = False
    start = 0
    current_size = 0
    
    for t in range(n_timepoints):
        if n_active[t] > 0:
            if not in_avalanche:
                in_avalanche = True
                start = t
                current_size = 0
            current_size += n_active[t]
        else:
            if in_avalanche:
                duration = t - start
                if duration >= min_duration:
                    avalanches.append({
                        "start": start,
                        "end": t,
                        "duration": duration,
                        "size": current_size,
                    })
                    sizes.append(current_size)
                    durations.append(duration)
                in_avalanche = False
    
    # Caso a série termine durante uma avalanche
    if in_avalanche:
        duration = n_timepoints - start
        if duration >= min_duration:
            sizes.append(current_size)
            durations.append(duration)
    
    return {
        "sizes": np.array(sizes),
        "durations": np.array(durations),
        "binary_activity": binary,
        "n_active_per_tr": n_active,
        "n_avalanches": len(sizes),
        "avalanches": avalanches,
    }


# ============================================================
# Executar para um sujeito
# ============================================================
aval = detect_avalanches(timeseries, threshold_sd=2.0)

print(f"\nAvalanches — {subject} ({atlas_name})")
print(f"  Total: {aval['n_avalanches']}")
print(f"  Rate: {aval['n_avalanches'] / (n_timepoints * TR / 60):.1f} /min")
if len(aval['sizes']) > 0:
    print(f"  Size: mean={aval['sizes'].mean():.1f}, "
          f"max={aval['sizes'].max()}, median={np.median(aval['sizes']):.0f}")
    print(f"  Duration: mean={aval['durations'].mean():.1f} TRs, "
          f"max={aval['durations'].max()}")
    
    
try:
    import powerlaw
    HAS_POWERLAW = True
except ImportError:
    HAS_POWERLAW = False
    print("pip install powerlaw")


def fit_powerlaw(data, xmin=None):
    """
    Ajustar lei de potência usando o método de Clauset et al. (2009).
    
    Compara automaticamente com exponencial e lognormal.
    
    Returns
    -------
    dict com alpha, xmin, sigma, KS, e comparações com alternativas.
    """
    if len(data) < 10:
        return {"alpha": np.nan, "xmin": np.nan, "sigma": np.nan}
    
    if HAS_POWERLAW:
        try:
            fit = powerlaw.Fit(data, xmin=xmin, discrete=True)
            
            R_exp, p_exp = fit.distribution_compare('power_law', 'exponential')
            R_ln, p_ln = fit.distribution_compare('power_law', 'lognormal')
            
            return {
                "alpha": fit.power_law.alpha,
                "xmin": fit.power_law.xmin,
                "sigma": fit.power_law.sigma,
                "ks_statistic": fit.power_law.KS(),
                "comparison_exponential": {"R": R_exp, "p": p_exp},
                "comparison_lognormal": {"R": R_ln, "p": p_ln},
                "fit_object": fit,
            }
        except Exception as e:
            print(f"  powerlaw fit error: {e}")
    
    # Fallback: MLE simples (Hill estimator)
    data = np.array(data, dtype=float)
    data = data[data > 0]
    if len(data) < 5:
        return {"alpha": np.nan, "xmin": np.nan, "sigma": np.nan}
    
    xmin_val = np.min(data) if xmin is None else xmin
    data_above = data[data >= xmin_val]
    if len(data_above) < 5:
        return {"alpha": np.nan, "xmin": xmin_val, "sigma": np.nan}
    
    alpha = 1 + len(data_above) / np.sum(np.log(data_above / (xmin_val - 0.5)))
    sigma = (alpha - 1) / np.sqrt(len(data_above))
    
    return {"alpha": alpha, "xmin": xmin_val, "sigma": sigma}


# ============================================================
# Executar
# ============================================================
size_fit = fit_powerlaw(aval["sizes"])
duration_fit = fit_powerlaw(aval["durations"])

print(f"\nPower-Law Fit — Sizes")
print(f"  α = {size_fit['alpha']:.3f} ± {size_fit.get('sigma', np.nan):.3f}")
print(f"  x_min = {size_fit['xmin']}")
if "comparison_exponential" in size_fit:
    print(f"  vs Exponential: R={size_fit['comparison_exponential']['R']:.2f}, "
          f"p={size_fit['comparison_exponential']['p']:.4f}")
    print(f"  vs Lognormal: R={size_fit['comparison_lognormal']['R']:.2f}, "
          f"p={size_fit['comparison_lognormal']['p']:.4f}")

print(f"\nPower-Law Fit — Durations")
print(f"  τ = {duration_fit['alpha']:.3f} ± {duration_fit.get('sigma', np.nan):.3f}")

def compute_branching_ratio(timeseries, threshold_sd=2.0):
    """
    Branching ratio: σ = <n_active(t+1) / n_active(t)>
    
    Calculado apenas para TRs onde n_active(t) > 0.
    """
    ts_z = stats.zscore(timeseries, axis=0)
    binary = (np.abs(ts_z) > threshold_sd).astype(int)
    n_active = binary.sum(axis=1)
    
    ratios = []
    for t in range(len(n_active) - 1):
        if n_active[t] > 0:
            ratios.append(n_active[t + 1] / n_active[t])
    
    ratios = np.array(ratios)
    
    return {
        "sigma": np.mean(ratios) if len(ratios) > 0 else np.nan,
        "sigma_std": np.std(ratios) if len(ratios) > 0 else np.nan,
        "sigma_median": np.median(ratios) if len(ratios) > 0 else np.nan,
        "n_ratios": len(ratios),
    }


br = compute_branching_ratio(timeseries)
print(f"\nBranching Ratio — {subject}")
print(f"  σ = {br['sigma']:.4f} ± {br['sigma_std']:.4f}")
print(f"  (Critical = 1.0, subcritical < 1, supercritical > 1)")

def compute_dfa(signal, scales=None):
    """
    Detrended Fluctuation Analysis.
    
    Parameters
    ----------
    signal : np.ndarray
        Se 2D (n_timepoints, n_rois), usa a média global.
        Se 1D, usa diretamente.
    scales : np.ndarray, optional
        Escalas de tempo (em TRs). Auto-detectadas se None.
    
    Returns
    -------
    dict com alpha, r_squared, scales, fluctuations.
    """
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    
    signal = signal - signal.mean()
    N = len(signal)
    
    # Perfil (cumulative sum)
    profile = np.cumsum(signal)
    
    # Escalas logaritmicamente espaçadas
    if scales is None:
        scales = np.unique(np.logspace(
            np.log10(4), np.log10(N // 4), 20
        ).astype(int))
    
    fluctuations = []
    valid_scales = []
    
    for s in scales:
        n_segments = N // s
        if n_segments < 2:
            continue
        
        # Dividir em segmentos e detrend
        rms_list = []
        for i in range(n_segments):
            segment = profile[i * s : (i + 1) * s]
            x = np.arange(s)
            
            # Detrend linear
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            residual = segment - trend
            
            rms = np.sqrt(np.mean(residual ** 2))
            rms_list.append(rms)
        
        if len(rms_list) > 0:
            fluctuations.append(np.mean(rms_list))
            valid_scales.append(s)
    
    scales = np.array(valid_scales)
    fluctuations = np.array(fluctuations)
    
    if len(scales) < 3:
        return {"alpha": np.nan, "scales": scales, "fluctuations": fluctuations}
    
    log_s = np.log(scales)
    log_f = np.log(fluctuations)
    
    valid = np.isfinite(log_s) & np.isfinite(log_f)
    if valid.sum() < 3:
        return {"alpha": np.nan, "scales": scales, "fluctuations": fluctuations}
    
    slope, intercept, r, p, se = stats.linregress(log_s[valid], log_f[valid])
    
    return {
        "alpha": slope,
        "alpha_se": se,
        "r_squared": r ** 2,
        "intercept": intercept,
        "scales": scales,
        "fluctuations": fluctuations,
    }


def compute_dfa_per_roi(timeseries):
    """DFA para cada ROI individualmente."""
    n_t, n_rois = timeseries.shape
    alphas = np.array([compute_dfa(timeseries[:, i])["alpha"] for i in range(n_rois)])
    return {
        "alpha_per_roi": alphas,
        "alpha_mean": np.nanmean(alphas),
        "alpha_std": np.nanstd(alphas),
    }


dfa = compute_dfa(timeseries)
print(f"\nDFA — {subject}")
print(f"  α_DFA = {dfa['alpha']:.4f} (R² = {dfa['r_squared']:.4f})")
print(f"  (Critical ≈ 1.0, white noise = 0.5, Brownian = 1.5)")

dfa_rois = compute_dfa_per_roi(timeseries)
print(f"  Per-ROI: mean={dfa_rois['alpha_mean']:.4f} ± {dfa_rois['alpha_std']:.4f}")

def analyze_criticality(timeseries, threshold_sd=2.0):
    """
    Pipeline completo de criticalidade para uma timeseries.
    
    Returns dict com todas as métricas + dados brutos para plots.
    """
    results = {}
    
    # 1. Avalanches
    aval = detect_avalanches(timeseries, threshold_sd)
    results["n_avalanches"] = aval["n_avalanches"]
    results["avalanche_rate"] = aval["n_avalanches"] / (len(timeseries) * TR / 60)
    results["_avalanche_data"] = aval  # guardar para plots
    
    if len(aval["sizes"]) > 0:
        results["mean_size"] = aval["sizes"].mean()
        results["max_size"] = int(aval["sizes"].max())
        results["mean_duration"] = aval["durations"].mean()
        results["max_duration"] = int(aval["durations"].max())
    
    # 2. Power-law fits
    if len(aval["sizes"]) >= 10:
        sf = fit_powerlaw(aval["sizes"])
        results["alpha_size"] = sf.get("alpha", np.nan)
        results["alpha_size_sigma"] = sf.get("sigma", np.nan)
        results["_size_fit"] = sf
        
        df = fit_powerlaw(aval["durations"])
        results["tau_duration"] = df.get("alpha", np.nan)
        results["tau_duration_sigma"] = df.get("sigma", np.nan)
        results["_duration_fit"] = df
    else:
        results["alpha_size"] = np.nan
        results["tau_duration"] = np.nan
    
    # 3. Scaling relation γ
    alpha = results.get("alpha_size", np.nan)
    tau = results.get("tau_duration", np.nan)
    if np.isfinite(alpha) and np.isfinite(tau) and tau != 1:
        results["gamma_theoretical"] = (alpha - 1) / (tau - 1)
    else:
        results["gamma_theoretical"] = np.nan
    
    # Empirical gamma (log-log regression size vs duration)
    if len(aval["sizes"]) > 5:
        valid = (aval["sizes"] > 0) & (aval["durations"] > 0)
        if valid.sum() > 5:
            slope, _, r, _, _ = stats.linregress(
                np.log(aval["durations"][valid]),
                np.log(aval["sizes"][valid])
            )
            results["gamma_empirical"] = slope
        else:
            results["gamma_empirical"] = np.nan
    else:
        results["gamma_empirical"] = np.nan
    
    # 4. Branching ratio
    br = compute_branching_ratio(timeseries, threshold_sd)
    results["branching_ratio"] = br["sigma"]
    results["branching_ratio_std"] = br["sigma_std"]
    
    # 5. DFA global
    dfa_result = compute_dfa(timeseries)
    results["dfa_alpha"] = dfa_result["alpha"]
    results["dfa_r_squared"] = dfa_result.get("r_squared", np.nan)
    results["_dfa_data"] = dfa_result
    
    # 6. DFA per ROI
    dfa_roi = compute_dfa_per_roi(timeseries)
    results["dfa_alpha_mean"] = dfa_roi["alpha_mean"]
    results["dfa_alpha_std"] = dfa_roi["alpha_std"]
    results["_dfa_per_roi"] = dfa_roi["alpha_per_roi"]
    
    # 7. Criticality Index (composto)
    br_score = max(0, 1 - abs(results["branching_ratio"] - 1)) if np.isfinite(results["branching_ratio"]) else 0
    dfa_score = max(0, 1 - abs(results["dfa_alpha"] - 1)) if np.isfinite(results["dfa_alpha"]) else 0
    gamma_val = results.get("gamma_theoretical", np.nan)
    gamma_score = max(0, 1 - abs(gamma_val - 2) / 2) if np.isfinite(gamma_val) else 0
    results["criticality_index"] = np.mean([br_score, dfa_score, gamma_score])
    
    return results


# Executar para um sujeito
crit = analyze_criticality(timeseries)

print(f"\n{'='*55}")
print(f"  CRITICALITY ANALYSIS — {subject} ({atlas_name})")
print(f"{'='*55}")
print(f"  Avalanches:       {crit['n_avalanches']}")
print(f"  Rate:             {crit['avalanche_rate']:.1f} /min")
print(f"  α (size):         {crit.get('alpha_size', np.nan):.3f}")
print(f"  τ (duration):     {crit.get('tau_duration', np.nan):.3f}")
print(f"  γ (theoretical):  {crit.get('gamma_theoretical', np.nan):.3f}")
print(f"  γ (empirical):    {crit.get('gamma_empirical', np.nan):.3f}")
print(f"  σ (branching):    {crit['branching_ratio']:.4f}")
print(f"  α_DFA:            {crit['dfa_alpha']:.4f}")
print(f"  Criticality Idx:  {crit['criticality_index']:.3f}")
print(f"{'='*55}")

def fig1_avalanche_raster(timeseries, aval_data, subject_id, atlas_name,
                           t_start=0, t_end=100, save_path=None):
    """
    FIGURA 1: Raster plot com detecção de avalanches.
    
    4 painéis verticais no estilo Brain/NeuroImage:
    A) Raw timeseries (heatmap)
    B) Binary raster (supra-threshold activity)
    C) Number of active ROIs per TR (avalanche profile)
    D) Avalanche events highlighted
    """
    binary = aval_data["binary_activity"]
    n_active = aval_data["n_active_per_tr"]
    avalanches = aval_data["avalanches"]
    
    # Recortar janela temporal
    t0, t1 = t_start, min(t_end, len(timeseries))
    time_axis = np.arange(t0, t1) * TR  # em segundos
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1.5, 1, 1], hspace=0.08)
    
    # --- Painel A: Raw timeseries heatmap ---
    ax_a = fig.add_subplot(gs[0])
    ts_z = stats.zscore(timeseries, axis=0)
    im = ax_a.imshow(ts_z[t0:t1].T, aspect='auto', cmap='RdBu_r',
                      vmin=-3, vmax=3, interpolation='none',
                      extent=[time_axis[0], time_axis[-1], 0, ts_z.shape[1]])
    ax_a.set_ylabel('ROI index', fontsize=11)
    ax_a.set_title(f'{subject_id} — {atlas_name}', fontsize=12, fontweight='bold', loc='left')
    ax_a.text(-0.02, 1.05, 'A', transform=ax_a.transAxes,
              fontsize=16, fontweight='bold', va='bottom')
    cb = plt.colorbar(im, ax=ax_a, shrink=0.6, pad=0.02)
    cb.set_label('z-score', fontsize=9)
    ax_a.set_xticklabels([])
    
    # --- Painel B: Binary raster ---
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_b.imshow(binary[t0:t1].T, aspect='auto', cmap='Greys',
                interpolation='none',
                extent=[time_axis[0], time_axis[-1], 0, binary.shape[1]])
    ax_b.set_ylabel('ROI index', fontsize=11)
    ax_b.text(-0.02, 1.05, 'B', transform=ax_b.transAxes,
              fontsize=16, fontweight='bold', va='bottom')
    ax_b.set_xticklabels([])
    
    # --- Painel C: Avalanche profile (n_active over time) ---
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)
    ax_c.fill_between(time_axis, n_active[t0:t1], alpha=0.6, color='steelblue')
    ax_c.plot(time_axis, n_active[t0:t1], color='steelblue', lw=0.8)
    ax_c.set_ylabel('Active\nROIs', fontsize=10)
    ax_c.text(-0.02, 1.05, 'C', transform=ax_c.transAxes,
              fontsize=16, fontweight='bold', va='bottom')
    ax_c.set_xticklabels([])
    
    # --- Painel D: Avalanches highlighted ---
    ax_d = fig.add_subplot(gs[3], sharex=ax_a)
    ax_d.fill_between(time_axis, n_active[t0:t1], alpha=0.15, color='gray')
    
    # Colorir cada avalanche
    cmap_aval = plt.cm.Set2
    for i, av in enumerate(avalanches):
        if av["start"] >= t0 and av["end"] <= t1:
            t_av = np.arange(av["start"], av["end"]) * TR
            ax_d.fill_between(t_av, n_active[av["start"]:av["end"]],
                             alpha=0.7, color=cmap_aval(i % 8))
    
    ax_d.set_xlabel('Time (s)', fontsize=11)
    ax_d.set_ylabel('Active\nROIs', fontsize=10)
    ax_d.text(-0.02, 1.05, 'D', transform=ax_d.transAxes,
              fontsize=16, fontweight='bold', va='bottom')
    
    # Annotation
    n_aval_shown = sum(1 for av in avalanches if av["start"] >= t0 and av["end"] <= t1)
    ax_d.text(0.98, 0.92, f'{n_aval_shown} avalanches\nthreshold = 2 SD',
              transform=ax_d.transAxes, fontsize=9, ha='right', va='top',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


# Executar:
fig1 = fig1_avalanche_raster(timeseries, aval, subject, atlas_name,
                               t_start=0, t_end=80,
                              save_path=FIGURES_DIR / "fig1_avalanche_raster.png")

def fig2_powerlaw_distributions(aval_data, size_fit, duration_fit,
                                 subject_id, save_path=None):
    """
    FIGURA 2: Distribuições de power-law no padrão Beggs & Plenz.
    
    3 painéis:
    A) P(S) em log-log com fit
    B) P(D) em log-log com fit
    C) Scaling relation: S vs D com γ
    """
    sizes = aval_data["sizes"]
    durations = aval_data["durations"]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # --- Painel A: Size distribution (CCDF) ---
    ax = axes[0]
    if HAS_POWERLAW and "fit_object" in size_fit:
        fit = size_fit["fit_object"]
        fit.plot_ccdf(ax=ax, color='steelblue', linewidth=0, marker='o',
                      markersize=4, label='Data')
        fit.power_law.plot_ccdf(ax=ax, color='crimson', linestyle='--',
                                linewidth=2, label=f'Power law (α={fit.power_law.alpha:.2f})')
    else:
        # Manual CCDF
        sorted_s = np.sort(sizes)[::-1]
        ccdf = np.arange(1, len(sorted_s) + 1) / len(sorted_s)
        ax.loglog(sorted_s, ccdf, 'o', color='steelblue', markersize=4, label='Data')
        alpha = size_fit.get("alpha", np.nan)
        if np.isfinite(alpha):
            x_fit = np.logspace(np.log10(sorted_s.min()), np.log10(sorted_s.max()), 100)
            y_fit = (x_fit / x_fit.min()) ** (1 - alpha)
            ax.loglog(x_fit, y_fit, '--', color='crimson', lw=2, label=f'α = {alpha:.2f}')
    
    ax.set_xlabel('Avalanche size S', fontsize=11)
    ax.set_ylabel('P(X ≥ S)', fontsize=11)
    ax.set_title('Size Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes,
            fontsize=16, fontweight='bold')
    
    # --- Painel B: Duration distribution (CCDF) ---
    ax = axes[1]
    if HAS_POWERLAW and "fit_object" in duration_fit:
        fit = duration_fit["fit_object"]
        fit.plot_ccdf(ax=ax, color='teal', linewidth=0, marker='s',
                      markersize=4, label='Data')
        fit.power_law.plot_ccdf(ax=ax, color='darkorange', linestyle='--',
                                linewidth=2, label=f'Power law (τ={fit.power_law.alpha:.2f})')
    else:
        sorted_d = np.sort(durations)[::-1]
        ccdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax.loglog(sorted_d, ccdf, 's', color='teal', markersize=4, label='Data')
        tau = duration_fit.get("alpha", np.nan)
        if np.isfinite(tau):
            x_fit = np.logspace(np.log10(sorted_d.min()), np.log10(sorted_d.max()), 100)
            y_fit = (x_fit / x_fit.min()) ** (1 - tau)
            ax.loglog(x_fit, y_fit, '--', color='darkorange', lw=2, label=f'τ = {tau:.2f}')
    
    ax.set_xlabel('Avalanche duration D (TRs)', fontsize=11)
    ax.set_ylabel('P(X ≥ D)', fontsize=11)
    ax.set_title('Duration Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes,
            fontsize=16, fontweight='bold')
    
    # --- Painel C: Size vs Duration (scaling relation) ---
    ax = axes[2]
    valid = (sizes > 0) & (durations > 0)
    ax.scatter(durations[valid], sizes[valid], s=15, alpha=0.5,
               c='slategray', edgecolors='none')
    
    if valid.sum() > 5:
        slope, intercept, r, _, _ = stats.linregress(
            np.log(durations[valid]), np.log(sizes[valid])
        )
        d_fit = np.logspace(np.log10(durations[valid].min()),
                           np.log10(durations[valid].max()), 50)
        s_fit = np.exp(intercept) * d_fit ** slope
        ax.plot(d_fit, s_fit, '--', color='crimson', lw=2,
                label=f'γ = {slope:.2f} (R² = {r**2:.2f})')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Duration D (TRs)', fontsize=11)
    ax.set_ylabel('Size S', fontsize=11)
    ax.set_title('Scaling Relation S ~ D^γ', fontsize=12)
    ax.legend(fontsize=9)
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes,
            fontsize=16, fontweight='bold')
    
    # Annotation: criticality expectation
    ax.text(0.95, 0.05, 'Critical: γ ≈ 2', transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom', style='italic', color='gray')
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


# Executar:
fig2 = fig2_powerlaw_distributions(aval, size_fit, duration_fit, subject,
                                    save_path=FIGURES_DIR / "fig2_powerlaw.png")


def fig3_dfa_analysis(dfa_data, dfa_per_roi, subject_id, save_path=None):
    """
    FIGURA 3: DFA com escalas teóricas de referência.
    
    2 painéis:
    A) DFA global com fit e linhas de referência (α=0.5, 1.0, 1.5)
    B) Distribuição dos expoentes DFA por ROI
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # --- Painel A: DFA global ---
    ax = axes[0]
    scales = dfa_data["scales"]
    fluct = dfa_data["fluctuations"]
    alpha = dfa_data["alpha"]
    r2 = dfa_data.get("r_squared", np.nan)
    intercept = dfa_data.get("intercept", 0)
    
    ax.loglog(scales, fluct, 'o', color='#2c3e50', markersize=5, label='Data')
    
    # Fit line
    x_fit = np.logspace(np.log10(scales.min()), np.log10(scales.max()), 100)
    y_fit = np.exp(intercept) * x_fit ** alpha
    ax.loglog(x_fit, y_fit, '-', color='crimson', lw=2.5,
              label=f'α = {alpha:.3f} (R² = {r2:.3f})')
    
    # Reference lines
    y_range = [fluct.min() * 0.5, fluct.max() * 2]
    for ref_alpha, label, color, ls in [
        (0.5, 'White noise (α=0.5)', '#95a5a6', ':'),
        (1.0, '1/f noise (α=1.0)', '#27ae60', '--'),
        (1.5, 'Brownian (α=1.5)', '#e67e22', ':'),
    ]:
        y_ref = np.exp(np.log(y_range[0])) * (x_fit / x_fit[0]) ** ref_alpha
        # Scale to overlap
        y_ref = y_ref * fluct[0] / y_ref[0]
        ax.loglog(x_fit, y_ref, ls, color=color, lw=1.2, alpha=0.7, label=label)
    
    ax.set_xlabel('Scale n (TRs)', fontsize=11)
    ax.set_ylabel('Fluctuation F(n)', fontsize=11)
    ax.set_title('Detrended Fluctuation Analysis', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.text(-0.12, 1.05, 'A', transform=ax.transAxes,
            fontsize=16, fontweight='bold')
    
    # --- Painel B: DFA per ROI distribution ---
    ax = axes[1]
    alphas = dfa_per_roi
    alphas_valid = alphas[np.isfinite(alphas)]
    
    ax.hist(alphas_valid, bins=25, color='#3498db', edgecolor='white',
            alpha=0.7, density=True)
    
    # KDE overlay
    if len(alphas_valid) > 5:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(alphas_valid)
        x_kde = np.linspace(alphas_valid.min(), alphas_valid.max(), 200)
        ax.plot(x_kde, kde(x_kde), '-', color='#2c3e50', lw=2)
    
    # Reference lines
    ax.axvline(1.0, color='#27ae60', ls='--', lw=2, label='Critical (α=1.0)')
    ax.axvline(0.5, color='#95a5a6', ls=':', lw=1.5, label='White noise')
    ax.axvline(np.nanmean(alphas), color='crimson', ls='-', lw=2,
               label=f'Mean = {np.nanmean(alphas):.3f}')
    
    ax.set_xlabel('DFA exponent α', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Per-ROI DFA Distribution (n={len(alphas_valid)})', fontsize=12)
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes,
            fontsize=16, fontweight='bold')
    
    fig.suptitle(f'{subject_id}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


# Executar:
fig3 = fig3_dfa_analysis(crit["_dfa_data"], crit["_dfa_per_roi"], subject,
                          save_path=FIGURES_DIR / "fig3_dfa.png")


def fig4_criticality_dashboard(crit_results, subject_id, atlas_name,
                                save_path=None):
    """
    FIGURA 4: Dashboard de criticalidade — todos os indicadores.
    
    Layout 2×3: size dist, duration dist, scaling, branching, DFA, summary.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    aval = crit_results["_avalanche_data"]
    sizes = aval["sizes"]
    durations = aval["durations"]
    
    # --- A: Size histogram (log-binned) ---
    ax = axes[0, 0]
    if len(sizes) > 0:
        ax.hist(sizes, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(sizes.mean(), color='red', ls='--', lw=1.5,
                   label=f'Mean: {sizes.mean():.1f}')
        ax.set_xlabel('Avalanche Size')
        ax.set_ylabel('Count')
        alpha_s = crit_results.get("alpha_size", np.nan)
        ax.set_title(f'Size Distribution  (α = {alpha_s:.2f})', fontsize=11)
        ax.legend(fontsize=8)
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # --- B: Duration histogram ---
    ax = axes[0, 1]
    if len(durations) > 0:
        ax.hist(durations, bins=30, color='teal', edgecolor='white', alpha=0.7)
        ax.axvline(durations.mean(), color='red', ls='--', lw=1.5,
                   label=f'Mean: {durations.mean():.1f}')
        ax.set_xlabel('Duration (TRs)')
        ax.set_ylabel('Count')
        tau = crit_results.get("tau_duration", np.nan)
        ax.set_title(f'Duration Distribution  (τ = {tau:.2f})', fontsize=11)
        ax.legend(fontsize=8)
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold')
    

    
    # --- C: Size vs Duration log-log ---
    ax = axes[0, 2]
    if len(sizes) > 0:
        valid = (sizes > 0) & (durations > 0)
        ax.scatter(durations[valid], sizes[valid], s=12, alpha=0.5, c='slategray')
        gamma = crit_results.get("gamma_empirical", np.nan)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Duration')
        ax.set_ylabel('Size')
        ax.set_title(f'Scaling Relation  (γ = {gamma:.2f})', fontsize=11)
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # --- D: Branching ratio ---
    ax = axes[1, 0]
    br = crit_results["branching_ratio"]
    colors_br = ['#27ae60' if abs(br - 1) < 0.1 else '#e74c3c']
    ax.bar(['σ'], [br], color=colors_br, edgecolor='black', lw=0.5, width=0.5)
    ax.axhline(1.0, color='#27ae60', ls='--', lw=2, label='Critical (σ=1)')
    ax.set_ylim(0, max(2.0, br + 0.5) if np.isfinite(br) else 2.0)
    ax.set_ylabel('Branching Ratio')
    ax.set_title(f'σ = {br:.4f}', fontsize=11)
    ax.legend(fontsize=8)
    ax.text(-0.15, 1.05, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # --- E: DFA ---
    ax = axes[1, 1]
    dfa_a = crit_results["dfa_alpha"]
    colors_dfa = ['#27ae60' if abs(dfa_a - 1) < 0.15 else '#e74c3c']
    ax.bar(['α_DFA'], [dfa_a], color=colors_dfa, edgecolor='black', lw=0.5, width=0.5)
    ax.axhline(1.0, color='#27ae60', ls='--', lw=2, label='1/f noise')
    ax.axhline(0.5, color='gray', ls=':', lw=1, label='White noise')
    ax.set_ylim(0, 2)
    ax.set_ylabel('DFA Exponent')
    ax.set_title(f'α_DFA = {dfa_a:.4f}', fontsize=11)
    ax.legend(fontsize=8)
    ax.text(-0.15, 1.05, 'E', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # --- F: Summary text box ---
    ax = axes[1, 2]
    ax.axis('off')
    
    ci = crit_results["criticality_index"]
    summary = (
        f"CRITICALITY METRICS\n"
        f"{'─'*32}\n\n"
        f"  Avalanches:       {crit_results['n_avalanches']}\n"
        f"  Rate:             {crit_results['avalanche_rate']:.1f} /min\n\n"
        f"  α (size):         {crit_results.get('alpha_size', np.nan):.3f}\n"
        f"  τ (duration):     {crit_results.get('tau_duration', np.nan):.3f}\n"
        f"  γ (theoretical):  {crit_results.get('gamma_theoretical', np.nan):.3f}\n"
        f"  γ (empirical):    {crit_results.get('gamma_empirical', np.nan):.3f}\n\n"
        f"  σ (branching):    {crit_results['branching_ratio']:.4f}\n"
        f"  α_DFA:            {crit_results['dfa_alpha']:.4f}\n\n"
        f"{'─'*32}\n"
        f"  Criticality Index: {ci:.3f}\n"
        f"  (1.0 = critical)"
    )
    
    ci_color = '#27ae60' if ci > 0.7 else '#e67e22' if ci > 0.4 else '#e74c3c'
    ax.text(0.1, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor=ci_color,
                      linewidth=2, alpha=0.9))
    ax.text(-0.05, 1.05, 'F', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    fig.suptitle(f'{subject_id} — Criticality Analysis ({atlas_name})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


# Executar:
fig4 = fig4_criticality_dashboard(crit, subject, atlas_name,
                                   save_path=FIGURES_DIR / "fig4_dashboard.png")


def fig5_group_criticality(all_crit_results, save_path=None):
    """
    FIGURA 5: Métricas de criticalidade para todos os 23 pacientes.
    
    Layout 2×3:
    A) Branching ratio por sujeito (stripplot + violino)
    B) DFA alpha por sujeito
    C) Alpha (size) vs Tau (duration) scatter
    D) Criticality index distribution
    E) Avalanche rate vs Criticality Index
    F) Brain map: DFA per ROI (group average)
    """
    subjects = sorted(all_crit_results.keys())
    
    # Extrair métricas
    df = pd.DataFrame([{
        "subject": sub,
        "branching_ratio": r["branching_ratio"],
        "dfa_alpha": r["dfa_alpha"],
        "alpha_size": r.get("alpha_size", np.nan),
        "tau_duration": r.get("tau_duration", np.nan),
        "gamma_emp": r.get("gamma_empirical", np.nan),
        "criticality_index": r["criticality_index"],
        "avalanche_rate": r["avalanche_rate"],
        "n_avalanches": r["n_avalanches"],
    } for sub, r in all_crit_results.items()])
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # --- A: Branching ratio ---
    ax = axes[0, 0]
    ax.axhline(1.0, color='#27ae60', ls='--', lw=2, alpha=0.5, zorder=0)
    ax.axhspan(0.9, 1.1, alpha=0.08, color='green', zorder=0)
    sns.stripplot(y=df["branching_ratio"], ax=ax, color='steelblue',
                  size=7, jitter=0.2, alpha=0.7)
    sns.boxplot(y=df["branching_ratio"], ax=ax, color='lightblue',
                width=0.3, fliersize=0, boxprops=dict(alpha=0.3))
    ax.set_ylabel('Branching Ratio (σ)', fontsize=11)
    ax.set_title(f'σ = {df["branching_ratio"].mean():.3f} ± {df["branching_ratio"].std():.3f}',
                 fontsize=11)
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # --- B: DFA alpha ---
    ax = axes[0, 1]
    ax.axhline(1.0, color='#27ae60', ls='--', lw=2, alpha=0.5, zorder=0)
    ax.axhspan(0.85, 1.15, alpha=0.08, color='green', zorder=0)
    sns.stripplot(y=df["dfa_alpha"], ax=ax, color='coral', size=7, jitter=0.2, alpha=0.7)
    sns.boxplot(y=df["dfa_alpha"], ax=ax, color='lightyellow',
                width=0.3, fliersize=0, boxprops=dict(alpha=0.3))
    ax.set_ylabel('DFA Exponent (α)', fontsize=11)
    ax.set_title(f'α = {df["dfa_alpha"].mean():.3f} ± {df["dfa_alpha"].std():.3f}',
                 fontsize=11)
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # --- C: Alpha vs Tau ---
    ax = axes[0, 2]
    valid = df["alpha_size"].notna() & df["tau_duration"].notna()
    sc = ax.scatter(df.loc[valid, "alpha_size"], df.loc[valid, "tau_duration"],
                    c=df.loc[valid, "criticality_index"], cmap='RdYlGn',
                    s=60, edgecolors='black', lw=0.5, vmin=0, vmax=1)
    ax.axvline(1.5, color='gray', ls=':', lw=1, alpha=0.5)
    ax.axhline(2.0, color='gray', ls=':', lw=1, alpha=0.5)
    ax.plot(1.5, 2.0, '*', color='gold', markersize=15, markeredgecolor='black',
            zorder=5, label='Critical point')
    ax.set_xlabel('α (size exponent)', fontsize=11)
    ax.set_ylabel('τ (duration exponent)', fontsize=11)
    ax.set_title('Exponent Space', fontsize=11)
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='Criticality Index', shrink=0.7)
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # --- D: Criticality Index distribution ---
    ax = axes[1, 0]
    ax.hist(df["criticality_index"], bins=12, color='#2ecc71', edgecolor='white', alpha=0.7)
    ax.axvline(df["criticality_index"].mean(), color='crimson', ls='--', lw=2,
               label=f'Mean: {df["criticality_index"].mean():.3f}')
    ax.set_xlabel('Criticality Index', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Criticality Index', fontsize=11)
    ax.legend(fontsize=8)
    ax.text(-0.15, 1.05, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # --- E: Avalanche rate vs CI ---
    ax = axes[1, 1]
    ax.scatter(df["avalanche_rate"], df["criticality_index"],
               s=50, c='steelblue', edgecolors='black', lw=0.5, alpha=0.7)
    rho, p = stats.spearmanr(df["avalanche_rate"].dropna(), df["criticality_index"].dropna())
    ax.set_xlabel('Avalanche Rate (/min)', fontsize=11)
    ax.set_ylabel('Criticality Index', fontsize=11)
    ax.set_title(f'Rate vs Criticality (ρ={rho:.3f}, p={p:.3f})', fontsize=11)
    ax.text(-0.15, 1.05, 'E', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # --- F: Summary table ---
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = (
        f"GROUP SUMMARY (N = {len(df)})\n"
        f"{'─'*36}\n\n"
        f"  σ (branching):    {df['branching_ratio'].mean():.3f} ± {df['branching_ratio'].std():.3f}\n"
        f"  α_DFA:            {df['dfa_alpha'].mean():.3f} ± {df['dfa_alpha'].std():.3f}\n"
        f"  α (size):         {df['alpha_size'].mean():.3f} ± {df['alpha_size'].std():.3f}\n"
        f"  τ (duration):     {df['tau_duration'].mean():.3f} ± {df['tau_duration'].std():.3f}\n"
        f"  γ (empirical):    {df['gamma_emp'].mean():.3f} ± {df['gamma_emp'].std():.3f}\n\n"
        f"  Criticality Idx:  {df['criticality_index'].mean():.3f} ± {df['criticality_index'].std():.3f}\n"
        f"  Aval rate:        {df['avalanche_rate'].mean():.1f} ± {df['avalanche_rate'].std():.1f} /min\n\n"
        f"{'─'*36}\n"
        f"  Critical expectations:\n"
        f"    σ ≈ 1, α ≈ 1.5, τ ≈ 2, γ ≈ 2\n"
        f"    α_DFA ≈ 1"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1',
                      edgecolor='#34495e', lw=1.5))
    ax.text(-0.05, 1.05, 'F', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    fig.suptitle('Brain Criticality — Post-COVID ICU Cohort',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig, df

def fig6_dfa_brain_map(dfa_per_roi, atlas_name, subject_id, save_path=None):
    """
    FIGURA 6: Mapa cerebral do DFA por ROI usando nilearn.
    
    Mostra quais regiões estão mais próximas da criticalidade (α ≈ 1).
    """
    try:
        from nilearn import plotting
    except ImportError:
        print("nilearn necessário para brain maps")
        return None
    
    # Carregar coordenadas do atlas
    labels_path = ATLASES[atlas_name]["labels_file"]
    labels_df = pd.read_csv(labels_path)
    
    # Coordenadas MNI (se disponíveis como arquivo separado)
    # Se não tiver coords, usar plot_markers com nilearn
    # Vamos usar o desvio de α em relação a 1.0 como métrica
    deviation = np.abs(dfa_per_roi - 1.0)  # 0 = critical, alto = não-critical
    criticality_map = 1.0 - deviation  # Mais alto = mais crítico
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Se tiver coordenadas MNI:
    # coords = np.load(ATLAS_DIR / f"{atlas_name}_coords_mni.npy")
    # plotting.plot_markers(criticality_map, coords, axes=axes[0], ...)
    
    # Alternativa: histograma com mapeamento por rede
    from sars.config import SCHAEFER_NETWORK_PREFIXES
    
    # Painel A: Distribuição de α por ROI
    ax = axes[0]
    n_rois = len(dfa_per_roi)
    colors = plt.cm.RdYlGn(Normalize(vmin=0.5, vmax=1.5)(dfa_per_roi))
    ax.bar(range(n_rois), dfa_per_roi, color=colors, width=1.0, edgecolor='none')
    ax.axhline(1.0, color='black', ls='--', lw=1.5)
    ax.axhspan(0.85, 1.15, alpha=0.1, color='green')
    ax.set_xlabel('ROI index', fontsize=11)
    ax.set_ylabel('DFA α', fontsize=11)
    ax.set_title('Per-ROI DFA Exponent', fontsize=12)
    ax.text(-0.12, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # Painel B: Sorted (rank plot)
    ax = axes[1]
    sorted_idx = np.argsort(np.abs(dfa_per_roi - 1.0))
    sorted_alpha = dfa_per_roi[sorted_idx]
    colors_sorted = plt.cm.RdYlGn(Normalize(vmin=0.5, vmax=1.5)(sorted_alpha))
    ax.bar(range(n_rois), sorted_alpha, color=colors_sorted, width=1.0, edgecolor='none')
    ax.axhline(1.0, color='black', ls='--', lw=1.5)
    ax.set_xlabel('ROI (ranked by |α - 1|)', fontsize=11)
    ax.set_ylabel('DFA α', fontsize=11)
    ax.set_title('Sorted by Proximity to Criticality', fontsize=12)
    ax.text(-0.12, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    # Painel C: Network-level mean
    if "schaefer" in atlas_name:
        networks = []
        for label in labels_df["label_roi"]:
            parts = label.split("_")
            networks.append(parts[2] if len(parts) >= 3 else "Unknown")
        
        net_df = pd.DataFrame({"network": networks, "alpha": dfa_per_roi[:len(networks)]})
        net_means = net_df.groupby("network")["alpha"].agg(["mean", "std"])
        
        yeo_colors = {
            "Vis": "#781286", "SomMot": "#4682B4", "DorsAttn": "#00760E",
            "SalVentAttn": "#C43AFA", "Limbic": "#DCF8A4",
            "Cont": "#E69422", "Default": "#CD3E4E"
        }
        
        ax = axes[2]
        nets = net_means.index.tolist()
        means = net_means["mean"].values
        stds = net_means["std"].values
        colors_net = [yeo_colors.get(n, "gray") for n in nets]
        
        ax.bar(range(len(nets)), means, yerr=stds, color=colors_net,
               edgecolor='black', lw=0.5, capsize=3)
        ax.axhline(1.0, color='black', ls='--', lw=1.5)
        ax.set_xticks(range(len(nets)))
        ax.set_xticklabels(nets, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean DFA α', fontsize=11)
        ax.set_title('DFA by Functional Network', fontsize=12)
    
    ax.text(-0.12, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    fig.suptitle(f'{subject_id} — Regional Criticality Map ({atlas_name})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig


# Executar:
fig6 = fig6_dfa_brain_map(crit["_dfa_per_roi"], atlas_name, subject,
                            save_path=FIGURES_DIR / "fig6_dfa_brain.png")


def run_criticality_pipeline(atlas_name="schaefer_100", threshold_sd=2.0):
    """
    Pipeline de criticalidade para todo o dataset SARS-CoV-2.
    """
    SAVE_DIR = OUTPUTS_DIR / "sars" / "criticality" / atlas_name
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR = FIGURES_DIR / "criticality" / atlas_name
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"  CRITICALITY PIPELINE — {atlas_name}")
    print("=" * 60)
    
    # 1. Carregar todas as timeseries
    all_ts = load_all_timeseries(atlas_name)
    
    # 2. Analisar cada sujeito
    all_results = {}
    for sub, ts in all_ts.items():
        print(f"\n[{sub}] {ts.shape[0]} TRs × {ts.shape[1]} ROIs")
        crit_sub = analyze_criticality(ts, threshold_sd)
        all_results[sub] = crit_sub
        
        # Dashboard individual
        fig4_criticality_dashboard(
            crit_sub, sub, atlas_name,
            save_path=FIG_DIR / f"{sub}_dashboard.png"
        )
        plt.close('all')
    
    # 3. Extrair features para CSV
    rows = []
    for sub, r in all_results.items():
        row = {k: v for k, v in r.items() if not k.startswith("_") and not isinstance(v, np.ndarray)}
        row["subject"] = sub
        rows.append(row)
    
    features_df = pd.DataFrame(rows)
    features_df.to_csv(SAVE_DIR / "criticality_features_per_subject.csv", index=False)
    
    # 4. Figura de grupo
    fig5, group_df = fig5_group_criticality(
        all_results, save_path=FIG_DIR / "group_criticality.png"
    )
    plt.close('all')
    
    print(f"\n{'='*60}")
    print(f"  RESULTADOS SALVOS")
    print(f"  CSV: {SAVE_DIR}")
    print(f"  Figuras: {FIG_DIR}")
    print(f"{'='*60}")
    
    return all_results, features_df


# Executar:
all_crit, crit_df = run_criticality_pipeline("schaefer_100")