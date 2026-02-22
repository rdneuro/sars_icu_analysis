# -*- coding: utf-8 -*-
"""
sars.utils
========================

Statistical utilities, null models, permutation testing, signal processing,
and general-purpose helpers shared across all analysis modules.

This module provides the statistical and numerical backbone for the
sars library.  All functions are designed to be modality-
agnostic and can be used with both functional and structural connectivity
data.

Includes
--------
- Multiple-comparison correction (FDR, Bonferroni, Holm-Bonferroni)
- Effect size computation (Cohen's d, Hedges' g, rank-biserial, η²)
- Surrogate / null model generation for timeseries
- Bootstrap confidence intervals (percentile, BCa)
- Permutation tests (one-sample, two-sample, paired)
- Fisher z-transform and inverse
- Z-scoring for matrices and timeseries
- Signal processing (bandpass, PSD, Hilbert, instantaneous metrics)
- Null network models (degree-preserving rewiring, Erdős-Rényi)
- Intraclass Correlation Coefficient (ICC)
- Correlation comparison (Steiger, Williams–Hotelling)
- Reproducibility / seed management
- Progress and logging helpers

References
----------
- Benjamini & Hochberg (1995). JRSS-B. FDR control.
- Benjamini & Yekutieli (2001). Ann Statist. FDR under dependence.
- Holm (1979). Scand J Statist. Sequentially rejective test procedure.
- Prichard & Theiler (1994). Phys Rev Lett. Phase-randomization surrogates.
- Theiler et al. (1992). Physica D. Surrogate data for nonlinear analysis.
- Lancaster et al. (2018). Phys Rev E. Surrogate data for multivariate TS.
- Schreiber & Schmitz (1996). Phys Rev Lett. IAAFT surrogates.
- Nichols & Holmes (2002). Hum Brain Mapp. Nonparametric permutation tests.
- Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.
- Maslov & Sneppen (2002). Science. Specificity and stability in network
  topology.
- Steiger (1980). Psychol Bull. Comparing correlations.
- Williams (1959). Biometrika. Regression Analysis.
- Shrout & Fleiss (1979). Psychol Bull. ICC forms.

Usage
-----
    from sars.utils import (
        fdr_correction, hedges_g, phase_randomize,
        permutation_test_twosample, bootstrap_ci,
        fisher_z, bandpass_filter, random_rewire,
    )
"""

import warnings
import numpy as np
from scipy import stats, signal
from typing import Optional, Tuple, List, Union, Callable, Dict, Any


# =============================================================================
# MULTIPLE-COMPARISON CORRECTION
# =============================================================================

def fdr_correction(
    pvalues: np.ndarray,
    alpha: float = 0.05,
    method: str = "bh",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    False Discovery Rate correction for multiple comparisons.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of uncorrected p-values.
    alpha : float
        Significance level.
    method : str
        'bh' : Benjamini-Hochberg (1995) — controls FDR.
        'by' : Benjamini-Yekutieli (2001) — controls FDR under dependence.

    Returns
    -------
    reject : np.ndarray of bool
        True where the null can be rejected.
    pvalues_corrected : np.ndarray
        Adjusted p-values.
    """
    pvals = np.asarray(pvalues).ravel()
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    ranks = np.arange(1, n + 1)

    if method == "bh":
        correction = n / ranks
    elif method == "by":
        c_m = np.sum(1.0 / np.arange(1, n + 1))
        correction = (n * c_m) / ranks
    else:
        raise ValueError(f"Unknown FDR method: {method}")

    corrected = np.minimum(sorted_pvals * correction, 1.0)

    # enforce monotonicity (largest → smallest)
    for i in range(n - 2, -1, -1):
        corrected[i] = min(corrected[i], corrected[i + 1])

    pvalues_corrected = np.empty(n)
    pvalues_corrected[sorted_idx] = corrected
    reject = pvalues_corrected <= alpha
    return reject, pvalues_corrected


def bonferroni_correction(
    pvalues: np.ndarray, alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bonferroni correction: p_adj = p × n_tests."""
    pvals = np.asarray(pvalues).ravel()
    corrected = np.minimum(pvals * len(pvals), 1.0)
    return corrected <= alpha, corrected


def holm_bonferroni_correction(
    pvalues: np.ndarray, alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Holm–Bonferroni step-down correction (Holm, 1979).

    Less conservative than Bonferroni while still controlling FWER.

    Returns
    -------
    reject, pvalues_corrected
    """
    pvals = np.asarray(pvalues).ravel()
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]

    corrected = np.minimum(sorted_pvals * (n - np.arange(n)), 1.0)

    # enforce monotonicity (ascending)
    for i in range(1, n):
        corrected[i] = max(corrected[i], corrected[i - 1])

    pvalues_corrected = np.empty(n)
    pvalues_corrected[sorted_idx] = corrected
    reject = pvalues_corrected <= alpha
    return reject, pvalues_corrected


# =============================================================================
# EFFECT SIZE MEASURES
# =============================================================================

def cohens_d(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    paired: bool = False,
) -> float:
    """
    Compute Cohen's d effect size.

    Parameters
    ----------
    x : array
        First sample.
    y : array, optional
        Second sample.  If None, one-sample test vs zero.
    paired : bool
        If True and y given, paired Cohen's d.
    """
    x = np.asarray(x, dtype=float)
    if y is None:
        s = np.std(x, ddof=1)
        return np.mean(x) / s if s > 0 else 0.0
    y = np.asarray(y, dtype=float)
    if paired:
        diff = x - y
        s = np.std(diff, ddof=1)
        return np.mean(diff) / s if s > 0 else 0.0
    nx, ny = len(x), len(y)
    pooled = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 0 else 0.0


def hedges_g(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    paired: bool = False,
) -> float:
    """
    Hedges' g — bias-corrected Cohen's d for small samples.

    Correction factor J = 1 − 3 / (4·df − 1).
    Recommended over Cohen's d for n < 50 (rsfmri_best_practices.pdf).
    """
    d = cohens_d(x, y, paired)
    if y is None or paired:
        df = len(x) - 1
    else:
        df = len(x) + len(y) - 2
    j = 1 - (3 / (4 * df - 1)) if df > 1 else 1.0
    return d * j


def rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Rank-biserial r — effect size for Mann-Whitney U.

    r = 1 − 2U / (n₁ · n₂).
    """
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 1 - (2 * u) / (len(x) * len(y))


def eta_squared(
    f_statistic: float, df_between: int, df_within: int,
) -> float:
    """
    Eta-squared (η²) effect size from F-statistic.

    η² = (F · df_between) / (F · df_between + df_within).
    """
    return (f_statistic * df_between) / (
        f_statistic * df_between + df_within
    )


def partial_eta_squared(
    f_statistic: float, df_effect: int, df_error: int,
) -> float:
    """
    Partial η² = (F · df_effect) / (F · df_effect + df_error).
    """
    return (f_statistic * df_effect) / (
        f_statistic * df_effect + df_error
    )


# =============================================================================
# SURROGATE DATA GENERATION
# =============================================================================

def phase_randomize(
    timeseries: np.ndarray,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a phase-randomized surrogate.

    Preserves the power spectrum (autocorrelation) but destroys
    temporal structure.  For multivariate (T, N) input, identical
    random phases are applied to all channels to preserve inter-channel
    power structure (Lancaster et al., 2018).

    Parameters
    ----------
    timeseries : np.ndarray, shape (T,) or (T, N)
    seed : int, optional

    Returns
    -------
    np.ndarray : same shape as input.
    """
    rng = np.random.default_rng(seed)
    ts = np.asarray(timeseries, dtype=float)
    is_1d = ts.ndim == 1
    if is_1d:
        ts = ts[:, np.newaxis]

    T, N = ts.shape
    ft = np.fft.rfft(ts, axis=0)
    n_freq = ft.shape[0]

    phases = rng.uniform(0, 2 * np.pi, size=n_freq)
    phases[0] = 0
    if T % 2 == 0:
        phases[-1] = 0

    ft_surr = ft * np.exp(1j * phases)[:, np.newaxis]
    surrogate = np.fft.irfft(ft_surr, n=T, axis=0)

    return surrogate.ravel() if is_1d else surrogate


def amplitude_adjusted_ft_surrogate(
    timeseries: np.ndarray,
    seed: Optional[int] = None,
    n_iterations: int = 50,
) -> np.ndarray:
    """
    Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogate.

    Preserves both the amplitude distribution and the power spectrum.
    Ref: Schreiber & Schmitz (1996). Phys Rev Lett.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T,)
    seed : int, optional
    n_iterations : int
    """
    rng = np.random.default_rng(seed)
    ts = np.asarray(timeseries, dtype=float).ravel()
    T = len(ts)

    sorted_values = np.sort(ts)
    original_amplitudes = np.abs(np.fft.rfft(ts))

    surrogate = rng.permutation(ts).copy()

    for _ in range(n_iterations):
        # Match power spectrum
        ft_surr = np.fft.rfft(surrogate)
        ft_adj = original_amplitudes * np.exp(1j * np.angle(ft_surr))
        surrogate = np.fft.irfft(ft_adj, n=T)
        # Match amplitude distribution
        rank_order = np.argsort(np.argsort(surrogate))
        surrogate = sorted_values[rank_order]

    return surrogate


def generate_surrogates(
    timeseries: np.ndarray,
    n_surrogates: int = 1000,
    method: str = "phase",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate an ensemble of surrogate timeseries.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T,) or (T, N)
    n_surrogates : int
    method : str
        'phase': Phase randomization (Prichard & Theiler 1994).
        'iaaft': Iterative AAFT (Schreiber & Schmitz 1996).
        'shuffle': Random temporal permutation.
    seed : int, optional

    Returns
    -------
    np.ndarray, shape (n_surrogates, T) for 1D or (n_surrogates, T, N).
    """
    rng = np.random.default_rng(seed)
    ts = np.asarray(timeseries, dtype=float)

    surrogates = []
    for i in range(n_surrogates):
        s = rng.integers(0, 2**31) + i
        if method == "phase":
            surrogates.append(phase_randomize(ts, seed=s))
        elif method == "iaaft":
            if ts.ndim == 1:
                surrogates.append(amplitude_adjusted_ft_surrogate(ts, seed=s))
            else:
                channels = [
                    amplitude_adjusted_ft_surrogate(ts[:, ch], seed=s + ch)
                    for ch in range(ts.shape[1])
                ]
                surrogates.append(np.column_stack(channels))
        elif method == "shuffle":
            idx = rng.permutation(ts.shape[0])
            surrogates.append(ts[idx] if ts.ndim == 1 else ts[idx, :])
        else:
            raise ValueError(f"Unknown surrogate method: {method}")

    return np.array(surrogates)


# =============================================================================
# PERMUTATION TESTS
# =============================================================================

def permutation_test_onesample(
    x: np.ndarray,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    seed: Optional[int] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Non-parametric one-sample permutation test against zero.

    Randomly flips signs to build the null distribution of the mean.

    Returns
    -------
    observed_stat, p_value, null_distribution
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    n = len(x)
    observed = np.mean(x)

    signs = rng.choice([-1, 1], size=(n_permutations, n))
    null_dist = np.mean(signs * x[np.newaxis, :], axis=1)

    if alternative == "two-sided":
        p = np.mean(np.abs(null_dist) >= np.abs(observed))
    elif alternative == "greater":
        p = np.mean(null_dist >= observed)
    elif alternative == "less":
        p = np.mean(null_dist <= observed)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return observed, max(p, 1.0 / (n_permutations + 1)), null_dist


def permutation_test_twosample(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 10000,
    stat_func: Optional[Callable] = None,
    alternative: str = "two-sided",
    seed: Optional[int] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Non-parametric two-sample permutation test.

    Parameters
    ----------
    x, y : arrays
    n_permutations : int
    stat_func : callable(x, y) → float.  Default: difference of means.
    alternative : str
    seed : int, optional

    Returns
    -------
    observed_stat, p_value, null_distribution
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if stat_func is None:
        stat_func = lambda a, b: np.mean(a) - np.mean(b)

    observed = stat_func(x, y)
    combined = np.concatenate([x, y])
    nx = len(x)

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null_dist[i] = stat_func(perm[:nx], perm[nx:])

    if alternative == "two-sided":
        p = np.mean(np.abs(null_dist) >= np.abs(observed))
    elif alternative == "greater":
        p = np.mean(null_dist >= observed)
    elif alternative == "less":
        p = np.mean(null_dist <= observed)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return observed, max(p, 1.0 / (n_permutations + 1)), null_dist


def permutation_test_paired(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    seed: Optional[int] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Paired permutation test via sign-flipping of within-pair differences.

    Returns
    -------
    observed_stat, p_value, null_distribution
    """
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    return permutation_test_onesample(
        diff, n_permutations, alternative, seed,
    )


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    data: np.ndarray,
    stat_func: Callable = np.mean,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    method: str = "percentile",
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval.

    Parameters
    ----------
    data : array
    stat_func : callable
    n_bootstrap : int
    ci : float
        Confidence level (e.g. 0.95).
    method : str
        'percentile' or 'bca' (bias-corrected & accelerated).
    seed : int, optional

    Returns
    -------
    point_estimate, ci_lower, ci_upper
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    point = stat_func(data)

    boot_stats = np.array([
        stat_func(data[rng.integers(0, n, size=n)])
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - ci

    if method == "percentile":
        lower = np.percentile(boot_stats, 100 * alpha / 2)
        upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    elif method == "bca":
        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_stats < point))
        # Acceleration (jackknife)
        jack = np.array([
            stat_func(np.delete(data, i)) for i in range(n)
        ])
        jack_mean = np.mean(jack)
        diffs = jack_mean - jack
        ss2 = np.sum(diffs**2)
        a = np.sum(diffs**3) / (6 * ss2**1.5) if ss2 > 0 else 0

        za_lo = stats.norm.ppf(alpha / 2)
        za_hi = stats.norm.ppf(1 - alpha / 2)

        def _adj(za):
            num = z0 + za
            return stats.norm.cdf(z0 + num / (1 - a * num))

        lower = np.percentile(boot_stats, 100 * _adj(za_lo))
        upper = np.percentile(boot_stats, 100 * _adj(za_hi))
    else:
        raise ValueError(f"Unknown bootstrap method: {method}")

    return float(point), float(lower), float(upper)


# =============================================================================
# TRANSFORMS AND NORMALIZATION
# =============================================================================

def fisher_z(r: np.ndarray) -> np.ndarray:
    """
    Fisher z-transform: z = arctanh(r).

    Variance-stabilizing transform for Pearson correlation coefficients.
    Required before group averaging / t-tests on correlation values.
    Clips r to (−0.9999, 0.9999) to avoid inf.
    """
    r = np.asarray(r, dtype=float)
    return np.arctanh(np.clip(r, -0.9999, 0.9999))


def inverse_fisher_z(z: np.ndarray) -> np.ndarray:
    """Inverse Fisher z-transform: r = tanh(z)."""
    return np.tanh(np.asarray(z, dtype=float))


def zscore_matrix(
    matrix: np.ndarray, per_row: bool = False,
) -> np.ndarray:
    """
    Z-score a matrix (globally or per row).
    """
    mat = np.asarray(matrix, dtype=float)
    if per_row:
        means = mat.mean(axis=1, keepdims=True)
        stds = mat.std(axis=1, keepdims=True)
        stds[stds == 0] = 1
        return (mat - means) / stds
    m, s = mat.mean(), mat.std()
    return (mat - m) / s if s > 0 else mat - m


def zscore_timeseries(
    timeseries: np.ndarray, axis: int = 0,
) -> np.ndarray:
    """
    Z-score a timeseries array.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T, N)
    axis : int
        0 = per ROI (temporal z-score), 1 = per timepoint.
    """
    return stats.zscore(timeseries, axis=axis, nan_policy="omit")


def robust_zscore(
    x: np.ndarray, axis: int = 0,
) -> np.ndarray:
    """
    Robust z-score using median and MAD (median absolute deviation).

    Less sensitive to outliers than standard z-score.

    z_robust = (x − median) / (1.4826 × MAD)
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    mad_scaled = 1.4826 * mad  # consistent estimator for σ
    mad_scaled[mad_scaled == 0] = 1
    return (x - med) / mad_scaled


# =============================================================================
# SIGNAL PROCESSING HELPERS
# =============================================================================

def bandpass_filter(
    timeseries: np.ndarray,
    tr: float,
    low_freq: float = 0.01,
    high_freq: float = 0.08,
    order: int = 5,
) -> np.ndarray:
    """
    Butterworth bandpass filter for BOLD timeseries.

    Standard rs-fMRI bandpass: 0.01–0.08 Hz (AJNR Task Force 2024).

    Parameters
    ----------
    timeseries : np.ndarray, shape (T,) or (T, N)
    tr : float
        Repetition time in seconds.
    low_freq, high_freq : float
        Cutoffs in Hz.
    order : int
    """
    fs = 1.0 / tr
    nyq = fs / 2.0
    low = max(low_freq / nyq, 1e-5)
    high = min(high_freq / nyq, 1.0 - 1e-5)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, timeseries, axis=0)


def compute_power_spectrum(
    timeseries: np.ndarray,
    tr: float,
    method: str = "welch",
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Power spectral density of a timeseries.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T,) or (T, N)
    tr : float
    method : str
        'welch' or 'periodogram'.
    nperseg : int, optional

    Returns
    -------
    freqs, psd
    """
    fs = 1.0 / tr
    if method == "welch":
        if nperseg is None:
            T = timeseries.shape[0]
            nperseg = min(T, max(64, T // 4))
        return signal.welch(timeseries, fs=fs, nperseg=nperseg, axis=0)
    elif method == "periodogram":
        return signal.periodogram(timeseries, fs=fs, axis=0)
    raise ValueError(f"Unknown PSD method: {method}")


def compute_analytic_signal(timeseries: np.ndarray) -> np.ndarray:
    """
    Analytic signal via Hilbert transform.

    Amplitude = np.abs(result), Phase = np.angle(result).
    """
    return signal.hilbert(timeseries, axis=0)


def compute_instantaneous_frequency(
    timeseries: np.ndarray, tr: float,
) -> np.ndarray:
    """
    Instantaneous frequency from the Hilbert analytic signal.

    f_inst = (1 / 2π) · dφ/dt

    Parameters
    ----------
    timeseries : np.ndarray, shape (T,) or (T, N)
    tr : float

    Returns
    -------
    np.ndarray, shape (T−1,) or (T−1, N)
    """
    analytic = compute_analytic_signal(timeseries)
    phase = np.unwrap(np.angle(analytic), axis=0)
    inst_freq = np.diff(phase, axis=0) / (2 * np.pi * tr)
    return inst_freq


def compute_functional_connectivity_dynamic(
    timeseries: np.ndarray,
    window_length: int,
    step: int = 1,
    method: str = "pearson",
) -> np.ndarray:
    """
    Sliding-window dynamic functional connectivity.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T, N)
    window_length : int
        Number of timepoints per window.
    step : int
        Stride between consecutive windows.
    method : str
        'pearson' or 'spearman'.

    Returns
    -------
    np.ndarray, shape (n_windows, N, N)
    """
    T, N = timeseries.shape
    windows = list(range(0, T - window_length + 1, step))
    dfc = np.zeros((len(windows), N, N))

    for w_idx, start in enumerate(windows):
        segment = timeseries[start : start + window_length]
        if method == "pearson":
            dfc[w_idx] = np.corrcoef(segment.T)
        elif method == "spearman":
            dfc[w_idx] = stats.spearmanr(segment).statistic
            if dfc[w_idx].shape != (N, N):
                # spearmanr returns scalar for 2 variables
                dfc[w_idx] = np.corrcoef(segment.T)
        else:
            raise ValueError(f"Unknown method: {method}")

    return dfc


# =============================================================================
# NULL MODEL UTILITIES FOR NETWORKS
# =============================================================================

def random_rewire(
    adjacency: np.ndarray,
    n_rewires: Optional[int] = None,
    preserve_strength: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Degree-preserving random rewiring (Maslov & Sneppen, 2002).

    Parameters
    ----------
    adjacency : np.ndarray (N, N)
        Binary or weighted symmetric adjacency matrix.
    n_rewires : int, optional
        Rewiring attempts.  Default: 5 × n_edges.
    preserve_strength : bool
        If True, swap weights rather than edges (weighted networks).
    seed : int, optional
    """
    rng = np.random.default_rng(seed)
    mat = adjacency.copy()
    n = mat.shape[0]
    np.fill_diagonal(mat, 0)

    triu = np.triu_indices_from(mat, k=1)
    edges = [
        (triu[0][i], triu[1][i])
        for i in range(len(triu[0]))
        if mat[triu[0][i], triu[1][i]] != 0
    ]

    if len(edges) < 2:
        return mat

    n_rewires = n_rewires or 5 * len(edges)

    for _ in range(n_rewires):
        idx = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[idx[0]]
        c, d = edges[idx[1]]

        if a == c or a == d or b == c or b == d:
            continue

        if rng.random() < 0.5:
            # (a,b),(c,d) → (a,d),(c,b)
            if mat[a, d] != 0 or mat[c, b] != 0:
                continue
            if preserve_strength:
                mat[a, d], mat[d, a] = mat[a, b], mat[b, a]
                mat[c, b], mat[b, c] = mat[c, d], mat[d, c]
            else:
                mat[a, d] = mat[d, a] = mat[a, b]
                mat[c, b] = mat[b, c] = mat[c, d]
            mat[a, b] = mat[b, a] = 0
            mat[c, d] = mat[d, c] = 0
            edges[idx[0]] = (min(a, d), max(a, d))
            edges[idx[1]] = (min(c, b), max(c, b))
        else:
            # (a,b),(c,d) → (a,c),(b,d)
            if mat[a, c] != 0 or mat[b, d] != 0:
                continue
            if preserve_strength:
                mat[a, c], mat[c, a] = mat[a, b], mat[b, a]
                mat[b, d], mat[d, b] = mat[c, d], mat[d, c]
            else:
                mat[a, c] = mat[c, a] = mat[a, b]
                mat[b, d] = mat[d, b] = mat[c, d]
            mat[a, b] = mat[b, a] = 0
            mat[c, d] = mat[d, c] = 0
            edges[idx[0]] = (min(a, c), max(a, c))
            edges[idx[1]] = (min(b, d), max(b, d))

    return mat


def generate_null_networks(
    adjacency: np.ndarray,
    n_null: int = 100,
    method: str = "rewire",
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Generate an ensemble of null-model networks.

    Parameters
    ----------
    adjacency : np.ndarray (N, N)
    n_null : int
    method : str
        'rewire': Maslov-Sneppen degree-preserving.
        'erdos_renyi': random graph with same density.
        'strength': weight-preserving rewiring.
    seed : int, optional
    """
    rng = np.random.default_rng(seed)
    nulls = []

    for i in range(n_null):
        s = rng.integers(0, 2**31) + i
        if method == "rewire":
            nulls.append(random_rewire(adjacency, seed=s))
        elif method == "strength":
            nulls.append(
                random_rewire(adjacency, preserve_strength=True, seed=s)
            )
        elif method == "erdos_renyi":
            mat = adjacency.copy()
            np.fill_diagonal(mat, 0)
            n = mat.shape[0]
            triu = np.triu_indices(n, k=1)
            density = np.mean(mat[triu] != 0)
            null = np.zeros((n, n))
            edges = rng.random(len(triu[0])) < density
            null[triu] = edges.astype(float)
            null = null + null.T
            nulls.append(null)
        else:
            raise ValueError(f"Unknown null method: {method}")

    return nulls


# =============================================================================
# INTRACLASS CORRELATION COEFFICIENT
# =============================================================================

def compute_icc(
    data: np.ndarray,
    icc_type: str = "ICC(3,1)",
) -> Tuple[float, float, float]:
    """
    Intraclass Correlation Coefficient.

    Parameters
    ----------
    data : np.ndarray, shape (n_subjects, n_measures)
    icc_type : str
        'ICC(1,1)', 'ICC(2,1)', 'ICC(3,1)'.

    Returns
    -------
    icc, ci_lower, ci_upper  (95% CI)
    """
    n, k = data.shape
    mean_total = np.mean(data)

    ss_rows = k * np.sum((np.mean(data, axis=1) - mean_total) ** 2)
    ss_cols = n * np.sum((np.mean(data, axis=0) - mean_total) ** 2)
    ss_total = np.sum((data - mean_total) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    if icc_type == "ICC(1,1)":
        ms_within = (ss_total - ss_rows) / (n * (k - 1))
        icc = (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within)
    elif icc_type == "ICC(2,1)":
        icc = (ms_rows - ms_error) / (
            ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
        )
    elif icc_type == "ICC(3,1)":
        icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
    else:
        raise ValueError(f"Unknown ICC type: {icc_type}")

    # F-based 95 % CI for ICC(3,1)
    f_val = ms_rows / ms_error if ms_error > 0 else np.inf
    df1, df2 = n - 1, (n - 1) * (k - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_lo = f_val / stats.f.ppf(0.975, df1, df2) if df2 > 0 else 0
        f_hi = f_val / stats.f.ppf(0.025, df1, df2) if df2 > 0 else 0
    ci_lo = (f_lo - 1) / (f_lo + k - 1)
    ci_hi = (f_hi - 1) / (f_hi + k - 1)

    return float(icc), float(ci_lo), float(ci_hi)


# =============================================================================
# CORRELATION UTILITIES
# =============================================================================

def correlation_with_pvalue(
    x: np.ndarray, y: np.ndarray, method: str = "pearson",
) -> Tuple[float, float]:
    """
    Correlation coefficient with p-value.

    Methods: 'pearson', 'spearman', 'kendall'.
    """
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    elif method == "kendall":
        r, p = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(r), float(p)


def compare_dependent_correlations(
    r_xy: float, r_xz: float, r_yz: float, n: int,
) -> Tuple[float, float]:
    """
    Steiger (1980) test for comparing two dependent correlations r(x,y)
    vs r(x,z) where all three variables are measured on the same sample.

    Tests H₀: ρ(x,y) = ρ(x,z).

    Parameters
    ----------
    r_xy, r_xz : float
        The two correlations to compare.
    r_yz : float
        Correlation between the two criterion variables.
    n : int
        Sample size.

    Returns
    -------
    t_statistic, p_value (two-tailed)
    """
    # Steiger's Z₂* test
    z_xy = np.arctanh(np.clip(r_xy, -0.9999, 0.9999))
    z_xz = np.arctanh(np.clip(r_xz, -0.9999, 0.9999))
    r_m = 0.5 * (r_xy + r_xz)
    f = (1 - r_yz) / (2 * (1 - r_m**2)) if abs(r_m) < 1 else 1
    f = max(f, 0)
    h = (1 - f * r_m**2) / (1 - r_m**2) if abs(r_m) < 1 else 1

    denominator = np.sqrt(2 * (1 - r_yz) * h / (n - 3))
    if denominator < 1e-15:
        return 0.0, 1.0

    t_stat = (z_xy - z_xz) / denominator
    p_val = 2 * stats.norm.sf(np.abs(t_stat))

    return float(t_stat), float(p_val)


def compare_independent_correlations(
    r1: float, n1: int, r2: float, n2: int,
) -> Tuple[float, float]:
    """
    Compare two independent Pearson correlations (Fisher's method).

    Tests H₀: ρ₁ = ρ₂ using Fisher z-transformation.

    Parameters
    ----------
    r1, r2 : float
        Correlation coefficients.
    n1, n2 : int
        Sample sizes.

    Returns
    -------
    z_statistic, p_value (two-tailed)
    """
    z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
    z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z = (z1 - z2) / se if se > 0 else 0.0
    p = 2 * stats.norm.sf(np.abs(z))
    return float(z), float(p)


# =============================================================================
# REPRODUCIBILITY AND SEED MANAGEMENT
# =============================================================================

class SeedManager:
    """
    Deterministic seed management for reproducible analyses.

    Uses a master seed to derive child seeds for different analysis
    modules, ensuring reproducibility while avoiding seed collisions.

    Usage
    -----
        seeds = SeedManager(master_seed=42)
        rng_criticality = seeds.get_rng("criticality")
        rng_graph = seeds.get_rng("graph_analysis")
        seed_for_nbs = seeds.get_seed("nbs_permutation")
    """

    def __init__(self, master_seed: int = 42):
        self.master_seed = master_seed
        self._rng = np.random.default_rng(master_seed)
        self._registry: Dict[str, int] = {}

    def get_seed(self, label: str) -> int:
        """
        Get a deterministic seed for a named analysis step.

        The same label always returns the same seed for a given master_seed.
        """
        if label not in self._registry:
            # Hash-based derivation for determinism
            h = hash((self.master_seed, label)) & 0x7FFFFFFF
            self._registry[label] = h
        return self._registry[label]

    def get_rng(self, label: str) -> np.random.Generator:
        """Get a numpy Generator seeded deterministically by label."""
        return np.random.default_rng(self.get_seed(label))

    def reset(self):
        """Reset the registry."""
        self._registry.clear()
        self._rng = np.random.default_rng(self.master_seed)


# =============================================================================
# PROGRESS AND LOGGING HELPERS
# =============================================================================

class ProgressTracker:
    """
    Simple progress tracker for long-running analyses.

    Usage
    -----
        tracker = ProgressTracker(n_total=23, description="Computing metrics")
        for sub in subjects:
            # ... do work ...
            tracker.update(sub)
        tracker.finish()
    """

    def __init__(self, n_total: int, description: str = "Processing"):
        self.n_total = n_total
        self.description = description
        self.n_done = 0
        self._start_time = None

    def start(self):
        import time
        self._start_time = time.time()
        print(f"[{self.description}] Starting ({self.n_total} items)…")

    def update(self, label: str = "", increment: int = 1):
        import time
        if self._start_time is None:
            self.start()
        self.n_done += increment
        pct = self.n_done / self.n_total * 100
        elapsed = time.time() - self._start_time
        eta = (elapsed / self.n_done * (self.n_total - self.n_done)
               if self.n_done > 0 else 0)
        print(
            f"\r  [{self.n_done}/{self.n_total}] {pct:5.1f}% "
            f"| {label:>10s} | "
            f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
            end="", flush=True,
        )

    def finish(self):
        import time
        elapsed = time.time() - self._start_time if self._start_time else 0
        print(
            f"\n[{self.description}] Done: "
            f"{self.n_done}/{self.n_total} in {elapsed:.1f}s."
        )


def format_pvalue(p: float) -> str:
    """Format a p-value for display in tables and figures."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def format_effect_size(
    d: float, name: str = "g", ci: Tuple[float, float] = None,
) -> str:
    """Format an effect size with optional CI for publication."""
    s = f"{name} = {d:.2f}"
    if ci is not None:
        s += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return s


# =============================================================================
# GENERAL NUMERICAL UTILITIES
# =============================================================================

def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill: float = 0.0,
) -> np.ndarray:
    """Element-wise division with zero-denominator protection."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.full_like(num, fill)
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out


def moving_average(
    x: np.ndarray, window: int, axis: int = 0,
) -> np.ndarray:
    """
    Compute a simple moving average along an axis.

    Uses uniform_filter1d from scipy.ndimage for efficiency.
    """
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(
        np.asarray(x, dtype=float), size=window, axis=axis, mode="nearest",
    )


def cosine_similarity(
    x: np.ndarray, y: np.ndarray,
) -> float:
    """Cosine similarity between two vectors."""
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    dot = np.dot(x, y)
    norms = np.linalg.norm(x) * np.linalg.norm(y)
    return float(dot / norms) if norms > 0 else 0.0


def auc_trapezoid(
    x: np.ndarray, y: np.ndarray,
) -> float:
    """
    Area under the curve via trapezoidal rule.

    Used for multi-threshold graph metric integration.
    """
    return float(np.trapz(y, x))


def get_mtx(sub, dtype, mod=None, atlas='schaefer100'):
    if mod == None:
        if dtype == 'fmri' or dtype == 'f':
            mod = "connectivity_correlation"
        elif dtype == 'dmri' or dtype == 'd':
            mod = "connectivity_sift2"
    fpath = f"/mnt/nvme1n1p1/sars_cov_2_project/data/proc/mtxs/{sub}/{atlas}/{dtype}/{mod}.npy"
    mtx = np.load(fpath)
    return mtx

def get_lsubs():
    subs_ls = [f"sub-{k:02n}" for k in range(1, 25)]
    subs_ls.remove('sub-21')
    return subs_ls

def treat_sc(sc, diagonal=np.nan, method='max'):
    sc_final = sc.copy()
    if method == 'max':
        sc_final = sc_final / np.nanmax(sc_final)
    elif method == 'log':
        sc_final = np.log1p(sc_final)
    np.fill_diagonal(sc_final, diagonal)
    return sc_final
    
