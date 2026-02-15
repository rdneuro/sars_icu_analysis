# -*- coding: utf-8 -*-
"""
sars.criticality.dfa
==================================

Detrended Fluctuation Analysis (DFA) and its multifractal extension
(MF-DFA) for quantifying long-range temporal correlations in BOLD
timeseries.

Theory
------
DFA decomposes a signal's fluctuations across temporal scales to
estimate the Hurst-like scaling exponent alpha:

    F(n) ~ n^alpha

where F(n) is the RMS fluctuation at scale n (window size).

Interpretation:
    alpha ~ 0.5  -> uncorrelated white noise
    alpha ~ 1.0  -> 1/f noise (long-range correlations, criticality)
    alpha ~ 1.5  -> Brownian motion (integrated white noise)
    0.5 < alpha < 1.0 -> persistent long-range correlations
    alpha > 1.0 -> non-stationary, stronger correlations

At the critical point, brain dynamics exhibit 1/f scaling (alpha ~ 1),
reflecting scale-free temporal organization.

Healthy resting fMRI: alpha = 0.65-0.95 (He 2011), with DMN regions
showing the highest values (~0.85-0.95).

Preprocessing Caveat
--------------------
**Bandpass filtering alters DFA exponents**. Low-pass filtering
artificially inflates alpha by removing high-frequency fluctuations.
For DFA analysis, use ONLY high-pass filtering (> 0.01 Hz), not the
typical 0.01-0.1 Hz bandpass used for FC analysis (Hardstone et al.
2012; He 2011). This module issues a warning if called with data that
may have been bandpass-filtered.

Scale Range for Short Timeseries
--------------------------------
For N ~ 200 timepoints:
    Minimum scale: 4 (hardcoded minimum for reliable detrending).
    Maximum scale: N // 4 = 50 (standard recommendation).
    This provides approximately one decade of scales (4-50), which is
    marginal but workable for DFA-1.

Multifractal DFA is NOT reliable for N ~ 200 (ICC < 0.3 per Guan et al.
2025, NeuroImage). This module includes MF-DFA for completeness but
issues an explicit warning and disables it by default for short series.

References
----------
- Peng et al. (1994). Phys Rev E 49:1685.
- Kantelhardt et al. (2002). Physica A 316:87. MF-DFA.
- Hardstone et al. (2012). Front Physiol 3:450. DFA for neuroimaging.
- Tagliazucchi et al. (2013). Hum Brain Mapp 34:2443. DFA in rs-fMRI.
- He (2011). PLoS ONE 6:e17645. Scale-free properties of BOLD signal.
- Ciuciu et al. (2012). Front Physiol 3:186. Multifractal in brain.
- Guan et al. (2025). NeuroImage. Reliability of MF-DFA in fMRI.

Usage
-----
    from sars.criticality.dfa import analyze_dfa

    results = analyze_dfa(timeseries, tr=2.217)
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple, List, Any
import warnings


# =============================================================================
# STANDARD DFA
# =============================================================================

def compute_dfa(
    signal: np.ndarray,
    scales: Optional[np.ndarray] = None,
    order: int = 1,
    min_scale: Optional[int] = None,
    max_scale: Optional[int] = None,
    n_scales: int = 20,
) -> Dict[str, Any]:
    """
    Detrended Fluctuation Analysis of a 1D signal.

    Algorithm (Peng et al. 1994):
    1. Compute the profile: Y(k) = sum_{i=1}^{k} (x_i - <x>)
    2. Divide Y into non-overlapping segments of size n.
    3. In each segment, fit a polynomial of order m and compute the
       RMS of the residuals.
    4. Average the RMS across segments to get F(n).
    5. Repeat for different n.
    6. Estimate alpha from the log-log regression of F(n) vs n.

    Both forward and backward traversals of the signal are used to
    maximize data utilization (Kantelhardt et al. 2002).

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
        1D signal to analyze (e.g., mean BOLD or single ROI).
    scales : np.ndarray, optional
        Array of window sizes. If None, computed automatically
        as logarithmically spaced integers.
    order : int
        Polynomial order for detrending. DFA-1 (order=1) is standard
        for fMRI data. Default: 1.
    min_scale : int, optional
        Minimum window size. Default: 4.
    max_scale : int, optional
        Maximum window size. Default: min(T // 4, 50) for short series.
    n_scales : int
        Number of logarithmically spaced scales. Default: 20.

    Returns
    -------
    dict
        'alpha'       : float -- DFA exponent.
        'intercept'   : float -- log-log intercept.
        'r_squared'   : float -- goodness of linear fit (> 0.95 desirable).
        'p_value'     : float -- p-value of slope.
        'std_err'     : float -- standard error of alpha.
        'scales'      : np.ndarray -- window sizes used.
        'fluctuations': np.ndarray -- F(n) for each scale.
        'log_scales'  : np.ndarray -- log10(scales).
        'log_fluctuations': np.ndarray -- log10(F(n)).
        'ci_alpha'    : tuple -- 95% CI for alpha.
        'order'       : int
        'crossover'   : dict or None -- crossover detection results.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    T = len(x)

    if T < 16:
        return _empty_dfa_result("signal too short (< 16 samples)")

    # Integrate the signal (cumulative sum of deviations from mean)
    y = np.cumsum(x - np.mean(x))

    # Determine scales
    if scales is None:
        min_s = min_scale or 4
        max_s = max_scale or min(T // 4, max(min_s + 1, T // 4))
        if max_s <= min_s:
            max_s = min_s + 1
        scales = np.unique(
            np.logspace(
                np.log10(min_s), np.log10(max_s), n_scales
            ).astype(int)
        )
        scales = scales[(scales >= 4) & (scales <= T // 2)]

    if len(scales) < 3:
        return _empty_dfa_result("insufficient valid scales (< 3)")

    # Compute fluctuation F(n) for each scale
    fluctuations = np.zeros(len(scales))

    for i, n in enumerate(scales):
        n_segments = T // n
        if n_segments < 1:
            fluctuations[i] = np.nan
            continue

        f2_all = []
        t_vec = np.arange(n, dtype=float)

        # Forward segments
        for seg in range(n_segments):
            start = seg * n
            segment = y[start:start + n]
            coeffs = np.polyfit(t_vec, segment, order)
            trend = np.polyval(coeffs, t_vec)
            f2_all.append(np.mean((segment - trend) ** 2))

        # Backward segments (from end of signal)
        for seg in range(n_segments):
            start = T - (seg + 1) * n
            if start < 0:
                break
            segment = y[start:start + n]
            coeffs = np.polyfit(t_vec, segment, order)
            trend = np.polyval(coeffs, t_vec)
            f2_all.append(np.mean((segment - trend) ** 2))

        fluctuations[i] = np.sqrt(np.mean(f2_all))

    # Remove invalid scales
    valid = np.isfinite(fluctuations) & (fluctuations > 0)
    scales_valid = scales[valid]
    fluct_valid = fluctuations[valid]

    if len(scales_valid) < 3:
        return _empty_dfa_result("too few valid fluctuation points")

    # Log-log regression
    log_scales = np.log10(scales_valid.astype(float))
    log_fluct = np.log10(fluct_valid)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_scales, log_fluct
    )

    # 95% CI for the slope
    n_pts = len(log_scales)
    t_val = stats.t.ppf(0.975, max(n_pts - 2, 1))
    ci_lower = slope - t_val * std_err
    ci_upper = slope + t_val * std_err

    # --- Crossover detection ---
    crossover = _detect_crossover(log_scales, log_fluct)

    result = {
        "alpha": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "scales": scales_valid,
        "fluctuations": fluct_valid,
        "log_scales": log_scales,
        "log_fluctuations": log_fluct,
        "ci_alpha": (float(ci_lower), float(ci_upper)),
        "order": order,
        "crossover": crossover,
    }

    # Quality warning
    if r_value ** 2 < 0.90:
        result["warning_r_squared"] = (
            f"R^2 = {r_value**2:.3f} < 0.90: poor log-log linearity. "
            f"DFA exponent may not be reliable."
        )

    return result


def _detect_crossover(
    log_scales: np.ndarray,
    log_fluct: np.ndarray,
    min_points_per_segment: int = 3,
) -> Optional[Dict[str, Any]]:
    """
    Detect crossover in the DFA log-log plot by testing whether a
    piecewise linear fit (two segments) significantly improves over
    a single linear fit.

    A crossover indicates different scaling regimes at different
    temporal scales (e.g., white noise at short scales, long-range
    correlations at long scales).

    Returns None if insufficient points or no improvement.
    """
    n = len(log_scales)
    if n < 2 * min_points_per_segment + 1:
        return None

    # Single-segment fit
    _, _, r_single, _, _ = stats.linregress(log_scales, log_fluct)
    ss_single = np.sum(
        (log_fluct - np.polyval(
            np.polyfit(log_scales, log_fluct, 1), log_scales
        )) ** 2
    )

    best_improvement = 0
    best_idx = None
    best_alphas = None

    for split in range(min_points_per_segment, n - min_points_per_segment):
        # Fit two segments
        s1, i1, r1, _, _ = stats.linregress(
            log_scales[:split], log_fluct[:split]
        )
        s2, i2, r2, _, _ = stats.linregress(
            log_scales[split:], log_fluct[split:]
        )

        pred1 = np.polyval([s1, i1], log_scales[:split])
        pred2 = np.polyval([s2, i2], log_scales[split:])
        ss_two = np.sum((log_fluct[:split] - pred1) ** 2) + \
                 np.sum((log_fluct[split:] - pred2) ** 2)

        improvement = 1 - ss_two / ss_single if ss_single > 0 else 0

        if improvement > best_improvement:
            best_improvement = improvement
            best_idx = split
            best_alphas = (s1, s2)

    # Only report if substantial improvement (> 10% reduction in SS)
    if best_improvement > 0.10 and best_idx is not None:
        crossover_scale = 10 ** log_scales[best_idx]
        return {
            "crossover_scale": float(crossover_scale),
            "alpha_short": float(best_alphas[0]),
            "alpha_long": float(best_alphas[1]),
            "improvement": float(best_improvement),
            "split_index": best_idx,
        }

    return None


def compute_dfa_per_roi(
    timeseries: np.ndarray,
    order: int = 1,
    n_scales: int = 20,
    min_scale: Optional[int] = None,
    max_scale: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute DFA exponent for each ROI independently.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T, N)
    order : int
    n_scales : int
    min_scale, max_scale : int, optional

    Returns
    -------
    dict
        'alpha_per_roi'   : np.ndarray (N,)
        'alpha_mean'      : float
        'alpha_std'       : float
        'alpha_median'    : float
        'r_squared_per_roi': np.ndarray (N,)
        'r_squared_mean'  : float
        'ci_per_roi'      : list of (lower, upper) tuples
    """
    ts = np.asarray(timeseries, dtype=np.float64)
    T, N = ts.shape

    alphas = np.full(N, np.nan)
    r2s = np.full(N, np.nan)
    cis = []

    for roi in range(N):
        result = compute_dfa(
            ts[:, roi], order=order, n_scales=n_scales,
            min_scale=min_scale, max_scale=max_scale,
        )
        alphas[roi] = result["alpha"]
        r2s[roi] = result.get("r_squared", np.nan)
        cis.append(result.get("ci_alpha", (np.nan, np.nan)))

    return {
        "alpha_per_roi": alphas,
        "alpha_mean": float(np.nanmean(alphas)),
        "alpha_std": float(np.nanstd(alphas)),
        "alpha_median": float(np.nanmedian(alphas)),
        "r_squared_per_roi": r2s,
        "r_squared_mean": float(np.nanmean(r2s)),
        "ci_per_roi": cis,
    }


# =============================================================================
# MULTIFRACTAL DFA (MF-DFA)
# =============================================================================

def compute_mfdfa(
    signal: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    order: int = 1,
    n_scales: int = 20,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Multifractal Detrended Fluctuation Analysis.

    Extends DFA by computing q-th order fluctuation functions F_q(n),
    yielding the generalized Hurst exponent h(q) and the multifractal
    singularity spectrum f(alpha_mf).

    **WARNING**: MF-DFA is NOT reliable for short timeseries (N < 1000).
    Guan et al. (2025, NeuroImage) showed ICC < 0.3 for typical fMRI
    scan lengths. Results should be interpreted with extreme caution
    and are disabled by default for N < 500 (set force=True to override).

    Parameters
    ----------
    signal : np.ndarray, shape (T,)
    q_values : np.ndarray, optional
        Moment orders. Default: linspace(-5, 5, 41). q=2 = standard DFA.
    scales : np.ndarray, optional
    order : int
    n_scales : int
    force : bool
        If True, compute MF-DFA even for short signals. Default: False.

    Returns
    -------
    dict
        'q_values', 'hq', 'tau_q', 'alpha_mf', 'f_alpha',
        'delta_h', 'h2', 'alpha_width', 'Fq_scales', 'scales'.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    T = len(x)

    if T < 500 and not force:
        warnings.warn(
            f"MF-DFA unreliable for N={T} (ICC < 0.3 per Guan et al. "
            f"2025). Returning NaN. Set force=True to compute anyway.",
            UserWarning,
        )
        if q_values is None:
            q_values = np.linspace(-5, 5, 41)
        return {
            "q_values": q_values,
            "hq": np.full_like(q_values, np.nan),
            "tau_q": np.full_like(q_values, np.nan),
            "alpha_mf": np.full(len(q_values) - 1, np.nan),
            "f_alpha": np.full(len(q_values) - 1, np.nan),
            "delta_h": np.nan, "h2": np.nan,
            "alpha_width": np.nan,
            "Fq_scales": np.array([[]]),
            "scales": np.array([]),
            "warning": "MF-DFA not computed: N < 500",
        }

    if q_values is None:
        q_values = np.linspace(-5, 5, 41)
    q_values = np.asarray(q_values, dtype=float)

    # Integrate
    y = np.cumsum(x - np.mean(x))

    # Scales
    if scales is None:
        min_s = 4
        max_s = max(T // 4, min_s + 1)
        scales = np.unique(
            np.logspace(
                np.log10(min_s), np.log10(max_s), n_scales
            ).astype(int)
        )
        scales = scales[(scales >= 4) & (scales <= T // 2)]

    if len(scales) < 3:
        return {
            "error": "insufficient scales",
            "hq": np.full_like(q_values, np.nan),
        }

    n_q = len(q_values)
    n_s = len(scales)
    Fq = np.zeros((n_q, n_s))

    for si, n in enumerate(scales):
        n_seg = T // n
        if n_seg < 1:
            Fq[:, si] = np.nan
            continue

        f2_segs = []
        t_vec = np.arange(n, dtype=float)

        # Forward + backward segments
        for seg in range(n_seg):
            segment = y[seg * n: (seg + 1) * n]
            coeffs = np.polyfit(t_vec, segment, order)
            trend = np.polyval(coeffs, t_vec)
            f2_segs.append(np.mean((segment - trend) ** 2))

        for seg in range(n_seg):
            start = T - (seg + 1) * n
            if start < 0:
                break
            segment = y[start: start + n]
            coeffs = np.polyfit(t_vec, segment, order)
            trend = np.polyval(coeffs, t_vec)
            f2_segs.append(np.mean((segment - trend) ** 2))

        f2_arr = np.array(f2_segs)
        f2_arr = f2_arr[f2_arr > 0]

        if len(f2_arr) < 2:
            Fq[:, si] = np.nan
            continue

        for qi, q in enumerate(q_values):
            if abs(q) < 1e-10:
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(f2_arr)))
            else:
                Fq[qi, si] = (np.mean(f2_arr ** (q / 2))) ** (1 / q)

    # Fit h(q) for each q
    hq = np.full(n_q, np.nan)
    for qi in range(n_q):
        valid = np.isfinite(Fq[qi, :]) & (Fq[qi, :] > 0)
        if np.sum(valid) >= 3:
            log_s = np.log10(scales[valid].astype(float))
            log_f = np.log10(Fq[qi, valid])
            slope, _, _, _, _ = stats.linregress(log_s, log_f)
            hq[qi] = slope

    # Derive multifractal spectrum
    tau_q = q_values * hq - 1

    # Singularity spectrum via numerical Legendre transform
    dq = np.diff(q_values)
    dtau = np.diff(tau_q)
    alpha_mf = dtau / dq
    q_mid = (q_values[:-1] + q_values[1:]) / 2
    f_alpha = q_mid * alpha_mf - (tau_q[:-1] + tau_q[1:]) / 2

    delta_h = (
        float(hq[0] - hq[-1])
        if np.isfinite(hq[0]) and np.isfinite(hq[-1])
        else np.nan
    )
    h2_idx = np.argmin(np.abs(q_values - 2))
    h2 = float(hq[h2_idx]) if np.isfinite(hq[h2_idx]) else np.nan

    valid_alpha = alpha_mf[np.isfinite(alpha_mf)]
    alpha_width = (
        float(np.max(valid_alpha) - np.min(valid_alpha))
        if len(valid_alpha) > 2 else np.nan
    )

    return {
        "q_values": q_values,
        "hq": hq,
        "tau_q": tau_q,
        "alpha_mf": alpha_mf,
        "f_alpha": f_alpha,
        "delta_h": delta_h,
        "h2": h2,
        "alpha_width": alpha_width,
        "Fq_scales": Fq,
        "scales": scales,
    }


# =============================================================================
# HELPERS
# =============================================================================

def _empty_dfa_result(reason: str) -> Dict[str, Any]:
    return {
        "alpha": np.nan, "intercept": np.nan,
        "r_squared": np.nan, "p_value": np.nan,
        "std_err": np.nan,
        "scales": np.array([]),
        "fluctuations": np.array([]),
        "log_scales": np.array([]),
        "log_fluctuations": np.array([]),
        "ci_alpha": (np.nan, np.nan),
        "order": 1, "reason": reason, "crossover": None,
    }


# =============================================================================
# MASTER ANALYSIS FUNCTION
# =============================================================================

def analyze_dfa(
    timeseries: np.ndarray,
    tr: Optional[float] = None,
    order: int = 1,
    compute_per_roi: bool = True,
    compute_multifractal: bool = False,
    n_scales: int = 20,
    bandpass_warning: bool = True,
) -> Dict[str, Any]:
    """
    Complete DFA analysis pipeline.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T, N)
    tr : float, optional
    order : int
        Polynomial detrending order. DFA-1 (default) recommended.
    compute_per_roi : bool
        Compute per-ROI DFA. Default: True.
    compute_multifractal : bool
        Compute MF-DFA. Default: False (unreliable for N < 500).
    n_scales : int
    bandpass_warning : bool
        If True, emit a warning about bandpass filtering effects.

    Returns
    -------
    dict with global DFA, per-ROI DFA, and optional MF-DFA results.
    """
    ts = np.asarray(timeseries, dtype=np.float64)
    T, N = ts.shape

    results = {"n_timepoints": T, "n_rois": N, "order": order}

    # --- Bandpass warning ---
    if bandpass_warning:
        results["preprocessing_note"] = (
            "IMPORTANT: Bandpass filtering (especially low-pass) "
            "artificially inflates DFA exponents. For DFA analysis, "
            "use ONLY high-pass filtered data (> 0.01 Hz), NOT the "
            "standard 0.01-0.1 Hz bandpass used for FC. See Hardstone "
            "et al. (2012) Front Physiol, He (2011) PLoS ONE."
        )

    # --- Scale range note for short timeseries ---
    if T < 300:
        results["scale_range_note"] = (
            f"With N = {T} timepoints, DFA scale range is limited to "
            f"approximately 4-{T // 4} (~{np.log10(T / 16):.1f} decades). "
            f"DFA-1 (linear detrending) is the most appropriate choice. "
            f"MF-DFA results would be unreliable."
        )

    # Global DFA on mean BOLD signal
    mean_signal = ts.mean(axis=1)
    global_dfa = compute_dfa(
        mean_signal, order=order, n_scales=n_scales
    )
    results["global_alpha"] = global_dfa["alpha"]
    results["global_r_squared"] = global_dfa["r_squared"]
    results["global_ci_alpha"] = global_dfa["ci_alpha"]
    results["global_crossover"] = global_dfa.get("crossover", None)
    results["global_dfa"] = global_dfa

    # Per-ROI DFA
    if compute_per_roi:
        roi_dfa = compute_dfa_per_roi(
            ts, order=order, n_scales=n_scales
        )
        results["alpha_per_roi"] = roi_dfa["alpha_per_roi"]
        results["alpha_mean"] = roi_dfa["alpha_mean"]
        results["alpha_std"] = roi_dfa["alpha_std"]
        results["alpha_median"] = roi_dfa["alpha_median"]
        results["r_squared_per_roi"] = roi_dfa["r_squared_per_roi"]
        results["r_squared_mean"] = roi_dfa["r_squared_mean"]

    # Multifractal DFA (disabled by default for short series)
    if compute_multifractal:
        mfdfa = compute_mfdfa(
            mean_signal, order=order, n_scales=n_scales
        )
        results["mfdfa_h2"] = mfdfa.get("h2", np.nan)
        results["mfdfa_delta_h"] = mfdfa.get("delta_h", np.nan)
        results["mfdfa_alpha_width"] = mfdfa.get("alpha_width", np.nan)
        results["mfdfa"] = mfdfa

    return results
