# -*- coding: utf-8 -*-
"""
sars.criticality.avalanches
=========================================

Neuronal avalanche detection and criticality analysis from parcellated
resting-state fMRI timeseries.

Theory
------
At criticality, cascades of neural activity (avalanches) follow
power-law distributions in both size S and duration T:

    P(S) ~ S^{-tau_s}  ,  P(T) ~ T^{-tau_t}

with a crackling-noise scaling relation linking the two exponents
(Friedman et al. 2012; Sethna, Dahmen & Myers 2001):

    gamma = (tau_t - 1) / (tau_s - 1)

This exponent also governs the average size as a function of duration:

    <S>(T) ~ T^gamma

Mean-field predictions: tau_s ~ 3/2, tau_t ~ 2, gamma ~ 2.
Recent in vivo evidence: gamma ~ 1.2-1.3 (Fontenele et al. 2019),
suggesting a non-mean-field universality class.

The branching ratio sigma ~ 1 at the critical point.

Pipeline
--------
1. Z-score the timeseries per ROI.
2. Binarize: |z_it| > threshold -> active.
3. At each timepoint, count active ROIs.
4. An avalanche is a contiguous sequence of timepoints with >= 1 active
   ROI, bounded by silent frames.
5. Avalanche SIZE S = total activations; DURATION T = number of frames.
6. Fit power-law to size and duration distributions using the
   Clauset-Shalizi-Newman (2009) framework:
   a. Discrete MLE with Hurwitz zeta normalization
   b. x_min estimated by KS minimization
   c. Semi-parametric bootstrap goodness-of-fit (p >= 0.1 = not rejected)
   d. Likelihood ratio tests vs exponential, lognormal, truncated PL
7. Assess branching ratio:
   a. Conventional: sigma = <n(t+1)/n(t)>
   b. MR estimator (Wilting & Priesemann 2018): exponential fit to
      autocorrelation function -- invariant to subsampling
8. Crackling-noise consistency check: |gamma_predicted - gamma_empirical|
9. Avalanche shape collapse: rescale temporal profiles

Threshold Selection for fMRI
-----------------------------
Unlike MEG/EEG where 3 SD is standard, the hemodynamic response
function smooths neural activity, requiring lower thresholds.
Recommended range: 1.0-1.4 SD (Shriki et al. 2013; Tagliazucchi et al.
2012). We default to 1.0 SD with a built-in sensitivity analysis
across [0.5, 1.0, 1.5, 2.0, 2.5] SD.

Data Requirements
-----------------
Individual-level power-law fitting requires >= 100 tail observations
(ideally 500-1000) for reliable exponent estimation. With ~200 volumes
and ~100 ROIs, individual subjects yield ~30-80 avalanches -- typically
insufficient for standalone fitting. Group-level pooling across subjects
is recommended (and provided via pool_avalanches_group()).

References
----------
- Beggs & Plenz (2003). J Neurosci 23(35):11167.
- Shriki et al. (2013). J Neurosci 33(16):7079.
- Clauset, Shalizi & Newman (2009). SIAM Rev 51(4):661.
- Alstott, Bullmore & Plenz (2014). PLoS ONE 9(1):e85777.
- Tagliazucchi et al. (2012). Front Physiol 3:15.
- Fontenele et al. (2019). Phys Rev Lett 122:208101.
- Friedman et al. (2012). Phys Rev Lett 108:208102.
- Palva et al. (2013). PNAS 110(9):3585.
- Shew et al. (2009). J Neurosci 29(49):15595.
- Wilting & Priesemann (2018). Nat Commun 9:1090.
- Sethna, Dahmen & Myers (2001). Nature 410:242.
- Touboul & Destexhe (2017). PLoS ONE 12:e0169930.

Usage
-----
    from sars.criticality.avalanches import (
        analyze_avalanches,
        pool_avalanches_group,
        sensitivity_analysis,
    )

    # Single subject
    results = analyze_avalanches(timeseries, threshold_sd=1.0)

    # Group pooling (recommended for power-law fitting)
    group_results = pool_avalanches_group(subject_timeseries_list)

    # Sensitivity across thresholds
    sensitivity = sensitivity_analysis(timeseries)
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple, List, Any, Union
import warnings


# =============================================================================
# AVALANCHE DETECTION
# =============================================================================

def detect_avalanches(
    timeseries: np.ndarray,
    threshold_sd: float = 1.0,
    min_duration: int = 1,
    temporal_bin: int = 1,
    zscore: bool = True,
) -> Dict[str, Any]:
    """
    Detect neuronal avalanches from parcellated BOLD timeseries.

    The input timeseries is z-scored per ROI, then binarized at the
    given threshold. An avalanche is defined as a contiguous sequence
    of timepoints with at least one active ROI, bounded by silent
    frames (no active ROI).

    Parameters
    ----------
    timeseries : np.ndarray, shape (T, N)
        Parcellated BOLD timeseries. T = timepoints, N = ROIs.
        If zscore=True (default), will be z-scored per ROI.
    threshold_sd : float
        Z-score threshold for binarization. An ROI is "active" at time
        t if |z_{it}| > threshold_sd.
        Recommended for fMRI: 1.0 SD (Shriki et al. 2013).
        Default: 1.0.
    min_duration : int
        Minimum avalanche duration in (binned) timepoints. Default: 1.
    temporal_bin : int
        Number of consecutive TRs to bin together before detection.
        If > 1, the binary matrix is coarse-grained first by taking
        the max (logical OR) within each bin. Default: 1 (no binning).
    zscore : bool
        If True, z-score the timeseries per ROI before thresholding.
        Set to False if timeseries are already z-scored.

    Returns
    -------
    dict
        'sizes'         : np.ndarray (int) -- total activations per avalanche.
        'durations'     : np.ndarray (int) -- number of binned frames.
        'n_avalanches'  : int -- total number of avalanches detected.
        'binary_matrix' : np.ndarray (T_eff, N) -- binarized activity.
        'activity_per_frame' : np.ndarray (T_eff,) -- # active ROIs/frame.
        'avalanche_indices'  : list of (start, end) tuples.
        'active_rois_per_avalanche' : list of lists of ROI indices.
        'temporal_profiles'  : list of np.ndarray -- activity profile per
                               avalanche (for shape analysis).
        'threshold_sd'  : float
        'silent_fraction' : float -- fraction of silent timepoints.
    """
    ts = np.asarray(timeseries, dtype=np.float64)
    if ts.ndim != 2:
        raise ValueError(f"Expected (T, N) array, got shape {ts.shape}")
    T, N = ts.shape

    # --- Z-score per ROI ---
    if zscore:
        means = ts.mean(axis=0, keepdims=True)
        stds = ts.std(axis=0, keepdims=True, ddof=0)
        stds[stds < 1e-12] = 1.0
        z = (ts - means) / stds
    else:
        z = ts.copy()

    # --- Temporal binning ---
    if temporal_bin > 1:
        n_bins = T // temporal_bin
        if n_bins < 2:
            raise ValueError(
                f"temporal_bin={temporal_bin} too large for T={T}"
            )
        z_binned = np.zeros((n_bins, N))
        for b in range(n_bins):
            # Max operation: active if ANY TR in bin exceeds threshold
            z_binned[b, :] = np.max(
                np.abs(z[b * temporal_bin:(b + 1) * temporal_bin, :]),
                axis=0,
            )
        # Re-threshold on binned absolute values
        binary = (z_binned > threshold_sd).astype(np.int8)
    else:
        binary = (np.abs(z) > threshold_sd).astype(np.int8)

    T_eff = binary.shape[0]

    # --- Count active ROIs per frame ---
    activity = binary.sum(axis=1)  # shape (T_eff,)

    # --- Identify avalanches ---
    is_active = (activity > 0).astype(np.int8)
    diff = np.diff(np.concatenate(([0], is_active, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    sizes = []
    durations = []
    avalanche_indices = []
    active_rois = []
    temporal_profiles = []

    for s, e in zip(starts, ends):
        dur = e - s
        if dur < min_duration:
            continue
        size = int(activity[s:e].sum())
        sizes.append(size)
        durations.append(dur)
        avalanche_indices.append((int(s), int(e)))

        # ROIs that participated at least once
        rois = np.where(binary[s:e, :].sum(axis=0) > 0)[0].tolist()
        active_rois.append(rois)

        # Temporal profile of activity
        temporal_profiles.append(activity[s:e].astype(float).copy())

    silent_frames = int(np.sum(is_active == 0))

    return {
        "sizes": np.array(sizes, dtype=int),
        "durations": np.array(durations, dtype=int),
        "n_avalanches": len(sizes),
        "binary_matrix": binary,
        "activity_per_frame": activity,
        "avalanche_indices": avalanche_indices,
        "active_rois_per_avalanche": active_rois,
        "temporal_profiles": temporal_profiles,
        "threshold_sd": threshold_sd,
        "silent_fraction": float(silent_frames / T_eff) if T_eff > 0 else 0.0,
        "T_effective": T_eff,
        "N_rois": N,
    }


# =============================================================================
# POWER-LAW FITTING (Clauset, Shalizi & Newman 2009)
# =============================================================================

def fit_powerlaw_distribution(
    data: np.ndarray,
    xmin: Optional[float] = None,
    discrete: bool = True,
    n_bootstrap_gof: int = 1000,
    compare_distributions: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fit a power-law distribution P(x) ~ x^{-alpha} using the Clauset-
    Shalizi-Newman (2009) framework.

    The method consists of three steps:
    1. MLE estimation of alpha given x_min (discrete case uses the
       Hurwitz zeta function for normalization).
    2. x_min estimation by minimizing the KS distance over candidate
       values (KS-minimization method).
    3. Goodness-of-fit via semi-parametric bootstrap: generate synthetic
       data from the fitted power-law + empirical below x_min, refit,
       and compare KS distances.

    Uses the ``powerlaw`` package (Alstott et al. 2014) when available,
    with a pure-numpy MLE fallback.

    Parameters
    ----------
    data : np.ndarray
        Observed values (e.g., avalanche sizes or durations).
        Must contain at least 10 positive values.
    xmin : float, optional
        If provided, use this x_min instead of estimating it.
    discrete : bool
        If True, fit discrete power-law (appropriate for avalanche
        sizes/durations which are integer-valued). Default: True.
    n_bootstrap_gof : int
        Number of bootstrap samples for the goodness-of-fit p-value.
        Set to 0 to skip (faster but no p-value). Default: 1000.
        Recommended: >= 1000 (Clauset et al. 2009).
    compare_distributions : bool
        If True, perform likelihood ratio tests against exponential,
        lognormal, and truncated power-law alternatives. Default: True.
    verbose : bool
        If True, print fitting progress.

    Returns
    -------
    dict
        'alpha'       : float -- power-law exponent.
        'xmin'        : float -- lower cutoff of the scaling region.
        'sigma'       : float -- standard error on alpha.
        'ks_stat'     : float -- KS distance between data and fit.
        'p_value'     : float -- bootstrap GOF p-value (p >= 0.1 to not
                        reject power-law hypothesis; Clauset et al. 2009).
        'n_tail'      : int -- number of observations >= x_min.
        'n_total'     : int -- total number of data points.
        'comparison_exponential'    : dict with 'R' and 'p'
        'comparison_lognormal'      : dict with 'R' and 'p'
        'comparison_truncated_pl'   : dict with 'R' and 'p'
        'method'      : str -- 'clauset' or 'mle_fallback'.
        'fit_object'  : powerlaw.Fit object (if available).

    Notes
    -----
    Interpretation of likelihood ratio R:
        R > 0: power-law preferred over alternative.
        R < 0: alternative preferred over power-law.
        p < 0.05: the sign of R is statistically significant.

    The Vuong test (normalized ratio) determines whether the likelihood
    ratio is significantly different from zero (Vuong 1989).

    **Critical caveat**: Passing the power-law GOF test (p >= 0.1) is
    *necessary but not sufficient* evidence for criticality. Alternative
    distributions must also be tested (Touboul & Destexhe 2017).
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    data = data[np.isfinite(data) & (data > 0)]

    if len(data) < 10:
        return _empty_powerlaw_result(
            "insufficient data (< 10 positive values)", n_total=len(data)
        )

    # --- Try powerlaw package (recommended) ---
    try:
        import powerlaw

        fit = powerlaw.Fit(
            data, xmin=xmin, discrete=discrete, verbose=verbose
        )

        alpha = float(fit.power_law.alpha)
        xmin_est = float(fit.power_law.xmin)
        sigma = float(fit.power_law.sigma)
        ks_stat = float(fit.power_law.KS())
        n_tail = int(np.sum(data >= xmin_est))

        # --- Goodness-of-fit p-value via semi-parametric bootstrap ---
        p_value = np.nan
        if n_bootstrap_gof > 0 and n_tail >= 20:
            try:
                ks_synth = []
                for _ in range(n_bootstrap_gof):
                    # Synthetic data: below xmin from empirical,
                    #                 above xmin from fitted PL
                    below = data[data < xmin_est]
                    n_above = n_tail
                    synthetic_above = fit.power_law.generate_random(n_above)
                    if len(below) > 0:
                        synthetic = np.concatenate([below, synthetic_above])
                    else:
                        synthetic = synthetic_above
                    synth_fit = powerlaw.Fit(
                        synthetic, xmin=xmin_est, discrete=discrete,
                        verbose=False
                    )
                    ks_synth.append(synth_fit.power_law.KS())
                p_value = float(np.mean(np.array(ks_synth) >= ks_stat))
            except Exception as e:
                if verbose:
                    warnings.warn(f"Bootstrap GOF failed: {e}")

        # --- Likelihood ratio tests ---
        comparisons = {}
        if compare_distributions:
            for alt_name in [
                "exponential", "lognormal", "truncated_power_law"
            ]:
                try:
                    R, p = fit.distribution_compare(
                        "power_law", alt_name, normalized_ratio=True
                    )
                    comparisons[alt_name] = {
                        "R": float(R), "p": float(p)
                    }
                except Exception:
                    comparisons[alt_name] = {"R": np.nan, "p": np.nan}

        return {
            "alpha": alpha,
            "xmin": xmin_est,
            "sigma": sigma,
            "ks_stat": ks_stat,
            "p_value": p_value,
            "n_tail": n_tail,
            "n_total": len(data),
            "comparison_exponential": comparisons.get(
                "exponential", {"R": np.nan, "p": np.nan}
            ),
            "comparison_lognormal": comparisons.get(
                "lognormal", {"R": np.nan, "p": np.nan}
            ),
            "comparison_truncated_pl": comparisons.get(
                "truncated_power_law", {"R": np.nan, "p": np.nan}
            ),
            "method": "clauset",
            "fit_object": fit,
        }

    except ImportError:
        warnings.warn(
            "powerlaw package not installed. Using MLE fallback. "
            "Install with: pip install powerlaw"
        )
    except Exception as e:
        warnings.warn(
            f"powerlaw package failed ({e}). Using MLE fallback."
        )

    # --- Fallback: pure-numpy MLE for discrete power-law ---
    return _mle_powerlaw_fallback(data, xmin)


def _mle_powerlaw_fallback(
    data: np.ndarray,
    xmin: Optional[float] = None,
) -> Dict[str, Any]:
    """
    MLE estimation of discrete power-law exponent (fallback method).

    Uses the Hill estimator adapted for discrete distributions:
        alpha_hat = 1 + n / sum(ln(x_i / (x_min - 0.5)))

    If x_min is not provided, attempts KS-minimization over unique
    data values.
    """
    data = data[data > 0]
    n_total = len(data)

    # --- x_min estimation via KS minimization ---
    if xmin is None:
        unique_vals = np.unique(data)
        if len(unique_vals) < 3:
            return _empty_powerlaw_result(
                "fewer than 3 unique values", n_total
            )

        best_ks = np.inf
        best_xmin = unique_vals[0]
        best_alpha = np.nan

        for candidate in unique_vals[:-2]:
            tail = data[data >= candidate]
            n = len(tail)
            if n < 5:
                continue
            denom = np.sum(np.log(tail / (candidate - 0.5)))
            if denom <= 0:
                continue
            alpha_c = 1.0 + n / denom
            sorted_tail = np.sort(tail)
            cdf_emp = np.arange(1, n + 1) / n
            cdf_theo = 1 - (sorted_tail / candidate) ** (-(alpha_c - 1))
            cdf_theo = np.clip(cdf_theo, 0, 1)
            ks = np.max(np.abs(cdf_emp - cdf_theo))
            if ks < best_ks:
                best_ks = ks
                best_xmin = candidate
                best_alpha = alpha_c

        xmin = best_xmin

    tail = data[data >= xmin]
    n = len(tail)

    if n < 5:
        return _empty_powerlaw_result(
            "fewer than 5 data points above x_min", n_total
        )

    # Discrete MLE
    denom = np.sum(np.log(tail / (xmin - 0.5)))
    if denom <= 0:
        return _empty_powerlaw_result("degenerate data for MLE", n_total)

    alpha = 1.0 + n / denom
    sigma = (alpha - 1.0) / np.sqrt(n)

    # KS statistic
    sorted_tail = np.sort(tail)
    cdf_emp = np.arange(1, n + 1) / n
    cdf_theo = 1 - (sorted_tail / xmin) ** (-(alpha - 1))
    cdf_theo = np.clip(cdf_theo, 0, 1)
    ks_stat = float(np.max(np.abs(cdf_emp - cdf_theo)))

    return {
        "alpha": float(alpha),
        "xmin": float(xmin),
        "sigma": float(sigma),
        "ks_stat": ks_stat,
        "p_value": np.nan,
        "n_tail": n,
        "n_total": n_total,
        "comparison_exponential": {"R": np.nan, "p": np.nan},
        "comparison_lognormal": {"R": np.nan, "p": np.nan},
        "comparison_truncated_pl": {"R": np.nan, "p": np.nan},
        "method": "mle_fallback",
        "fit_object": None,
    }


def _empty_powerlaw_result(
    reason: str, n_total: int = 0,
) -> Dict[str, Any]:
    """Return NaN result dict with reason."""
    return {
        "alpha": np.nan, "xmin": np.nan, "sigma": np.nan,
        "ks_stat": np.nan, "p_value": np.nan,
        "n_tail": 0, "n_total": n_total,
        "comparison_exponential": {"R": np.nan, "p": np.nan},
        "comparison_lognormal": {"R": np.nan, "p": np.nan},
        "comparison_truncated_pl": {"R": np.nan, "p": np.nan},
        "method": "none", "reason": reason, "fit_object": None,
    }


# =============================================================================
# BRANCHING RATIO
# =============================================================================

def compute_branching_ratio(
    timeseries: np.ndarray,
    threshold_sd: float = 1.0,
    method: str = "both",
    mr_max_lag: int = 10,
    zscore: bool = True,
) -> Dict[str, Any]:
    """
    Compute the branching ratio sigma from binarized BOLD timeseries.

    Two estimators are provided:

    1. **Conventional** (Harris 1963):
       sigma_hat = <n(t+1) / n(t)> for all t where n(t) > 0.
       At criticality sigma = 1. Simple but biased downward by
       subsampling (Priesemann et al. 2014; Levina & Priesemann 2017).

    2. **Multistep Regression (MR)** (Wilting & Priesemann 2018):
       Fit r_k = b * m^k to the autocorrelation function at multiple
       lags k. The estimated m is invariant to spatial subsampling,
       providing a more accurate estimate of the branching parameter.
       Brain typically shows m ~ 0.98-0.999 (slightly subcritical
       "reverberating regime").

    Parameters
    ----------
    timeseries : np.ndarray, shape (T, N)
    threshold_sd : float
        Z-score threshold. Default: 1.0.
    method : str
        'conventional', 'mr', or 'both' (default).
    mr_max_lag : int
        Maximum lag for MR estimator autocorrelation fit. Default: 10.
    zscore : bool
        Whether to z-score timeseries before binarization.

    Returns
    -------
    dict
        'sigma_conventional': float -- naive estimator.
        'sigma_conventional_std': float
        'sigma_conventional_median': float
        'sigma_mr': float -- MR estimator (if computed).
        'sigma_mr_b': float -- MR intercept.
        'sigma_mr_r_squared': float -- quality of exponential fit.
        'n_transitions': int -- number of t->t+1 transitions used.
        'method': str
    """
    ts = np.asarray(timeseries, dtype=np.float64)
    if ts.ndim != 2:
        raise ValueError(f"Expected (T, N), got shape {ts.shape}")
    T, N = ts.shape

    # Z-score and binarize
    if zscore:
        z = (ts - ts.mean(axis=0, keepdims=True)) / np.maximum(
            ts.std(axis=0, keepdims=True, ddof=0), 1e-12
        )
    else:
        z = ts.copy()

    activity = (np.abs(z) > threshold_sd).sum(axis=1).astype(float)

    result = {"n_transitions": 0, "method": method}

    # --- Conventional estimator ---
    if method in ("conventional", "both"):
        ratios = []
        for t in range(T - 1):
            if activity[t] > 0:
                ratios.append(activity[t + 1] / activity[t])

        ratios = np.array(ratios) if ratios else np.array([np.nan])
        result["sigma_conventional"] = float(np.nanmean(ratios))
        result["sigma_conventional_std"] = float(np.nanstd(ratios))
        result["sigma_conventional_median"] = float(np.nanmedian(ratios))
        result["n_transitions"] = int(np.sum(np.isfinite(ratios)))

    # --- MR estimator (Wilting & Priesemann 2018) ---
    if method in ("mr", "both"):
        a_centered = activity - np.mean(activity)
        var_a = np.var(activity)

        if var_a > 0:
            lags = np.arange(1, min(mr_max_lag + 1, T // 3))
            r_k = np.zeros(len(lags))

            for i, lag in enumerate(lags):
                r_k[i] = np.mean(
                    a_centered[:T - lag] * a_centered[lag:]
                ) / var_a

            # Fit exponential: r_k = b * m^k
            # -> log(r_k) = log(b) + k*log(m)
            pos_mask = r_k > 0
            if np.sum(pos_mask) >= 3:
                log_r = np.log(r_k[pos_mask])
                lags_pos = lags[pos_mask].astype(float)

                slope, intercept, r_value, _, _ = stats.linregress(
                    lags_pos, log_r
                )
                m_hat = np.exp(slope)
                b_hat = np.exp(intercept)
                r_sq = r_value ** 2

                result["sigma_mr"] = float(m_hat)
                result["sigma_mr_b"] = float(b_hat)
                result["sigma_mr_r_squared"] = float(r_sq)
                result["mr_lags"] = lags
                result["mr_autocorrelation"] = r_k
            else:
                result["sigma_mr"] = np.nan
                result["sigma_mr_b"] = np.nan
                result["sigma_mr_r_squared"] = np.nan
        else:
            result["sigma_mr"] = np.nan
            result["sigma_mr_b"] = np.nan
            result["sigma_mr_r_squared"] = np.nan

    return result


# =============================================================================
# CRACKLING NOISE SCALING RELATION
# =============================================================================

def compute_scaling_relation(
    sizes: np.ndarray,
    durations: np.ndarray,
    tau_s: Optional[float] = None,
    tau_t: Optional[float] = None,
    min_count_per_duration: int = 2,
) -> Dict[str, Any]:
    """
    Compute the crackling-noise scaling relation between size and
    duration exponents.

    Theory predicts (Friedman et al. 2012; Sethna et al. 2001):

        gamma_predicted = (tau_t - 1) / (tau_s - 1)

    and empirically from the data:

        <S>(T) ~ T^{gamma_empirical}

    Consistency between the two (gamma_ratio ~ 1) is a necessary
    condition for criticality, beyond power-law distributions alone.

    For mean-field branching process: gamma = 2.
    Recent cortical data: gamma ~ 1.2-1.3 (Fontenele et al. 2019).

    Parameters
    ----------
    sizes, durations : np.ndarray
        Avalanche sizes and durations (same length).
    tau_s : float, optional
        Power-law exponent for size distribution. If None, only
        empirical gamma is computed.
    tau_t : float, optional
        Power-law exponent for duration distribution.
    min_count_per_duration : int
        Minimum number of avalanches at a given duration to include
        that duration in the empirical regression. Default: 2.

    Returns
    -------
    dict
        'gamma_predicted' : float -- (tau_t - 1) / (tau_s - 1).
        'gamma_empirical' : float -- from <S>(T) vs T regression.
        'gamma_ratio'     : float -- empirical / predicted (~ 1 at crit.).
        'gamma_deviation' : float -- |1 - gamma_ratio|.
        'r_squared'       : float -- quality of log-log fit.
        'durations_used'  : np.ndarray -- durations in the regression.
        'mean_sizes_used' : np.ndarray -- mean sizes at each duration.
    """
    sizes = np.asarray(sizes, dtype=float)
    durations = np.asarray(durations, dtype=float)

    if len(sizes) != len(durations):
        raise ValueError("sizes and durations must have the same length")

    # --- Predicted gamma from exponents ---
    gamma_predicted = np.nan
    if tau_s is not None and tau_t is not None:
        denom = tau_s - 1.0
        if abs(denom) > 1e-10:
            gamma_predicted = (tau_t - 1.0) / denom

    # --- Empirical gamma: <S>(T) vs T ---
    gamma_empirical = np.nan
    r_squared = np.nan
    durations_used = np.array([])
    mean_sizes_used = np.array([])

    unique_durations = np.unique(durations)
    if len(unique_durations) >= 3:
        dur_vals = []
        mean_s_vals = []
        for d in unique_durations:
            mask = durations == d
            if np.sum(mask) >= min_count_per_duration:
                dur_vals.append(d)
                mean_s_vals.append(np.mean(sizes[mask]))

        if len(dur_vals) >= 3:
            dur_vals = np.array(dur_vals)
            mean_s_vals = np.array(mean_s_vals)
            pos_mask = (dur_vals > 0) & (mean_s_vals > 0)
            if np.sum(pos_mask) >= 3:
                log_d = np.log(dur_vals[pos_mask])
                log_s = np.log(mean_s_vals[pos_mask])
                slope, _, r_value, _, _ = stats.linregress(log_d, log_s)
                gamma_empirical = float(slope)
                r_squared = float(r_value ** 2)
                durations_used = dur_vals[pos_mask]
                mean_sizes_used = mean_s_vals[pos_mask]

    # --- Consistency ratio ---
    gamma_ratio = np.nan
    gamma_deviation = np.nan
    if np.isfinite(gamma_predicted) and np.isfinite(gamma_empirical):
        if abs(gamma_predicted) > 1e-10:
            gamma_ratio = gamma_empirical / gamma_predicted
            gamma_deviation = abs(1.0 - gamma_ratio)

    return {
        "gamma_predicted": float(gamma_predicted),
        "gamma_empirical": float(gamma_empirical),
        "gamma_ratio": float(gamma_ratio),
        "gamma_deviation": float(gamma_deviation),
        "r_squared": float(r_squared),
        "durations_used": durations_used,
        "mean_sizes_used": mean_sizes_used,
    }


# =============================================================================
# KAPPA STATISTIC (Shew et al. 2009)
# =============================================================================

def compute_kappa(
    sizes: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute kappa -- a deviation-from-criticality index based on the
    empirical vs theoretical cumulative distribution of avalanche sizes.

    Following Shew et al. (2009, J Neurosci):
        kappa = 1 + (1/n_bins) * sum_i (F_empirical(s_i) - F_PL(s_i))

    kappa = 1 at criticality.
    kappa > 1 -> supercritical (too many large avalanches).
    kappa < 1 -> subcritical (too few large avalanches).

    Parameters
    ----------
    sizes : np.ndarray
        Avalanche sizes.
    n_bins : int
        Number of logarithmically spaced comparison points. Default: 10.

    Returns
    -------
    dict with 'kappa': float
    """
    sizes = np.asarray(sizes, dtype=float)
    sizes = sizes[sizes > 0]

    if len(sizes) < 10:
        return {"kappa": np.nan}

    pl_fit = fit_powerlaw_distribution(
        sizes, n_bootstrap_gof=0, compare_distributions=False
    )
    alpha = pl_fit.get("alpha", np.nan)
    xmin = pl_fit.get("xmin", 1.0)

    if not np.isfinite(alpha) or alpha <= 1:
        return {"kappa": np.nan}

    s_min = max(sizes.min(), xmin)
    s_max = sizes.max()
    if s_min >= s_max:
        return {"kappa": np.nan}

    bin_edges = np.logspace(np.log10(s_min), np.log10(s_max), n_bins)

    f_emp = np.array([np.mean(sizes <= b) for b in bin_edges])
    f_theo = np.array([
        1.0 - (b / xmin) ** (-(alpha - 1)) if b >= xmin else 0.0
        for b in bin_edges
    ])
    f_theo = np.clip(f_theo, 0, 1)

    kappa = 1.0 + np.mean(f_emp - f_theo)

    return {"kappa": float(kappa)}


# =============================================================================
# AVALANCHE SHAPE ANALYSIS & COLLAPSE
# =============================================================================

def compute_avalanche_shapes(
    temporal_profiles: List[np.ndarray],
    durations: np.ndarray,
    gamma: Optional[float] = None,
    min_avalanches_per_duration: int = 3,
    duration_range: Optional[Tuple[int, int]] = None,
    n_resample_points: int = 50,
) -> Dict[str, Any]:
    """
    Compute average avalanche shapes and test shape collapse.

    At criticality, the temporal profile of avalanches follows a
    universal scaling function. When rescaled:

        T^{-(gamma-1)} * <S(t|T)>  vs  t/T

    all avalanches of different duration T collapse onto a single
    universal curve F(t/T).

    Shape collapse is a more stringent test of criticality than
    power-law distributions alone (Friedman et al. 2012).

    Parameters
    ----------
    temporal_profiles : list of np.ndarray
        Activity profile of each avalanche.
    durations : np.ndarray
        Duration of each avalanche.
    gamma : float, optional
        Scaling exponent for collapse. If None, uses gamma=2
        (mean-field prediction). Pass empirical gamma from
        compute_scaling_relation for data-driven collapse.
    min_avalanches_per_duration : int
        Minimum avalanches per duration to compute mean shape. Default: 3.
    duration_range : tuple, optional
        (min_dur, max_dur).
    n_resample_points : int
        Points in the rescaled time axis. Default: 50.

    Returns
    -------
    dict
        'shapes_by_duration'  : dict {dur: list of profile arrays}
        'mean_shapes'         : dict {dur: mean_profile}
        'collapsed_shapes'    : dict {dur: dict with 't_rescaled',
                                 'amplitude_rescaled'}
        'collapse_quality'    : float -- coefficient of variation across
                                 collapsed shapes (lower = better).
        'gamma_used'          : float
    """
    if gamma is None:
        gamma = 2.0

    durations = np.asarray(durations, dtype=int)

    shapes_by_dur = {}
    for profile, dur in zip(temporal_profiles, durations):
        if duration_range is not None:
            if dur < duration_range[0] or dur > duration_range[1]:
                continue
        shapes_by_dur.setdefault(int(dur), []).append(profile)

    mean_shapes = {}
    collapsed = {}
    collapse_curves = []

    t_common = np.linspace(0, 1, n_resample_points)

    for dur, profiles in shapes_by_dur.items():
        if len(profiles) < min_avalanches_per_duration:
            continue
        arr = np.array(profiles)
        mean_profile = arr.mean(axis=0)
        mean_shapes[dur] = mean_profile

        t_original = np.linspace(0, 1, dur)
        interp_profile = np.interp(t_common, t_original, mean_profile)

        # Rescale amplitude: T^{-(gamma-1)} * <S(t|T)>
        amp_rescaled = interp_profile / (dur ** (gamma - 1))

        collapsed[dur] = {
            "t_rescaled": t_common,
            "amplitude_rescaled": amp_rescaled,
        }
        collapse_curves.append(amp_rescaled)

    # Collapse quality: coefficient of variation
    collapse_quality = np.nan
    if len(collapse_curves) >= 2:
        curves_matrix = np.array(collapse_curves)
        point_std = np.std(curves_matrix, axis=0)
        point_mean = np.mean(curves_matrix, axis=0)
        safe_mean = np.where(
            np.abs(point_mean) > 1e-12, point_mean, 1e-12
        )
        cv_per_point = point_std / np.abs(safe_mean)
        collapse_quality = float(np.mean(cv_per_point))

    return {
        "shapes_by_duration": shapes_by_dur,
        "mean_shapes": mean_shapes,
        "collapsed_shapes": collapsed,
        "collapse_quality": collapse_quality,
        "n_durations_with_shapes": len(mean_shapes),
        "gamma_used": gamma,
    }


# =============================================================================
# CRITICALITY INDEX (COMPOSITE)
# =============================================================================

def compute_criticality_index(
    branching_ratio: float,
    tau_size: float,
    tau_duration: float,
    gamma_deviation: Optional[float] = None,
    dfa_alpha: Optional[float] = None,
    kappa: Optional[float] = None,
    method: str = "weighted",
) -> Dict[str, Any]:
    """
    Compute a composite criticality index from multiple signatures.

    Two methods are available:

    1. **Weighted** (default): Each component scored as
       1 - |observed - theoretical| / theoretical, clipped to [0, 1],
       combined by weighted average. Higher weights for the most
       diagnostic measures (branching ratio, crackling relation).

    2. **Euclidean**: Distance from the theoretical critical point in a
       normalized multi-dimensional space. Smaller distance = closer.

    Theoretical critical values:
        sigma ~ 1.0, tau_s ~ 1.5, tau_t ~ 2.0,
        gamma_dev ~ 0, DFA alpha ~ 1.0, kappa ~ 1.0

    Parameters
    ----------
    branching_ratio, tau_size, tau_duration : float
    gamma_deviation, dfa_alpha, kappa : float, optional
    method : str -- 'weighted' (default) or 'euclidean'.

    Returns
    -------
    dict with 'criticality_index', 'component_scores', 'method'.
    """
    components = {
        "branching_ratio": {
            "value": branching_ratio, "target": 1.0, "weight": 2.5
        },
        "tau_size": {
            "value": tau_size, "target": 1.5, "weight": 1.5
        },
        "tau_duration": {
            "value": tau_duration, "target": 2.0, "weight": 1.5
        },
    }
    if gamma_deviation is not None:
        components["gamma_deviation"] = {
            "value": gamma_deviation, "target": 0.0, "weight": 2.0
        }
    if dfa_alpha is not None:
        components["dfa_alpha"] = {
            "value": dfa_alpha, "target": 1.0, "weight": 1.0
        }
    if kappa is not None:
        components["kappa"] = {
            "value": kappa, "target": 1.0, "weight": 1.5
        }

    scores = {}

    if method == "weighted":
        total_weight = 0
        weighted_sum = 0
        for name, comp in components.items():
            val = comp["value"]
            target = comp["target"]
            weight = comp["weight"]
            if not np.isfinite(val):
                scores[name] = np.nan
                continue
            if target == 0:
                s = max(0.0, 1.0 - abs(val))
            else:
                s = max(0.0, 1.0 - abs(val - target) / abs(target))
            scores[name] = float(s)
            weighted_sum += s * weight
            total_weight += weight
        ci = (
            float(weighted_sum / total_weight) if total_weight > 0
            else np.nan
        )

    elif method == "euclidean":
        diffs_sq = []
        for name, comp in components.items():
            val = comp["value"]
            target = comp["target"]
            if not np.isfinite(val):
                scores[name] = np.nan
                continue
            norm = abs(target) if abs(target) > 1e-10 else 1.0
            diff = (val - target) / norm
            scores[name] = float(diff)
            diffs_sq.append(diff ** 2)
        ci = float(np.sqrt(np.sum(diffs_sq))) if diffs_sq else np.nan

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "criticality_index": ci,
        "component_scores": scores,
        "method": method,
    }


# =============================================================================
# SURROGATE COMPARISON
# =============================================================================

def compare_with_surrogates(
    timeseries: np.ndarray,
    threshold_sd: float = 1.0,
    n_surrogates: int = 99,
    surrogate_type: str = "phase",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compare avalanche metrics against surrogate (null) timeseries.

    Generate surrogates that preserve certain statistical properties
    but destroy temporal correlations or nonlinear structure, then
    re-detect avalanches and compare distributions.

    At criticality, real data should show power-law avalanche
    distributions that surrogates (which typically follow exponential
    distributions) do not (Tagliazucchi et al. 2012; Xu et al. 2022).

    Parameters
    ----------
    timeseries : np.ndarray (T, N)
    threshold_sd : float
    n_surrogates : int
        Number of surrogates. Default: 99 (for p < 0.01).
        Minimum for significance: 19 (p < 0.05).
    surrogate_type : str
        'phase'   : phase randomization (preserves power spectrum).
        'iaaft'   : IAAFT (preserves spectrum + amplitude distribution).
        'shuffle' : simple time-shuffle per ROI.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        'real_metrics'     : dict -- real data metrics.
        'surrogate_metrics': dict -- arrays of surrogate metrics.
        'z_scores'         : dict -- z-score of real vs surrogates.
        'p_values'         : dict -- two-sided p-values.
        'n_surrogates'     : int
        'surrogate_type'   : str
    """
    from ..utils import phase_randomize, amplitude_adjusted_ft_surrogate

    rng = np.random.default_rng(seed)
    ts = np.asarray(timeseries, dtype=np.float64)

    # --- Real data metrics ---
    real_aval = detect_avalanches(ts, threshold_sd=threshold_sd)
    real_br = compute_branching_ratio(
        ts, threshold_sd=threshold_sd, method="conventional"
    )

    real_metrics = {
        "n_avalanches": real_aval["n_avalanches"],
        "mean_size": float(np.mean(real_aval["sizes"]))
            if len(real_aval["sizes"]) > 0 else 0.0,
        "max_size": int(np.max(real_aval["sizes"]))
            if len(real_aval["sizes"]) > 0 else 0,
        "branching": real_br["sigma_conventional"],
    }

    # --- Surrogate metrics ---
    surr_metrics = {k: [] for k in real_metrics}

    for i in range(n_surrogates):
        s = int(rng.integers(0, 2**31))
        if surrogate_type == "phase":
            surr_ts = phase_randomize(ts, seed=s)
        elif surrogate_type == "iaaft":
            surr_ts = amplitude_adjusted_ft_surrogate(ts, seed=s)
        elif surrogate_type == "shuffle":
            surr_ts = ts.copy()
            for col in range(ts.shape[1]):
                rng.shuffle(surr_ts[:, col])
        else:
            raise ValueError(f"Unknown surrogate_type: {surrogate_type}")

        s_aval = detect_avalanches(surr_ts, threshold_sd=threshold_sd)
        s_br = compute_branching_ratio(
            surr_ts, threshold_sd=threshold_sd, method="conventional"
        )

        surr_metrics["n_avalanches"].append(s_aval["n_avalanches"])
        if len(s_aval["sizes"]) > 0:
            surr_metrics["mean_size"].append(
                float(np.mean(s_aval["sizes"]))
            )
            surr_metrics["max_size"].append(
                int(np.max(s_aval["sizes"]))
            )
        else:
            surr_metrics["mean_size"].append(0.0)
            surr_metrics["max_size"].append(0)
        surr_metrics["branching"].append(s_br["sigma_conventional"])

    surr_metrics = {k: np.array(v) for k, v in surr_metrics.items()}

    # --- Z-scores and p-values ---
    z_scores = {}
    p_values = {}
    for key in real_metrics:
        surr_arr = surr_metrics[key]
        surr_mean = np.nanmean(surr_arr)
        surr_std = np.nanstd(surr_arr)
        if surr_std > 1e-12:
            z_scores[key] = float(
                (real_metrics[key] - surr_mean) / surr_std
            )
        else:
            z_scores[key] = np.nan
        n_extreme = np.sum(
            np.abs(surr_arr - surr_mean)
            >= np.abs(real_metrics[key] - surr_mean)
        )
        p_values[key] = float((n_extreme + 1) / (n_surrogates + 1))

    return {
        "real_metrics": real_metrics,
        "surrogate_metrics": surr_metrics,
        "z_scores": z_scores,
        "p_values": p_values,
        "n_surrogates": n_surrogates,
        "surrogate_type": surrogate_type,
    }


# =============================================================================
# GROUP-LEVEL POOLING
# =============================================================================

def pool_avalanches_group(
    timeseries_list: List[np.ndarray],
    threshold_sd: float = 1.0,
    temporal_bin: int = 1,
    subject_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Pool avalanche detections across multiple subjects for group-level
    power-law fitting.

    Individual subjects with ~200 volumes typically yield 30-80
    avalanches -- insufficient for reliable power-law estimation.
    Pooling across N_subjects gives N x more avalanches, reaching the
    > 100 tail observations needed for the CSN framework.

    Parameters
    ----------
    timeseries_list : list of np.ndarray
        Each element is a (T, N) timeseries for one subject.
    threshold_sd : float
    temporal_bin : int
    subject_ids : list of str, optional

    Returns
    -------
    dict with pooled sizes/durations, per-subject results,
    group-level power-law fits, scaling relation, and branching stats.
    """
    if subject_ids is None:
        subject_ids = [
            f"sub_{i:03d}" for i in range(len(timeseries_list))
        ]

    all_sizes = []
    all_durations = []
    per_subject = []
    all_br_conv = []
    all_br_mr = []

    for i, ts in enumerate(timeseries_list):
        aval = detect_avalanches(
            ts, threshold_sd=threshold_sd, temporal_bin=temporal_bin,
        )
        br = compute_branching_ratio(
            ts, threshold_sd=threshold_sd, method="both"
        )

        per_subject.append({
            "subject_id": subject_ids[i],
            "n_avalanches": aval["n_avalanches"],
            "sizes": aval["sizes"],
            "durations": aval["durations"],
            "branching_conventional": br.get(
                "sigma_conventional", np.nan
            ),
            "branching_mr": br.get("sigma_mr", np.nan),
        })

        all_sizes.append(aval["sizes"])
        all_durations.append(aval["durations"])
        all_br_conv.append(br.get("sigma_conventional", np.nan))
        all_br_mr.append(br.get("sigma_mr", np.nan))

    pooled_sizes = (
        np.concatenate(all_sizes) if all_sizes else np.array([])
    )
    pooled_durations = (
        np.concatenate(all_durations) if all_durations else np.array([])
    )

    result = {
        "pooled_sizes": pooled_sizes,
        "pooled_durations": pooled_durations,
        "n_total_avalanches": len(pooled_sizes),
        "per_subject": per_subject,
    }

    # Group-level power-law fitting
    if len(pooled_sizes) >= 20:
        result["size_fit"] = fit_powerlaw_distribution(
            pooled_sizes, discrete=True
        )
        result["duration_fit"] = fit_powerlaw_distribution(
            pooled_durations, discrete=True
        )

        tau_s = result["size_fit"]["alpha"]
        tau_t = result["duration_fit"]["alpha"]
        result["scaling_relation"] = compute_scaling_relation(
            pooled_sizes, pooled_durations, tau_s=tau_s, tau_t=tau_t,
        )
        result["kappa"] = compute_kappa(pooled_sizes)
    else:
        result["size_fit"] = _empty_powerlaw_result(
            "insufficient pooled avalanches", len(pooled_sizes)
        )
        result["duration_fit"] = _empty_powerlaw_result(
            "insufficient pooled avalanches", len(pooled_durations)
        )

    # Branching statistics across subjects
    br_conv = np.array(all_br_conv)
    br_mr = np.array(all_br_mr)
    result["branching_stats"] = {
        "conventional_mean": float(np.nanmean(br_conv)),
        "conventional_std": float(np.nanstd(br_conv)),
        "conventional_median": float(np.nanmedian(br_conv)),
        "mr_mean": float(np.nanmean(br_mr)),
        "mr_std": float(np.nanstd(br_mr)),
        "mr_median": float(np.nanmedian(br_mr)),
        "per_subject_conventional": br_conv,
        "per_subject_mr": br_mr,
    }

    return result


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity_analysis(
    timeseries: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    temporal_bins: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run avalanche detection across multiple thresholds and temporal
    binning factors to assess robustness.

    Metrics that vary wildly across reasonable parameter choices may
    not be robust signatures of the underlying dynamics.

    Parameters
    ----------
    timeseries : np.ndarray (T, N)
    thresholds : np.ndarray, optional
        Default: [0.5, 1.0, 1.5, 2.0, 2.5].
    temporal_bins : list of int, optional
        Default: [1, 2].

    Returns
    -------
    dict with 'thresholds', 'temporal_bins', 'results', 'summary'.
    """
    ts = np.asarray(timeseries, dtype=np.float64)

    if thresholds is None:
        thresholds = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    if temporal_bins is None:
        temporal_bins = [1, 2]

    results = {}
    n_avals = []
    sigmas = []

    for th in thresholds:
        for tb in temporal_bins:
            try:
                aval = detect_avalanches(
                    ts, threshold_sd=th, temporal_bin=tb
                )
                br = compute_branching_ratio(
                    ts, threshold_sd=th, method="conventional"
                )

                key = (float(th), int(tb))
                results[key] = {
                    "n_avalanches": aval["n_avalanches"],
                    "mean_size": float(np.mean(aval["sizes"]))
                        if len(aval["sizes"]) > 0 else 0.0,
                    "max_size": int(np.max(aval["sizes"]))
                        if len(aval["sizes"]) > 0 else 0,
                    "mean_duration": float(np.mean(aval["durations"]))
                        if len(aval["durations"]) > 0 else 0.0,
                    "silent_fraction": aval["silent_fraction"],
                    "branching_ratio": br["sigma_conventional"],
                }
                n_avals.append(aval["n_avalanches"])
                sigmas.append(br["sigma_conventional"])
            except Exception as e:
                results[(float(th), int(tb))] = {"error": str(e)}

    summary = {
        "n_avalanches_range": (
            min(n_avals), max(n_avals)
        ) if n_avals else (0, 0),
        "branching_ratio_range": (
            float(np.nanmin(sigmas)), float(np.nanmax(sigmas)),
        ) if sigmas else (np.nan, np.nan),
    }

    return {
        "thresholds": thresholds,
        "temporal_bins": temporal_bins,
        "results": results,
        "summary": summary,
    }


# =============================================================================
# MASTER ANALYSIS FUNCTION
# =============================================================================

def analyze_avalanches(
    timeseries: np.ndarray,
    threshold_sd: float = 1.0,
    min_duration: int = 1,
    temporal_bin: int = 1,
    tr: Optional[float] = None,
    compute_shapes: bool = True,
    dfa_alpha: Optional[float] = None,
    n_bootstrap_gof: int = 1000,
    run_sensitivity: bool = False,
) -> Dict[str, Any]:
    """
    Complete avalanche-based criticality analysis pipeline for a
    single subject.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T, N)
        Parcellated BOLD timeseries.
    threshold_sd : float
        Z-score threshold. Default: 1.0 (recommended for fMRI).
    min_duration : int
        Minimum avalanche duration. Default: 1.
    temporal_bin : int
        Temporal binning factor. Default: 1.
    tr : float, optional
        Repetition time in seconds. If None, imported from config.
    compute_shapes : bool
        Whether to perform avalanche shape analysis. Default: True.
    dfa_alpha : float, optional
        Pre-computed DFA exponent (for criticality index).
    n_bootstrap_gof : int
        Bootstrap iterations for power-law GOF. Default: 1000.
    run_sensitivity : bool
        If True, also run sensitivity analysis. Default: False.

    Returns
    -------
    dict : Comprehensive results including detection statistics,
           power-law fits, branching ratio, scaling relation, kappa,
           criticality index, and optionally shapes + sensitivity.
    """
    try:
        from .. import config
        tr = tr or config.TR
    except (ImportError, AttributeError):
        tr = tr or 2.0

    ts = np.asarray(timeseries, dtype=np.float64)

    # === 1. Detect avalanches ===
    aval = detect_avalanches(
        ts, threshold_sd=threshold_sd,
        min_duration=min_duration,
        temporal_bin=temporal_bin,
    )

    sizes = aval["sizes"]
    durations = aval["durations"]

    results = {
        "n_avalanches": aval["n_avalanches"],
        "threshold_sd": threshold_sd,
        "temporal_bin": temporal_bin,
        "silent_fraction": aval["silent_fraction"],
        "T_effective": aval["T_effective"],
        "N_rois": aval["N_rois"],
    }

    if len(sizes) == 0:
        results["warning"] = "No avalanches detected"
        results["alpha_size"] = np.nan
        results["tau_duration"] = np.nan
        results["branching_ratio"] = np.nan
        results["criticality_index"] = np.nan
        return results

    # === 2. Descriptive statistics ===
    scan_duration = ts.shape[0] * tr / 60.0
    results["avalanche_rate_per_min"] = float(
        aval["n_avalanches"] / scan_duration
    )
    results["mean_size"] = float(np.mean(sizes))
    results["std_size"] = float(np.std(sizes))
    results["max_size"] = int(np.max(sizes))
    results["median_size"] = float(np.median(sizes))
    results["mean_duration"] = float(np.mean(durations))
    results["std_duration"] = float(np.std(durations))
    results["max_duration"] = int(np.max(durations))
    results["median_duration"] = float(np.median(durations))

    # === 3. Power-law fits ===
    if len(sizes) < 50:
        results["warning_sample_size"] = (
            f"Only {len(sizes)} avalanches detected. Individual "
            f"power-law fitting may be unreliable (< 100 tail obs). "
            f"Consider pool_avalanches_group()."
        )

    size_fit = fit_powerlaw_distribution(
        sizes, discrete=True, n_bootstrap_gof=n_bootstrap_gof
    )
    dur_fit = fit_powerlaw_distribution(
        durations, discrete=True, n_bootstrap_gof=n_bootstrap_gof
    )

    results["size_fit"] = size_fit
    results["alpha_size"] = size_fit["alpha"]
    results["alpha_size_sigma"] = size_fit["sigma"]
    results["alpha_size_ks"] = size_fit["ks_stat"]
    results["alpha_size_pvalue"] = size_fit["p_value"]
    results["alpha_size_n_tail"] = size_fit["n_tail"]
    results["alpha_size_method"] = size_fit["method"]

    results["duration_fit"] = dur_fit
    results["tau_duration"] = dur_fit["alpha"]
    results["tau_duration_sigma"] = dur_fit["sigma"]
    results["tau_duration_ks"] = dur_fit["ks_stat"]
    results["tau_duration_pvalue"] = dur_fit["p_value"]

    # === 4. Scaling relation ===
    scaling = compute_scaling_relation(
        sizes, durations,
        tau_s=size_fit["alpha"], tau_t=dur_fit["alpha"],
    )
    results["gamma_predicted"] = scaling["gamma_predicted"]
    results["gamma_empirical"] = scaling["gamma_empirical"]
    results["gamma_ratio"] = scaling["gamma_ratio"]
    results["gamma_deviation"] = scaling["gamma_deviation"]
    results["gamma_r_squared"] = scaling["r_squared"]

    # === 5. Branching ratio ===
    br = compute_branching_ratio(
        ts, threshold_sd=threshold_sd, method="both"
    )
    results["branching_conventional"] = br.get(
        "sigma_conventional", np.nan
    )
    results["branching_conventional_std"] = br.get(
        "sigma_conventional_std", np.nan
    )
    results["branching_mr"] = br.get("sigma_mr", np.nan)
    results["branching_mr_r_squared"] = br.get(
        "sigma_mr_r_squared", np.nan
    )
    results["n_transitions"] = br.get("n_transitions", 0)

    # === 6. Kappa ===
    kappa_res = compute_kappa(sizes)
    results["kappa"] = kappa_res["kappa"]

    # === 7. Criticality index ===
    crit_idx = compute_criticality_index(
        branching_ratio=br.get("sigma_conventional", np.nan),
        tau_size=size_fit["alpha"],
        tau_duration=dur_fit["alpha"],
        gamma_deviation=scaling["gamma_deviation"],
        dfa_alpha=dfa_alpha,
        kappa=kappa_res["kappa"],
    )
    results["criticality_index"] = crit_idx["criticality_index"]
    results["criticality_components"] = crit_idx["component_scores"]

    # === 8. Avalanche shapes ===
    if compute_shapes and len(aval["temporal_profiles"]) > 0:
        shapes = compute_avalanche_shapes(
            aval["temporal_profiles"],
            durations,
            gamma=(
                scaling["gamma_empirical"]
                if np.isfinite(scaling["gamma_empirical"])
                else None
            ),
        )
        results["shape_collapse_quality"] = shapes["collapse_quality"]
        results["n_durations_with_shapes"] = shapes[
            "n_durations_with_shapes"
        ]

    # === 9. Sensitivity analysis (optional) ===
    if run_sensitivity:
        results["sensitivity"] = sensitivity_analysis(ts)

    # === 10. Store raw data ===
    results["sizes"] = sizes
    results["durations"] = durations
    results["avalanche_indices"] = aval["avalanche_indices"]
    results["temporal_profiles"] = aval["temporal_profiles"]

    return results
