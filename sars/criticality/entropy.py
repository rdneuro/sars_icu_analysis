# -*- coding: utf-8 -*-
"""
sars.criticality.entropy
=======================================

Information-theoretic complexity measures for BOLD timeseries.

At the critical point, the brain maximizes information capacity and
shows characteristic entropy signatures:
  - Elevated sample entropy (temporal unpredictability)
  - High permutation entropy (complexity of ordinal patterns)
  - Intermediate Lempel-Ziv complexity (between random and periodic)
  - Scale-free multi-scale entropy profiles

Measures
--------
- Sample Entropy (SampEn): Richman & Moorman (2000)
- Permutation Entropy (PermEn): Bandt & Pompe (2002)
- Lempel-Ziv Complexity (LZC): Lempel & Ziv (1976)
- Multi-Scale Entropy (MSE): Costa et al. (2005)
- Spectral Entropy: Inouye et al. (1991)

References
----------
- Richman & Moorman (2000). Am J Physiol. Sample entropy.
- Bandt & Pompe (2002). Phys Rev Lett. Permutation entropy.
- Lempel & Ziv (1976). IEEE Trans Inf Theory. Compression complexity.
- Costa, Goldberger & Peng (2005). Phys Rev E. Multi-scale entropy.
- Inouye et al. (1991). Electroencephalogr Clin Neurophysiol.
- Nezafati, Temmar & Bhatt (2020). HBM. Functional MRI signal complexity.

Usage
-----
    from sars.criticality.entropy import analyze_entropy

    results = analyze_entropy(timeseries)
"""

import numpy as np
from scipy import signal as sig
from typing import Dict, Optional, List, Any
from itertools import permutations
import warnings


# =============================================================================
# SAMPLE ENTROPY
# =============================================================================

def sample_entropy(
    x: np.ndarray,
    m: int = 2,
    r: float = 0.2,
    normalize_r: bool = True,
) -> float:
    """
    Compute Sample Entropy (SampEn) of a time series.

    SampEn quantifies the regularity/predictability of a signal by
    measuring the conditional probability that patterns similar for
    m points remain similar at m+1 points.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Time series.
    m : int
        Embedding dimension (pattern length). Default: 2.
    r : float
        Tolerance threshold. If normalize_r is True, this is a
        fraction of the signal's standard deviation.
    normalize_r : bool
        If True, r_eff = r * std(x).

    Returns
    -------
    float : SampEn value. Higher → more complex/irregular.
            0 → perfectly regular. Inf → insufficient data.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)

    if N < m + 2:
        return np.nan

    if normalize_r:
        r_eff = r * np.std(x)
    else:
        r_eff = r

    if r_eff <= 0:
        return np.nan

    def _count_matches(dim):
        """Count template matches at given embedding dimension."""
        templates = np.array([x[i:i + dim] for i in range(N - dim)])
        n_templates = len(templates)
        count = 0
        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                if np.max(np.abs(templates[i] - templates[j])) <= r_eff:
                    count += 1
        return count

    # Use vectorized Chebyshev distance for efficiency
    def _count_matches_fast(dim):
        n_temp = N - dim
        if n_temp < 2:
            return 0
        templates = np.array([x[i:i + dim] for i in range(n_temp)])
        count = 0
        for i in range(n_temp - 1):
            diffs = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += np.sum(diffs <= r_eff)
        return count

    B = _count_matches_fast(m)      # matches of length m
    A = _count_matches_fast(m + 1)  # matches of length m+1

    if B == 0:
        return np.nan

    return -np.log(A / B) if A > 0 else np.inf


# =============================================================================
# PERMUTATION ENTROPY
# =============================================================================

def permutation_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """
    Compute Permutation Entropy (PermEn) of a time series.

    PermEn captures the complexity of ordinal patterns (rank permutations)
    in the signal. Robust to noise and amplitude scaling.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
    order : int
        Permutation order (embedding dimension). Default: 3.
        Typical: 3-7 for fMRI (limited by T).
    delay : int
        Time delay between elements of each ordinal pattern.
    normalize : bool
        If True, normalize by log(order!), yielding values in [0, 1].

    Returns
    -------
    float : PermEn. 0 → perfectly regular, 1 → maximally complex.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    n_patterns = N - (order - 1) * delay

    if n_patterns < 1:
        return np.nan

    # Extract ordinal patterns
    pattern_counts = {}
    for i in range(n_patterns):
        indices = [i + j * delay for j in range(order)]
        pattern = tuple(np.argsort(x[indices]))
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Compute Shannon entropy of pattern distribution
    counts = np.array(list(pattern_counts.values()), dtype=float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))

    if normalize:
        max_entropy = np.log2(np.math.factorial(order))
        return entropy / max_entropy if max_entropy > 0 else np.nan

    return entropy


# =============================================================================
# LEMPEL-ZIV COMPLEXITY
# =============================================================================

def lempel_ziv_complexity(
    x: np.ndarray,
    threshold: Optional[str] = "median",
    normalize: bool = True,
) -> float:
    """
    Compute Lempel-Ziv Complexity (LZC) of a time series.

    LZC measures the number of distinct substrings in a binary sequence,
    quantifying the algorithmic complexity of the signal.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
    threshold : str or float
        Method to binarize the signal:
        'median' : threshold at the median.
        'mean'   : threshold at the mean.
        float    : explicit threshold value.
    normalize : bool
        If True, normalize by n/log2(n) — the expected complexity for
        a random binary sequence.

    Returns
    -------
    float : LZC. Higher → more complex.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)

    if N < 2:
        return np.nan

    # Binarize
    if isinstance(threshold, str):
        if threshold == "median":
            thresh_val = np.median(x)
        elif threshold == "mean":
            thresh_val = np.mean(x)
        else:
            raise ValueError(f"Unknown threshold method: {threshold}")
    else:
        thresh_val = float(threshold)

    binary = (x > thresh_val).astype(int)

    # Lempel-Ziv76 algorithm
    s = "".join(map(str, binary))
    n = len(s)
    i = 0
    c = 1  # complexity counter
    l = 1  # current substring length

    while i + l <= n:
        # Check if s[i:i+l] appears in s[0:i+l-1]
        substring = s[i:i + l]
        search_in = s[0:i + l - 1]

        if substring in search_in:
            l += 1
        else:
            c += 1
            i += l
            l = 1

    if normalize:
        # Normalize by expected complexity of random sequence
        expected = n / np.log2(n) if n > 1 else 1
        return c / expected
    return c


# =============================================================================
# MULTI-SCALE ENTROPY
# =============================================================================

def multiscale_entropy(
    x: np.ndarray,
    max_scale: int = 20,
    m: int = 2,
    r: float = 0.2,
) -> Dict[str, Any]:
    """
    Compute Multi-Scale Entropy (MSE) profile.

    MSE applies Sample Entropy to coarse-grained versions of the signal
    at multiple temporal scales, revealing complexity across scales.

    At criticality, MSE profiles remain high across scales (scale-free).
    Diseased/altered states often show specific scale-dependent changes.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
    max_scale : int
        Maximum coarse-graining scale.
    m : int
        SampEn embedding dimension.
    r : float
        SampEn tolerance (as fraction of std).

    Returns
    -------
    dict with 'scales', 'entropy', 'ci' (complexity index = area under MSE).
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)

    max_scale = min(max_scale, N // (m + 2))
    scales = np.arange(1, max_scale + 1)

    entropies = np.full(len(scales), np.nan)

    for i, scale in enumerate(scales):
        # Coarse-grain: average consecutive non-overlapping windows
        n_samples = N // scale
        if n_samples < m + 2:
            break
        coarse = np.mean(
            x[:n_samples * scale].reshape(n_samples, scale), axis=1
        )
        entropies[i] = sample_entropy(coarse, m=m, r=r, normalize_r=True)

    # Complexity index: area under the MSE curve
    valid = np.isfinite(entropies)
    ci = float(np.trapz(entropies[valid], scales[valid])) if np.sum(valid) > 1 else np.nan

    return {
        "scales": scales,
        "entropy": entropies,
        "complexity_index": ci,
        "m": m,
        "r": r,
    }


# =============================================================================
# SPECTRAL ENTROPY
# =============================================================================

def spectral_entropy(
    x: np.ndarray,
    fs: Optional[float] = None,
    method: str = "welch",
    normalize: bool = True,
    nperseg: Optional[int] = None,
    freq_band: Optional[tuple] = None,
) -> float:
    """
    Compute Spectral Entropy of a signal.

    Spectral entropy measures the flatness of the power spectrum.
    A flat spectrum (white noise) → maximum spectral entropy.
    A single-frequency signal → minimum spectral entropy.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
    fs : float, optional
        Sampling frequency (1/TR for fMRI).
    method : str
        'welch' or 'fft'.
    normalize : bool
        If True, normalize to [0, 1].
    nperseg : int, optional
        Segment length for Welch.
    freq_band : tuple, optional
        (low, high) Hz. Only compute entropy within this band.

    Returns
    -------
    float : Spectral entropy.
    """
    x = np.asarray(x, dtype=float).ravel()

    if fs is None:
        from .. import config
        fs = 1.0 / config.TR

    if method == "welch":
        if nperseg is None:
            nperseg = min(len(x), max(64, len(x) // 4))
        freqs, psd = sig.welch(x, fs=fs, nperseg=nperseg)
    elif method == "fft":
        freqs, psd = sig.periodogram(x, fs=fs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply frequency band filter
    if freq_band is not None:
        mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        psd = psd[mask]
        freqs = freqs[mask]

    # Normalize PSD to probability distribution
    psd = psd[psd > 0]
    if len(psd) == 0:
        return np.nan

    psd_norm = psd / psd.sum()

    # Shannon entropy
    se = -np.sum(psd_norm * np.log2(psd_norm))

    if normalize:
        se /= np.log2(len(psd_norm))

    return float(se)


# =============================================================================
# MASTER ANALYSIS FUNCTION
# =============================================================================

def analyze_entropy(
    timeseries: np.ndarray,
    tr: Optional[float] = None,
    sampen_m: int = 2,
    sampen_r: float = 0.2,
    permen_order: int = 3,
    mse_max_scale: int = 20,
) -> Dict[str, Any]:
    """
    Complete entropy-based complexity analysis pipeline.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T, N)
    tr : float, optional
    sampen_m : int
    sampen_r : float
    permen_order : int
    mse_max_scale : int

    Returns
    -------
    dict with per-ROI and global entropy measures.
    """
    from .. import config
    tr = tr or config.TR

    ts = np.asarray(timeseries, dtype=float)
    T, N = ts.shape
    fs = 1.0 / tr

    results = {"n_timepoints": T, "n_rois": N}

    # Per-ROI measures
    sampen_vals = np.full(N, np.nan)
    permen_vals = np.full(N, np.nan)
    lzc_vals = np.full(N, np.nan)
    specen_vals = np.full(N, np.nan)

    for roi in range(N):
        x = ts[:, roi]
        sampen_vals[roi] = sample_entropy(x, m=sampen_m, r=sampen_r)
        permen_vals[roi] = permutation_entropy(x, order=permen_order)
        lzc_vals[roi] = lempel_ziv_complexity(x)
        specen_vals[roi] = spectral_entropy(x, fs=fs)

    results["sample_entropy_per_roi"] = sampen_vals
    results["sample_entropy_mean"] = float(np.nanmean(sampen_vals))
    results["sample_entropy_std"] = float(np.nanstd(sampen_vals))

    results["permutation_entropy_per_roi"] = permen_vals
    results["permutation_entropy_mean"] = float(np.nanmean(permen_vals))
    results["permutation_entropy_std"] = float(np.nanstd(permen_vals))

    results["lzc_per_roi"] = lzc_vals
    results["lzc_mean"] = float(np.nanmean(lzc_vals))
    results["lzc_std"] = float(np.nanstd(lzc_vals))

    results["spectral_entropy_per_roi"] = specen_vals
    results["spectral_entropy_mean"] = float(np.nanmean(specen_vals))
    results["spectral_entropy_std"] = float(np.nanstd(specen_vals))

    # Global MSE on mean signal
    mean_signal = ts.mean(axis=1)
    mse_result = multiscale_entropy(mean_signal, max_scale=mse_max_scale,
                                     m=sampen_m, r=sampen_r)
    results["mse_scales"] = mse_result["scales"]
    results["mse_entropy"] = mse_result["entropy"]
    results["mse_complexity_index"] = mse_result["complexity_index"]

    # Global spectral entropy
    results["global_spectral_entropy"] = spectral_entropy(mean_signal, fs=fs)

    return results
