# -*- coding: utf-8 -*-
"""
sars.criticality.point_process
=============================================

Point-process analysis of BOLD timeseries following the framework of
Tagliazucchi et al. (2012, 2016).

Theory
------
The continuous BOLD signal is reduced to discrete events at timepoints
where the signal crosses a threshold (typically ±1 SD), capturing
"significant" activations. This yields a binary raster of spatiotemporal
events that preserves the essential dynamics while enabling analysis of:

  - Spatiotemporal clustering of events (functional connectivity from
    co-occurrences)
  - Order parameter: global synchrony (fraction of simultaneously active
    ROIs)
  - Control parameter: temporal density of events
  - Phase transition detection: divergence of susceptibility and
    correlation length near criticality
  - Critical slowing down: increase of autocorrelation time

This approach bridges the gap between continuous fMRI and discrete
neural avalanche frameworks.

References
----------
- Tagliazucchi et al. (2012). Front Physiol 3:15. Point process in
  large-scale brain dynamics.
- Tagliazucchi et al. (2016). HBM 37:4487. Voxel-wise point process.
- Petridou et al. (2013). NeuroImage 82:3.
- Cifre et al. (2020). NeuroImage 209:116518.

Usage
-----
    from sars.criticality.point_process import analyze_point_process

    results = analyze_point_process(timeseries, threshold_sd=1.0)
"""

import numpy as np
from scipy import stats, signal
from typing import Dict, Optional, Tuple, List, Any


# =============================================================================
# POINT PROCESS EXTRACTION
# =============================================================================

def extract_point_process(
    timeseries: np.ndarray,
    threshold_sd: float = 1.0,
    mode: str = "crossing",
) -> Dict[str, Any]:
    """
    Convert continuous BOLD timeseries to a discrete point process.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T, N)
        Z-scored parcellated BOLD timeseries.
    threshold_sd : float
        Threshold in standard deviations. Events are detected at
        timepoints where |z_i(t)| > threshold_sd.
    mode : str
        'crossing' : Events at positive threshold crossings (upward).
        'exceedance': All timepoints exceeding threshold.
        'peak'     : Local maxima exceeding threshold.

    Returns
    -------
    dict with keys:
        'events'          : np.ndarray (T, N) — binary event matrix.
        'event_times'     : list of arrays — event timepoints per ROI.
        'event_rate'      : np.ndarray (N,) — events per timepoint per ROI.
        'global_rate'     : np.ndarray (T,) — fraction of active ROIs per t.
        'n_events_total'  : int
        'threshold_sd'    : float
    """
    ts = np.asarray(timeseries, dtype=float)
    T, N = ts.shape

    # Z-score per ROI
    means = ts.mean(axis=0, keepdims=True)
    stds = ts.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    z = (ts - means) / stds

    if mode == "exceedance":
        events = (np.abs(z) > threshold_sd).astype(int)

    elif mode == "crossing":
        # Detect upward threshold crossings
        above = (z > threshold_sd).astype(int)
        events = np.zeros_like(above)
        events[1:, :] = np.diff(above, axis=0)
        events = (events == 1).astype(int)  # only onset frames

    elif mode == "peak":
        events = np.zeros((T, N), dtype=int)
        for roi in range(N):
            # Find local maxima
            peaks, _ = signal.find_peaks(z[:, roi], height=threshold_sd)
            events[peaks, roi] = 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Statistics
    event_times = [np.where(events[:, roi])[0] for roi in range(N)]
    event_rate = events.sum(axis=0) / T
    global_rate = events.sum(axis=1) / N

    return {
        "events": events,
        "event_times": event_times,
        "event_rate": event_rate,
        "global_rate": global_rate,
        "n_events_total": int(events.sum()),
        "threshold_sd": threshold_sd,
        "mode": mode,
    }


# =============================================================================
# POINT-PROCESS FUNCTIONAL CONNECTIVITY
# =============================================================================

def point_process_fc(
    events: np.ndarray,
    method: str = "jaccard",
) -> np.ndarray:
    """
    Compute functional connectivity from point-process co-activations.

    Parameters
    ----------
    events : np.ndarray, shape (T, N)
        Binary event matrix.
    method : str
        'jaccard' : Jaccard similarity between ROI event sets.
        'correlation': Pearson correlation of binary event series.
        'cosine'  : Cosine similarity of event vectors.
        'coincidence': Fraction of co-active timepoints.

    Returns
    -------
    np.ndarray (N, N) : Point-process FC matrix.
    """
    T, N = events.shape

    if method == "correlation":
        fc = np.corrcoef(events.T)
        np.fill_diagonal(fc, 0)
        return fc

    fc = np.zeros((N, N))

    if method == "jaccard":
        for i in range(N):
            for j in range(i + 1, N):
                intersection = np.sum(events[:, i] & events[:, j])
                union = np.sum(events[:, i] | events[:, j])
                fc[i, j] = fc[j, i] = intersection / union if union > 0 else 0

    elif method == "cosine":
        norms = np.linalg.norm(events, axis=0)
        norms[norms == 0] = 1
        normed = events / norms[np.newaxis, :]
        fc = normed.T @ normed
        np.fill_diagonal(fc, 0)

    elif method == "coincidence":
        for i in range(N):
            for j in range(i + 1, N):
                n_i = events[:, i].sum()
                n_j = events[:, j].sum()
                n_both = np.sum(events[:, i] & events[:, j])
                denom = min(n_i, n_j) if min(n_i, n_j) > 0 else 1
                fc[i, j] = fc[j, i] = n_both / denom
    else:
        raise ValueError(f"Unknown FC method: {method}")

    return fc


# =============================================================================
# SPATIOTEMPORAL CLUSTERING
# =============================================================================

def spatiotemporal_clustering(
    events: np.ndarray,
    adjacency: Optional[np.ndarray] = None,
    temporal_window: int = 1,
) -> Dict[str, Any]:
    """
    Cluster point-process events into spatiotemporal cascades.

    Events occurring within `temporal_window` timepoints and in
    connected ROIs (via adjacency matrix) are grouped into clusters.

    Parameters
    ----------
    events : np.ndarray (T, N)
        Binary event matrix.
    adjacency : np.ndarray (N, N), optional
        Structural or functional connectivity for spatial proximity.
        If None, all ROIs are considered spatially connected.
    temporal_window : int
        Maximum temporal gap for merging events.

    Returns
    -------
    dict with 'cluster_sizes', 'cluster_durations', 'n_clusters',
              'cluster_labels' (T, N) array.
    """
    T, N = events.shape

    if adjacency is None:
        adj = np.ones((N, N))
    else:
        adj = (adjacency != 0).astype(int)
    np.fill_diagonal(adj, 1)

    # Label connected events
    labels = np.zeros((T, N), dtype=int)
    current_label = 0
    visited = set()

    def _flood_fill(t, r, label):
        """BFS to find connected events."""
        queue = [(t, r)]
        while queue:
            ct, cr = queue.pop(0)
            if (ct, cr) in visited:
                continue
            if ct < 0 or ct >= T or cr < 0 or cr >= N:
                continue
            if events[ct, cr] == 0:
                continue
            visited.add((ct, cr))
            labels[ct, cr] = label

            # Temporal neighbors
            for dt in range(-temporal_window, temporal_window + 1):
                nt = ct + dt
                if 0 <= nt < T:
                    # Spatial neighbors
                    for nr in range(N):
                        if adj[cr, nr] and (nt, nr) not in visited and events[nt, nr]:
                            queue.append((nt, nr))

    for t in range(T):
        for r in range(N):
            if events[t, r] and (t, r) not in visited:
                current_label += 1
                _flood_fill(t, r, current_label)

    # Extract cluster statistics
    cluster_sizes = []
    cluster_durations = []
    for lbl in range(1, current_label + 1):
        mask = labels == lbl
        cluster_sizes.append(int(mask.sum()))
        times_active = np.where(mask.any(axis=1))[0]
        if len(times_active) > 0:
            cluster_durations.append(int(times_active[-1] - times_active[0] + 1))
        else:
            cluster_durations.append(0)

    return {
        "cluster_sizes": np.array(cluster_sizes),
        "cluster_durations": np.array(cluster_durations),
        "n_clusters": current_label,
        "cluster_labels": labels,
    }


# =============================================================================
# ORDER AND CONTROL PARAMETERS
# =============================================================================

def compute_order_parameter(
    events: np.ndarray,
    method: str = "global_sync",
) -> Dict[str, Any]:
    """
    Compute the order parameter — a measure of global synchronization.

    At the critical point, the order parameter is intermediate (neither
    fully synchronized nor fully desynchronized) and exhibits large
    fluctuations (susceptibility peak).

    Parameters
    ----------
    events : np.ndarray (T, N)
        Binary event matrix.
    method : str
        'global_sync' : Fraction of active ROIs per timepoint.
        'kuramoto'    : Kuramoto-like parameter (mean phase coherence).

    Returns
    -------
    dict with 'order_param' (timeseries), 'mean', 'std', 'susceptibility'.
    """
    T, N = events.shape

    if method == "global_sync":
        op = events.sum(axis=1) / N  # fraction of active ROIs
    elif method == "kuramoto":
        # Treat events as 0/π phases and compute Kuramoto R
        phases = events * np.pi
        complex_op = np.exp(1j * phases)
        op = np.abs(complex_op.mean(axis=1))
    else:
        raise ValueError(f"Unknown method: {method}")

    mean_op = float(np.mean(op))
    std_op = float(np.std(op))

    # Susceptibility: χ = N * Var(order_parameter)
    susceptibility = float(N * np.var(op))

    return {
        "order_param": op,
        "mean": mean_op,
        "std": std_op,
        "susceptibility": susceptibility,
    }


def compute_control_parameter(
    events: np.ndarray,
    timeseries: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute the control parameter — a measure that drives the system
    through the phase transition.

    For neural systems, the control parameter can be:
    - Event density (temporal rate of activations)
    - Effective connectivity strength
    - Input drive

    Parameters
    ----------
    events : np.ndarray (T, N)
    timeseries : np.ndarray (T, N), optional
        Original continuous signal for variance-based measures.

    Returns
    -------
    dict with 'control_param', 'mean', 'std'.
    """
    T, N = events.shape

    # Event density: total events per timepoint
    density = events.sum(axis=1).astype(float)

    # Temporal smoothing with Gaussian kernel
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(density, sigma=3)

    result = {
        "control_param": density,
        "control_param_smoothed": smoothed,
        "mean": float(np.mean(density)),
        "std": float(np.std(density)),
    }

    # If original timeseries provided, also compute global variance
    if timeseries is not None:
        global_var = np.var(timeseries, axis=1)
        result["global_variance"] = global_var
        result["global_variance_mean"] = float(np.mean(global_var))

    return result


# =============================================================================
# CRITICAL DYNAMICS INDICATORS
# =============================================================================

def compute_autocorrelation_decay(
    order_param: np.ndarray,
    max_lag: int = 50,
) -> Dict[str, Any]:
    """
    Compute the autocorrelation function of the order parameter and
    estimate the autocorrelation time τ.

    Near criticality, τ diverges (critical slowing down).

    Parameters
    ----------
    order_param : np.ndarray (T,)
    max_lag : int

    Returns
    -------
    dict with 'lags', 'acf', 'tau' (autocorrelation time).
    """
    op = np.asarray(order_param, dtype=float).ravel()
    T = len(op)
    max_lag = min(max_lag, T // 2)

    op_centered = op - np.mean(op)
    var = np.var(op)

    if var == 0:
        return {"lags": np.arange(max_lag), "acf": np.zeros(max_lag), "tau": np.nan}

    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        acf[lag] = np.mean(op_centered[:T - lag] * op_centered[lag:]) / var

    # Estimate τ: first lag where ACF drops below 1/e
    threshold = 1 / np.e
    below = np.where(acf < threshold)[0]
    tau = float(below[0]) if len(below) > 0 else float(max_lag)

    return {
        "lags": np.arange(max_lag),
        "acf": acf,
        "tau": tau,
    }


def compute_susceptibility_vs_threshold(
    timeseries: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute susceptibility (variance of order parameter) as a function
    of the binarization threshold.

    A peak in susceptibility indicates the critical threshold.

    Parameters
    ----------
    timeseries : np.ndarray (T, N)
    thresholds : np.ndarray, optional
        Array of z-score thresholds to scan.

    Returns
    -------
    dict with 'thresholds', 'susceptibility', 'critical_threshold'.
    """
    ts = np.asarray(timeseries, dtype=float)
    T, N = ts.shape

    if thresholds is None:
        thresholds = np.arange(0.5, 4.1, 0.25)

    z = (ts - ts.mean(axis=0, keepdims=True)) / np.maximum(
        ts.std(axis=0, keepdims=True), 1e-10
    )

    suscept = np.zeros(len(thresholds))
    for i, th in enumerate(thresholds):
        events = (np.abs(z) > th).astype(int)
        op = events.sum(axis=1) / N
        suscept[i] = N * np.var(op)

    # Find critical threshold (max susceptibility)
    idx_peak = np.argmax(suscept)
    critical_th = float(thresholds[idx_peak])

    return {
        "thresholds": thresholds,
        "susceptibility": suscept,
        "critical_threshold": critical_th,
        "peak_susceptibility": float(suscept[idx_peak]),
    }


# =============================================================================
# MASTER ANALYSIS FUNCTION
# =============================================================================

def analyze_point_process(
    timeseries: np.ndarray,
    threshold_sd: float = 1.0,
    mode: str = "exceedance",
    adjacency: Optional[np.ndarray] = None,
    scan_thresholds: bool = True,
) -> Dict[str, Any]:
    """
    Complete point-process analysis pipeline.

    Parameters
    ----------
    timeseries : np.ndarray (T, N)
    threshold_sd : float
    mode : str
    adjacency : np.ndarray, optional
    scan_thresholds : bool

    Returns
    -------
    dict with all point-process analysis results.
    """
    ts = np.asarray(timeseries, dtype=float)
    T, N = ts.shape

    results = {"n_timepoints": T, "n_rois": N, "threshold_sd": threshold_sd}

    # 1. Extract point process
    pp = extract_point_process(ts, threshold_sd=threshold_sd, mode=mode)
    results["n_events"] = pp["n_events_total"]
    results["event_rate_per_roi"] = pp["event_rate"]
    results["mean_event_rate"] = float(np.mean(pp["event_rate"]))

    # 2. Point-process FC
    pp_fc = point_process_fc(pp["events"], method="correlation")
    results["pp_fc"] = pp_fc

    # 3. Order parameter
    op_result = compute_order_parameter(pp["events"])
    results["order_param_mean"] = op_result["mean"]
    results["order_param_std"] = op_result["std"]
    results["susceptibility"] = op_result["susceptibility"]

    # 4. Control parameter
    cp_result = compute_control_parameter(pp["events"], ts)
    results["control_param_mean"] = cp_result["mean"]

    # 5. Autocorrelation (critical slowing down)
    ac_result = compute_autocorrelation_decay(op_result["order_param"])
    results["autocorrelation_time"] = ac_result["tau"]

    # 6. Spatiotemporal clustering
    st_result = spatiotemporal_clustering(pp["events"], adjacency=adjacency)
    results["n_st_clusters"] = st_result["n_clusters"]
    if len(st_result["cluster_sizes"]) > 0:
        results["mean_cluster_size"] = float(np.mean(st_result["cluster_sizes"]))
        results["max_cluster_size"] = int(np.max(st_result["cluster_sizes"]))
    else:
        results["mean_cluster_size"] = 0.0
        results["max_cluster_size"] = 0

    # 7. Susceptibility scan (optional)
    if scan_thresholds:
        scan = compute_susceptibility_vs_threshold(ts)
        results["critical_threshold"] = scan["critical_threshold"]
        results["peak_susceptibility"] = scan["peak_susceptibility"]
        results["susceptibility_scan"] = scan

    # Store events matrix reference
    results["events"] = pp["events"]
    results["global_rate"] = pp["global_rate"]

    return results
