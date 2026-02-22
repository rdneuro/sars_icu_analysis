# -*- coding: utf-8 -*-
"""
sars.io
=======

Input / Output utilities for the SARS-CoV-2 neuroimaging library.

Provides a unified interface for loading preprocessed data (timeseries,
functional connectivity matrices, structural connectivity matrices, atlas
labels) and for persisting analysis results and figures. All functions
rely on paths defined in ``sars.config`` so that downstream modules never
need to hard-code file locations.

Functions
---------
- ``load_timeseries``       : single-subject timeseries → ndarray (T, N)
- ``load_fc``               : single-subject FC matrix → ndarray (N, N)
- ``load_sc``               : single-subject SC matrix → ndarray (N, N)
- ``get_roi_names``         : atlas labels → list[str]
- ``get_roi_coordinates``   : atlas MNI coordinates → ndarray (N, 3)
- ``save_results``          : dict / array → .json / .npy / .npz
- ``save_figure``           : matplotlib figure → .png / .svg / .pdf

Usage
-----
    from sars import io

    ts = io.load_timeseries("sub-01", "schaefer_100")
    fc = io.load_fc("sub-01", "schaefer_100", "correlation")
    labels = io.get_roi_names("schaefer_100")
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Any, Union


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _resolve_timeseries_path(
    subject_id: str, atlas: str, strategy: str
) -> Path:
    """Try several path conventions to locate a timeseries .npy file."""
    from . import config

    candidates = [
        # Standard: connectivity/<atlas>/<strategy>/<sub>_<atlas>_timeseries.npy
        config.CONNECTIVITY_DIR / atlas / strategy
        / f"{subject_id}_{atlas}_timeseries.npy",
        # Flat: connectivity/<atlas>/<sub>_<atlas>_timeseries.npy
        config.CONNECTIVITY_DIR / atlas
        / f"{subject_id}_{atlas}_timeseries.npy",
        # Alternative nesting
        config.CONNECTIVITY_DIR / strategy / atlas
        / f"{subject_id}_{atlas}_timeseries.npy",
        # Legacy
        config.CONNECTIVITY_DIR / f"{subject_id}" / atlas
        / f"{subject_id}_{atlas}_timeseries.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Timeseries file not found for {subject_id} / {atlas} / {strategy}.  "
        f"Searched:\n" + "\n".join(f"  {c}" for c in candidates)
    )


def _resolve_fc_path(
    subject_id: str, atlas: str, kind: str, strategy: str
) -> Path:
    """Try several conventions to locate an FC .npy file."""
    from . import config

    candidates = [
        config.CONNECTIVITY_DIR / atlas / strategy
        / f"{subject_id}_{atlas}_{kind}.npy",
        config.CONNECTIVITY_DIR / atlas
        / f"{subject_id}_{atlas}_{kind}.npy",
        config.CONNECTIVITY_DIR / strategy / atlas
        / f"{subject_id}_{atlas}_{kind}.npy",
        config.CONNECTIVITY_DIR / f"{subject_id}" / atlas
        / f"{subject_id}_{atlas}_{kind}.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"FC matrix not found for {subject_id}/{atlas}/{kind}/{strategy}.  "
        f"Searched:\n" + "\n".join(f"  {c}" for c in candidates)
    )


def _resolve_sc_path(
    subject_id: str, atlas: str, weight: str
) -> Path:
    """Try several conventions to locate an SC .npy file."""
    from . import config

    sc_base = (
        config.PROJECT_ROOT / "data" / "outputs" / "diffusion"
        / "connectivity"
    )
    candidates = [
        sc_base / atlas / f"{subject_id}_{atlas}_{weight}.npy",
        sc_base / f"{subject_id}" / atlas
        / f"{subject_id}_{atlas}_{weight}.npy",
        sc_base / atlas / f"{subject_id}_{weight}.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"SC matrix not found for {subject_id}/{atlas}/{weight}.  "
        f"Searched:\n" + "\n".join(f"  {c}" for c in candidates)
    )


# =============================================================================
# TIMESERIES
# =============================================================================

def load_timeseries(
    subject_id: str,
    atlas: str,
    strategy: Optional[str] = None,
) -> np.ndarray:
    """
    Load preprocessed BOLD timeseries for one subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., ``"sub-01"``).
    atlas : str
        Atlas key (``"schaefer_100"``, ``"aal3"``, ``"brainnetome"``,
        ``"synthseg"``).
    strategy : str, optional
        Denoising strategy.  Defaults to ``config.DENOISING_STRATEGY``.

    Returns
    -------
    np.ndarray, shape (T, N)
        Timeseries matrix — T timepoints × N ROIs.

    Raises
    ------
    FileNotFoundError
        If no matching .npy file is found.
    """
    from . import config
    strategy = strategy or config.DENOISING_STRATEGY

    path = _resolve_timeseries_path(subject_id, atlas, strategy)
    ts = np.load(path)

    # Ensure (T, N) orientation — some pipelines store (N, T)
    if ts.ndim == 2 and ts.shape[0] < ts.shape[1]:
        ts = ts.T

    return ts


# =============================================================================
# FUNCTIONAL CONNECTIVITY
# =============================================================================

def load_fc(
    subject_id: str,
    atlas: str,
    kind: str = "correlation",
    strategy: Optional[str] = None,
) -> np.ndarray:
    """
    Load a functional connectivity matrix for one subject.

    Parameters
    ----------
    subject_id : str
    atlas : str
    kind : str
        FC estimator (``"correlation"``, ``"partial_correlation"``,
        ``"covariance"``, ``"tangent"``).
    strategy : str, optional

    Returns
    -------
    np.ndarray, shape (N, N)

    Raises
    ------
    FileNotFoundError
    """
    from . import config
    strategy = strategy or config.DENOISING_STRATEGY

    path = _resolve_fc_path(subject_id, atlas, kind, strategy)
    return np.load(path)


# =============================================================================
# STRUCTURAL CONNECTIVITY
# =============================================================================

def load_sc(
    subject_id: str,
    atlas: str,
    weight: str = "streamline_count",
) -> np.ndarray:
    """
    Load a structural connectivity matrix for one subject.

    Parameters
    ----------
    subject_id : str
    atlas : str
    weight : str
        Edge-weight definition (``"streamline_count"``, ``"fa"``,
        ``"length"``).

    Returns
    -------
    np.ndarray, shape (N, N)

    Raises
    ------
    FileNotFoundError
    """
    path = _resolve_sc_path(subject_id, atlas, weight)
    return np.load(path)


# =============================================================================
# ATLAS LABELS
# =============================================================================

def get_roi_names(atlas: str) -> List[str]:
    """
    Return the ordered list of ROI label strings for *atlas*.

    The function reads the CSV / TSV shipped with the atlas definition
    in ``config.ATLASES``.  Labels are cached after the first call.

    Parameters
    ----------
    atlas : str

    Returns
    -------
    list of str
        ROI labels in the order matching matrix rows/columns.
    """
    from . import config

    if atlas not in config.ATLASES:
        raise ValueError(
            f"Unknown atlas '{atlas}'.  "
            f"Available: {list(config.ATLASES.keys())}"
        )

    info = config.ATLASES[atlas]
    labels_file = Path(info["labels_file"])

    if not labels_file.exists():
        # Attempt nilearn fallback for Schaefer
        if atlas == "schaefer_100":
            try:
                from nilearn import datasets
                sch = datasets.fetch_atlas_schaefer_2018(
                    n_rois=100, yeo_networks=7, resolution_mm=2
                )
                return [
                    l.decode() if isinstance(l, bytes) else str(l)
                    for l in sch.labels
                ]
            except Exception:
                pass
        # Generic fallback: numbered labels
        warnings.warn(
            f"Label file not found: {labels_file}.  "
            f"Returning generic ROI_001 … ROI_{info['n_rois']:03d} labels.",
            UserWarning,
        )
        return [f"ROI_{i + 1:03d}" for i in range(info["n_rois"])]

    df = pd.read_csv(labels_file, sep=info["labels_sep"])
    col = info["label_col"]
    if col in df.columns:
        return df[col].astype(str).tolist()
    # Fallback to second column
    return df.iloc[:, 1].astype(str).tolist()


# label cache (module-level)
_label_cache: Dict[str, List[str]] = {}


def get_roi_names_cached(atlas: str) -> List[str]:
    """Cached wrapper around :func:`get_roi_names`."""
    if atlas not in _label_cache:
        _label_cache[atlas] = get_roi_names(atlas)
    return _label_cache[atlas]


# =============================================================================
# ROI COORDINATES
# =============================================================================

def get_roi_coordinates(atlas: str) -> np.ndarray:
    """
    Compute MNI centroid coordinates for each ROI in *atlas*.

    Uses ``nilearn.plotting.find_parcellation_cut_coords`` when the
    NIfTI parcellation is available; otherwise returns NaN array.

    Parameters
    ----------
    atlas : str

    Returns
    -------
    np.ndarray, shape (N, 3)
        MNI (x, y, z) coordinates per ROI.
    """
    from . import config

    info = config.ATLASES[atlas]
    nifti_path = Path(info["nifti"])

    if nifti_path.exists():
        try:
            from nilearn.plotting import find_parcellation_cut_coords
            coords = find_parcellation_cut_coords(str(nifti_path))
            return np.asarray(coords)
        except Exception as exc:
            warnings.warn(
                f"Could not extract coordinates for {atlas}: {exc}",
                UserWarning,
            )

    return np.full((info["n_rois"], 3), np.nan)


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    data: Any,
    filename: str,
    output_dir: Optional[Path] = None,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Persist analysis results to disk.

    - ``dict`` → ``.json``  (numpy arrays converted to lists)
    - ``np.ndarray`` → ``.npy``
    - Multiple arrays → ``.npz`` (pass a dict of arrays)

    Parameters
    ----------
    data : dict | np.ndarray
    filename : str
        Output filename **without extension** (extension is added
        automatically).
    output_dir : Path, optional
        Defaults to ``config.METRICS_DIR / "sars_results"``.
    overwrite : bool
        Overwrite existing files.

    Returns
    -------
    Path
        Path to the saved file.
    """
    from . import config

    if output_dir is None:
        output_dir = config.METRICS_DIR / "sars_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON dict
    if isinstance(data, dict):
        path = output_dir / f"{filename}.json"
        if path.exists() and not overwrite:
            return path

        def _default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Non-serialisable: {type(obj)}")

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_default)
        return path

    # Single array
    if isinstance(data, np.ndarray):
        path = output_dir / f"{filename}.npy"
        if path.exists() and not overwrite:
            return path
        np.save(path, data)
        return path

    raise TypeError(f"Unsupported data type for saving: {type(data)}")


# =============================================================================
# SAVE FIGURE
# =============================================================================

def save_figure(
    fig,
    filename: str,
    output_dir: Optional[Path] = None,
    *,
    formats: tuple = ("png", "svg"),
    dpi: int = 300,
    close: bool = True,
) -> List[Path]:
    """
    Save a matplotlib figure in one or more formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filename : str
        Filename **without extension**.
    output_dir : Path, optional
        Defaults to ``config.FIGURES_DIR``.
    formats : tuple of str
        File extensions to produce.
    dpi : int
    close : bool
        If True, close the figure after saving.

    Returns
    -------
    list of Path
    """
    import matplotlib.pyplot as plt
    from . import config

    if output_dir is None:
        output_dir = config.FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for fmt in formats:
        p = output_dir / f"{filename}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(p)

    if close:
        plt.close(fig)

    return paths
