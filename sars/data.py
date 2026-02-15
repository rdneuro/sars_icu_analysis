# -*- coding: utf-8 -*-
"""
sars.data
======================

Data structures, atlas management, quality control, group-level containers,
and matrix preprocessing utilities for the SARS-CoV-2 neuroimaging library.

This module provides the primary interfaces for organizing, validating,
and preprocessing multimodal brain connectivity data. It integrates with
the I/O layer (sars.io) and configuration (sars.config)
to offer a streamlined workflow from raw pipeline outputs to analysis-ready
data structures.

Includes
--------
- ConnectivityData : single-subject multimodal container
- GroupData : group-level 3D matrix stack with statistical summaries
- Atlas management : network assignment, hemisphere parsing, coordinates
- Matrix operations : thresholding, binarization, symmetrization, normalization
- Quality control : QC-FC correlation, motion, tSNR, outlier detection
- Data validation : integrity checks, NaN/Inf handling, dimension verification
- Batch loading : sequential loading with progress reporting

References
----------
- Power et al. (2012). NeuroImage. Spurious but systematic correlations
  in functional connectivity MRI networks arise from subject motion.
- Ciric et al. (2017). NeuroImage. Benchmarking of participant-level
  confound regression strategies.
- Parkes et al. (2018). NeuroImage. An evaluation of the efficacy,
  reliability, and sensitivity of motion correction strategies.
- Satterthwaite et al. (2013). NeuroImage. An improved framework
  for confound regression and filtering.
- Schaefer et al. (2018). Cerebral Cortex. Local-global parcellation
  of the human cerebral cortex.
- Finn et al. (2015). Nature Neuroscience. Functional connectome
  fingerprinting.

Usage
-----
    from sars.data import ConnectivityData, GroupData
    from sars.data import threshold_matrix, get_schaefer_network_indices

    # Single subject
    sub = ConnectivityData.from_subject("sub-01", "schaefer_100", load_sc=True)
    sub.validate()
    sub.summary()

    # Group level
    group = GroupData.from_subjects(atlas="schaefer_100", fc_kind="correlation")
    group.apply_fisher_z()
    group.compute_group_stats()
    group.qc_fc(mean_fd_values)
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Any

from . import config


# =============================================================================
# CONNECTIVITY DATA — SINGLE-SUBJECT CONTAINER
# =============================================================================

@dataclass
class ConnectivityData:
    """
    Container holding a single subject's multimodal connectivity data.

    This is the primary unit of data in the sars library.
    It encapsulates timeseries, functional connectivity, structural
    connectivity, ROI labels, and associated metadata / QC metrics for
    one subject under one atlas parcellation.

    Attributes
    ----------
    subject_id : str
        Subject identifier (e.g. 'sub-01').
    atlas : str
        Atlas key as defined in config.ATLASES.
    timeseries : np.ndarray or None
        BOLD timeseries, shape (T, N).
    fc : dict
        Functional connectivity matrices.  Keys: 'correlation', 'partial', …
    sc : dict
        Structural connectivity matrices.  Keys: 'streamline', 'fa', …
    roi_labels : list of str
        Ordered ROI names from the atlas.
    n_rois : int
        Number of brain regions.
    n_timepoints : int
        Number of fMRI volumes.
    metadata : dict
        Denoising strategy, TR, acquisition parameters, etc.
    qc : dict
        Quality control metrics (mean_fd, tsnr, n_censored_volumes, …).
    is_valid : bool
        Whether the data passed validation checks.
    validation_notes : list of str
        Warnings or issues found during validation.
    """
    subject_id: str
    atlas: str
    timeseries: Optional[np.ndarray] = None
    fc: Dict[str, np.ndarray] = field(default_factory=dict)
    sc: Dict[str, np.ndarray] = field(default_factory=dict)
    roi_labels: List[str] = field(default_factory=list)
    n_rois: int = 0
    n_timepoints: int = 0
    metadata: Dict = field(default_factory=dict)
    qc: Dict = field(default_factory=dict)
    is_valid: bool = True
    validation_notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.timeseries is not None:
            self.n_timepoints, self.n_rois = self.timeseries.shape
        elif self.fc:
            first = next(iter(self.fc.values()))
            self.n_rois = first.shape[0]
        elif self.sc:
            first = next(iter(self.sc.values()))
            self.n_rois = first.shape[0]

    # -----------------------------------------------------------------
    # Factory constructor
    # -----------------------------------------------------------------
    @classmethod
    def from_subject(
        cls,
        subject_id: str,
        atlas: str,
        load_timeseries: bool = True,
        load_fc: bool = True,
        fc_kinds: List[str] = None,
        load_sc: bool = False,
        sc_weights: List[str] = None,
        strategy: str = None,
        load_qc: bool = True,
        validate: bool = True,
    ) -> "ConnectivityData":
        """
        Factory method: load all available data for a subject.

        Parameters
        ----------
        subject_id : str
            Subject identifier (e.g. 'sub-01').
        atlas : str
            Atlas key (e.g. 'schaefer_100', 'brainnetome', 'aal3').
        load_timeseries : bool
            Whether to load parcellated BOLD timeseries.
        load_fc : bool
            Whether to load functional connectivity matrices.
        fc_kinds : list of str
            FC types to load.  Default: ['correlation'].
        load_sc : bool
            Whether to load structural connectivity matrices.
        sc_weights : list of str
            SC weights to load.  Default: ['streamline'].
        strategy : str
            Denoising strategy.  Default from config.DENOISING_STRATEGY.
        load_qc : bool
            Whether to attempt loading QC metrics (mean FD, tSNR, etc.).
        validate : bool
            Whether to run validation after loading.
        """
        from . import io

        fc_kinds = fc_kinds or ["correlation"]
        sc_weights = sc_weights or ["streamline"]
        strategy = strategy or config.DENOISING_STRATEGY

        ts, fc_dict, sc_dict, qc_dict = None, {}, {}, {}
        notes = []

        # --- Load timeseries ---
        if load_timeseries:
            try:
                ts = io.load_timeseries(subject_id, atlas, strategy)
            except FileNotFoundError as e:
                notes.append(f"Timeseries not found: {e}")

        # --- Load FC ---
        if load_fc:
            for kind in fc_kinds:
                try:
                    fc_dict[kind] = io.load_fc(
                        subject_id, atlas, kind, strategy
                    )
                except FileNotFoundError:
                    notes.append(f"FC '{kind}' not found.")

        # --- Load SC ---
        if load_sc:
            for w in sc_weights:
                try:
                    sc_dict[w] = io.load_sc(subject_id, atlas, w)
                except FileNotFoundError:
                    notes.append(f"SC '{w}' not found.")

        # --- Load atlas labels ---
        try:
            labels = io.get_roi_names(atlas)
        except Exception:
            labels = []
            notes.append("ROI labels could not be loaded.")

        # --- Load QC metrics ---
        if load_qc:
            qc_dict = _load_subject_qc(subject_id, atlas, strategy)

        meta = {
            "strategy": strategy,
            "tr": config.TR,
            "bandpass": config.BANDPASS,
            "atlas_info": config.ATLASES.get(atlas),
        }

        obj = cls(
            subject_id=subject_id, atlas=atlas,
            timeseries=ts, fc=fc_dict, sc=sc_dict,
            roi_labels=labels, metadata=meta,
            qc=qc_dict, validation_notes=notes,
        )

        if validate:
            obj.validate()
        return obj

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------
    def validate(self) -> bool:
        """
        Run comprehensive data integrity checks.

        Checks: NaN/Inf, dimension consistency, symmetry, diagonal,
        minimum timeseries length (4 min per rsfmri_best_practices.pdf),
        zero-variance and constant-signal ROIs.

        Returns
        -------
        bool : True if all critical checks pass.
        """
        self.is_valid = True

        # --- Timeseries ---
        if self.timeseries is not None:
            ts = self.timeseries
            n_nan = int(np.sum(np.isnan(ts)))
            n_inf = int(np.sum(np.isinf(ts)))
            if n_nan > 0:
                self.validation_notes.append(
                    f"⚠ Timeseries contains {n_nan} NaN values."
                )
            if n_inf > 0:
                self.validation_notes.append(
                    f"⚠ Timeseries contains {n_inf} Inf values."
                )

            min_volumes = int(np.ceil(240.0 / config.TR))  # 4 min
            if ts.shape[0] < min_volumes:
                self.validation_notes.append(
                    f"⚠ Short timeseries: {ts.shape[0]} vols "
                    f"({ts.shape[0] * config.TR:.0f}s).  "
                    f"Min recommended: {min_volumes} vols (240 s)."
                )

            variances = np.var(ts, axis=0)
            zero_var = np.where(variances < 1e-10)[0]
            if len(zero_var) > 0:
                self.validation_notes.append(
                    f"⚠ {len(zero_var)} ROIs with zero/near-zero variance."
                )
                self.is_valid = False

        # --- FC ---
        for kind, mat in self.fc.items():
            notes = _validate_square_matrix(mat, f"FC({kind})")
            self.validation_notes.extend(notes)
            if any("✗" in n for n in notes):
                self.is_valid = False
            if self.n_rois > 0 and mat.shape[0] != self.n_rois:
                self.validation_notes.append(
                    f"✗ FC({kind}) has {mat.shape[0]} ROIs, "
                    f"expected {self.n_rois}."
                )
                self.is_valid = False

        # --- SC ---
        for weight, mat in self.sc.items():
            notes = _validate_square_matrix(mat, f"SC({weight})")
            self.validation_notes.extend(notes)
            if any("✗" in n for n in notes):
                self.is_valid = False
            if np.any(mat < 0):
                n_neg = int(np.sum(mat < 0))
                self.validation_notes.append(
                    f"⚠ SC({weight}) has {n_neg} negative values."
                )

        return self.is_valid

    # -----------------------------------------------------------------
    # Sanitization
    # -----------------------------------------------------------------
    def sanitize(self, fill_nan: float = 0.0, verbose: bool = True):
        """
        Replace NaN/Inf, enforce symmetry, zero diagonal, clamp SC ≥ 0.
        """
        actions = []

        if self.timeseries is not None:
            n_bad = int(np.sum(~np.isfinite(self.timeseries)))
            if n_bad > 0:
                self.timeseries = np.nan_to_num(
                    self.timeseries, nan=fill_nan,
                    posinf=fill_nan, neginf=fill_nan,
                )
                actions.append(
                    f"Replaced {n_bad} NaN/Inf in timeseries."
                )

        for kind in self.fc:
            mat = self.fc[kind]
            n_bad = int(np.sum(~np.isfinite(mat)))
            if n_bad > 0:
                mat = np.nan_to_num(
                    mat, nan=fill_nan, posinf=fill_nan, neginf=fill_nan,
                )
                actions.append(f"Replaced {n_bad} NaN/Inf in FC({kind}).")
            self.fc[kind] = _clean_square(mat)

        for weight in self.sc:
            mat = self.sc[weight]
            n_bad = int(np.sum(~np.isfinite(mat)))
            if n_bad > 0:
                mat = np.nan_to_num(
                    mat, nan=fill_nan, posinf=fill_nan, neginf=fill_nan,
                )
                actions.append(
                    f"Replaced {n_bad} NaN/Inf in SC({weight})."
                )
            mat = _clean_square(mat)
            n_neg = int(np.sum(mat < 0))
            if n_neg > 0:
                mat[mat < 0] = 0
                actions.append(
                    f"Zeroed {n_neg} negative values in SC({weight})."
                )
            self.sc[weight] = mat

        if verbose and actions:
            print(f"[sanitize] {self.subject_id}/{self.atlas}:")
            for a in actions:
                print(f"  → {a}")

    # -----------------------------------------------------------------
    # Convenience getters
    # -----------------------------------------------------------------
    def get_fc(
        self, kind: str = "correlation", fisher_z: bool = False,
    ) -> np.ndarray:
        """
        Retrieve an FC matrix, optionally Fisher z-transformed.

        Parameters
        ----------
        kind : str
            FC type key.
        fisher_z : bool
            Apply arctanh transform (variance-stabilizing for group stats).
        """
        if kind not in self.fc:
            raise KeyError(
                f"FC kind '{kind}' not loaded.  "
                f"Available: {list(self.fc.keys())}"
            )
        mat = self.fc[kind].copy()
        if fisher_z:
            np.fill_diagonal(mat, 0)
            mat = np.clip(mat, -0.9999, 0.9999)
            mat = np.arctanh(mat)
            np.fill_diagonal(mat, 0)
        return mat

    def get_sc(self, weight: str = "streamline") -> np.ndarray:
        """Retrieve an SC matrix."""
        if weight not in self.sc:
            raise KeyError(
                f"SC weight '{weight}' not loaded.  "
                f"Available: {list(self.sc.keys())}"
            )
        return self.sc[weight].copy()

    def get_timeseries(self, zscore: bool = False) -> np.ndarray:
        """
        Retrieve the BOLD timeseries, optionally z-scored per ROI.

        Note: avoid double z-scoring if data was already standardized
        during denoising (standardize=False in the rs-fMRI pipeline).
        """
        if self.timeseries is None:
            raise ValueError(f"No timeseries loaded for {self.subject_id}.")
        ts = self.timeseries.copy()
        if zscore:
            from scipy import stats as sp_stats
            ts = sp_stats.zscore(ts, axis=0, nan_policy="omit")
        return ts

    # -----------------------------------------------------------------
    # Computed properties
    # -----------------------------------------------------------------
    @property
    def duration_seconds(self) -> float:
        return self.n_timepoints * self.metadata.get("tr", config.TR)

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    @property
    def has_timeseries(self) -> bool:
        return self.timeseries is not None

    @property
    def has_fc(self) -> bool:
        return len(self.fc) > 0

    @property
    def has_sc(self) -> bool:
        return len(self.sc) > 0

    @property
    def available_fc(self) -> List[str]:
        return list(self.fc.keys())

    @property
    def available_sc(self) -> List[str]:
        return list(self.sc.keys())

    # -----------------------------------------------------------------
    # Timeseries quality metrics
    # -----------------------------------------------------------------
    def compute_tsnr(self) -> np.ndarray:
        """
        Temporal signal-to-noise ratio per ROI: tSNR = mean / std.

        Returns
        -------
        np.ndarray, shape (N,)
        """
        if self.timeseries is None:
            raise ValueError("No timeseries loaded.")
        ts = self.timeseries
        mean_ts = np.mean(ts, axis=0)
        std_ts = np.std(ts, axis=0)
        std_ts[std_ts == 0] = np.inf
        tsnr = mean_ts / std_ts
        self.qc["tsnr_per_roi"] = tsnr
        self.qc["tsnr_mean"] = float(np.nanmean(tsnr[np.isfinite(tsnr)]))
        return tsnr

    def identify_outlier_rois(
        self, method: str = "variance", threshold_sd: float = 3.0,
    ) -> np.ndarray:
        """
        Identify ROIs with anomalous signal properties.

        Parameters
        ----------
        method : str
            'variance', 'tsnr', or 'range'.
        threshold_sd : float

        Returns
        -------
        np.ndarray : Boolean mask, True = outlier.
        """
        if self.timeseries is None:
            raise ValueError("No timeseries loaded.")
        ts = self.timeseries

        if method == "variance":
            metric = np.var(ts, axis=0)
        elif method == "tsnr":
            metric = -self.compute_tsnr()  # low tSNR → high score
        elif method == "range":
            metric = np.ptp(ts, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        z = (metric - np.nanmean(metric)) / (np.nanstd(metric) + 1e-15)
        return np.abs(z) > threshold_sd

    # -----------------------------------------------------------------
    # Subset / mask operations
    # -----------------------------------------------------------------
    def drop_rois(
        self, roi_indices: Union[List[int], np.ndarray],
    ) -> "ConnectivityData":
        """
        Return a new ConnectivityData with specified ROIs removed.
        """
        roi_indices = np.asarray(roi_indices)
        keep = np.setdiff1d(np.arange(self.n_rois), roi_indices)

        new = deepcopy(self)
        new.n_rois = len(keep)
        if new.timeseries is not None:
            new.timeseries = new.timeseries[:, keep]
            new.n_timepoints = new.timeseries.shape[0]
        for kind in new.fc:
            new.fc[kind] = new.fc[kind][np.ix_(keep, keep)]
        for w in new.sc:
            new.sc[w] = new.sc[w][np.ix_(keep, keep)]
        if new.roi_labels:
            new.roi_labels = [new.roi_labels[i] for i in keep]
        new.metadata["dropped_rois"] = roi_indices.tolist()
        return new

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    def summary(self) -> str:
        """Print a formatted summary of loaded data."""
        lines = [
            f"{'=' * 60}",
            f"  ConnectivityData: {self.subject_id} | {self.atlas}",
            f"{'=' * 60}",
        ]
        if self.timeseries is not None:
            lines.append(
                f"  Timeseries : ({self.n_timepoints}, {self.n_rois})"
                f"  [{self.duration_minutes:.1f} min]"
            )
        else:
            lines.append("  Timeseries : not loaded")

        if self.fc:
            fc_str = ", ".join(
                f"{k} ({v.shape[0]}×{v.shape[1]})"
                for k, v in self.fc.items()
            )
            lines.append(f"  FC         : {fc_str}")
        else:
            lines.append("  FC         : not loaded")

        if self.sc:
            sc_str = ", ".join(
                f"{k} ({v.shape[0]}×{v.shape[1]})"
                for k, v in self.sc.items()
            )
            lines.append(f"  SC         : {sc_str}")
        else:
            lines.append("  SC         : not loaded")

        lines.append(f"  ROI labels : {len(self.roi_labels)} labels")
        lines.append(f"  Valid      : {'✓' if self.is_valid else '✗'}")
        lines.append(
            f"  Strategy   : {self.metadata.get('strategy', 'N/A')}"
        )

        if self.qc:
            qc_parts = []
            if "mean_fd" in self.qc:
                qc_parts.append(f"FD={self.qc['mean_fd']:.3f}mm")
            if "tsnr_mean" in self.qc:
                qc_parts.append(f"tSNR={self.qc['tsnr_mean']:.1f}")
            if "n_censored" in self.qc:
                qc_parts.append(f"censored={self.qc['n_censored']} vols")
            if qc_parts:
                lines.append(f"  QC         : {', '.join(qc_parts)}")

        if self.validation_notes:
            lines.append(f"  Notes ({len(self.validation_notes)}):")
            for note in self.validation_notes[:5]:
                lines.append(f"    {note}")
            if len(self.validation_notes) > 5:
                lines.append(
                    f"    … and {len(self.validation_notes) - 5} more."
                )
        lines.append(f"{'=' * 60}")
        text = "\n".join(lines)
        print(text)
        return text

    def __repr__(self):
        return (
            f"ConnectivityData('{self.subject_id}', '{self.atlas}', "
            f"n_rois={self.n_rois}, fc={list(self.fc.keys())}, "
            f"sc={list(self.sc.keys())}, valid={self.is_valid})"
        )


# =============================================================================
# GROUP DATA CONTAINER
# =============================================================================

class GroupData:
    """
    Group-level container for multi-subject connectivity data.

    Holds a 3D stack of connectivity matrices (subjects × ROIs × ROIs)
    with subject IDs, atlas info, and group statistics.  Supports QC-FC
    analysis, outlier detection, and Fisher z-transformation.

    Follows rsfmri_best_practices.pdf recommendations:
    - Fisher z-transform before group averaging / statistics
    - QC-FC correlation < 0.05 median, < 5 % FDR-significant edges
    - Distance-dependence |r| < 0.1
    """

    def __init__(
        self,
        matrices: np.ndarray,
        subject_ids: List[str],
        atlas: str,
        kind: str = "correlation",
        modality: str = "fc",
        roi_labels: List[str] = None,
    ):
        assert matrices.ndim == 3
        assert matrices.shape[0] == len(subject_ids)
        assert matrices.shape[1] == matrices.shape[2]

        self.matrices = matrices
        self.subject_ids = list(subject_ids)
        self.atlas = atlas
        self.kind = kind
        self.modality = modality
        self.roi_labels = roi_labels or []
        self.n_subjects = matrices.shape[0]
        self.n_rois = matrices.shape[1]
        self.is_fisher_z = False
        self.group_stats = None
        self._qc_fc = None

    # -----------------------------------------------------------------
    # Factory constructors
    # -----------------------------------------------------------------
    @classmethod
    def from_subjects(
        cls, atlas: str, fc_kind: str = None, sc_weight: str = None,
        strategy: str = None, subjects: List[str] = None,
        fisher_z: bool = False,
    ) -> "GroupData":
        """Load and stack connectivity matrices for all subjects."""
        from . import io

        if fc_kind is None and sc_weight is None:
            fc_kind = "correlation"
        subjects = subjects or config.ALL_SUBJECT_IDS
        strategy = strategy or config.DENOISING_STRATEGY

        matrices, loaded_subs = [], []
        for sub in subjects:
            try:
                if fc_kind:
                    mat = io.load_fc(sub, atlas, fc_kind, strategy)
                    modality, kind = "fc", fc_kind
                else:
                    mat = io.load_sc(sub, atlas, sc_weight)
                    modality, kind = "sc", sc_weight
                matrices.append(mat)
                loaded_subs.append(sub)
            except FileNotFoundError:
                pass

        if not matrices:
            raise FileNotFoundError(
                f"No matrices found for atlas='{atlas}', "
                f"kind='{fc_kind or sc_weight}'."
            )

        try:
            labels = io.get_roi_names(atlas)
        except Exception:
            labels = []

        obj = cls(
            np.array(matrices), loaded_subs, atlas, kind, modality, labels,
        )
        if fisher_z and modality == "fc":
            obj.apply_fisher_z()
        return obj

    @classmethod
    def from_connectivity_data_list(
        cls, data_list: List[ConnectivityData],
        fc_kind: str = "correlation", sc_weight: str = None,
        fisher_z: bool = False,
    ) -> "GroupData":
        """Build GroupData from a list of ConnectivityData objects."""
        if not data_list:
            raise ValueError("data_list is empty.")
        atlas = data_list[0].atlas
        matrices, subs = [], []
        for d in data_list:
            if d.atlas != atlas:
                raise ValueError(
                    f"Inconsistent atlases: {d.atlas} vs {atlas}."
                )
            if sc_weight:
                matrices.append(d.get_sc(sc_weight))
                modality, kind = "sc", sc_weight
            else:
                matrices.append(d.get_fc(fc_kind, fisher_z=False))
                modality, kind = "fc", fc_kind
            subs.append(d.subject_id)

        obj = cls(
            np.array(matrices), subs, atlas, kind, modality,
            data_list[0].roi_labels,
        )
        if fisher_z and modality == "fc":
            obj.apply_fisher_z()
        return obj

    # -----------------------------------------------------------------
    # Transforms
    # -----------------------------------------------------------------
    def apply_fisher_z(self):
        """Apply Fisher z-transform in-place (arctanh)."""
        if self.is_fisher_z:
            warnings.warn("Fisher z-transform already applied.")
            return
        mats = self.matrices.copy()
        for i in range(self.n_subjects):
            np.fill_diagonal(mats[i], 0)
        mats = np.clip(mats, -0.9999, 0.9999)
        mats = np.arctanh(mats)
        for i in range(self.n_subjects):
            np.fill_diagonal(mats[i], 0)
        self.matrices = mats
        self.is_fisher_z = True

    def inverse_fisher_z(self):
        """Reverse Fisher z-transform in-place: r = tanh(z)."""
        if not self.is_fisher_z:
            warnings.warn("Data is not Fisher z-transformed.")
            return
        self.matrices = np.tanh(self.matrices)
        for i in range(self.n_subjects):
            np.fill_diagonal(self.matrices[i], 0)
        self.is_fisher_z = False

    # -----------------------------------------------------------------
    # Group statistics
    # -----------------------------------------------------------------
    def compute_group_stats(self) -> Dict[str, np.ndarray]:
        """Mean, std, median, SEM across subjects."""
        self.group_stats = {
            "mean": np.mean(self.matrices, axis=0),
            "std": np.std(self.matrices, axis=0, ddof=1),
            "median": np.median(self.matrices, axis=0),
            "sem": (np.std(self.matrices, axis=0, ddof=1)
                    / np.sqrt(self.n_subjects)),
            "n_subjects": self.n_subjects,
        }
        return self.group_stats

    def get_mean_matrix(self) -> np.ndarray:
        if self.group_stats is None:
            self.compute_group_stats()
        return self.group_stats["mean"]

    # -----------------------------------------------------------------
    # QC-FC correlation
    # -----------------------------------------------------------------
    def qc_fc(
        self, motion_values: np.ndarray, method: str = "spearman",
    ) -> Dict[str, Any]:
        """
        Compute QC-FC correlation to assess residual motion artifacts.

        After successful denoising (rsfmri_best_practices.pdf):
        - median |QC-FC| < 0.05
        - < 5 % edges significant at FDR q < 0.05

        Parameters
        ----------
        motion_values : np.ndarray, shape (n_subjects,)
            Per-subject mean framewise displacement (mm).
        method : str
            'spearman' (recommended) or 'pearson'.

        Returns
        -------
        dict with qc_fc_matrix, median_abs_qc_fc, pct_significant_fdr,
        passes_threshold.
        """
        from scipy import stats as sp_stats

        assert len(motion_values) == self.n_subjects
        n = self.n_rois
        qc_fc = np.zeros((n, n))
        p_vals = np.zeros((n, n))
        corr_fn = (sp_stats.spearmanr if method == "spearman"
                   else sp_stats.pearsonr)

        triu = np.triu_indices(n, k=1)
        for idx in range(len(triu[0])):
            i, j = triu[0][idx], triu[1][idx]
            r, p = corr_fn(motion_values, self.matrices[:, i, j])
            qc_fc[i, j] = qc_fc[j, i] = r
            p_vals[i, j] = p_vals[j, i] = p

        p_upper = p_vals[triu]
        r_upper = np.abs(qc_fc[triu])

        from .utils import fdr_correction
        reject, _ = fdr_correction(p_upper, alpha=0.05, method="bh")

        median_abs = float(np.median(r_upper))
        pct_fdr = float(np.mean(reject) * 100)

        self._qc_fc = {
            "qc_fc_matrix": qc_fc,
            "median_abs_qc_fc": median_abs,
            "pct_significant": float(np.mean(p_upper < 0.05) * 100),
            "pct_significant_fdr": pct_fdr,
            "passes_threshold": median_abs < 0.05 and pct_fdr < 5.0,
            "motion_values": motion_values,
            "method": method,
        }
        return self._qc_fc

    def qc_fc_distance_dependence(
        self, roi_coordinates: np.ndarray,
    ) -> Dict[str, float]:
        """
        Test distance-dependence of QC-FC.  |r| should be < 0.1.

        Parameters
        ----------
        roi_coordinates : np.ndarray, shape (N, 3)
            MNI coordinates per ROI.
        """
        from scipy import stats as sp_stats
        from scipy.spatial.distance import pdist, squareform

        if self._qc_fc is None:
            raise ValueError("Run qc_fc() first.")
        qc_fc_mat = self._qc_fc["qc_fc_matrix"]
        dist_mat = squareform(pdist(roi_coordinates, metric="euclidean"))
        triu = np.triu_indices(self.n_rois, k=1)
        r, p = sp_stats.spearmanr(dist_mat[triu], np.abs(qc_fc_mat[triu]))
        return {
            "r_distance_qcfc": float(r),
            "p_distance_qcfc": float(p),
            "passes": abs(r) < 0.1,
        }

    # -----------------------------------------------------------------
    # Outlier detection
    # -----------------------------------------------------------------
    def identify_outlier_subjects(
        self, threshold_sd: float = 3.0,
    ) -> Dict[str, Any]:
        """Flag subjects whose mean connectivity deviates > threshold SDs."""
        triu = np.triu_indices(self.n_rois, k=1)
        sub_means = np.array([
            np.mean(np.abs(self.matrices[s][triu]))
            for s in range(self.n_subjects)
        ])
        mu, sigma = np.mean(sub_means), np.std(sub_means, ddof=1)
        z = (sub_means - mu) / sigma if sigma > 0 else np.zeros_like(sub_means)
        mask = np.abs(z) > threshold_sd
        return {
            "outlier_mask": mask,
            "outlier_subjects": [
                self.subject_ids[i] for i in range(self.n_subjects) if mask[i]
            ],
            "subject_means": sub_means,
            "z_scores": z,
            "group_mean": float(mu),
            "group_std": float(sigma),
        }

    # -----------------------------------------------------------------
    # Subsetting
    # -----------------------------------------------------------------
    def exclude_subjects(self, to_exclude: List[str]) -> "GroupData":
        """Return a new GroupData without specified subjects."""
        keep = np.array([s not in to_exclude for s in self.subject_ids])
        return GroupData(
            self.matrices[keep],
            [s for s, k in zip(self.subject_ids, keep) if k],
            self.atlas, self.kind, self.modality, self.roi_labels,
        )

    def get_subject_matrix(self, subject_id: str) -> np.ndarray:
        idx = self.subject_ids.index(subject_id)
        return self.matrices[idx].copy()

    # -----------------------------------------------------------------
    # Strongest connections
    # -----------------------------------------------------------------
    def get_strongest_connections(
        self, n_top: int = 20, absolute: bool = True,
    ) -> pd.DataFrame:
        """Top N connections from the group mean matrix."""
        if self.group_stats is None:
            self.compute_group_stats()
        mat = self.group_stats["mean"]
        triu = np.triu_indices_from(mat, k=1)
        values = mat[triu]
        order = (np.argsort(np.abs(values))[::-1] if absolute
                 else np.argsort(values)[::-1])
        labels = self.roi_labels
        rows = []
        for i in order[:n_top]:
            r, c = triu[0][i], triu[1][i]
            rows.append({
                "region1": labels[r] if r < len(labels) else f"ROI_{r}",
                "region2": labels[c] if c < len(labels) else f"ROI_{c}",
                "region1_idx": int(r), "region2_idx": int(c),
                "value": float(values[i]),
                "abs_value": float(np.abs(values[i])),
            })
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    def summary(self) -> str:
        lines = [
            f"{'=' * 60}",
            f"  GroupData: {self.atlas} | {self.modality.upper()} ({self.kind})",
            f"{'=' * 60}",
            f"  Subjects   : {self.n_subjects}",
            f"  ROIs       : {self.n_rois}",
            f"  Shape      : {self.matrices.shape}",
            f"  Fisher-z   : {'✓' if self.is_fisher_z else '✗'}",
        ]
        if self._qc_fc:
            qc = self._qc_fc
            status = "✓" if qc["passes_threshold"] else "✗"
            lines.append(
                f"  QC-FC      : {status} median|r|="
                f"{qc['median_abs_qc_fc']:.3f}, "
                f"FDR sig={qc['pct_significant_fdr']:.1f}%"
            )
        lines.append(f"{'=' * 60}")
        text = "\n".join(lines)
        print(text)
        return text

    def __repr__(self):
        return (
            f"GroupData(atlas='{self.atlas}', kind='{self.kind}', "
            f"n={self.n_subjects}, rois={self.n_rois}, "
            f"fz={self.is_fisher_z})"
        )


# =============================================================================
# BATCH LOADING
# =============================================================================

def load_all_subjects(
    atlas: str,
    load_timeseries: bool = True,
    load_fc: bool = True,
    fc_kinds: List[str] = None,
    load_sc: bool = False,
    sc_weights: List[str] = None,
    strategy: str = None,
    subjects: List[str] = None,
    validate: bool = True,
    verbose: bool = True,
) -> Dict[str, ConnectivityData]:
    """
    Load data for all (or specified) subjects into ConnectivityData objects.

    Returns
    -------
    dict : {subject_id: ConnectivityData}
    """
    subjects = subjects or config.ALL_SUBJECT_IDS
    n_total = len(subjects)
    loaded, n_failed = {}, 0

    for i, sub in enumerate(subjects):
        if verbose:
            print(
                f"  [{i+1:>3}/{n_total}] Loading {sub} ({atlas})…",
                end="", flush=True,
            )
        try:
            cd = ConnectivityData.from_subject(
                sub, atlas,
                load_timeseries=load_timeseries, load_fc=load_fc,
                fc_kinds=fc_kinds, load_sc=load_sc,
                sc_weights=sc_weights, strategy=strategy,
                validate=validate,
            )
            loaded[sub] = cd
            print(f" {'✓' if cd.is_valid else '⚠'}" if verbose else "", end="")
            if verbose:
                print()
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f" ✗ {e}")

    if verbose:
        print(
            f"\n  Loaded: {len(loaded)}/{n_total} "
            f"({n_failed} failed)."
        )
    return loaded


def load_all_subjects_as_group(
    atlas: str,
    fc_kind: str = "correlation",
    fisher_z: bool = True,
    strategy: str = None,
    subjects: List[str] = None,
    verbose: bool = True,
) -> GroupData:
    """
    Convenience function: load FC matrices directly into a GroupData.
    """
    gd = GroupData.from_subjects(
        atlas=atlas, fc_kind=fc_kind, strategy=strategy,
        subjects=subjects, fisher_z=fisher_z,
    )
    if verbose:
        gd.summary()
    return gd


# =============================================================================
# ATLAS NETWORK ASSIGNMENT
# =============================================================================

def get_schaefer_network_indices(
    atlas: str = "schaefer_100",
) -> Dict[str, List[int]]:
    """
    Parse Schaefer labels → {network_full_name: [0-based ROI indices]}.

    Labels follow: "{Hemisphere}_{Network}_{Index}" pattern.
    """
    from . import io
    labels = io.get_roi_names(atlas)
    networks = {}
    for i, label in enumerate(labels):
        parts = label.split("_")
        if len(parts) >= 2:
            full = config.SCHAEFER_NETWORK_PREFIXES.get(parts[1], parts[1])
            networks.setdefault(full, []).append(i)
    return networks


def get_network_order(atlas: str = "schaefer_100") -> np.ndarray:
    """Sorting index that groups ROIs by Schaefer 7-network membership."""
    networks = get_schaefer_network_indices(atlas)
    order = []
    for net in config.SCHAEFER_NETWORK_PREFIXES.values():
        if net in networks:
            order.extend(networks[net])
    # remaining unmatched ROIs
    from . import io
    n = len(io.get_roi_names(atlas))
    assigned = set(order)
    order.extend(i for i in range(n) if i not in assigned)
    return np.array(order)


def get_network_boundaries(
    atlas: str = "schaefer_100",
) -> Tuple[List[int], List[str]]:
    """Cumulative boundary positions and names for network-ordered matrix."""
    networks = get_schaefer_network_indices(atlas)
    bounds, names = [0], []
    for net in config.SCHAEFER_NETWORK_PREFIXES.values():
        if net in networks:
            names.append(net)
            bounds.append(bounds[-1] + len(networks[net]))
    return bounds, names


def get_roi_hemisphere(atlas: str) -> np.ndarray:
    """
    Hemisphere assignment per ROI: 'L', 'R', or 'B' (midline).

    Heuristics: LH_/RH_ prefix (Schaefer), _L/_R suffix (AAL/BN),
    or Left/Right keywords.
    """
    from . import io
    labels = io.get_roi_names(atlas)
    hemi = []
    for label in labels:
        lu = label.upper()
        if (lu.startswith("LH_") or lu.endswith("_L")
                or "LEFT" in lu or lu.startswith("L_")):
            hemi.append("L")
        elif (lu.startswith("RH_") or lu.endswith("_R")
                or "RIGHT" in lu or lu.startswith("R_")):
            hemi.append("R")
        else:
            hemi.append("B")
    return np.array(hemi)


def get_roi_coordinates(atlas: str) -> np.ndarray:
    """
    MNI centroid coordinates per ROI from atlas NIfTI.

    Uses nilearn.plotting.find_parcellation_cut_coords.

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    info = config.ATLASES[atlas]
    if info.nifti_file is None or not info.nifti_file.exists():
        raise FileNotFoundError(
            f"Atlas NIfTI not found for '{atlas}': {info.nifti_file}"
        )
    from nilearn.plotting import find_parcellation_cut_coords
    return find_parcellation_cut_coords(str(info.nifti_file))


def get_interhemispheric_mask(atlas: str) -> np.ndarray:
    """Boolean (N, N) mask: True where i and j are in different hemispheres."""
    hemi = get_roi_hemisphere(atlas)
    n = len(hemi)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if hemi[i] != hemi[j] and "B" not in (hemi[i], hemi[j]):
                mask[i, j] = mask[j, i] = True
    return mask


# =============================================================================
# MATRIX OPERATIONS
# =============================================================================

def threshold_matrix(
    matrix: np.ndarray,
    method: str = "density",
    value: float = 0.1,
    binarize: bool = False,
    absolute: bool = False,
) -> np.ndarray:
    """
    Threshold a connectivity matrix.

    Follows van den Heuvel et al. (2017) for proportional thresholding.

    Parameters
    ----------
    matrix : np.ndarray (N, N)
    method : str
        'density'/'proportional': keep top ``value`` fraction of edges.
        'absolute': keep edges with weight > ``value``.
    value : float
    binarize : bool
    absolute : bool
        Threshold on absolute values.
    """
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    vals = np.abs(mat) if absolute else mat

    if method in ("density", "proportional"):
        triu_idx = np.triu_indices_from(mat, k=1)
        triu_vals = np.abs(mat[triu_idx]) if absolute else mat[triu_idx]
        n_keep = max(1, int(np.ceil(value * len(triu_vals))))
        thresh = np.sort(triu_vals)[::-1][min(n_keep - 1, len(triu_vals) - 1)]
        mask = vals >= thresh
    elif method == "absolute":
        mask = vals > value
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    mat[~mask] = 0
    mat = (mat + mat.T) / 2
    if binarize:
        mat = (mat != 0).astype(float)
    return mat


def threshold_matrix_negative(
    matrix: np.ndarray, strategy: str = "zero",
) -> np.ndarray:
    """
    Handle negative values in a connectivity matrix.

    Parameters
    ----------
    strategy : str
        'zero': set negatives to 0 (van den Heuvel et al., 2017).
        'absolute': take absolute value.
        'separate': return positive part only.
    """
    mat = matrix.copy()
    if strategy == "zero":
        mat[mat < 0] = 0
    elif strategy == "absolute":
        mat = np.abs(mat)
    elif strategy == "separate":
        mat[mat < 0] = 0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return mat


def symmetrize_matrix(matrix: np.ndarray, method: str = "average") -> np.ndarray:
    """Symmetrize: 'average', 'maximum', or 'minimum'."""
    if method == "average":
        return (matrix + matrix.T) / 2
    elif method == "maximum":
        return np.maximum(matrix, matrix.T)
    elif method == "minimum":
        return np.minimum(matrix, matrix.T)
    raise ValueError(f"Unknown method: {method}")


def normalize_matrix(matrix: np.ndarray, method: str = "max") -> np.ndarray:
    """
    Normalize a connectivity matrix.

    Methods: 'max', 'spectral', 'row_sum', 'minmax', 'log'.
    """
    mat = matrix.copy().astype(float)
    if method == "max":
        m = np.max(np.abs(mat))
        return mat / m if m > 0 else mat
    elif method == "spectral":
        sr = np.max(np.abs(np.linalg.eigvalsh(mat)))
        return mat / sr if sr > 0 else mat
    elif method == "row_sum":
        rs = mat.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return mat / rs
    elif method == "minmax":
        mn, mx = mat.min(), mat.max()
        rng = mx - mn
        return (mat - mn) / rng if rng > 0 else mat - mn
    elif method == "log":
        mat[mat > 0] = np.log1p(mat[mat > 0])
        return mat
    raise ValueError(f"Unknown method: {method}")


def get_upper_triangle(matrix: np.ndarray, k: int = 1) -> np.ndarray:
    """Extract upper triangle as 1-D vector."""
    return matrix[np.triu_indices_from(matrix, k=k)]


def upper_triangle_to_matrix(values: np.ndarray, n: int, k: int = 1) -> np.ndarray:
    """Reconstruct symmetric matrix from upper triangle vector."""
    mat = np.zeros((n, n))
    idx = np.triu_indices(n, k=k)
    mat[idx] = values
    return mat + mat.T


def compute_density(matrix: np.ndarray) -> float:
    """Proportion of non-zero edges in upper triangle."""
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    n = mat.shape[0]
    triu = np.triu_indices(n, k=1)
    n_possible = n * (n - 1) / 2
    return float(np.sum(mat[triu] != 0) / n_possible) if n_possible > 0 else 0.0


def compute_edge_weight_stats(matrix: np.ndarray) -> Dict[str, float]:
    """Descriptive statistics of non-zero edge weights."""
    from scipy import stats as sp_stats
    mat = matrix.copy()
    np.fill_diagonal(mat, 0)
    triu = np.triu_indices_from(mat, k=1)
    nz = mat[triu][mat[triu] != 0]
    n = mat.shape[0]
    n_possible = n * (n - 1) / 2
    out = {"n_edges": len(nz), "density": len(nz) / n_possible if n_possible else 0}
    if len(nz) > 0:
        out.update({
            "mean": float(np.mean(nz)), "std": float(np.std(nz)),
            "median": float(np.median(nz)),
            "min": float(np.min(nz)), "max": float(np.max(nz)),
            "skewness": float(sp_stats.skew(nz)),
            "kurtosis": float(sp_stats.kurtosis(nz)),
            "pct_negative": float(np.mean(nz < 0) * 100),
        })
    return out


# =============================================================================
# MATRIX COMPARISON UTILITIES
# =============================================================================

def compute_matrix_similarity(
    mat1: np.ndarray, mat2: np.ndarray, method: str = "pearson",
) -> Tuple[float, float]:
    """
    Similarity between two matrices (upper triangle comparison).

    Methods: 'pearson', 'spearman', 'cosine'.
    Returns (similarity, p_value).  p is NaN for cosine.
    """
    from scipy import stats as sp_stats
    triu = np.triu_indices_from(mat1, k=1)
    v1, v2 = mat1[triu], mat2[triu]
    if method == "pearson":
        return tuple(map(float, sp_stats.pearsonr(v1, v2)))
    elif method == "spearman":
        return tuple(map(float, sp_stats.spearmanr(v1, v2)))
    elif method == "cosine":
        from scipy.spatial.distance import cosine
        return float(1 - cosine(v1, v2)), np.nan
    raise ValueError(f"Unknown method: {method}")


def compute_fingerprinting_accuracy(
    sess1: np.ndarray, sess2: np.ndarray, method: str = "pearson",
) -> Dict[str, Any]:
    """
    Connectome fingerprinting accuracy (Finn et al., 2015, Nat Neurosci).

    For each subject in session 1, identifies most similar matrix in
    session 2.  Accuracy = fraction correctly identified.
    """
    n = sess1.shape[0]
    id_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            id_mat[i, j], _ = compute_matrix_similarity(
                sess1[i], sess2[j], method,
            )
    best = np.argmax(id_mat, axis=1)
    correct = best == np.arange(n)
    return {
        "accuracy": float(np.mean(correct)),
        "identification_matrix": id_mat,
        "correct_ids": correct,
        "best_match": best,
    }


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _validate_square_matrix(
    matrix: np.ndarray, name: str, check_symmetry: bool = True,
) -> List[str]:
    """Run validation checks on a square matrix."""
    notes = []
    if matrix.ndim != 2:
        notes.append(f"✗ {name}: expected 2D, got {matrix.ndim}D.")
        return notes
    if matrix.shape[0] != matrix.shape[1]:
        notes.append(
            f"✗ {name}: not square ({matrix.shape[0]}×{matrix.shape[1]})."
        )
        return notes
    n_nan = int(np.sum(np.isnan(matrix)))
    n_inf = int(np.sum(np.isinf(matrix)))
    if n_nan > 0:
        notes.append(f"⚠ {name}: {n_nan} NaN values.")
    if n_inf > 0:
        notes.append(f"⚠ {name}: {n_inf} Inf values.")
    if check_symmetry:
        asym = np.max(np.abs(matrix - matrix.T))
        if asym > 1e-6:
            notes.append(f"⚠ {name}: asymmetric (max diff = {asym:.2e}).")
    return notes


def _clean_square(mat: np.ndarray) -> np.ndarray:
    """Symmetrize and zero diagonal."""
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 0)
    return mat


def _load_subject_qc(
    subject_id: str, atlas: str, strategy: str,
) -> Dict:
    """Attempt to load QC metrics from fMRIPrep confounds or pipeline JSON."""
    qc = {}
    confounds_candidates = [
        config.DERIVATIVES_DIR / "fmriprep" / subject_id / "func" /
        f"{subject_id}_task-rest_desc-confounds_timeseries.tsv",
        config.DERIVATIVES_DIR / subject_id / "func" /
        f"{subject_id}_task-rest_desc-confounds_timeseries.tsv",
    ]
    for path in confounds_candidates:
        if path.exists():
            try:
                df = pd.read_csv(path, sep="\t")
                if "framewise_displacement" in df.columns:
                    fd = df["framewise_displacement"].dropna().values
                    qc["mean_fd"] = float(np.mean(fd))
                    qc["max_fd"] = float(np.max(fd))
                    qc["median_fd"] = float(np.median(fd))
                    fd_thresh = 0.2
                    qc["n_censored"] = int(np.sum(fd > fd_thresh))
                    qc["pct_censored"] = float(np.mean(fd > fd_thresh) * 100)
                break
            except Exception:
                pass

    qc_candidates = [
        config.RSFMRI_OUT / "qc" / atlas / strategy /
        subject_id / "qc_summary.json",
        config.RSFMRI_OUT / "qc" / subject_id / "qc_summary.json",
    ]
    for path in qc_candidates:
        if path.exists():
            try:
                with open(path) as f:
                    qc.update(json.load(f))
                break
            except Exception:
                pass
    return qc
