#!/usr/bin/env python3
"""
==============================================================================
GNN MULTIMODAL ANALYSIS - DATA LOADER
==============================================================================

Converts structural and functional connectivity matrices into
PyTorch Geometric Data objects suitable for GNN processing.

Handles:
- Loading SC matrices (streamline count, FA-weighted, etc.)
- Loading FC matrices (Pearson correlation, Fisher-z transformed)
- Constructing graph topology from SC
- Node feature engineering from SC/FC properties
- Multi-atlas support with dimension validation

Data Flow:
----------
    SC matrix (N×N) ──→ edge_index + edge_attr (structural graph)
    FC matrix (N×N) ──→ node features + regression targets
    Atlas labels    ──→ node metadata

Author: SARS-CoV-2 Neuroimaging Study
Date: February 2026
==============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings

import torch
from torch_geometric.data import Data, Dataset

from .config import (
    GNNMultimodalConfig,
    ATLAS_REGISTRY,
    SUBJECTS,
)

# Direct imports from sars.config for standalone function usage
from sars.config import (
    PROJECT_ROOT,
    CONNECTIVITY_DIR,
    ATLAS_DIR,
    ATLASES,
    DENOISING_STRATEGY,
    get_connectivity_path,
    get_sc_path,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# MATRIX LOADING UTILITIES
# ==============================================================================

def load_sc_matrix(
    subject: str,
    atlas: str,
    metric: str = "sift2",
    sc_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Load structural connectivity matrix for a subject/atlas combination.

    Uses ``sars.config.get_sc_path`` as the primary lookup, then falls
    back to searching common naming conventions.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-01').
    atlas : str
        Atlas name (e.g., 'schaefer_100').
    metric : str
        SC edge weight metric (maps to the ``weight`` parameter in
        ``sars.config.get_sc_path``).  Common values: 'sift2', 'count',
        'mean_fa', 'mean_length', 'invlength'.
    sc_dir : Path, optional
        Override base directory for SC outputs.

    Returns
    -------
    np.ndarray or None
        SC matrix of shape (N, N), or None if not found.
    """
    # Primary: use sars.config canonical path
    primary = get_sc_path(subject, atlas, weight=metric)
    search_paths = [primary]

    # Secondary: common alternative patterns
    if sc_dir is None:
        sc_dir = PROJECT_ROOT / "data" / "output" / "diffusion" / "subjects"

    atlas_dir_map = {
        "synthseg": "synthseg",
        "schaefer_100": "schaefer100",
        "aal3": "aal3",
        "brainnetome": "brainnetome",
    }
    atlas_dirname = atlas_dir_map.get(atlas, atlas)

    search_paths += [
        sc_dir / subject / "matrices" / atlas_dirname / f"connectivity_{metric}.npy",
        sc_dir / subject / "matrices" / atlas_dirname / f"{subject}_{atlas_dirname}_{metric}.csv",
        sc_dir / subject / "matrices" / atlas_dirname / f"{metric}.csv",
        sc_dir / subject / "matrices" / f"{atlas_dirname}_{metric}.csv",
    ]

    for fpath in search_paths:
        if fpath.exists():
            try:
                if fpath.suffix == ".csv":
                    mat = pd.read_csv(fpath, header=None).values
                elif fpath.suffix == ".npy":
                    mat = np.load(fpath)
                elif fpath.suffix == ".npz":
                    data = np.load(fpath)
                    mat = data[list(data.keys())[0]]
                else:
                    continue

                if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
                    logger.debug(f"Loaded SC: {fpath} ({mat.shape})")
                    return mat.astype(np.float32)
                else:
                    logger.warning(f"Non-square matrix at {fpath}: {mat.shape}")
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}")

    logger.warning(f"SC matrix not found for {subject}/{atlas}/{metric}")
    return None


def load_fc_matrix(
    subject: str,
    atlas: str,
    kind: str = "correlation",
    strategy: str = None,
    fc_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Load functional connectivity matrix for a subject/atlas combination.

    Uses ``sars.config.get_connectivity_path`` as the primary lookup.

    Parameters
    ----------
    subject : str
        Subject ID.
    atlas : str
        Atlas name.
    kind : str
        FC estimation kind ('correlation', 'partial', 'tangent').
    strategy : str, optional
        Denoising strategy (defaults to sars.config.DENOISING_STRATEGY).
    fc_dir : Path, optional
        Override base directory for FC outputs.

    Returns
    -------
    np.ndarray or None
        FC matrix of shape (N, N), or None if not found.
    """
    strategy = strategy or DENOISING_STRATEGY

    # Primary: use sars.config canonical path
    primary = get_connectivity_path(subject, atlas, kind=kind, strategy=strategy)
    search_paths = [primary]

    # Secondary: alternative patterns
    base = fc_dir or CONNECTIVITY_DIR
    search_paths += [
        base / atlas / strategy / subject / f"connectivity_{kind}.npy",
        base / subject / atlas / f"{kind}_correlation.npy",
        base / subject / atlas / f"{kind}_matrix.npy",
        base / subject / f"{atlas}_{kind}.npy",
    ]

    for fpath in search_paths:
        if fpath.exists():
            try:
                if fpath.suffix == ".npy":
                    mat = np.load(fpath)
                elif fpath.suffix == ".csv":
                    mat = pd.read_csv(fpath, header=None).values
                elif fpath.suffix == ".npz":
                    data = np.load(fpath)
                    mat = data[list(data.keys())[0]]
                else:
                    continue

                if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
                    logger.debug(f"Loaded FC: {fpath} ({mat.shape})")
                    return mat.astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}")

    logger.warning(f"FC matrix not found for {subject}/{atlas}/{kind}")
    return None


def load_atlas_labels(
    atlas: str,
    labels_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """
    Load atlas ROI labels.

    Parameters
    ----------
    atlas : str
        Atlas name.
    labels_dir : Path, optional
        Directory containing label files.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns 'value_roi' and 'label_roi'.
    """
    if labels_dir is None:
        from .config import _detect_project_root
        labels_dir = _detect_project_root() / "info" / "atlases"

    info = ATLAS_REGISTRY.get(atlas)
    if info is None:
        return None

    label_file = labels_dir / info["label_file"]
    if not label_file.exists():
        # Try project knowledge directory
        alt_paths = [
            Path("/mnt/project") / info["label_file"],
            labels_dir.parent / info["label_file"],
        ]
        for alt in alt_paths:
            if alt.exists():
                label_file = alt
                break

    if not label_file.exists():
        logger.warning(f"Label file not found: {label_file}")
        return None

    try:
        sep = "\t" if label_file.suffix == ".tsv" else ","
        return pd.read_csv(label_file, sep=sep)
    except Exception as e:
        logger.warning(f"Failed to load labels: {e}")
        return None


# ==============================================================================
# MATRIX PREPROCESSING
# ==============================================================================

def preprocess_sc(
    sc: np.ndarray,
    threshold: float = 0.0,
    log_transform: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess structural connectivity matrix.

    Steps:
    1. Ensure symmetry: SC = (SC + SC^T) / 2
    2. Remove self-connections (diagonal = 0)
    3. Threshold weak connections
    4. Log-transform: log(1 + SC) to compress dynamic range
    5. Row-normalize to unit sum

    Parameters
    ----------
    sc : np.ndarray
        Raw SC matrix (N, N).
    threshold : float
        Minimum value to retain.
    log_transform : bool
        Apply log(1 + x) transform.
    normalize : bool
        Row-normalize.

    Returns
    -------
    np.ndarray
        Preprocessed SC matrix.
    """
    sc = sc.copy()

    # Symmetrize
    sc = (sc + sc.T) / 2.0

    # Zero diagonal
    np.fill_diagonal(sc, 0.0)

    # Threshold
    sc[sc < threshold] = 0.0

    # Log transform
    if log_transform:
        sc = np.log1p(sc)

    # Row-normalize
    if normalize:
        row_sums = sc.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid division by zero
        sc = sc / row_sums

    return sc.astype(np.float32)


def preprocess_fc(
    fc: np.ndarray,
    fisher_z: bool = True,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Preprocess functional connectivity matrix.

    Steps:
    1. Ensure symmetry
    2. Clip to [-1, 1] (for Pearson correlation)
    3. Fisher-z transform: z = arctanh(r)
    4. Zero diagonal
    5. Optionally threshold weak connections

    Parameters
    ----------
    fc : np.ndarray
        Raw FC matrix (N, N).
    fisher_z : bool
        Apply Fisher-z transformation.
    threshold : float
        Minimum absolute value to retain.

    Returns
    -------
    np.ndarray
        Preprocessed FC matrix.
    """
    fc = fc.copy()

    # Symmetrize
    fc = (fc + fc.T) / 2.0

    # Zero diagonal
    np.fill_diagonal(fc, 0.0)

    # Fisher-z transform
    if fisher_z:
        fc = np.clip(fc, -0.9999, 0.9999)
        fc = np.arctanh(fc)

    # Threshold
    if threshold > 0:
        fc[np.abs(fc) < threshold] = 0.0

    return fc.astype(np.float32)


# ==============================================================================
# PyTorch Geometric DATA CONSTRUCTION
# ==============================================================================

def sc_to_edge_index(
    sc: np.ndarray,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert SC matrix to edge_index and edge_attr tensors.

    For the GAT, edges are defined by non-zero structural connections.
    Edge attributes carry the SC weights.

    Parameters
    ----------
    sc : np.ndarray
        Preprocessed SC matrix (N, N).
    threshold : float
        Minimum SC value to create an edge.

    Returns
    -------
    edge_index : torch.Tensor
        Edge indices [2, E] in COO format.
    edge_attr : torch.Tensor
        Edge weights [E, 1].
    """
    # Find non-zero entries
    rows, cols = np.where(sc > threshold)

    edge_index = torch.tensor(
        np.stack([rows, cols], axis=0), dtype=torch.long
    )
    edge_attr = torch.tensor(
        sc[rows, cols].reshape(-1, 1), dtype=torch.float32
    )

    return edge_index, edge_attr


def build_subject_graph(
    subject: str,
    atlas: str,
    config: GNNMultimodalConfig,
    sc_metric: str = "sift2_count",
) -> Optional[Data]:
    """
    Build a PyTorch Geometric Data object for one subject.

    The graph structure:
    - Nodes: brain regions (ROIs from the atlas)
    - Edges: structural connections (from SC matrix)
    - Node features: SC degree profile (row of preprocessed SC)
    - Target: FC matrix (to be predicted by GAT)

    Parameters
    ----------
    subject : str
        Subject ID.
    atlas : str
        Atlas name.
    config : GNNMultimodalConfig
        Pipeline configuration.
    sc_metric : str
        SC edge weight metric.

    Returns
    -------
    Data or None
        PyG Data object, or None if data unavailable.
    """
    # Load matrices
    sc_raw = load_sc_matrix(
        subject, atlas, metric=sc_metric, sc_dir=config.sc_dir
    )
    fc_raw = load_fc_matrix(
        subject, atlas, kind=config.fc_method, fc_dir=config.fc_dir
    )

    if sc_raw is None or fc_raw is None:
        logger.warning(f"Missing data for {subject}/{atlas}")
        return None

    # Validate dimensions match
    n_rois_expected = ATLAS_REGISTRY[atlas]["n_rois"]
    if sc_raw.shape[0] != fc_raw.shape[0]:
        logger.warning(
            f"Dimension mismatch for {subject}/{atlas}: "
            f"SC={sc_raw.shape}, FC={fc_raw.shape}"
        )
        # Attempt to reconcile by taking the minimum dimension
        n_min = min(sc_raw.shape[0], fc_raw.shape[0])
        sc_raw = sc_raw[:n_min, :n_min]
        fc_raw = fc_raw[:n_min, :n_min]

    n_rois = sc_raw.shape[0]

    # Preprocess
    sc = preprocess_sc(
        sc_raw,
        threshold=config.gat.sc_threshold,
        log_transform=config.gat.sc_log_transform,
        normalize=config.gat.sc_normalize,
    )
    fc = preprocess_fc(
        fc_raw,
        fisher_z=config.gat.fc_fisher_z,
        threshold=config.gat.fc_threshold,
    )

    # Build edges from SC
    edge_index, edge_attr = sc_to_edge_index(sc, threshold=1e-6)

    if edge_index.shape[1] == 0:
        logger.warning(f"No edges for {subject}/{atlas} after thresholding")
        return None

    # Node features: SC degree profile (each node's row of connections)
    # This captures the structural connectivity pattern of each region
    x = torch.tensor(sc, dtype=torch.float32)  # (N, N)

    # Target: FC matrix
    y = torch.tensor(fc, dtype=torch.float32)  # (N, N)

    # Raw matrices for analysis
    sc_raw_tensor = torch.tensor(
        preprocess_sc(sc_raw, log_transform=False, normalize=False),
        dtype=torch.float32,
    )

    # Build Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=n_rois,
    )

    # Metadata
    data.subject = subject
    data.atlas = atlas
    data.sc_metric = sc_metric
    data.n_rois = n_rois
    data.sc_raw = sc_raw_tensor

    return data


def build_paired_graphs(
    subject: str,
    atlas: str,
    config: GNNMultimodalConfig,
    sc_metric: str = "sift2_count",
) -> Optional[Tuple[Data, Data]]:
    """
    Build paired SC and FC graph representations for contrastive learning.

    For the contrastive framework, we need two separate graphs per subject:
    - SC graph: edges from structural connections, features from SC profile
    - FC graph: edges from functional correlations, features from FC profile

    Parameters
    ----------
    subject : str
        Subject ID.
    atlas : str
        Atlas name.
    config : GNNMultimodalConfig
        Configuration.
    sc_metric : str
        SC metric for edge weights.

    Returns
    -------
    Tuple[Data, Data] or None
        (sc_graph, fc_graph) pair, or None if unavailable.
    """
    # Load raw matrices
    sc_raw = load_sc_matrix(subject, atlas, metric=sc_metric, sc_dir=config.sc_dir)
    fc_raw = load_fc_matrix(subject, atlas, kind=config.fc_method, fc_dir=config.fc_dir)

    if sc_raw is None or fc_raw is None:
        return None

    # Reconcile dimensions
    if sc_raw.shape[0] != fc_raw.shape[0]:
        n_min = min(sc_raw.shape[0], fc_raw.shape[0])
        sc_raw = sc_raw[:n_min, :n_min]
        fc_raw = fc_raw[:n_min, :n_min]

    n_rois = sc_raw.shape[0]

    # Preprocess SC
    sc = preprocess_sc(sc_raw, log_transform=True, normalize=True)
    sc_edge_index, sc_edge_attr = sc_to_edge_index(sc, threshold=1e-6)

    # Preprocess FC — build graph from positive correlations
    fc = preprocess_fc(fc_raw, fisher_z=True)
    # For FC graph: use positive correlations above median as edges
    fc_pos = fc.copy()
    fc_pos[fc_pos <= 0] = 0
    median_fc = np.median(fc_pos[fc_pos > 0]) if np.any(fc_pos > 0) else 0
    fc_pos[fc_pos < median_fc] = 0
    fc_edge_index, fc_edge_attr = sc_to_edge_index(fc_pos, threshold=1e-6)

    if sc_edge_index.shape[1] == 0 or fc_edge_index.shape[1] == 0:
        return None

    # SC graph
    sc_graph = Data(
        x=torch.tensor(sc, dtype=torch.float32),
        edge_index=sc_edge_index,
        edge_attr=sc_edge_attr,
        num_nodes=n_rois,
    )
    sc_graph.subject = subject
    sc_graph.modality = "SC"

    # FC graph
    fc_graph = Data(
        x=torch.tensor(fc, dtype=torch.float32),
        edge_index=fc_edge_index,
        edge_attr=fc_edge_attr,
        num_nodes=n_rois,
    )
    fc_graph.subject = subject
    fc_graph.modality = "FC"

    return sc_graph, fc_graph


# ==============================================================================
# DATASET CLASSES
# ==============================================================================

class BrainConnectomeDataset(Dataset):
    """
    PyTorch Geometric Dataset for brain connectome analysis.

    Loads all subjects for a given atlas and constructs
    graph representations suitable for GAT SC→FC prediction.
    """

    def __init__(
        self,
        config: GNNMultimodalConfig,
        atlas: str,
        sc_metric: str = "sift2_count",
        subjects: Optional[List[str]] = None,
    ):
        super().__init__()
        self.config = config
        self.atlas = atlas
        self.sc_metric = sc_metric
        self.subjects = subjects or config.subjects

        # Pre-load all graphs
        self._graphs: List[Data] = []
        self._valid_subjects: List[str] = []

        for subj in self.subjects:
            graph = build_subject_graph(subj, atlas, config, sc_metric)
            if graph is not None:
                self._graphs.append(graph)
                self._valid_subjects.append(subj)

        logger.info(
            f"BrainConnectomeDataset: {len(self._graphs)}/{len(self.subjects)} "
            f"subjects loaded for {atlas}"
        )

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx]

    @property
    def valid_subjects(self) -> List[str]:
        return self._valid_subjects

    @property
    def n_rois(self) -> int:
        if self._graphs:
            return self._graphs[0].n_rois
        return ATLAS_REGISTRY[self.atlas]["n_rois"]


class MultimodalPairedDataset(Dataset):
    """
    Dataset providing paired SC-FC graphs for contrastive learning.

    Each sample is a tuple of (sc_graph, fc_graph) from the same subject.
    """

    def __init__(
        self,
        config: GNNMultimodalConfig,
        atlas: str,
        sc_metric: str = "sift2_count",
        subjects: Optional[List[str]] = None,
    ):
        super().__init__()
        self.config = config
        self.atlas = atlas

        self._pairs: List[Tuple[Data, Data]] = []
        self._valid_subjects: List[str] = []

        subjects = subjects or config.subjects
        for subj in subjects:
            result = build_paired_graphs(subj, atlas, config, sc_metric)
            if result is not None:
                self._pairs.append(result)
                self._valid_subjects.append(subj)

        logger.info(
            f"MultimodalPairedDataset: {len(self._pairs)}/{len(subjects)} "
            f"pairs loaded for {atlas}"
        )

    def len(self) -> int:
        return len(self._pairs)

    def get(self, idx: int) -> Tuple[Data, Data]:
        return self._pairs[idx]

    @property
    def valid_subjects(self) -> List[str]:
        return self._valid_subjects
