# -*- coding: utf-8 -*-
"""
sars.reservoir.reservoir_dynamics
=================================

Reservoir Computing Module for SC-FC Decoupling Analysis
=========================================================

This module implements Echo State Network (ESN)-based methods for quantifying
and simulating structural-functional connectivity (SC-FC) coupling and
decoupling in brain networks, with specific applications to COVID-19
neuroimaging studies.

Theoretical Foundations
-----------------------
The relationship between structural connectivity (SC) and functional
connectivity (FC) is non-trivial: FC emerges from neural dynamics propagating
through the SC backbone, but many strong functional connections exist between
regions without direct structural links (Sporns, 2013). This module leverages
reservoir computing — specifically Echo State Networks — to model this
structure-function relationship.

The core insight is that an ESN whose reservoir weight matrix is derived from
the empirical structural connectome can simulate functional dynamics. The
discrepancy between ESN-simulated FC and empirical FC constitutes a principled
measure of SC-FC decoupling (Suárez et al., Nature Communications, 2024).

Additionally, the Structural Decoupling Index (SDI) is computed via graph
signal processing (GSP), decomposing functional signals into structurally
coupled and decoupled components (Preti & Van De Ville, 2019; Dong et al.,
Human Brain Mapping, 2024).

Key Features
------------
- SC-FC coupling quantification via reservoir-simulated FC
- Structural Decoupling Index (SDI) via graph signal processing
- Regional and network-level decoupling profiles
- Perturbation (virtual lesion) analysis
- Group comparison (COVID vs. Control) with statistical testing
- Multi-atlas support (SynthSeg 86, Schaefer 100, AAL3 170, Brainnetome 246)
- Hebbian Adaptive Reservoir dynamics (HAG-inspired)

References
----------
- Suárez et al. (2024). Connectome-based reservoir computing with the
  conn2res toolbox. Nature Communications, 15.
- Preti & Van De Ville (2019). Decoupling of brain function from structure
  reveals regional behavioral specialization in humans. Nature Communications.
- Dong et al. (2024). How brain structure-function decoupling supports
  individual cognition. Human Brain Mapping, 45(2).
- Sporns (2013). Structure and function of complex brain networks. Dialogues
  in Clinical Neuroscience, 15(3), 247-262.
- Takahashi et al. (2025). The Role of Brain Connectivity Patterns in
  Applying Connectome-Based Reservoir Computing. IJCNN.
- Nature Communications (Dec 2025). Hebbian architecture for adaptive
  reservoir computing (HAG).

Author: Velho Mago
Project: SARS-CoV-2 Brain Connectivity Analysis Library
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg, stats, sparse
from scipy.spatial.distance import squareform
from scipy.signal import detrend
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    import reservoirpy as rpy
    from reservoirpy.nodes import Reservoir, Ridge as RPYRidge, Input
    HAS_RESERVOIRPY = True
except ImportError:
    HAS_RESERVOIRPY = False
    warnings.warn(
        "ReservoirPy not installed. Install with: pip install reservoirpy. "
        "Some functionality will use fallback NumPy-based ESN.",
        ImportWarning,
    )

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SCFCCouplingResult:
    """Results from SC-FC coupling analysis via reservoir computing.

    Attributes
    ----------
    coupling_global : float
        Global SC-FC coupling (Pearson correlation between SC and empirical FC).
    coupling_reservoir : float
        Reservoir-predicted SC-FC coupling (correlation between simulated and
        empirical FC).
    decoupling_index_global : float
        Global decoupling = 1 - coupling_reservoir. Higher values indicate that
        empirical FC deviates more from what the SC-constrained reservoir predicts.
    regional_coupling : np.ndarray
        Per-node coupling strength (correlation between row of simulated FC and
        row of empirical FC).
    regional_decoupling : np.ndarray
        Per-node decoupling index = 1 - regional_coupling.
    fc_simulated : np.ndarray
        Reservoir-simulated functional connectivity matrix.
    fc_empirical : np.ndarray
        Empirical functional connectivity matrix.
    sc_matrix : np.ndarray
        Input structural connectivity matrix.
    reservoir_states : Optional[np.ndarray]
        Internal reservoir state trajectories (T x N_nodes), if retained.
    spectral_radius : float
        Spectral radius used for reservoir construction.
    prediction_r2 : float
        R² of the readout layer predicting empirical FC from reservoir states.
    memory_capacity : Optional[float]
        Memory capacity of the reservoir, if computed.
    network_coupling : Optional[Dict[str, float]]
        Per-network coupling values if atlas network labels are provided.
    network_decoupling : Optional[Dict[str, float]]
        Per-network decoupling values.
    """
    coupling_global: float
    coupling_reservoir: float
    decoupling_index_global: float
    regional_coupling: np.ndarray
    regional_decoupling: np.ndarray
    fc_simulated: np.ndarray
    fc_empirical: np.ndarray
    sc_matrix: np.ndarray
    reservoir_states: Optional[np.ndarray] = None
    spectral_radius: float = 0.9
    prediction_r2: float = 0.0
    memory_capacity: Optional[float] = None
    network_coupling: Optional[Dict[str, float]] = None
    network_decoupling: Optional[Dict[str, float]] = None


@dataclass
class SDIResult:
    """Results from Structural Decoupling Index (SDI) analysis.

    The SDI quantifies, for each brain region, the extent to which its
    functional connectivity profile depends on the underlying structural
    connectome. Low SDI indicates tight SC-FC coupling (primary sensorimotor
    regions); high SDI indicates decoupling (association cortex).

    Attributes
    ----------
    sdi : np.ndarray
        Regional SDI values (N_nodes,). Normalized to [0, 1].
    sdi_raw : np.ndarray
        Raw (unnormalized) SDI values.
    fc_coupled : np.ndarray
        Structurally coupled component of FC (N_nodes x N_nodes).
    fc_decoupled : np.ndarray
        Structurally decoupled component of FC (N_nodes x N_nodes).
    sc_eigenvalues : np.ndarray
        Eigenvalues of the SC graph Laplacian.
    sc_eigenvectors : np.ndarray
        Eigenvectors of the SC graph Laplacian.
    coupling_ratio : float
        Global ratio of coupled to total FC energy.
    n_coupled_components : int
        Number of low-frequency graph eigenmodes used for coupled component.
    network_sdi : Optional[Dict[str, float]]
        Mean SDI per functional network, if labels provided.
    gradient_correlation : Optional[float]
        Correlation of SDI with the principal functional gradient.
    """
    sdi: np.ndarray
    sdi_raw: np.ndarray
    fc_coupled: np.ndarray
    fc_decoupled: np.ndarray
    sc_eigenvalues: np.ndarray
    sc_eigenvectors: np.ndarray
    coupling_ratio: float
    n_coupled_components: int
    network_sdi: Optional[Dict[str, float]] = None
    gradient_correlation: Optional[float] = None


@dataclass
class PerturbationResult:
    """Results from virtual lesion / perturbation analysis.

    Attributes
    ----------
    baseline_coupling : float
        SC-FC coupling before perturbation.
    perturbed_coupling : np.ndarray
        SC-FC coupling after removing each node (N_nodes,).
    coupling_change : np.ndarray
        Change in coupling per node removal (positive = coupling decreased).
    vulnerability_index : np.ndarray
        Normalized vulnerability: how much each node's removal disrupts SC-FC
        coupling. Higher = more critical for maintaining coupling.
    most_vulnerable_nodes : np.ndarray
        Indices of top-k most vulnerable nodes.
    most_resilient_nodes : np.ndarray
        Indices of top-k most resilient nodes.
    edge_vulnerability : Optional[np.ndarray]
        If edge-level perturbation was performed, vulnerability per edge.
    """
    baseline_coupling: float
    perturbed_coupling: np.ndarray
    coupling_change: np.ndarray
    vulnerability_index: np.ndarray
    most_vulnerable_nodes: np.ndarray
    most_resilient_nodes: np.ndarray
    edge_vulnerability: Optional[np.ndarray] = None


@dataclass
class GroupComparisonResult:
    """Results from group comparison of SC-FC coupling metrics.

    Attributes
    ----------
    group1_name : str
        Name of group 1 (e.g., "COVID").
    group2_name : str
        Name of group 2 (e.g., "Control").
    regional_tstat : np.ndarray
        T-statistic per region for coupling difference.
    regional_pvalue : np.ndarray
        Uncorrected p-value per region.
    regional_pvalue_fdr : np.ndarray
        FDR-corrected p-values (Benjamini-Hochberg).
    significant_regions : np.ndarray
        Boolean mask of significant regions after FDR correction.
    effect_size_cohen_d : np.ndarray
        Cohen's d effect size per region.
    group1_mean : np.ndarray
        Mean coupling/SDI per region for group 1.
    group2_mean : np.ndarray
        Mean coupling/SDI per region for group 2.
    global_tstat : float
        T-statistic for global coupling difference.
    global_pvalue : float
        P-value for global coupling difference.
    network_results : Optional[Dict[str, Dict]]
        Per-network group comparison results.
    """
    group1_name: str
    group2_name: str
    regional_tstat: np.ndarray
    regional_pvalue: np.ndarray
    regional_pvalue_fdr: np.ndarray
    significant_regions: np.ndarray
    effect_size_cohen_d: np.ndarray
    group1_mean: np.ndarray
    group2_mean: np.ndarray
    global_tstat: float
    global_pvalue: float
    network_results: Optional[Dict[str, Dict]] = None


@dataclass
class ReservoirDynamicsResult:
    """Comprehensive results from full reservoir dynamics analysis pipeline.

    Attributes
    ----------
    coupling : SCFCCouplingResult
        SC-FC coupling results from reservoir simulation.
    sdi : SDIResult
        Structural Decoupling Index results.
    perturbation : Optional[PerturbationResult]
        Virtual lesion results, if computed.
    lyapunov_exponent : Optional[float]
        Estimated maximal Lyapunov exponent of reservoir dynamics.
    edge_of_chaos_distance : Optional[float]
        Distance of the reservoir operating point from the edge of chaos.
    reservoir_entropy : Optional[float]
        Shannon entropy of the reservoir state distribution.
    atlas_name : str
        Name of the atlas used.
    n_regions : int
        Number of brain regions.
    """
    coupling: SCFCCouplingResult
    sdi: SDIResult
    perturbation: Optional[PerturbationResult] = None
    lyapunov_exponent: Optional[float] = None
    edge_of_chaos_distance: Optional[float] = None
    reservoir_entropy: Optional[float] = None
    atlas_name: str = "unknown"
    n_regions: int = 0


# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def _validate_connectivity_matrix(
    matrix: np.ndarray,
    name: str = "matrix",
    symmetric: bool = False,
    nonneg: bool = False,
) -> np.ndarray:
    """Validate and clean a connectivity matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input connectivity matrix.
    name : str
        Name for error messages.
    symmetric : bool
        If True, enforce symmetry by averaging with transpose.
    nonneg : bool
        If True, ensure all values are non-negative.

    Returns
    -------
    np.ndarray
        Validated matrix.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {matrix.shape}")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"{name} must be square, got shape {matrix.shape}"
        )
    if np.any(np.isnan(matrix)):
        warnings.warn(f"{name} contains NaN values; replacing with 0.")
        matrix = np.nan_to_num(matrix, nan=0.0)
    if symmetric:
        matrix = (matrix + matrix.T) / 2.0
    if nonneg:
        matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _normalize_sc_for_reservoir(
    sc: np.ndarray,
    spectral_radius: float = 0.9,
    normalize_method: str = "spectral",
) -> np.ndarray:
    """Normalize structural connectivity matrix for use as reservoir weights.

    The normalization ensures the echo state property (ESP) is satisfied,
    which requires that the effect of previous reservoir states vanishes
    over time. The standard approach scales the weight matrix such that
    its spectral radius (largest absolute eigenvalue) is below or at
    the desired value.

    Parameters
    ----------
    sc : np.ndarray
        Structural connectivity matrix (N x N).
    spectral_radius : float
        Target spectral radius. Values near 1.0 operate at the edge of
        chaos, maximizing computational capacity. Default 0.9.
    normalize_method : str
        Normalization method:
        - "spectral": Scale by largest eigenvalue magnitude (standard).
        - "norm": Scale by matrix norm.
        - "log": Apply log(1 + x) transformation before spectral scaling.

    Returns
    -------
    np.ndarray
        Normalized weight matrix with target spectral radius.

    Notes
    -----
    Operating near spectral_radius = 1.0 places the reservoir at the
    "edge of chaos", which theoretical and empirical work suggests
    maximizes memory capacity and computational power (Woo et al., 2024).
    For connectome-based reservoirs, Suárez et al. (2024) demonstrated
    that the biological topology confers robustness to increasing
    spectral radius compared to random null networks.
    """
    sc = sc.copy()
    np.fill_diagonal(sc, 0.0)

    if normalize_method == "log":
        sc = np.log1p(sc)
    elif normalize_method == "norm":
        norm = np.linalg.norm(sc)
        if norm > 0:
            sc = sc / norm * spectral_radius
        return sc

    # Spectral normalization (default)
    eigenvalues = linalg.eigvals(sc)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    if max_eigenvalue > 0:
        sc = sc * (spectral_radius / max_eigenvalue)
    else:
        warnings.warn(
            "SC matrix has zero spectral radius; returning unnormalized."
        )

    return sc


def _compute_graph_laplacian(
    sc: np.ndarray,
    normalized: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the graph Laplacian and its eigendecomposition.

    Parameters
    ----------
    sc : np.ndarray
        Symmetric structural connectivity matrix (N x N).
    normalized : bool
        If True, compute the normalized (symmetric) Laplacian:
        L_sym = D^{-1/2} L D^{-1/2} where L = D - W.
        If False, compute the combinatorial Laplacian L = D - W.

    Returns
    -------
    laplacian : np.ndarray
        Graph Laplacian matrix (N x N).
    eigenvalues : np.ndarray
        Sorted eigenvalues (ascending).
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns).

    Notes
    -----
    The graph Laplacian eigenvectors serve as the "Fourier basis" for
    graph signal processing. Low-frequency eigenmodes (small eigenvalues)
    capture smooth variations aligned with the structural topology,
    while high-frequency eigenmodes capture rapid variations that deviate
    from the structural backbone.
    """
    sc = _validate_connectivity_matrix(sc, "SC for Laplacian", symmetric=True, nonneg=True)
    n = sc.shape[0]

    degree = np.sum(sc, axis=1)
    D = np.diag(degree)
    L = D - sc  # Combinatorial Laplacian

    if normalized:
        # Symmetric normalized Laplacian: L_sym = D^{-1/2} L D^{-1/2}
        with np.errstate(divide="ignore", invalid="ignore"):
            d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = D_inv_sqrt @ L @ D_inv_sqrt

    # Eigendecomposition (symmetric matrix → real eigenvalues)
    eigenvalues, eigenvectors = linalg.eigh(L)

    # Ensure non-negative eigenvalues (numerical precision)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    return L, eigenvalues, eigenvectors


def _fisher_z(fc: np.ndarray) -> np.ndarray:
    """Apply Fisher z-transformation to a correlation matrix.

    Parameters
    ----------
    fc : np.ndarray
        Correlation matrix with values in [-1, 1].

    Returns
    -------
    np.ndarray
        Fisher z-transformed matrix.
    """
    fc = np.clip(fc, -0.9999, 0.9999)
    return np.arctanh(fc)


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Extract upper triangle of a matrix (excluding diagonal).

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix.

    Returns
    -------
    np.ndarray
        Flattened upper triangle values.
    """
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


# =============================================================================
# STRUCTURAL DECOUPLING INDEX (SDI)
# =============================================================================

def compute_sdi(
    sc: np.ndarray,
    fc: np.ndarray,
    n_components: Optional[int] = None,
    frequency_threshold: float = 0.5,
    normalize: bool = True,
    network_labels: Optional[np.ndarray] = None,
    network_names: Optional[Dict[int, str]] = None,
) -> SDIResult:
    """Compute the Structural Decoupling Index (SDI) via graph signal processing.

    The SDI quantifies, for each brain region, how much its functional
    connectivity depends on the underlying structural connectome. This is
    achieved by decomposing the FC signal into structurally coupled (low
    graph-frequency) and structurally decoupled (high graph-frequency)
    components using the eigenmodes of the SC graph Laplacian.

    Mathematically, let U be the matrix of SC Laplacian eigenvectors and
    Λ the corresponding eigenvalues. The FC matrix F can be decomposed as:
        F̂ = U^T F  (graph Fourier transform)
    The coupled component uses only the first k low-frequency eigenmodes:
        F_coupled = U_k U_k^T F
    The decoupled component is the remainder:
        F_decoupled = F - F_coupled

    The SDI for node i is defined as the ratio of decoupled to total
    energy in that node's connectivity profile:
        SDI_i = ||F_decoupled[i,:]||² / ||F[i,:]||²

    Parameters
    ----------
    sc : np.ndarray
        Structural connectivity matrix (N x N), symmetric, non-negative.
    fc : np.ndarray
        Functional connectivity matrix (N x N), symmetric.
    n_components : int, optional
        Number of low-frequency eigenmodes for the coupled component.
        If None, determined by frequency_threshold.
    frequency_threshold : float
        Fraction of eigenmodes to consider as "low frequency" (coupled).
        Default 0.5 (lower half of spectrum).
    normalize : bool
        If True, normalize SDI values to [0, 1].
    network_labels : np.ndarray, optional
        Integer labels assigning each node to a functional network.
    network_names : dict, optional
        Mapping from network label integers to network names.

    Returns
    -------
    SDIResult
        Complete SDI analysis results.

    Notes
    -----
    The SDI was introduced by Preti & Van De Ville (2019) and has been
    shown to follow a hierarchical gradient across the cortex: primary
    sensorimotor regions exhibit low SDI (tight SC-FC coupling), while
    association cortex (frontoparietal, default mode, limbic) exhibits
    high SDI (loose coupling) (Dong et al., 2024). This gradient
    correlates with the principal functional gradient (Margulies et al.,
    2016) and with neurotransmitter receptor distributions (D2, NET,
    MOR, mGluR5).

    In the context of COVID-19, altered SDI patterns may reflect
    neuroinflammation-induced disruption of structure-function relationships.
    """
    sc = _validate_connectivity_matrix(sc, "SC", symmetric=True, nonneg=True)
    fc = _validate_connectivity_matrix(fc, "FC", symmetric=True)
    n = sc.shape[0]

    if sc.shape != fc.shape:
        raise ValueError(
            f"SC and FC must have same shape: SC={sc.shape}, FC={fc.shape}"
        )

    # Step 1: Compute SC graph Laplacian eigendecomposition
    _, eigenvalues, eigenvectors = _compute_graph_laplacian(sc, normalized=True)

    # Step 2: Determine split point between coupled and decoupled
    if n_components is None:
        n_components = max(1, int(np.ceil(n * frequency_threshold)))
    n_components = min(n_components, n)

    # Step 3: Graph Fourier decomposition of FC
    # U_k = first k eigenvectors (low frequency / coupled)
    U_low = eigenvectors[:, :n_components]   # (N x k)
    U_high = eigenvectors[:, n_components:]  # (N x (N-k))

    # Coupled component: projection onto low-frequency subspace
    # F_coupled = U_low @ U_low^T @ F
    fc_coupled = U_low @ (U_low.T @ fc)

    # Decoupled component
    fc_decoupled = fc - fc_coupled

    # Step 4: Compute regional SDI
    # Energy of decoupled component relative to total, per node
    total_energy = np.sum(fc ** 2, axis=1)
    decoupled_energy = np.sum(fc_decoupled ** 2, axis=1)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        sdi_raw = np.where(total_energy > 0, decoupled_energy / total_energy, 0.0)

    # Normalize to [0, 1]
    if normalize and (sdi_raw.max() - sdi_raw.min()) > 0:
        sdi = (sdi_raw - sdi_raw.min()) / (sdi_raw.max() - sdi_raw.min())
    else:
        sdi = sdi_raw.copy()

    # Global coupling ratio
    total_fc_energy = np.sum(fc ** 2)
    coupled_fc_energy = np.sum(fc_coupled ** 2)
    coupling_ratio = (
        coupled_fc_energy / total_fc_energy if total_fc_energy > 0 else 0.0
    )

    # Per-network SDI
    network_sdi = None
    if network_labels is not None:
        network_sdi = {}
        unique_labels = np.unique(network_labels)
        for label in unique_labels:
            mask = network_labels == label
            name = (
                network_names.get(label, f"Network_{label}")
                if network_names
                else f"Network_{label}"
            )
            network_sdi[name] = float(np.mean(sdi[mask]))

    return SDIResult(
        sdi=sdi,
        sdi_raw=sdi_raw,
        fc_coupled=fc_coupled,
        fc_decoupled=fc_decoupled,
        sc_eigenvalues=eigenvalues,
        sc_eigenvectors=eigenvectors,
        coupling_ratio=coupling_ratio,
        n_coupled_components=n_components,
        network_sdi=network_sdi,
    )


# =============================================================================
# ECHO STATE NETWORK — NUMPY FALLBACK
# =============================================================================

class _NumpyESN:
    """Minimal Echo State Network implementation as fallback.

    This class provides a basic ESN when ReservoirPy is not available.
    It implements the standard leaky-integrator ESN dynamics:

        x(t) = (1 - α) * x(t-1) + α * tanh(W_in * u(t) + W * x(t-1))

    where x is the reservoir state, u is input, W_in is the input weight
    matrix, W is the recurrent weight matrix, and α is the leaking rate.

    Parameters
    ----------
    W : np.ndarray
        Reservoir weight matrix (N x N), already scaled to desired
        spectral radius.
    input_scaling : float
        Scaling factor for input weights.
    leaking_rate : float
        Leaking rate α ∈ (0, 1]. Controls the speed of reservoir dynamics.
    noise_level : float
        Amplitude of additive noise injected into reservoir dynamics.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        W: np.ndarray,
        input_scaling: float = 0.1,
        leaking_rate: float = 0.3,
        noise_level: float = 0.01,
        seed: int = 42,
    ):
        self.W = W.copy()
        self.n_nodes = W.shape[0]
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

        # Input weights: sparse random
        self.W_in = self.rng.uniform(
            -input_scaling, input_scaling, (self.n_nodes, 1)
        )

    def run(
        self,
        input_signal: np.ndarray,
        washout: int = 100,
    ) -> np.ndarray:
        """Drive the reservoir with an input signal.

        Parameters
        ----------
        input_signal : np.ndarray
            Input signal of shape (T,) or (T, n_inputs).
        washout : int
            Number of initial timesteps to discard.

        Returns
        -------
        np.ndarray
            Reservoir states of shape (T - washout, N_nodes).
        """
        if input_signal.ndim == 1:
            input_signal = input_signal[:, np.newaxis]

        T = input_signal.shape[0]
        n_inputs = input_signal.shape[1]

        # Adjust W_in if needed
        if self.W_in.shape[1] != n_inputs:
            self.W_in = self.rng.uniform(
                -self.input_scaling,
                self.input_scaling,
                (self.n_nodes, n_inputs),
            )

        states = np.zeros((T, self.n_nodes))
        x = np.zeros(self.n_nodes)

        for t in range(T):
            pre = self.W_in @ input_signal[t] + self.W @ x
            noise = self.noise_level * self.rng.standard_normal(self.n_nodes)
            x = (
                (1 - self.leaking_rate) * x
                + self.leaking_rate * np.tanh(pre + noise)
            )
            states[t] = x

        return states[washout:]

    def run_autonomous(
        self,
        n_steps: int,
        initial_state: Optional[np.ndarray] = None,
        noise_level: Optional[float] = None,
    ) -> np.ndarray:
        """Run the reservoir in autonomous mode (no external input).

        This mode uses noise as the sole driving force, simulating
        spontaneous (resting-state) dynamics on the structural backbone.

        Parameters
        ----------
        n_steps : int
            Number of timesteps to simulate.
        initial_state : np.ndarray, optional
            Initial reservoir state. If None, uses random initialization.
        noise_level : float, optional
            Override noise amplitude.

        Returns
        -------
        np.ndarray
            Reservoir states (n_steps, N_nodes).
        """
        if noise_level is None:
            noise_level = self.noise_level

        states = np.zeros((n_steps, self.n_nodes))
        x = (
            initial_state.copy()
            if initial_state is not None
            else self.rng.standard_normal(self.n_nodes) * 0.01
        )

        for t in range(n_steps):
            noise = noise_level * self.rng.standard_normal(self.n_nodes)
            x = (
                (1 - self.leaking_rate) * x
                + self.leaking_rate * np.tanh(self.W @ x + noise)
            )
            states[t] = x

        return states


# =============================================================================
# CONNECTOME RESERVOIR CLASS
# =============================================================================

class ConnectomeReservoir:
    """Echo State Network with reservoir topology derived from a brain connectome.

    This class implements the core conn2res paradigm (Suárez et al., 2024):
    the structural connectome is used as the reservoir weight matrix of an
    ESN, allowing the network topology to directly shape computational
    dynamics. This provides a principled framework for studying how brain
    structure constrains and enables function.

    The reservoir can be driven by:
    1. External input (e.g., task stimuli) to assess computational capacity
    2. Noise alone (autonomous mode) to simulate resting-state dynamics
    3. Empirical BOLD time series to learn structure-function mappings

    Parameters
    ----------
    sc_matrix : np.ndarray
        Structural connectivity matrix (N x N). Can be weighted or binary.
    spectral_radius : float
        Target spectral radius for the reservoir weight matrix. Values
        near 1.0 place the reservoir at the edge of chaos. Default 0.9.
    input_scaling : float
        Scaling for the input weight matrix. Default 0.1.
    leaking_rate : float
        Leaking rate α ∈ (0, 1] for the leaky integrator dynamics.
        Lower values create slower, more memory-retaining dynamics.
        Default 0.3.
    noise_level : float
        Amplitude of additive noise. Default 0.01.
    normalize_method : str
        Method for SC normalization: "spectral", "norm", or "log".
    use_reservoirpy : bool
        If True and ReservoirPy is installed, use it as backend.
    seed : int
        Random seed for reproducibility.

    Attributes
    ----------
    W : np.ndarray
        Normalized reservoir weight matrix.
    n_nodes : int
        Number of brain regions (reservoir nodes).
    esn : object
        Underlying ESN implementation.

    Examples
    --------
    >>> import numpy as np
    >>> sc = np.random.rand(86, 86)  # Example SC matrix
    >>> sc = (sc + sc.T) / 2
    >>> np.fill_diagonal(sc, 0)
    >>> reservoir = ConnectomeReservoir(sc, spectral_radius=0.95)
    >>> states = reservoir.simulate_resting_state(n_steps=1000, washout=200)
    >>> fc_sim = np.corrcoef(states.T)
    """

    def __init__(
        self,
        sc_matrix: np.ndarray,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        leaking_rate: float = 0.3,
        noise_level: float = 0.01,
        normalize_method: str = "spectral",
        use_reservoirpy: bool = True,
        seed: int = 42,
    ):
        self.sc_raw = _validate_connectivity_matrix(
            sc_matrix, "SC matrix", symmetric=True, nonneg=True
        )
        self.n_nodes = self.sc_raw.shape[0]
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.noise_level = noise_level
        self.seed = seed

        # Normalize SC for reservoir
        self.W = _normalize_sc_for_reservoir(
            self.sc_raw, spectral_radius, normalize_method
        )

        # Verify spectral radius
        actual_sr = np.max(np.abs(linalg.eigvals(self.W)))
        logger.info(
            f"ConnectomeReservoir initialized: {self.n_nodes} nodes, "
            f"target ρ={spectral_radius:.3f}, actual ρ={actual_sr:.3f}"
        )

        # Create ESN backend
        self._use_rpy = use_reservoirpy and HAS_RESERVOIRPY
        if self._use_rpy:
            rpy.set_seed(seed)
            self.esn = Reservoir(
                units=self.n_nodes,
                sr=spectral_radius,
                lr=leaking_rate,
                input_scaling=input_scaling,
                W=self.W,
                seed=seed,
            )
        else:
            self.esn = _NumpyESN(
                W=self.W,
                input_scaling=input_scaling,
                leaking_rate=leaking_rate,
                noise_level=noise_level,
                seed=seed,
            )

    def simulate_resting_state(
        self,
        n_steps: int = 1200,
        washout: int = 200,
        noise_level: Optional[float] = None,
        return_fc: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate resting-state dynamics via noise-driven autonomous evolution.

        The reservoir is driven solely by noise, producing spontaneous
        fluctuations on the SC topology. The resulting simulated time series
        can be correlated to produce a simulated FC matrix, analogous to
        computational models of resting-state dynamics (Deco et al., 2009;
        Sporns, 2013).

        Parameters
        ----------
        n_steps : int
            Total number of simulation timesteps (including washout).
        washout : int
            Initial timesteps to discard (transient dynamics).
        noise_level : float, optional
            Override noise amplitude for this simulation.
        return_fc : bool
            If True, also return the simulated FC matrix.

        Returns
        -------
        states : np.ndarray
            Reservoir state trajectories (n_steps - washout, N_nodes).
        fc_sim : np.ndarray, optional
            Simulated FC matrix (N_nodes x N_nodes), if return_fc=True.
        """
        if noise_level is None:
            noise_level = self.noise_level

        if self._use_rpy:
            # Drive with noise as input
            rng = np.random.default_rng(self.seed)
            noise_input = noise_level * rng.standard_normal((n_steps, 1))
            states = self.esn.run(noise_input)
            states = np.array(states)
            if states.ndim == 3:
                states = states.squeeze()
            states = states[washout:]
        else:
            states = self.esn.run_autonomous(
                n_steps=n_steps,
                noise_level=noise_level,
            )
            states = states[washout:]

        if return_fc:
            fc_sim = np.corrcoef(states.T)
            np.fill_diagonal(fc_sim, 0.0)
            return states, fc_sim
        return states

    def simulate_driven(
        self,
        input_signal: np.ndarray,
        washout: int = 100,
    ) -> np.ndarray:
        """Drive the reservoir with an external input signal.

        Parameters
        ----------
        input_signal : np.ndarray
            External signal of shape (T,) or (T, n_inputs).
        washout : int
            Initial timesteps to discard.

        Returns
        -------
        np.ndarray
            Reservoir states (T - washout, N_nodes).
        """
        if input_signal.ndim == 1:
            input_signal = input_signal[:, np.newaxis]

        if self._use_rpy:
            states = self.esn.run(input_signal)
            states = np.array(states)
            if states.ndim == 3:
                states = states.squeeze()
            return states[washout:]
        else:
            return self.esn.run(input_signal, washout=washout)

    def predict_fc(
        self,
        fc_empirical: np.ndarray,
        n_steps: int = 2000,
        washout: int = 500,
        alpha: float = 1.0,
        cv_folds: int = 5,
    ) -> SCFCCouplingResult:
        """Predict empirical FC from reservoir dynamics and quantify SC-FC coupling.

        This implements the core SC-FC decoupling analysis:
        1. Simulate resting-state dynamics on the SC-based reservoir
        2. Train a readout layer to predict empirical FC from reservoir states
        3. Quantify coupling as the correlation between predicted and empirical FC
        4. The decoupling index captures what the SC reservoir cannot explain

        Parameters
        ----------
        fc_empirical : np.ndarray
            Empirical functional connectivity matrix (N x N).
        n_steps : int
            Total simulation timesteps.
        washout : int
            Initial timesteps to discard.
        alpha : float
            Ridge regularization parameter for readout training.
        cv_folds : int
            Number of cross-validation folds for readout optimization.

        Returns
        -------
        SCFCCouplingResult
            Comprehensive coupling analysis results.
        """
        fc_empirical = _validate_connectivity_matrix(
            fc_empirical, "FC empirical", symmetric=True
        )

        if fc_empirical.shape[0] != self.n_nodes:
            raise ValueError(
                f"FC size ({fc_empirical.shape[0]}) != reservoir size ({self.n_nodes})"
            )

        # Step 1: Simulate resting state
        states, fc_sim = self.simulate_resting_state(
            n_steps=n_steps, washout=washout, return_fc=True
        )

        # Step 2: Train readout to predict FC from reservoir states
        # Use cross-validated Ridge regression
        fc_upper = _upper_triangle(fc_empirical)
        states_fc = np.corrcoef(states.T)
        np.fill_diagonal(states_fc, 0.0)
        sim_upper = _upper_triangle(states_fc)

        # Ridge regression: predict empirical FC edges from simulated FC edges
        scaler = StandardScaler()
        X = scaler.fit_transform(sim_upper.reshape(-1, 1))
        y = fc_upper

        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=min(cv_folds, 5))
        ridge.fit(X, y)
        y_pred = ridge.predict(X)
        r2 = r2_score(y, y_pred)

        # Step 3: Global coupling metrics
        # Direct SC-FC correlation
        sc_upper = _upper_triangle(self.sc_raw)
        coupling_global = float(np.corrcoef(sc_upper, fc_upper)[0, 1])

        # Reservoir-based coupling
        coupling_reservoir = float(np.corrcoef(sim_upper, fc_upper)[0, 1])
        decoupling_global = 1.0 - max(0.0, coupling_reservoir)

        # Step 4: Regional coupling (per-node)
        regional_coupling = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            mask = np.ones(self.n_nodes, dtype=bool)
            mask[i] = False
            r, _ = stats.pearsonr(fc_sim[i, mask], fc_empirical[i, mask])
            regional_coupling[i] = max(0.0, r)

        regional_decoupling = 1.0 - regional_coupling

        return SCFCCouplingResult(
            coupling_global=coupling_global,
            coupling_reservoir=coupling_reservoir,
            decoupling_index_global=decoupling_global,
            regional_coupling=regional_coupling,
            regional_decoupling=regional_decoupling,
            fc_simulated=fc_sim,
            fc_empirical=fc_empirical,
            sc_matrix=self.sc_raw,
            reservoir_states=states,
            spectral_radius=self.spectral_radius,
            prediction_r2=float(r2),
        )

    def memory_capacity(
        self,
        max_delay: int = 50,
        n_samples: int = 5000,
        washout: int = 500,
    ) -> Tuple[float, np.ndarray]:
        """Compute the memory capacity of the reservoir.

        Memory capacity (MC) measures the reservoir's ability to reconstruct
        past inputs from current states. It is defined as the sum of R²
        values across all delays:
            MC = Σ_k r²(y(t), u(t-k))

        For a reservoir with N nodes, the theoretical maximum MC = N.
        Connectome-based reservoirs typically achieve MC values that depend
        on the network's small-world properties (Morra & Daley, 2022).

        Parameters
        ----------
        max_delay : int
            Maximum delay to test. Default 50.
        n_samples : int
            Number of input samples. Default 5000.
        washout : int
            Initial samples to discard.

        Returns
        -------
        mc_total : float
            Total memory capacity (sum of R² across delays).
        mc_per_delay : np.ndarray
            R² value per delay (max_delay,).
        """
        rng = np.random.default_rng(self.seed + 1)
        u = rng.uniform(-0.5, 0.5, n_samples)

        states = self.simulate_driven(u, washout=0)
        states = states[washout:]
        u = u[washout:]

        n_valid = len(u) - max_delay
        mc_per_delay = np.zeros(max_delay)

        for delay in range(1, max_delay + 1):
            target = u[:n_valid]
            X = states[delay : delay + n_valid]

            ridge = Ridge(alpha=1e-4)
            ridge.fit(X, target)
            y_pred = ridge.predict(X)

            ss_res = np.sum((target - y_pred) ** 2)
            ss_tot = np.sum((target - target.mean()) ** 2)
            mc_per_delay[delay - 1] = max(0.0, 1.0 - ss_res / ss_tot)

        mc_total = float(np.sum(mc_per_delay))
        return mc_total, mc_per_delay

    def kernel_quality(
        self,
        n_samples: int = 1000,
        washout: int = 200,
    ) -> float:
        """Compute the kernel quality (separation property) of the reservoir.

        Kernel quality measures the rank of the reservoir state matrix,
        which indicates the reservoir's ability to separate different
        input patterns. Higher rank means better separation.

        Parameters
        ----------
        n_samples : int
            Number of input samples.
        washout : int
            Initial samples to discard.

        Returns
        -------
        float
            Kernel quality ∈ (0, 1], the effective rank normalized by N_nodes.
        """
        rng = np.random.default_rng(self.seed + 2)
        u = rng.standard_normal(n_samples)

        states = self.simulate_driven(u, washout=washout)

        # Effective rank via singular values
        sv = linalg.svdvals(states)
        sv = sv[sv > 1e-10]

        # Normalize singular values
        p = sv / sv.sum()
        # Shannon entropy of normalized singular values
        entropy = -np.sum(p * np.log(p))
        # Effective rank
        eff_rank = np.exp(entropy)

        return float(eff_rank / self.n_nodes)


# =============================================================================
# HEBBIAN ADAPTIVE RESERVOIR
# =============================================================================

class HebbianAdaptiveReservoir(ConnectomeReservoir):
    """Reservoir with Hebbian plasticity for adaptive SC-FC coupling analysis.

    Inspired by the HAG framework (Nature Communications, Dec 2025), this
    reservoir incorporates online Hebbian learning rules that allow the
    reservoir weights to adapt during simulation. This models the biological
    process of experience-dependent plasticity sculpting structure-function
    relationships (Sporns, 2013).

    The Hebbian update rule is:
        ΔW_ij = η * (x_i * x_j - λ * W_ij)

    where η is the learning rate, x_i and x_j are pre- and post-synaptic
    activities, and λ is a decay term preventing unbounded weight growth.

    Parameters
    ----------
    sc_matrix : np.ndarray
        Structural connectivity matrix.
    learning_rate : float
        Hebbian learning rate η. Default 0.001.
    decay_rate : float
        Weight decay λ. Default 0.01.
    plasticity_mask : np.ndarray, optional
        Binary mask indicating which connections are plastic.
        If None, all existing connections are plastic.
    **kwargs
        Additional arguments passed to ConnectomeReservoir.
    """

    def __init__(
        self,
        sc_matrix: np.ndarray,
        learning_rate: float = 0.001,
        decay_rate: float = 0.01,
        plasticity_mask: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(sc_matrix, **kwargs)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        if plasticity_mask is not None:
            self.plasticity_mask = plasticity_mask.astype(bool)
        else:
            self.plasticity_mask = self.W != 0

        self.W_initial = self.W.copy()
        self.W_history: List[np.ndarray] = []

    def simulate_with_plasticity(
        self,
        n_steps: int = 2000,
        washout: int = 500,
        record_weights_every: int = 100,
        noise_level: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Simulate reservoir dynamics with online Hebbian weight adaptation.

        Parameters
        ----------
        n_steps : int
            Number of simulation timesteps.
        washout : int
            Initial timesteps to discard from output.
        record_weights_every : int
            Record weight snapshots every N steps.
        noise_level : float, optional
            Override noise level.

        Returns
        -------
        states : np.ndarray
            Reservoir state trajectories (n_steps - washout, N_nodes).
        weight_history : list of np.ndarray
            Weight matrix snapshots during simulation.
        """
        if noise_level is None:
            noise_level = self.noise_level

        rng = np.random.default_rng(self.seed)
        W = self.W_initial.copy()
        states = np.zeros((n_steps, self.n_nodes))
        x = rng.standard_normal(self.n_nodes) * 0.01
        weight_history = [W.copy()]

        for t in range(n_steps):
            noise = noise_level * rng.standard_normal(self.n_nodes)
            x_new = (
                (1 - self.leaking_rate) * x
                + self.leaking_rate * np.tanh(W @ x + noise)
            )

            # Hebbian update: ΔW = η (x_post ⊗ x_pre - λ W)
            delta_W = self.learning_rate * (
                np.outer(x_new, x) - self.decay_rate * W
            )
            delta_W *= self.plasticity_mask
            W += delta_W

            # Maintain spectral radius constraint
            sr = np.max(np.abs(linalg.eigvals(W)))
            if sr > self.spectral_radius * 1.1:
                W *= self.spectral_radius / sr

            x = x_new
            states[t] = x

            if (t + 1) % record_weights_every == 0:
                weight_history.append(W.copy())

        self.W = W.copy()
        self.W_history = weight_history

        return states[washout:], weight_history

    def plasticity_analysis(
        self,
        fc_empirical: np.ndarray,
        n_steps: int = 3000,
        washout: int = 500,
    ) -> Dict:
        """Analyze how Hebbian plasticity reshapes SC-FC coupling.

        Runs the adaptive reservoir and compares the SC-FC coupling
        before and after plasticity, providing insight into how
        activity-dependent remodeling can explain structure-function
        deviations.

        Parameters
        ----------
        fc_empirical : np.ndarray
            Empirical FC matrix.
        n_steps : int
            Simulation timesteps.
        washout : int
            Washout period.

        Returns
        -------
        dict
            Dictionary containing:
            - 'coupling_before': SC-FC coupling before plasticity
            - 'coupling_after': SC-FC coupling after plasticity
            - 'coupling_improvement': Change in coupling
            - 'weight_change': Mean absolute weight change
            - 'plasticity_map': Spatial pattern of weight changes
            - 'fc_before': Simulated FC before plasticity
            - 'fc_after': Simulated FC after plasticity
        """
        fc_empirical = _validate_connectivity_matrix(
            fc_empirical, "FC", symmetric=True
        )

        # FC before plasticity
        self.W = self.W_initial.copy()
        _, fc_before = self.simulate_resting_state(
            n_steps=1500, washout=washout, return_fc=True
        )

        fc_emp_upper = _upper_triangle(fc_empirical)
        coupling_before = float(
            np.corrcoef(_upper_triangle(fc_before), fc_emp_upper)[0, 1]
        )

        # Run with plasticity
        states, w_history = self.simulate_with_plasticity(
            n_steps=n_steps, washout=washout
        )

        # FC after plasticity
        fc_after = np.corrcoef(states.T)
        np.fill_diagonal(fc_after, 0.0)

        coupling_after = float(
            np.corrcoef(_upper_triangle(fc_after), fc_emp_upper)[0, 1]
        )

        # Weight change analysis
        weight_change = np.abs(self.W - self.W_initial)
        plasticity_map = np.sum(weight_change, axis=1)  # Per-node change

        return {
            "coupling_before": coupling_before,
            "coupling_after": coupling_after,
            "coupling_improvement": coupling_after - coupling_before,
            "weight_change_mean": float(np.mean(weight_change[weight_change > 0])),
            "plasticity_map": plasticity_map,
            "fc_before": fc_before,
            "fc_after": fc_after,
            "n_weights_changed": int(np.sum(weight_change > 1e-6)),
        }


# =============================================================================
# PERTURBATION / VIRTUAL LESION ANALYSIS
# =============================================================================

def virtual_lesion_analysis(
    sc: np.ndarray,
    fc_empirical: np.ndarray,
    spectral_radius: float = 0.9,
    n_steps: int = 1500,
    washout: int = 300,
    top_k: int = 10,
    seed: int = 42,
    lesion_type: str = "node",
) -> PerturbationResult:
    """Perform virtual lesion analysis to identify critical nodes for SC-FC coupling.

    For each node (or edge), the corresponding row/column is zeroed out
    in the SC matrix, a new reservoir is constructed, and the change in
    SC-FC coupling is measured. Nodes whose removal most reduces coupling
    are considered critical for maintaining the structure-function relationship.

    This is particularly relevant for COVID-19 studies, as SARS-CoV-2
    neurotropism may preferentially target specific brain regions, and
    this analysis can identify which structural perturbations would most
    disrupt functional organization.

    Parameters
    ----------
    sc : np.ndarray
        Structural connectivity matrix (N x N).
    fc_empirical : np.ndarray
        Empirical FC matrix (N x N).
    spectral_radius : float
        Spectral radius for reservoir construction.
    n_steps : int
        Simulation timesteps per reservoir.
    washout : int
        Washout period.
    top_k : int
        Number of top vulnerable/resilient nodes to return.
    seed : int
        Random seed.
    lesion_type : str
        Type of lesion: "node" (remove entire node) or "edge" (remove
        individual edges).

    Returns
    -------
    PerturbationResult
        Perturbation analysis results.
    """
    sc = _validate_connectivity_matrix(sc, "SC", symmetric=True, nonneg=True)
    fc_empirical = _validate_connectivity_matrix(fc_empirical, "FC", symmetric=True)
    n = sc.shape[0]

    # Baseline coupling
    baseline_res = ConnectomeReservoir(
        sc, spectral_radius=spectral_radius, seed=seed
    )
    _, fc_baseline = baseline_res.simulate_resting_state(
        n_steps=n_steps, washout=washout, return_fc=True
    )
    fc_emp_upper = _upper_triangle(fc_empirical)
    baseline_coupling = float(
        np.corrcoef(_upper_triangle(fc_baseline), fc_emp_upper)[0, 1]
    )

    if lesion_type == "node":
        perturbed_coupling = np.zeros(n)
        for node in range(n):
            sc_lesioned = sc.copy()
            sc_lesioned[node, :] = 0.0
            sc_lesioned[:, node] = 0.0

            if np.sum(sc_lesioned) < 1e-10:
                perturbed_coupling[node] = 0.0
                continue

            try:
                res = ConnectomeReservoir(
                    sc_lesioned, spectral_radius=spectral_radius, seed=seed
                )
                _, fc_lesioned = res.simulate_resting_state(
                    n_steps=n_steps, washout=washout, return_fc=True
                )
                perturbed_coupling[node] = float(
                    np.corrcoef(
                        _upper_triangle(fc_lesioned), fc_emp_upper
                    )[0, 1]
                )
            except Exception:
                perturbed_coupling[node] = 0.0

        coupling_change = baseline_coupling - perturbed_coupling
        vulnerability = coupling_change / max(abs(baseline_coupling), 1e-10)

        top_k = min(top_k, n)
        most_vulnerable = np.argsort(vulnerability)[-top_k:][::-1]
        most_resilient = np.argsort(vulnerability)[:top_k]

        return PerturbationResult(
            baseline_coupling=baseline_coupling,
            perturbed_coupling=perturbed_coupling,
            coupling_change=coupling_change,
            vulnerability_index=vulnerability,
            most_vulnerable_nodes=most_vulnerable,
            most_resilient_nodes=most_resilient,
        )
    else:
        raise NotImplementedError(
            "Edge-level lesion analysis not yet implemented. Use lesion_type='node'."
        )


# =============================================================================
# GROUP COMPARISON
# =============================================================================

def compare_groups_coupling(
    sc_group1: List[np.ndarray],
    fc_group1: List[np.ndarray],
    sc_group2: List[np.ndarray],
    fc_group2: List[np.ndarray],
    group1_name: str = "COVID",
    group2_name: str = "Control",
    metric: str = "sdi",
    spectral_radius: float = 0.9,
    alpha_fdr: float = 0.05,
    network_labels: Optional[np.ndarray] = None,
    network_names: Optional[Dict[int, str]] = None,
) -> GroupComparisonResult:
    """Compare SC-FC coupling/decoupling between two groups.

    Computes either SDI or reservoir-based coupling for each subject
    in both groups, then performs mass-univariate t-tests with FDR
    correction at the regional level.

    Parameters
    ----------
    sc_group1, fc_group1 : list of np.ndarray
        SC and FC matrices for group 1 subjects.
    sc_group2, fc_group2 : list of np.ndarray
        SC and FC matrices for group 2 subjects.
    group1_name, group2_name : str
        Names for the two groups.
    metric : str
        Which metric to compare: "sdi" (Structural Decoupling Index) or
        "reservoir" (reservoir-based regional coupling).
    spectral_radius : float
        Spectral radius for reservoir construction (if metric="reservoir").
    alpha_fdr : float
        FDR significance threshold.
    network_labels : np.ndarray, optional
        Network assignments for each node.
    network_names : dict, optional
        Network label-to-name mapping.

    Returns
    -------
    GroupComparisonResult
        Statistical comparison results.
    """
    n1, n2 = len(sc_group1), len(sc_group2)
    n_nodes = sc_group1[0].shape[0]

    # Compute per-subject metrics
    def _compute_metric(sc, fc):
        if metric == "sdi":
            result = compute_sdi(sc, fc)
            return result.sdi
        elif metric == "reservoir":
            res = ConnectomeReservoir(sc, spectral_radius=spectral_radius)
            coupling_result = res.predict_fc(fc)
            return coupling_result.regional_coupling
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # Group 1
    metrics_g1 = np.zeros((n1, n_nodes))
    for i in range(n1):
        metrics_g1[i] = _compute_metric(sc_group1[i], fc_group1[i])

    # Group 2
    metrics_g2 = np.zeros((n2, n_nodes))
    for i in range(n2):
        metrics_g2[i] = _compute_metric(sc_group2[i], fc_group2[i])

    # Regional t-tests
    t_stats = np.zeros(n_nodes)
    p_values = np.zeros(n_nodes)
    for r in range(n_nodes):
        t, p = stats.ttest_ind(metrics_g1[:, r], metrics_g2[:, r])
        t_stats[r] = t
        p_values[r] = p

    # FDR correction (Benjamini-Hochberg)
    sorted_idx = np.argsort(p_values)
    p_fdr = np.zeros(n_nodes)
    for i, idx in enumerate(sorted_idx):
        p_fdr[idx] = p_values[idx] * n_nodes / (i + 1)
    p_fdr = np.minimum.accumulate(p_fdr[sorted_idx[::-1]])[::-1]
    p_fdr_final = np.zeros(n_nodes)
    for i, idx in enumerate(sorted_idx):
        p_fdr_final[idx] = p_fdr[i]
    p_fdr_final = np.clip(p_fdr_final, 0, 1)

    significant = p_fdr_final < alpha_fdr

    # Effect size (Cohen's d)
    mean_g1 = np.mean(metrics_g1, axis=0)
    mean_g2 = np.mean(metrics_g2, axis=0)
    std_pooled = np.sqrt(
        ((n1 - 1) * np.var(metrics_g1, axis=0, ddof=1)
         + (n2 - 1) * np.var(metrics_g2, axis=0, ddof=1))
        / (n1 + n2 - 2)
    )
    cohen_d = np.where(std_pooled > 0, (mean_g1 - mean_g2) / std_pooled, 0.0)

    # Global test
    global_g1 = np.mean(metrics_g1, axis=1)
    global_g2 = np.mean(metrics_g2, axis=1)
    global_t, global_p = stats.ttest_ind(global_g1, global_g2)

    # Network-level results
    network_results = None
    if network_labels is not None:
        network_results = {}
        for label in np.unique(network_labels):
            mask = network_labels == label
            name = (
                network_names.get(label, f"Network_{label}")
                if network_names
                else f"Network_{label}"
            )
            net_g1 = np.mean(metrics_g1[:, mask], axis=1)
            net_g2 = np.mean(metrics_g2[:, mask], axis=1)
            net_t, net_p = stats.ttest_ind(net_g1, net_g2)
            network_results[name] = {
                "t_statistic": float(net_t),
                "p_value": float(net_p),
                "mean_group1": float(np.mean(net_g1)),
                "mean_group2": float(np.mean(net_g2)),
                "cohen_d": float(
                    (np.mean(net_g1) - np.mean(net_g2))
                    / max(np.std(np.concatenate([net_g1, net_g2])), 1e-10)
                ),
            }

    return GroupComparisonResult(
        group1_name=group1_name,
        group2_name=group2_name,
        regional_tstat=t_stats,
        regional_pvalue=p_values,
        regional_pvalue_fdr=p_fdr_final,
        significant_regions=significant,
        effect_size_cohen_d=cohen_d,
        group1_mean=mean_g1,
        group2_mean=mean_g2,
        global_tstat=float(global_t),
        global_pvalue=float(global_p),
        network_results=network_results,
    )


# =============================================================================
# SPECTRAL RADIUS SWEEP (EDGE OF CHAOS ANALYSIS)
# =============================================================================

def spectral_radius_sweep(
    sc: np.ndarray,
    fc_empirical: np.ndarray,
    radii: Optional[np.ndarray] = None,
    n_steps: int = 1500,
    washout: int = 300,
    seed: int = 42,
) -> Dict:
    """Sweep spectral radius to find optimal operating point for SC-FC coupling.

    The spectral radius controls the dynamical regime of the reservoir:
    - Low ρ: overdamped, rapid forgetting, low memory
    - ρ ≈ 1: edge of chaos, maximal computational capacity
    - High ρ: chaotic, unstable dynamics

    The optimal ρ for SC-FC coupling reveals where the brain's structural
    topology best supports its functional dynamics. Connectome-based
    reservoirs show robustness to increasing ρ compared to random networks
    (Takahashi et al., 2025).

    Parameters
    ----------
    sc : np.ndarray
        Structural connectivity matrix.
    fc_empirical : np.ndarray
        Empirical FC matrix.
    radii : np.ndarray, optional
        Array of spectral radii to test. Default: 0.1 to 1.5.
    n_steps : int
        Simulation timesteps per trial.
    washout : int
        Washout period.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dictionary with:
        - 'radii': tested spectral radii
        - 'coupling': SC-FC coupling at each radius
        - 'optimal_radius': radius with highest coupling
        - 'optimal_coupling': best coupling value
        - 'memory_capacity': MC at each radius (if computed)
    """
    if radii is None:
        radii = np.linspace(0.1, 1.5, 15)

    sc = _validate_connectivity_matrix(sc, "SC", symmetric=True, nonneg=True)
    fc_empirical = _validate_connectivity_matrix(fc_empirical, "FC", symmetric=True)
    fc_upper = _upper_triangle(fc_empirical)

    couplings = np.zeros(len(radii))
    for i, rho in enumerate(radii):
        try:
            res = ConnectomeReservoir(sc, spectral_radius=rho, seed=seed)
            _, fc_sim = res.simulate_resting_state(
                n_steps=n_steps, washout=washout, return_fc=True
            )
            sim_upper = _upper_triangle(fc_sim)
            couplings[i] = float(np.corrcoef(sim_upper, fc_upper)[0, 1])
        except Exception as e:
            logger.warning(f"Failed at ρ={rho:.2f}: {e}")
            couplings[i] = np.nan

    best_idx = np.nanargmax(couplings)
    return {
        "radii": radii,
        "coupling": couplings,
        "optimal_radius": float(radii[best_idx]),
        "optimal_coupling": float(couplings[best_idx]),
    }


# =============================================================================
# FULL PIPELINE
# =============================================================================

def analyze_sc_fc_decoupling(
    sc: np.ndarray,
    fc: np.ndarray,
    timeseries: Optional[np.ndarray] = None,
    spectral_radius: float = 0.9,
    n_steps: int = 2000,
    washout: int = 500,
    compute_sdi_flag: bool = True,
    compute_perturbation: bool = False,
    compute_memory: bool = False,
    atlas_name: str = "unknown",
    network_labels: Optional[np.ndarray] = None,
    network_names: Optional[Dict[int, str]] = None,
    seed: int = 42,
) -> ReservoirDynamicsResult:
    """Full SC-FC decoupling analysis pipeline.

    Integrates reservoir-based coupling, SDI, optional perturbation
    analysis, and dynamical regime characterization (Lyapunov exponent,
    edge-of-chaos distance, reservoir entropy) into a single workflow.

    Parameters
    ----------
    sc : np.ndarray
        Structural connectivity matrix (N x N).
    fc : np.ndarray
        Functional connectivity matrix (N x N).
    timeseries : np.ndarray, optional
        BOLD time series (T x N), for driving the reservoir.
    spectral_radius : float
        Spectral radius for reservoir. Default 0.9.
    n_steps : int
        Simulation timesteps.
    washout : int
        Washout period.
    compute_sdi_flag : bool
        Whether to compute SDI. Default True.
    compute_perturbation : bool
        Whether to run virtual lesion analysis. Default False (slow).
    compute_memory : bool
        Whether to compute memory capacity. Default False.
    atlas_name : str
        Name of the atlas used.
    network_labels : np.ndarray, optional
        Network assignments per node.
    network_names : dict, optional
        Network label-to-name mapping.
    seed : int
        Random seed.

    Returns
    -------
    ReservoirDynamicsResult
        Comprehensive analysis results with all fields populated.

    Examples
    --------
    >>> result = analyze_sc_fc_decoupling(
    ...     sc=sc_matrix, fc=fc_matrix,
    ...     spectral_radius=0.95,
    ...     compute_sdi_flag=True,
    ...     compute_perturbation=False,
    ...     atlas_name="schaefer_100",
    ... )
    >>> print(f"Global coupling: {result.coupling.coupling_reservoir:.3f}")
    >>> print(f"Lyapunov: {result.lyapunov_exponent:.4f}")
    >>> print(f"Edge of chaos: {result.edge_of_chaos_distance:.4f}")
    >>> print(f"Reservoir entropy: {result.reservoir_entropy:.4f}")
    """
    n_nodes = sc.shape[0]
    logger.info(
        f"Starting SC-FC decoupling analysis: {n_nodes} nodes, "
        f"atlas={atlas_name}, ρ={spectral_radius}"
    )

    # =====================================================================
    # 1. Reservoir-based SC-FC coupling
    # =====================================================================
    reservoir = ConnectomeReservoir(
        sc, spectral_radius=spectral_radius, seed=seed
    )
    coupling_result = reservoir.predict_fc(
        fc, n_steps=n_steps, washout=washout
    )

    # Add network-level metrics if labels provided
    if network_labels is not None:
        net_coupling = {}
        net_decoupling = {}
        for label in np.unique(network_labels):
            mask = network_labels == label
            name = (
                network_names.get(label, f"Network_{label}")
                if network_names
                else f"Network_{label}"
            )
            net_coupling[name] = float(
                np.mean(coupling_result.regional_coupling[mask])
            )
            net_decoupling[name] = float(
                np.mean(coupling_result.regional_decoupling[mask])
            )
        coupling_result.network_coupling = net_coupling
        coupling_result.network_decoupling = net_decoupling

    # Memory capacity (optional — adds ~10s per call)
    if compute_memory:
        mc_total, mc_delays = reservoir.memory_capacity()
        coupling_result.memory_capacity = mc_total

    # =====================================================================
    # 2. Dynamical regime characterization
    # =====================================================================
    # Use reservoir states from the coupling analysis (already computed)
    states = coupling_result.reservoir_states  # shape (T_eff, N)

    # ── 2a. Lyapunov exponent (maximum) ──────────────────────────────────
    # Measures divergence rate of nearby trajectories in state space.
    #   λ_max < 0 → ordered (stable fixed point / limit cycle)
    #   λ_max ≈ 0 → edge of chaos (critical regime)
    #   λ_max > 0 → chaotic
    #
    # Method: Jacobian-based estimation from reservoir weight matrix
    # and the derivative of the activation function at each state.
    lyapunov = _compute_lyapunov_exponent(reservoir.W, states)
    logger.info(f"  Lyapunov exponent: {lyapunov:.4f}")

    # ── 2b. Edge of chaos distance ───────────────────────────────────────
    # Quantifies how far the reservoir operates from the critical point.
    # The critical spectral radius for tanh ESNs is ρ_c ≈ 1.0.
    #
    # We use two complementary measures:
    #   (i)  Parameter distance: |ρ - 1.0|
    #   (ii) Dynamical distance: based on Lyapunov (|λ_max|)
    # Combined as geometric mean for a robust single metric.
    rho_distance = abs(spectral_radius - 1.0)
    lambda_distance = abs(lyapunov)
    edge_of_chaos = float(np.sqrt(rho_distance * lambda_distance))
    logger.info(f"  Edge of chaos distance: {edge_of_chaos:.4f}")

    # ── 2c. Reservoir entropy ────────────────────────────────────────────
    # Shannon entropy of the reservoir state distribution.
    # Higher entropy → richer dynamical repertoire → more computational
    # capacity. Low entropy → degenerate dynamics (collapsed states).
    #
    # We discretize each neuron's activation into bins and compute the
    # average entropy across all neurons.
    reservoir_entropy = _compute_reservoir_entropy(states)
    logger.info(f"  Reservoir entropy: {reservoir_entropy:.4f}")

    # =====================================================================
    # 3. Structural Decoupling Index (SDI)
    # =====================================================================
    sdi_result = None
    if compute_sdi_flag:
        sdi_result = compute_sdi(
            sc, fc,
            network_labels=network_labels,
            network_names=network_names,
        )
    else:
        sdi_result = SDIResult(
            sdi=np.zeros(n_nodes),
            sdi_raw=np.zeros(n_nodes),
            fc_coupled=np.zeros((n_nodes, n_nodes)),
            fc_decoupled=np.zeros((n_nodes, n_nodes)),
            sc_eigenvalues=np.zeros(n_nodes),
            sc_eigenvectors=np.eye(n_nodes),
            coupling_ratio=0.0,
            n_coupled_components=0,
        )

    # =====================================================================
    # 4. Perturbation analysis (optional — slow)
    # =====================================================================
    perturbation_result = None
    if compute_perturbation:
        logger.info("Running virtual lesion analysis (this may take a while)...")
        perturbation_result = virtual_lesion_analysis(
            sc, fc, spectral_radius=spectral_radius,
            n_steps=n_steps, washout=washout, seed=seed,
        )

    # =====================================================================
    # 5. Assemble result
    # =====================================================================
    return ReservoirDynamicsResult(
        coupling=coupling_result,
        sdi=sdi_result,
        perturbation=perturbation_result,
        lyapunov_exponent=lyapunov,
        edge_of_chaos_distance=edge_of_chaos,
        reservoir_entropy=reservoir_entropy,
        atlas_name=atlas_name,
        n_regions=n_nodes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS (private)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_lyapunov_exponent(
    W: np.ndarray,
    states: np.ndarray,
    leak_rate: float = 0.3,
) -> float:
    """Estimate the maximum Lyapunov exponent from reservoir dynamics.

    Uses the Jacobian-based method: at each timestep, the local Jacobian
    of the ESN update rule is:

        J(t) = (1 - α)·I + α · diag(f'(h(t))) · W

    where f'(h) = 1 - tanh²(h) for tanh activation.

    The maximum Lyapunov exponent is estimated as the time-averaged
    log of the largest singular value of J(t).

    Parameters
    ----------
    W : np.ndarray, shape (N, N)
        Reservoir weight matrix (already scaled by spectral radius).
    states : np.ndarray, shape (T, N)
        Reservoir states from a simulation run.
    leak_rate : float
        Leak rate α used during simulation.

    Returns
    -------
    float
        Estimated maximum Lyapunov exponent.
    """
    T, N = states.shape
    alpha = leak_rate

    # Use a subsample for efficiency (every 5th timestep)
    step = max(1, T // 200)
    indices = range(0, T, step)

    log_svs = []
    I = np.eye(N)
    for t in indices:
        x = states[t]
        # Derivative of tanh: f'(h) = 1 - x² (since x = tanh(h))
        f_prime = 1.0 - x ** 2
        # Local Jacobian
        J = (1.0 - alpha) * I + alpha * np.diag(f_prime) @ W
        # Largest singular value
        s_max = np.linalg.norm(J, ord=2)
        if s_max > 0:
            log_svs.append(np.log(s_max))

    if not log_svs:
        return 0.0

    return float(np.mean(log_svs))


def _compute_reservoir_entropy(
    states: np.ndarray,
    n_bins: int = 30,
) -> float:
    """Compute average Shannon entropy of reservoir neuron activations.

    Each neuron's activation distribution is discretized into bins,
    and the Shannon entropy is computed. The result is averaged across
    all neurons and normalized to [0, 1] (dividing by log(n_bins)).

    Higher values indicate richer dynamical repertoire.

    Parameters
    ----------
    states : np.ndarray, shape (T, N)
        Reservoir states.
    n_bins : int
        Number of bins for histogram. Default 30.

    Returns
    -------
    float
        Mean normalized entropy across neurons, in [0, 1].
    """
    T, N = states.shape
    entropies = np.zeros(N)

    for i in range(N):
        counts, _ = np.histogram(states[:, i], bins=n_bins)
        # Normalize to probability
        p = counts / counts.sum()
        # Remove zeros (log(0) is undefined)
        p = p[p > 0]
        # Shannon entropy
        entropies[i] = -np.sum(p * np.log2(p))

    # Normalize by maximum possible entropy
    max_entropy = np.log2(n_bins)
    mean_entropy = float(np.mean(entropies) / max_entropy) if max_entropy > 0 else 0.0

    return mean_entropy

# =============================================================================
# MULTI-ATLAS ANALYSIS
# =============================================================================

def multi_atlas_decoupling(
    sc_dict: Dict[str, np.ndarray],
    fc_dict: Dict[str, np.ndarray],
    spectral_radius: float = 0.9,
    network_labels_dict: Optional[Dict[str, np.ndarray]] = None,
    network_names_dict: Optional[Dict[str, Dict[int, str]]] = None,
    seed: int = 42,
) -> Dict[str, ReservoirDynamicsResult]:
    """Run SC-FC decoupling analysis across multiple brain atlases.

    Parameters
    ----------
    sc_dict : dict
        Atlas name → SC matrix mapping.
    fc_dict : dict
        Atlas name → FC matrix mapping (same keys as sc_dict).
    spectral_radius : float
        Spectral radius for all reservoirs.
    network_labels_dict : dict, optional
        Atlas name → network labels mapping.
    network_names_dict : dict, optional
        Atlas name → network names mapping.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Atlas name → ReservoirDynamicsResult mapping.
    """
    results = {}
    for atlas_name in sc_dict:
        if atlas_name not in fc_dict:
            logger.warning(f"Atlas '{atlas_name}' not found in FC dict, skipping.")
            continue

        net_labels = (
            network_labels_dict.get(atlas_name)
            if network_labels_dict
            else None
        )
        net_names = (
            network_names_dict.get(atlas_name)
            if network_names_dict
            else None
        )

        logger.info(f"Analyzing atlas: {atlas_name}")
        results[atlas_name] = analyze_sc_fc_decoupling(
            sc=sc_dict[atlas_name],
            fc=fc_dict[atlas_name],
            spectral_radius=spectral_radius,
            atlas_name=atlas_name,
            network_labels=net_labels,
            network_names=net_names,
            seed=seed,
        )

    return results


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "SCFCCouplingResult",
    "SDIResult",
    "PerturbationResult",
    "GroupComparisonResult",
    "ReservoirDynamicsResult",
    # Core functions
    "compute_sdi",
    "virtual_lesion_analysis",
    "compare_groups_coupling",
    "spectral_radius_sweep",
    "analyze_sc_fc_decoupling",
    "multi_atlas_decoupling",
    # Classes
    "ConnectomeReservoir",
    "HebbianAdaptiveReservoir",
]
