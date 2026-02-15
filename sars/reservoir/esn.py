# -*- coding: utf-8 -*-
"""
sars.reservoir.esn
==================

Core Echo State Network (ESN) module for connectome-based reservoir computing.

This module provides the computational engine for reservoir computing analyses
in the SARS-CoV-2 neuroimaging study. It implements ESN classes that use real
brain structural connectivity matrices as reservoir topologies, following the
conn2res framework (Suárez et al., 2024, Nature Communications).

The key insight is that only the readout layer is trained (via ridge regression),
while the reservoir weights are fixed to the patient's structural connectome —
making this approach uniquely suited for small samples (N=22) where traditional
deep learning would overfit.

Classes
-------
ConnectomeReservoir
    Standard ESN using a structural connectome as reservoir topology.
    Supports multiple input/output configurations, spectral radius scaling,
    and leak-rate integration.

AdaptiveReservoir
    ESN with Hebbian synaptic plasticity (HAG-inspired), where reservoir
    weights are updated online via a local learning rule. This models
    activity-dependent structural remodeling observed in neuroplasticity.

Functions
---------
memory_capacity
    Compute the short-term memory capacity (MC) of a reservoir, measuring
    its ability to reconstruct delayed versions of past inputs.

kernel_quality
    Evaluate the kernel quality / separation property of a reservoir,
    quantifying its ability to map distinct inputs to distinct internal states.

echo_state_property_index
    Empirical verification of the echo state property (ESP) — the condition
    that reservoir states asymptotically depend only on driving inputs,
    not on initial conditions.

lyapunov_exponent
    Compute the maximum Lyapunov exponent of the reservoir dynamics to
    assess stability at the edge of chaos.

spectral_analysis
    Analyze the eigenspectrum of the reservoir weight matrix for understanding
    dynamic properties (e.g., timescale hierarchy, oscillatory modes).

compare_reservoir_architectures
    Systematic comparison of different reservoir configurations (connectome
    vs random vs small-world vs ring) across performance metrics.

References
----------
- Suárez et al. (2024). conn2res: A toolbox for connectome-based reservoir
  computing. Nature Communications, 15(1), 656.
- Jaeger (2001). The "echo state" approach to analysing and training recurrent
  neural networks. GMD Report 148.
- Damicelli et al. (2022). Brain connectivity meets reservoir computing.
  PLOS Computational Biology, 18(11), e1010639.
- Suárez et al. (2021). Learning function from structure in neuromorphic
  networks. Nature Machine Intelligence, 3, 771–786.
- Woo et al. (2024). Echo State Property upon Noisy Driving Input.
  Complexity, 2024(1), 5593925.
- Lukoševičius & Jaeger (2009). Reservoir computing approaches to recurrent
  neural network training. Computer Science Review, 3(3), 127–149.

Notes
-----
Primary backend: ReservoirPy (Trouvain et al., 2020)
Fallback: Pure NumPy implementation for environments without ReservoirPy.

Author: Velho Mago
"""

import numpy as np
import warnings
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass, field
from scipy import linalg, sparse, stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Check optional dependencies
# ─────────────────────────────────────────────────────────────────────────────
_HAS_RESERVOIRPY = False
try:
    import reservoirpy as rpy
    from reservoirpy.nodes import Reservoir, Ridge as RpyRidge
    _HAS_RESERVOIRPY = True
    try:
        rpy.verbosity(0)
    except Exception:
        pass  # Removido na v0.4+
except ImportError:
    pass


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ReservoirMetrics:
    """Container for reservoir characterization metrics.

    Attributes
    ----------
    memory_capacity : float
        Total short-term memory capacity (MC ∈ [0, N_neurons]).
        Measures the reservoir's ability to linearly reconstruct
        delayed versions of past inputs. Theoretical upper bound
        equals the number of reservoir neurons (Jaeger, 2001).
    memory_profile : np.ndarray
        MC_δ for each delay δ (coefficient of determination R² of
        reconstructing u_{k−δ} from reservoir states x_k).
    kernel_quality : float
        Kernel rank / separation property (KQ ∈ [0, 1]).
        Quantifies how well distinct inputs are mapped to distinct
        internal states. A value close to 1.0 indicates maximal
        separation (i.e., the reservoir acts as a high-dimensional
        kernel). Computed as the effective rank of the state matrix
        divided by the number of neurons.
    generalization_rank : float
        Generalization capability via effective dimensionality of
        the reservoir state space. Related to the ratio of explained
        variance by the top singular values.
    spectral_radius : float
        Actual spectral radius ρ(W) of the reservoir weight matrix.
        Controls the timescale of reservoir dynamics: ρ < 1 ensures
        fading memory; ρ ≈ 1 operates at the "edge of chaos".
    lyapunov_exponent : Optional[float]
        Maximum Lyapunov exponent λ_max of the reservoir dynamics.
        λ_max < 0: stable (ordered regime); λ_max ≈ 0: edge of chaos;
        λ_max > 0: chaotic (unstable regime).
    esp_index : Optional[float]
        Echo State Property index (ESP ∈ [0, 1]). Measures the degree
        to which reservoir states converge regardless of initial conditions.
        Values close to 1.0 indicate strong ESP (Woo et al., 2024).
    """
    memory_capacity: float = 0.0
    memory_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    kernel_quality: float = 0.0
    generalization_rank: float = 0.0
    spectral_radius: float = 0.0
    lyapunov_exponent: Optional[float] = None
    esp_index: Optional[float] = None


@dataclass
class ReservoirComparison:
    """Results from comparing multiple reservoir architectures.

    Attributes
    ----------
    architecture_names : List[str]
        Labels for each tested architecture.
    metrics : Dict[str, ReservoirMetrics]
        ReservoirMetrics for each architecture.
    task_scores : Dict[str, Dict[str, float]]
        Performance on each task for each architecture.
        Outer key: architecture name; inner key: task name → R² score.
    best_architecture : str
        Name of the best-performing architecture (by mean task R²).
    """
    architecture_names: List[str] = field(default_factory=list)
    metrics: Dict[str, ReservoirMetrics] = field(default_factory=dict)
    task_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_architecture: str = ""


# =============================================================================
# CONNECTOME RESERVOIR (Standard ESN)
# =============================================================================

class ConnectomeReservoir:
    """Echo State Network using a structural connectome as reservoir topology.

    This class implements the conn2res approach (Suárez et al., 2024),
    where the reservoir weight matrix W is derived from the patient's
    structural connectivity matrix. Only the readout layer is trained,
    making this approach suitable for small samples.

    The reservoir dynamics follow the standard ESN equations:

        x(t) = (1 − α) · x(t−1) + α · f(W_in · u(t) + W · x(t−1) + b)

    where:
        x(t)  = reservoir state vector at time t
        u(t)  = input signal at time t
        α     = leak rate (controls integration timescale)
        f(·)  = activation function (default: tanh)
        W_in  = input weight matrix
        W     = reservoir weight matrix (from connectome)
        b     = bias vector

    Parameters
    ----------
    connectivity : np.ndarray, shape (N, N)
        Structural connectivity matrix. Can be weighted (streamline count,
        FA-weighted, etc.) or binary. Diagonal is zeroed automatically.
    spectral_radius : float, default=0.9
        Target spectral radius ρ for scaling W. Controls the timescale
        and memory properties of the reservoir. Values near 1.0 operate
        at the edge of chaos; values << 1.0 produce faster forgetting.
    leak_rate : float, default=0.3
        Leak rate α ∈ (0, 1]. Controls the speed of reservoir state
        updates. α = 1.0 gives no leaky integration (standard ESN);
        smaller values produce slower, more smoothed dynamics, which
        is appropriate for the hemodynamic timescale of BOLD fMRI.
    input_scaling : float, default=0.1
        Scaling factor for the input weight matrix W_in. Controls the
        nonlinearity regime: small values → linear regime; large values
        → highly nonlinear transformation.
    bias_scaling : float, default=0.0
        Scaling factor for bias terms. Default 0 (no bias) as per
        standard connectome-based RC practice.
    input_connectivity : float, default=1.0
        Fraction of reservoir neurons that receive input (density of W_in).
        For connectome reservoirs, typically all nodes receive input.
    ridge_alpha : float, default=1e-5
        L2 regularization parameter for the readout ridge regression.
        Prevents overfitting of the linear readout layer.
    activation : str, default='tanh'
        Activation function: 'tanh', 'sigmoid', 'relu', or 'identity'.
    normalize_connectivity : str, default='spectral'
        How to normalize W before scaling to target spectral radius:
        'spectral': divide by actual spectral radius (standard);
        'max': divide by max absolute value;
        'none': use raw connectivity (not recommended).
    symmetrize : bool, default=True
        Whether to symmetrize the connectivity matrix before use.
        Brain structural connectivity is inherently undirected (DTI
        cannot distinguish afferent from efferent fibers).
    seed : Optional[int], default=None
        Random seed for reproducibility (affects W_in generation).

    Attributes
    ----------
    W : np.ndarray
        Scaled reservoir weight matrix.
    W_in : np.ndarray
        Input weight matrix.
    n_neurons : int
        Number of reservoir neurons (= number of ROIs in the atlas).
    states : np.ndarray
        Reservoir states from the most recent run, shape (T, N).
    readout : Ridge
        Trained readout model.

    Examples
    --------
    >>> from sars.config import get_sc_path
    >>> import numpy as np
    >>> sc = np.load(get_sc_path("sub-01", "schaefer_100"))
    >>> esn = ConnectomeReservoir(sc, spectral_radius=0.9, leak_rate=0.3)
    >>> ts = np.load(get_timeseries_path("sub-01", "schaefer_100"))
    >>> # Use first half of ROIs as input, predict second half
    >>> u_train = ts[:100, :50]
    >>> y_train = ts[:100, 50:]
    >>> esn.fit(u_train, y_train)
    >>> y_pred = esn.predict(ts[100:150, :50])

    References
    ----------
    Suárez et al. (2024). Nature Communications, 15, 656.
    Jaeger (2001). GMD Report 148.
    """

    def __init__(
        self,
        connectivity: np.ndarray,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        input_scaling: float = 0.1,
        bias_scaling: float = 0.0,
        input_connectivity: float = 1.0,
        ridge_alpha: float = 1e-5,
        activation: str = "tanh",
        normalize_connectivity: str = "spectral",
        symmetrize: bool = True,
        seed: Optional[int] = None,
    ):
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.input_connectivity = input_connectivity
        self.ridge_alpha = ridge_alpha
        self.activation_name = activation
        self.seed = seed

        self._rng = np.random.RandomState(seed)

        # ── Process connectivity matrix ─────────────────────────────────
        W = connectivity.copy().astype(np.float64)
        np.fill_diagonal(W, 0.0)

        if symmetrize:
            W = (W + W.T) / 2.0

        # Normalize to unit spectral radius, then scale
        if normalize_connectivity == "spectral":
            eigs = linalg.eigvalsh(W) if np.allclose(W, W.T) else linalg.eigvals(W)
            sr = np.max(np.abs(eigs))
            if sr > 0:
                W = W / sr
        elif normalize_connectivity == "max":
            m = np.max(np.abs(W))
            if m > 0:
                W = W / m

        self.W = W * spectral_radius
        self.n_neurons = W.shape[0]

        # ── Activation function ─────────────────────────────────────────
        self._activation = self._get_activation(activation)

        # ── Internal state ──────────────────────────────────────────────
        self.states: Optional[np.ndarray] = None
        self.readout: Optional[Ridge] = None
        self.W_in: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._is_fitted = False

        logger.info(
            f"ConnectomeReservoir initialized: {self.n_neurons} neurons, "
            f"ρ={spectral_radius:.2f}, α={leak_rate:.2f}"
        )

    # ─── Activation functions ───────────────────────────────────────────
    @staticmethod
    def _get_activation(name: str):
        """Return activation function by name."""
        activations = {
            "tanh": np.tanh,
            "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
            "relu": lambda x: np.maximum(0, x),
            "identity": lambda x: x,
        }
        if name not in activations:
            raise ValueError(
                f"Unknown activation '{name}'. "
                f"Choose from: {list(activations.keys())}"
            )
        return activations[name]

    # ─── Input weight matrix generation ─────────────────────────────────
    def _generate_input_weights(self, n_inputs: int) -> np.ndarray:
        """Generate the input weight matrix W_in.

        Creates a sparse random matrix with entries drawn from
        {-input_scaling, 0, +input_scaling} following the standard
        ESN initialization scheme.

        Parameters
        ----------
        n_inputs : int
            Dimensionality of the input signal.

        Returns
        -------
        np.ndarray, shape (n_neurons, n_inputs)
        """
        W_in = self._rng.uniform(-1, 1, (self.n_neurons, n_inputs))

        # Apply input connectivity (sparsify)
        if self.input_connectivity < 1.0:
            mask = self._rng.rand(self.n_neurons, n_inputs) < self.input_connectivity
            W_in *= mask

        W_in *= self.input_scaling
        return W_in

    # ─── Core dynamics: drive the reservoir ─────────────────────────────
    def _run(
        self,
        inputs: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        wash_out: int = 0,
    ) -> np.ndarray:
        """Drive the reservoir with input signals and record states.

        Implements the leaky integrator ESN update equation:
            x(t) = (1 − α) · x(t−1) + α · f(W_in · u(t) + W · x(t−1) + b)

        Parameters
        ----------
        inputs : np.ndarray, shape (T, n_inputs)
            Input time series.
        initial_state : np.ndarray, optional
            Initial reservoir state. Defaults to zeros.
        wash_out : int, default=0
            Number of initial timesteps to discard (transient).

        Returns
        -------
        np.ndarray, shape (T − wash_out, n_neurons)
            Reservoir states after wash_out.
        """
        T, n_inputs = inputs.shape

        # Initialize input weights if needed
        if self.W_in is None or self.W_in.shape[1] != n_inputs:
            self.W_in = self._generate_input_weights(n_inputs)

        # Initialize bias
        if self._bias is None:
            self._bias = self._rng.uniform(
                -self.bias_scaling, self.bias_scaling, self.n_neurons
            ) if self.bias_scaling > 0 else np.zeros(self.n_neurons)

        # Initialize state
        x = initial_state if initial_state is not None else np.zeros(self.n_neurons)
        states = np.zeros((T, self.n_neurons))

        # Drive the reservoir
        for t in range(T):
            pre_activation = (
                self.W_in @ inputs[t]
                + self.W @ x
                + self._bias
            )
            x = (1 - self.leak_rate) * x + self.leak_rate * self._activation(
                pre_activation
            )
            states[t] = x

        self.states = states
        return states[wash_out:]

    # ─── Training (fit readout) ─────────────────────────────────────────
    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        wash_out: int = 50,
        add_bias_feature: bool = True,
    ) -> "ConnectomeReservoir":
        """Train the readout layer via ridge regression.

        The reservoir weights W are FIXED (from the connectome). Only the
        readout weights are learned, making this approach equivalent to
        a high-dimensional nonlinear kernel followed by linear regression.

        Parameters
        ----------
        inputs : np.ndarray, shape (T, n_inputs)
            Input time series for training.
        targets : np.ndarray, shape (T, n_outputs)
            Target time series for training.
        wash_out : int, default=50
            Initial transient to discard.
        add_bias_feature : bool, default=True
            Whether to append a constant 1 column to reservoir states
            (intercept term for the readout).

        Returns
        -------
        self
        """
        if inputs.ndim == 1:
            inputs = inputs[:, np.newaxis]
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]

        states = self._run(inputs, wash_out=wash_out)
        y = targets[wash_out:]

        if add_bias_feature:
            states = np.hstack([states, np.ones((states.shape[0], 1))])

        self.readout = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        self.readout.fit(states, y)
        self._is_fitted = True
        self._add_bias_feature = add_bias_feature

        # Report training performance
        y_pred = self.readout.predict(states)
        r2 = r2_score(y, y_pred, multioutput="variance_weighted")
        logger.info(f"Readout trained: R² = {r2:.4f} (train)")

        return self

    # ─── Prediction ─────────────────────────────────────────────────────
    def predict(
        self,
        inputs: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        wash_out: int = 0,
    ) -> np.ndarray:
        """Predict target from new inputs.

        Parameters
        ----------
        inputs : np.ndarray, shape (T, n_inputs)
        initial_state : np.ndarray, optional
        wash_out : int, default=0

        Returns
        -------
        np.ndarray, shape (T − wash_out, n_outputs)
        """
        if not self._is_fitted:
            raise RuntimeError("Reservoir not fitted. Call .fit() first.")

        if inputs.ndim == 1:
            inputs = inputs[:, np.newaxis]

        states = self._run(inputs, initial_state=initial_state, wash_out=wash_out)

        if self._add_bias_feature:
            states = np.hstack([states, np.ones((states.shape[0], 1))])

        return self.readout.predict(states)

    # ─── Score ──────────────────────────────────────────────────────────
    def score(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        wash_out: int = 0,
    ) -> float:
        """Compute R² score on test data.

        Parameters
        ----------
        inputs : np.ndarray, shape (T, n_inputs)
        targets : np.ndarray, shape (T, n_outputs)
        wash_out : int

        Returns
        -------
        float
            R² (variance-weighted across output dimensions).
        """
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]

        y_pred = self.predict(inputs, wash_out=wash_out)
        y_true = targets[wash_out:]

        return r2_score(y_true, y_pred, multioutput="variance_weighted")

    # ─── Cross-validated evaluation ─────────────────────────────────────
    def cross_validate(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        n_splits: int = 5,
        wash_out: int = 50,
    ) -> Dict[str, Any]:
        """Time-series cross-validation (walk-forward).

        Splits the time series into contiguous folds to respect temporal
        ordering (no random shuffling). Reports R² and RMSE per fold.

        Parameters
        ----------
        inputs : np.ndarray, shape (T, n_inputs)
        targets : np.ndarray, shape (T, n_outputs)
        n_splits : int, default=5
        wash_out : int, default=50

        Returns
        -------
        dict with keys:
            'r2_scores': list of R² per fold
            'rmse_scores': list of RMSE per fold
            'mean_r2': float
            'std_r2': float
        """
        if inputs.ndim == 1:
            inputs = inputs[:, np.newaxis]
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]

        T = inputs.shape[0]
        fold_size = T // (n_splits + 1)
        r2_scores = []
        rmse_scores = []

        for i in range(n_splits):
            train_end = fold_size * (i + 1)
            test_end = min(fold_size * (i + 2), T)

            u_train = inputs[:train_end]
            y_train = targets[:train_end]
            u_test = inputs[train_end:test_end]
            y_test = targets[train_end:test_end]

            # Fresh reservoir for each fold (same W, new readout)
            self.fit(u_train, y_train, wash_out=wash_out)

            if self.states is not None:
                init_state = self.states[-1]
            else:
                init_state = None

            y_pred = self.predict(u_test, initial_state=init_state)
            r2 = r2_score(y_test, y_pred, multioutput="variance_weighted")
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            r2_scores.append(r2)
            rmse_scores.append(rmse)

        return {
            "r2_scores": r2_scores,
            "rmse_scores": rmse_scores,
            "mean_r2": np.mean(r2_scores),
            "std_r2": np.std(r2_scores),
        }

    # ─── ReservoirPy backend ────────────────────────────────────────────
    def to_reservoirpy(self) -> Any:
        """Create an equivalent ReservoirPy ESN pipeline.

        Returns
        -------
        reservoirpy.nodes.Model
            Reservoir >> Ridge pipeline.

        Raises
        ------
        ImportError
            If ReservoirPy is not installed.
        """
        if not _HAS_RESERVOIRPY:
            raise ImportError(
                "ReservoirPy not installed. Install with: pip install reservoirpy"
            )

        reservoir = Reservoir(
            units=self.n_neurons,
            W=self.W,
            sr=self.spectral_radius,
            lr=self.leak_rate,
            input_scaling=self.input_scaling,
            seed=self.seed,
        )
        readout = RpyRidge(ridge=self.ridge_alpha)
        model = reservoir >> readout

        logger.info("Created ReservoirPy model pipeline.")
        return model


# =============================================================================
# ADAPTIVE RESERVOIR (Hebbian Architecture — HAG-inspired)
# =============================================================================

class AdaptiveReservoir(ConnectomeReservoir):
    """Echo State Network with Hebbian synaptic plasticity.

    Extends the standard ConnectomeReservoir by allowing the reservoir
    weight matrix W to update online via a local Hebbian learning rule.
    This models activity-dependent synaptic modification observed in
    biological neural networks, inspired by the Hebbian Architecture
    for Graphs (HAG) framework (Nature Communications, 2025).

    The update rule at each timestep is:

        ΔW_ij = η · (x_i(t) · x_j(t) − λ · W_ij)

    where:
        η = learning rate (controls plasticity speed)
        λ = decay rate (prevents unbounded growth, enforces weight homeostasis)

    After each update, W is re-normalized to maintain the target spectral
    radius, ensuring dynamic stability (echo state property is preserved).

    This is neuroscientifically relevant for the COVID-19 study because
    post-ICU patients may exhibit altered neuroplasticity patterns, and
    the adaptive reservoir can reveal whether a patient's structural
    connectome supports or impairs activity-dependent remodeling.

    Parameters
    ----------
    connectivity : np.ndarray, shape (N, N)
        Structural connectivity matrix.
    learning_rate : float, default=0.001
        Hebbian learning rate η. Small values ensure gradual adaptation;
        larger values model rapid plasticity (e.g., post-injury).
    decay_rate : float, default=0.01
        Weight decay λ for homeostatic regulation. Prevents runaway
        Hebbian potentiation and enforces sparse connectivity.
    plasticity_mask : Optional[np.ndarray], shape (N, N)
        Binary mask specifying which connections are plastic (1) vs
        fixed (0). Defaults to all connections being plastic. Can be
        used to restrict plasticity to specific networks (e.g., only
        allow intra-DMN plasticity).
    renormalize_every : int, default=10
        Re-normalize W to target spectral radius every N timesteps.
        Balances stability (frequent renorm) vs computational cost.
    **kwargs
        Additional keyword arguments passed to ConnectomeReservoir.

    Examples
    --------
    >>> sc = np.load(get_sc_path("sub-01", "schaefer_100"))
    >>> adaptive_esn = AdaptiveReservoir(
    ...     sc, spectral_radius=0.9, learning_rate=0.001, decay_rate=0.01
    ... )
    >>> adaptive_esn.fit(u_train, y_train)
    >>> # Access the evolved weight matrix
    >>> W_evolved = adaptive_esn.W.copy()
    >>> # Compare with initial: ΔW reveals plasticity patterns
    >>> delta_W = W_evolved - sc_normalized
    """

    def __init__(
        self,
        connectivity: np.ndarray,
        learning_rate: float = 0.001,
        decay_rate: float = 0.01,
        plasticity_mask: Optional[np.ndarray] = None,
        renormalize_every: int = 10,
        **kwargs,
    ):
        super().__init__(connectivity, **kwargs)

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.renormalize_every = renormalize_every

        # Store initial weights for comparison
        self.W_initial = self.W.copy()

        # Plasticity mask: which connections can change
        if plasticity_mask is not None:
            self.plasticity_mask = plasticity_mask.astype(np.float64)
        else:
            self.plasticity_mask = np.ones_like(self.W)
            np.fill_diagonal(self.plasticity_mask, 0.0)

        # Track weight evolution
        self.weight_history: List[float] = []  # Frobenius norm of ΔW

        logger.info(
            f"AdaptiveReservoir: η={learning_rate}, λ={decay_rate}, "
            f"plastic_fraction={self.plasticity_mask.sum() / self.plasticity_mask.size:.2%}"
        )

    def _hebbian_update(self, x: np.ndarray) -> None:
        """Apply one step of Hebbian plasticity to W.

        Parameters
        ----------
        x : np.ndarray, shape (n_neurons,)
            Current reservoir state vector.
        """
        # Outer product of activations (Hebbian term)
        hebbian = np.outer(x, x)

        # Weight update: ΔW = η * (x_i * x_j − λ * W_ij)
        dW = self.learning_rate * (hebbian - self.decay_rate * self.W)

        # Apply plasticity mask
        dW *= self.plasticity_mask

        # Update weights
        self.W += dW

    def _renormalize_weights(self) -> None:
        """Re-normalize W to maintain target spectral radius."""
        if np.allclose(self.W, self.W.T):
            eigs = linalg.eigvalsh(self.W)
        else:
            eigs = linalg.eigvals(self.W)

        current_sr = np.max(np.abs(eigs))
        if current_sr > 0:
            self.W = self.W * (self.spectral_radius / current_sr)

    def _run(
        self,
        inputs: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        wash_out: int = 0,
    ) -> np.ndarray:
        """Drive reservoir with Hebbian plasticity active.

        Overrides ConnectomeReservoir._run() to include Hebbian
        weight updates during the forward pass.
        """
        T, n_inputs = inputs.shape

        if self.W_in is None or self.W_in.shape[1] != n_inputs:
            self.W_in = self._generate_input_weights(n_inputs)

        if self._bias is None:
            self._bias = (
                self._rng.uniform(-self.bias_scaling, self.bias_scaling, self.n_neurons)
                if self.bias_scaling > 0
                else np.zeros(self.n_neurons)
            )

        x = initial_state if initial_state is not None else np.zeros(self.n_neurons)
        states = np.zeros((T, self.n_neurons))

        for t in range(T):
            pre_activation = self.W_in @ inputs[t] + self.W @ x + self._bias
            x = (1 - self.leak_rate) * x + self.leak_rate * self._activation(
                pre_activation
            )
            states[t] = x

            # Hebbian update (after wash-out period to avoid transient noise)
            if t >= wash_out:
                self._hebbian_update(x)

                # Periodic renormalization for stability
                if (t - wash_out) % self.renormalize_every == 0:
                    self._renormalize_weights()

            # Track weight evolution (every 100 steps to reduce overhead)
            if t % 100 == 0:
                delta_norm = np.linalg.norm(self.W - self.W_initial, "fro")
                self.weight_history.append(delta_norm)

        self.states = states
        return states[wash_out:]

    def get_weight_change(self) -> np.ndarray:
        """Return the matrix of weight changes ΔW = W_current − W_initial.

        This reveals which connections were strengthened (ΔW > 0) or
        weakened (ΔW < 0) by the Hebbian process. In the COVID-19
        context, this maps which structural connections are functionally
        reinforced versus suppressed.

        Returns
        -------
        np.ndarray, shape (N, N)
            ΔW matrix (signed).
        """
        return self.W - self.W_initial

    def get_plasticity_summary(self) -> Dict[str, float]:
        """Summarize the Hebbian plasticity effects.

        Returns
        -------
        dict with keys:
            'total_change_fro': Frobenius norm of ΔW
            'mean_potentiation': mean of positive ΔW entries
            'mean_depression': mean of negative ΔW entries
            'fraction_potentiated': fraction of edges that strengthened
            'fraction_depressed': fraction of edges that weakened
            'spectral_radius_final': final spectral radius of W
        """
        dW = self.get_weight_change()
        mask = self.plasticity_mask > 0
        dW_plastic = dW[mask]

        potentiated = dW_plastic[dW_plastic > 0]
        depressed = dW_plastic[dW_plastic < 0]

        if np.allclose(self.W, self.W.T):
            sr_final = np.max(np.abs(linalg.eigvalsh(self.W)))
        else:
            sr_final = np.max(np.abs(linalg.eigvals(self.W)))

        return {
            "total_change_fro": np.linalg.norm(dW, "fro"),
            "mean_potentiation": potentiated.mean() if len(potentiated) > 0 else 0.0,
            "mean_depression": depressed.mean() if len(depressed) > 0 else 0.0,
            "fraction_potentiated": len(potentiated) / max(len(dW_plastic), 1),
            "fraction_depressed": len(depressed) / max(len(dW_plastic), 1),
            "spectral_radius_final": sr_final,
        }


# =============================================================================
# RESERVOIR CHARACTERIZATION METRICS
# =============================================================================

def memory_capacity(
    reservoir: ConnectomeReservoir,
    max_delay: int = 50,
    n_timesteps: int = 5000,
    wash_out: int = 500,
    seed: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """Compute the short-term memory capacity (MC) of a reservoir.

    Memory capacity measures the reservoir's ability to linearly
    reconstruct delayed versions of a random input signal u(t−δ)
    from its current state x(t). The total MC is:

        MC = Σ_{δ=1}^{δ_max} MC_δ

    where MC_δ = R²(u_{t−δ}, ŷ_δ(t)) is the coefficient of determination
    of the ridge regression readout trained to reconstruct u_{t−δ}.

    Theoretical bounds: MC ≤ N_neurons (Jaeger, 2001). For networks
    operating at the edge of chaos, MC approaches N_neurons.

    Parameters
    ----------
    reservoir : ConnectomeReservoir
        The reservoir to evaluate.
    max_delay : int, default=50
        Maximum delay δ_max to evaluate.
    n_timesteps : int, default=5000
        Length of the random input signal.
    wash_out : int, default=500
        Transient timesteps to discard.
    seed : int, optional
        Random seed for the input signal.

    Returns
    -------
    total_mc : float
        Total memory capacity.
    mc_profile : np.ndarray, shape (max_delay,)
        MC_δ for each delay.
    """
    rng = np.random.RandomState(seed or reservoir.seed)
    u = rng.uniform(-0.5, 0.5, (n_timesteps, 1))

    states = reservoir._run(u, wash_out=wash_out)
    u_effective = u[wash_out:]

    T = states.shape[0]
    mc_profile = np.zeros(max_delay)

    for delta in range(1, max_delay + 1):
        if delta >= T:
            break

        # Target: u(t − δ)
        y_target = u_effective[:-delta].ravel()
        X = states[delta:]

        # Ridge regression
        reg = Ridge(alpha=1e-6, fit_intercept=True)
        reg.fit(X, y_target)
        y_pred = reg.predict(X)

        r2 = r2_score(y_target, y_pred)
        mc_profile[delta - 1] = max(r2, 0.0)  # Clamp negatives

    total_mc = mc_profile.sum()
    logger.info(
        f"Memory capacity: MC = {total_mc:.2f} "
        f"(theoretical max = {reservoir.n_neurons})"
    )
    return total_mc, mc_profile


def kernel_quality(
    reservoir: ConnectomeReservoir,
    n_patterns: int = 200,
    pattern_length: int = 50,
    wash_out: int = 50,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluate the kernel quality (separation property) of a reservoir.

    Measures how well the reservoir separates distinct input patterns
    into distinct internal states. This is quantified via the effective
    rank of the state matrix X collected from many input patterns.

    Kernel quality = effective_rank(X) / N_neurons

    A high kernel quality means the reservoir acts as a rich nonlinear
    kernel — essential for downstream classification/regression tasks.

    Parameters
    ----------
    reservoir : ConnectomeReservoir
        The reservoir to evaluate.
    n_patterns : int, default=200
        Number of distinct random input patterns.
    pattern_length : int, default=50
        Length of each input pattern (timesteps).
    wash_out : int, default=50
        Transient to discard before recording the terminal state.
    seed : int, optional

    Returns
    -------
    kq : float
        Kernel quality ∈ [0, 1].
    gen_rank : float
        Generalization rank (effective dimensionality / N_neurons).
    """
    rng = np.random.RandomState(seed or reservoir.seed)

    terminal_states = np.zeros((n_patterns, reservoir.n_neurons))

    for i in range(n_patterns):
        # Generate random input pattern
        u = rng.uniform(-0.5, 0.5, (pattern_length + wash_out, 1))
        states = reservoir._run(u, wash_out=wash_out)
        terminal_states[i] = states[-1]

    # Compute effective rank via singular values
    _, s, _ = linalg.svd(terminal_states, full_matrices=False)

    # Normalize singular values
    s_normalized = s / s.sum()

    # Effective rank (exponential of entropy of normalized singular values)
    # Roy & Vetterli (2007)
    entropy = -np.sum(s_normalized * np.log(s_normalized + 1e-15))
    effective_rank = np.exp(entropy)

    kq = effective_rank / reservoir.n_neurons

    # Generalization rank: fraction of variance explained by top 90% SVs
    cumvar = np.cumsum(s ** 2) / np.sum(s ** 2)
    n_90 = np.searchsorted(cumvar, 0.90) + 1
    gen_rank = n_90 / reservoir.n_neurons

    logger.info(
        f"Kernel quality: KQ = {kq:.3f}, "
        f"effective rank = {effective_rank:.1f}/{reservoir.n_neurons}, "
        f"90% variance in {n_90} dimensions"
    )
    return kq, gen_rank


def echo_state_property_index(
    reservoir: ConnectomeReservoir,
    n_trials: int = 10,
    n_timesteps: int = 1000,
    wash_out: int = 200,
    seed: Optional[int] = None,
) -> float:
    """Empirical verification of the Echo State Property (ESP).

    The ESP requires that reservoir states asymptotically depend only
    on the driving input, not on initial conditions. We verify this by
    driving the reservoir with the SAME input sequence from DIFFERENT
    random initial conditions, and measuring the convergence of states.

    ESP_index = 1 − mean(||x_i(T) − x_j(T)||) / normalization

    Values close to 1.0 indicate strong ESP. Based on Woo et al. (2024).

    Parameters
    ----------
    reservoir : ConnectomeReservoir
    n_trials : int, default=10
        Number of different initial conditions to test.
    n_timesteps : int, default=1000
    wash_out : int, default=200
    seed : int, optional

    Returns
    -------
    float
        ESP index ∈ [0, 1].
    """
    rng = np.random.RandomState(seed or reservoir.seed)

    # Same input for all trials
    u = rng.uniform(-0.5, 0.5, (n_timesteps, 1))

    terminal_states = np.zeros((n_trials, reservoir.n_neurons))

    for trial in range(n_trials):
        # Different random initial state
        x0 = rng.uniform(-1, 1, reservoir.n_neurons) * 0.5
        states = reservoir._run(u, initial_state=x0, wash_out=0)
        terminal_states[trial] = states[-1]

    # Compute pairwise distances between terminal states
    from scipy.spatial.distance import pdist

    distances = pdist(terminal_states, metric="euclidean")

    if len(distances) == 0:
        return 1.0

    # Normalize by expected distance for random states
    mean_dist = distances.mean()
    max_possible = np.sqrt(reservoir.n_neurons) * 2  # rough upper bound

    esp_idx = 1.0 - np.clip(mean_dist / max_possible, 0, 1)

    logger.info(f"Echo State Property index: {esp_idx:.4f}")
    return esp_idx


def lyapunov_exponent(
    reservoir: ConnectomeReservoir,
    n_timesteps: int = 5000,
    wash_out: int = 500,
    seed: Optional[int] = None,
) -> float:
    """Estimate the maximum Lyapunov exponent of the reservoir dynamics.

    The Lyapunov exponent λ_max characterizes the rate of divergence
    of nearby trajectories in state space:

        λ_max < 0  →  ordered (stable fixed point)
        λ_max ≈ 0  →  edge of chaos (optimal for computation)
        λ_max > 0  →  chaotic (sensitive dependence on initial conditions)

    Estimated via the Jacobian method: at each timestep, we compute the
    local Jacobian of the reservoir update and track the growth rate of
    perturbation vectors.

    Parameters
    ----------
    reservoir : ConnectomeReservoir
    n_timesteps : int
    wash_out : int
    seed : int, optional

    Returns
    -------
    float
        Estimated maximum Lyapunov exponent.
    """
    rng = np.random.RandomState(seed or reservoir.seed)
    u = rng.uniform(-0.5, 0.5, (n_timesteps, 1))

    # Initialize
    if reservoir.W_in is None:
        reservoir.W_in = reservoir._generate_input_weights(1)
    if reservoir._bias is None:
        reservoir._bias = np.zeros(reservoir.n_neurons)

    x = np.zeros(reservoir.n_neurons)
    alpha = reservoir.leak_rate

    # Perturbation vector (random unit vector)
    delta = rng.randn(reservoir.n_neurons)
    delta /= np.linalg.norm(delta)

    lyap_sum = 0.0
    count = 0

    for t in range(n_timesteps):
        pre = reservoir.W_in @ u[t] + reservoir.W @ x + reservoir._bias
        x_new = (1 - alpha) * x + alpha * reservoir._activation(pre)

        if t >= wash_out:
            # Jacobian: J = (1−α)I + α · diag(f'(pre)) · W
            if reservoir.activation_name == "tanh":
                f_prime = 1 - np.tanh(pre) ** 2
            elif reservoir.activation_name == "sigmoid":
                sig = 1.0 / (1.0 + np.exp(-np.clip(pre, -500, 500)))
                f_prime = sig * (1 - sig)
            elif reservoir.activation_name == "relu":
                f_prime = (pre > 0).astype(float)
            else:
                f_prime = np.ones_like(pre)

            J = (1 - alpha) * np.eye(reservoir.n_neurons) + alpha * np.diag(f_prime) @ reservoir.W

            # Evolve perturbation
            delta = J @ delta
            norm_delta = np.linalg.norm(delta)

            if norm_delta > 0:
                lyap_sum += np.log(norm_delta)
                delta /= norm_delta  # Re-normalize
                count += 1

        x = x_new

    lyap_max = lyap_sum / max(count, 1)
    logger.info(f"Max Lyapunov exponent: λ_max = {lyap_max:.4f}")
    return lyap_max


def spectral_analysis(
    connectivity: np.ndarray,
    spectral_radius: float = 0.9,
    symmetrize: bool = True,
) -> Dict[str, Any]:
    """Analyze the eigenspectrum of the reservoir weight matrix.

    The eigenvalues of the reservoir weight matrix determine its
    fundamental dynamic properties: timescale hierarchy, oscillatory
    modes, and stability boundaries.

    Parameters
    ----------
    connectivity : np.ndarray, shape (N, N)
        Structural connectivity matrix.
    spectral_radius : float
        Target spectral radius for scaling.
    symmetrize : bool

    Returns
    -------
    dict with keys:
        'eigenvalues': np.ndarray (complex or real)
        'spectral_radius': float (actual ρ)
        'spectral_gap': float (gap between largest and second eigenvalue)
        'n_oscillatory_modes': int (eigenvalues with imaginary part > 0.01)
        'effective_timescales': np.ndarray (1 / (1 − |λ_i|) for each mode)
        'condition_number': float
    """
    W = connectivity.copy().astype(np.float64)
    np.fill_diagonal(W, 0.0)

    if symmetrize:
        W = (W + W.T) / 2.0
        eigenvalues = linalg.eigvalsh(W)
        eigenvalues = eigenvalues.astype(complex)  # Uniform type
    else:
        eigenvalues = linalg.eigvals(W)

    # Scale to target spectral radius
    sr_actual = np.max(np.abs(eigenvalues))
    if sr_actual > 0:
        eigenvalues = eigenvalues * (spectral_radius / sr_actual)
        sr_actual = spectral_radius

    # Sort by magnitude (descending)
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]

    abs_eigs = np.abs(eigenvalues)

    # Spectral gap
    spectral_gap = abs_eigs[0] - abs_eigs[1] if len(abs_eigs) > 1 else abs_eigs[0]

    # Oscillatory modes (complex eigenvalues)
    n_osc = np.sum(np.abs(eigenvalues.imag) > 0.01)

    # Effective timescales: τ_i = 1 / (1 − |λ_i|)
    # Only meaningful for |λ_i| < 1
    safe_abs = np.clip(abs_eigs, 0, 0.999)
    timescales = 1.0 / (1.0 - safe_abs)

    # Condition number
    cond = abs_eigs[0] / abs_eigs[-1] if abs_eigs[-1] > 1e-10 else np.inf

    return {
        "eigenvalues": eigenvalues,
        "spectral_radius": sr_actual,
        "spectral_gap": spectral_gap,
        "n_oscillatory_modes": int(n_osc),
        "effective_timescales": timescales,
        "condition_number": cond,
    }


def characterize_reservoir(
    reservoir: ConnectomeReservoir,
    max_delay: int = 50,
    n_mc_timesteps: int = 3000,
    n_kq_patterns: int = 150,
    n_esp_trials: int = 10,
    compute_lyapunov: bool = True,
    seed: Optional[int] = None,
) -> ReservoirMetrics:
    """Complete characterization of a reservoir's computational properties.

    Computes all standard metrics in a single call: memory capacity,
    kernel quality, ESP index, Lyapunov exponent, and spectral properties.

    This is the recommended entry point for evaluating a single reservoir.

    Parameters
    ----------
    reservoir : ConnectomeReservoir
    max_delay : int
    n_mc_timesteps : int
    n_kq_patterns : int
    n_esp_trials : int
    compute_lyapunov : bool
    seed : int, optional

    Returns
    -------
    ReservoirMetrics
        Complete characterization.
    """
    logger.info("Characterizing reservoir...")

    # Memory capacity
    mc_total, mc_profile = memory_capacity(
        reservoir, max_delay=max_delay, n_timesteps=n_mc_timesteps, seed=seed
    )

    # Kernel quality
    kq, gen_rank = kernel_quality(
        reservoir, n_patterns=n_kq_patterns, seed=seed
    )

    # Echo state property
    esp = echo_state_property_index(
        reservoir, n_trials=n_esp_trials, seed=seed
    )

    # Spectral radius (actual, from current W)
    if np.allclose(reservoir.W, reservoir.W.T):
        sr = np.max(np.abs(linalg.eigvalsh(reservoir.W)))
    else:
        sr = np.max(np.abs(linalg.eigvals(reservoir.W)))

    # Lyapunov exponent
    lyap = None
    if compute_lyapunov:
        lyap = lyapunov_exponent(reservoir, seed=seed)

    metrics = ReservoirMetrics(
        memory_capacity=mc_total,
        memory_profile=mc_profile,
        kernel_quality=kq,
        generalization_rank=gen_rank,
        spectral_radius=sr,
        lyapunov_exponent=lyap,
        esp_index=esp,
    )

    logger.info(
        f"Characterization complete: MC={mc_total:.2f}, KQ={kq:.3f}, "
        f"ESP={esp:.3f}, ρ={sr:.3f}"
        + (f", λ_max={lyap:.4f}" if lyap is not None else "")
    )
    return metrics


# =============================================================================
# RESERVOIR ARCHITECTURE COMPARISON
# =============================================================================

def _generate_random_reservoir(n: int, density: float, seed: int) -> np.ndarray:
    """Generate a random Erdős–Rényi reservoir."""
    rng = np.random.RandomState(seed)
    W = rng.randn(n, n) * (rng.rand(n, n) < density)
    np.fill_diagonal(W, 0.0)
    W = (W + W.T) / 2
    return W


def _generate_smallworld_reservoir(
    n: int, k: int = 6, p: float = 0.1, seed: int = 42
) -> np.ndarray:
    """Generate a Watts-Strogatz small-world reservoir."""
    try:
        import networkx as nx

        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        W = nx.to_numpy_array(G).astype(np.float64)
        # Add random weights
        rng = np.random.RandomState(seed)
        weights = rng.uniform(0.1, 1.0, W.shape)
        W = W * weights
        W = (W + W.T) / 2
        return W
    except ImportError:
        warnings.warn("NetworkX not available; falling back to random network.")
        return _generate_random_reservoir(n, k / n, seed)


def _generate_ring_reservoir(n: int, bandwidth: int = 3) -> np.ndarray:
    """Generate a ring/band-diagonal reservoir (chain topology)."""
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(1, bandwidth + 1):
            W[i, (i + j) % n] = 1.0 / j
            W[i, (i - j) % n] = 1.0 / j
    return W


def compare_reservoir_architectures(
    connectome: np.ndarray,
    input_signal: np.ndarray,
    target_signal: np.ndarray,
    architectures: Optional[List[str]] = None,
    spectral_radius: float = 0.9,
    leak_rate: float = 0.3,
    ridge_alpha: float = 1e-5,
    wash_out: int = 50,
    max_delay_mc: int = 50,
    seed: int = 42,
) -> ReservoirComparison:
    """Compare connectome-based reservoir against synthetic architectures.

    This is a key analysis for demonstrating that the patient's actual
    brain structural connectivity provides computational advantages (or
    disadvantages) compared to generic network topologies.

    If the connectome reservoir outperforms random/small-world alternatives,
    it suggests that the patient's brain structure is computationally
    well-organized. Conversely, if it underperforms, this may indicate
    structural disorganization (e.g., from COVID-19 white matter damage).

    Parameters
    ----------
    connectome : np.ndarray, shape (N, N)
        Patient's structural connectivity matrix.
    input_signal : np.ndarray, shape (T, n_inputs)
        Input signal for the prediction task.
    target_signal : np.ndarray, shape (T, n_outputs)
        Target signal for the prediction task.
    architectures : list of str, optional
        Which architectures to compare. Default: all four.
        Options: 'connectome', 'random', 'small_world', 'ring'
    spectral_radius : float
    leak_rate : float
    ridge_alpha : float
    wash_out : int
    max_delay_mc : int
    seed : int

    Returns
    -------
    ReservoirComparison
        Comprehensive comparison results.
    """
    if architectures is None:
        architectures = ["connectome", "random", "small_world", "ring"]

    n = connectome.shape[0]
    density = (connectome > 0).sum() / (n * (n - 1))

    # Generate architecture matrices
    arch_matrices = {}
    if "connectome" in architectures:
        arch_matrices["connectome"] = connectome

    if "random" in architectures:
        arch_matrices["random"] = _generate_random_reservoir(n, density, seed)

    if "small_world" in architectures:
        k = max(2, int(density * n))
        arch_matrices["small_world"] = _generate_smallworld_reservoir(n, k, 0.1, seed)

    if "ring" in architectures:
        arch_matrices["ring"] = _generate_ring_reservoir(n, bandwidth=3)

    # Evaluate each architecture
    comparison = ReservoirComparison()
    comparison.architecture_names = list(arch_matrices.keys())

    best_r2 = -np.inf

    for name, W in arch_matrices.items():
        logger.info(f"Evaluating architecture: {name}")

        # Create reservoir
        res = ConnectomeReservoir(
            W,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            ridge_alpha=ridge_alpha,
            seed=seed,
        )

        # ── Characterization metrics ────────────────────────────────────
        metrics = characterize_reservoir(
            res,
            max_delay=max_delay_mc,
            compute_lyapunov=False,  # Skip for speed in comparison
            seed=seed,
        )
        comparison.metrics[name] = metrics

        # ── Task performance ────────────────────────────────────────────
        T = input_signal.shape[0]
        split = int(T * 0.7)

        u_train = input_signal[:split]
        y_train = target_signal[:split]
        u_test = input_signal[split:]
        y_test = target_signal[split:]

        res.fit(u_train, y_train, wash_out=wash_out)

        # Get last training state for continuity
        init = res.states[-1] if res.states is not None else None

        y_pred = res.predict(u_test, initial_state=init)
        r2_test = r2_score(
            y_test if y_test.ndim > 1 else y_test[:, np.newaxis],
            y_pred,
            multioutput="variance_weighted",
        )

        comparison.task_scores[name] = {"prediction_r2": r2_test}

        if r2_test > best_r2:
            best_r2 = r2_test
            comparison.best_architecture = name

        logger.info(
            f"  {name}: R²={r2_test:.4f}, MC={metrics.memory_capacity:.2f}, "
            f"KQ={metrics.kernel_quality:.3f}"
        )

    logger.info(f"Best architecture: {comparison.best_architecture} (R²={best_r2:.4f})")
    return comparison


# =============================================================================
# CONVENIENCE: FC PREDICTION TASK
# =============================================================================

def fc_prediction_task(
    sc: np.ndarray,
    timeseries: np.ndarray,
    spectral_radius: float = 0.9,
    leak_rate: float = 0.3,
    ridge_alpha: float = 1e-5,
    wash_out: int = 50,
    train_fraction: float = 0.7,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Predict functional connectivity from structural connectivity via ESN.

    This is the canonical conn2res task: use a subset of fMRI ROIs as
    input, predict the remaining ROIs' time series. The reservoir topology
    is the patient's structural connectome.

    The prediction accuracy R² reflects how well the structural network
    supports the observed functional dynamics — a direct measure of
    SC-FC coupling through the lens of computational capacity.

    Parameters
    ----------
    sc : np.ndarray, shape (N, N)
        Structural connectivity matrix.
    timeseries : np.ndarray, shape (T, N)
        BOLD time series (parcellated, denoised).
    spectral_radius : float
    leak_rate : float
    ridge_alpha : float
    wash_out : int
    train_fraction : float
    seed : int, optional

    Returns
    -------
    dict with keys:
        'r2_global': float — overall prediction R²
        'r2_per_roi': np.ndarray — R² for each output ROI
        'reservoir': ConnectomeReservoir — the fitted reservoir
        'y_true': np.ndarray — true test targets
        'y_pred': np.ndarray — predicted test targets
        'states_test': np.ndarray — reservoir states during test
    """
    rng = np.random.RandomState(seed)
    N = sc.shape[0]
    T = timeseries.shape[0]

    # Split ROIs into input and output (random halves)
    roi_indices = rng.permutation(N)
    n_input = N // 2
    input_rois = roi_indices[:n_input]
    output_rois = roi_indices[n_input:]

    u = timeseries[:, input_rois]
    y = timeseries[:, output_rois]

    # Train/test split (temporal)
    split = int(T * train_fraction)
    u_train, u_test = u[:split], u[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train reservoir
    esn = ConnectomeReservoir(
        sc,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        ridge_alpha=ridge_alpha,
        seed=seed,
    )
    esn.fit(u_train, y_train, wash_out=wash_out)

    # Predict
    init = esn.states[-1] if esn.states is not None else None
    y_pred = esn.predict(u_test, initial_state=init)

    # Per-ROI R²
    r2_per_roi = np.array([
        r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])
    ])

    r2_global = r2_score(y_test, y_pred, multioutput="variance_weighted")

    logger.info(
        f"FC prediction: R²_global={r2_global:.4f}, "
        f"R²_per_roi: mean={r2_per_roi.mean():.4f}, "
        f"std={r2_per_roi.std():.4f}"
    )

    return {
        "r2_global": r2_global,
        "r2_per_roi": r2_per_roi,
        "input_rois": input_rois,
        "output_rois": output_rois,
        "reservoir": esn,
        "y_true": y_test,
        "y_pred": y_pred,
    }


def spectral_radius_sweep(
    connectivity: np.ndarray,
    sr_values: Optional[np.ndarray] = None,
    leak_rate: float = 0.3,
    n_timesteps: int = 3000,
    max_delay: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Sweep spectral radius and measure memory capacity at each value.

    This reveals the "sweet spot" for the reservoir — the spectral radius
    that maximizes computational capacity for the given connectome topology.
    Brain networks with small-world structure typically have broader optimal
    ranges than random networks (Damicelli et al., 2022).

    Parameters
    ----------
    connectivity : np.ndarray, shape (N, N)
    sr_values : np.ndarray, optional
        Spectral radii to test. Default: np.arange(0.1, 1.5, 0.1)
    leak_rate : float
    n_timesteps : int
    max_delay : int
    seed : int, optional

    Returns
    -------
    dict with keys:
        'sr_values': np.ndarray
        'memory_capacities': np.ndarray
        'kernel_qualities': np.ndarray
        'optimal_sr': float (spectral radius at max MC)
    """
    if sr_values is None:
        sr_values = np.arange(0.1, 1.5, 0.1)

    mc_values = np.zeros(len(sr_values))
    kq_values = np.zeros(len(sr_values))

    for i, sr in enumerate(sr_values):
        res = ConnectomeReservoir(
            connectivity,
            spectral_radius=sr,
            leak_rate=leak_rate,
            seed=seed,
        )
        mc, _ = memory_capacity(res, max_delay=max_delay, n_timesteps=n_timesteps, seed=seed)
        kq, _ = kernel_quality(res, n_patterns=100, seed=seed)

        mc_values[i] = mc
        kq_values[i] = kq

        logger.info(f"  ρ={sr:.2f}: MC={mc:.2f}, KQ={kq:.3f}")

    optimal_sr = sr_values[np.argmax(mc_values)]

    return {
        "sr_values": sr_values,
        "memory_capacities": mc_values,
        "kernel_qualities": kq_values,
        "optimal_sr": optimal_sr,
    }
