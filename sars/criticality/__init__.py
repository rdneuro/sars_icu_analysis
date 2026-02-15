# -*- coding: utf-8 -*-
"""
sars.criticality
==============================

Subpackage for brain criticality analysis from resting-state fMRI data.

The brain operates near a critical point — a phase transition between
ordered and disordered dynamics — which maximizes information processing
capacity, dynamic range, and sensitivity to perturbations.

Modules
-------
avalanches
    Neuronal avalanche detection, power-law fitting (Clauset-Shalizi-
    Newman 2009 framework), branching ratio (conventional + MR
    estimator), crackling-noise scaling relation, kappa statistic,
    avalanche shape collapse, surrogate comparison, group-level pooling,
    sensitivity analysis, and composite criticality index.
dfa
    Detrended Fluctuation Analysis (DFA-1/2) and multifractal extension
    (MF-DFA) for long-range temporal correlations. Includes crossover
    detection and per-ROI analysis.
entropy
    Information-theoretic complexity measures: sample entropy,
    permutation entropy, Lempel-Ziv complexity, multi-scale entropy,
    and spectral entropy.
point_process
    Point-process extraction from continuous BOLD (crossing, exceedance,
    peak modes), spatiotemporal clustering, order/control parameters,
    susceptibility analysis, and autocorrelation decay.

Pipeline
--------
For a complete single-subject analysis:

    from sars.criticality import (
        analyze_avalanches,
        analyze_dfa,
        analyze_entropy,
        analyze_point_process,
    )

    aval = analyze_avalanches(timeseries, threshold_sd=1.0)
    dfa  = analyze_dfa(timeseries)
    ent  = analyze_entropy(timeseries)
    pp   = analyze_point_process(timeseries)

For group-level power-law fitting (recommended):

    from sars.criticality import pool_avalanches_group
    group = pool_avalanches_group(timeseries_list)

References
----------
- Beggs & Plenz (2003). J Neurosci. Neuronal avalanches in neocortical
  circuits.
- Clauset, Shalizi & Newman (2009). SIAM Rev. Power-law distributions
  in empirical data.
- Alstott, Bullmore & Plenz (2014). PLoS ONE. powerlaw: a Python
  package for analysis of heavy-tailed distributions.
- Tagliazucchi et al. (2012). Front Physiol. Criticality in large-scale
  brain fMRI dynamics unveiled by a novel point process analysis.
- Shew & Plenz (2013). Neuroscientist. The functional benefits of
  criticality in the cortex.
- Cocchi et al. (2017). Prog Neurobiol. Criticality in the brain.
- Wilting & Priesemann (2018). Nat Commun. Inferring collective
  dynamical states from widely unobserved systems.
- Peng et al. (1994). Phys Rev E. Mosaic organization of DNA.
- Bandt & Pompe (2002). Phys Rev Lett. Permutation entropy.
- Fontenele et al. (2019). Phys Rev Lett. Criticality between cortical
  states.
"""

# === avalanches ===
from .avalanches import (
    detect_avalanches,
    fit_powerlaw_distribution,
    compute_branching_ratio,
    compute_scaling_relation,
    compute_kappa,
    compute_avalanche_shapes,
    compute_criticality_index,
    compare_with_surrogates,
    pool_avalanches_group,
    sensitivity_analysis,
    analyze_avalanches,
)

# === dfa ===
from .dfa import (
    compute_dfa,
    compute_dfa_per_roi,
    compute_mfdfa,
    analyze_dfa,
)

# === entropy ===
from .entropy import (
    sample_entropy,
    permutation_entropy,
    lempel_ziv_complexity,
    multiscale_entropy,
    spectral_entropy,
    analyze_entropy,
)

# === point_process ===
from .point_process import (
    extract_point_process,
    point_process_fc,
    spatiotemporal_clustering,
    compute_order_parameter,
    compute_control_parameter,
    compute_autocorrelation_decay,
    compute_susceptibility_vs_threshold,
    analyze_point_process,
)


__all__ = [
    # --- avalanches ---
    "detect_avalanches",
    "fit_powerlaw_distribution",
    "compute_branching_ratio",
    "compute_scaling_relation",
    "compute_kappa",
    "compute_avalanche_shapes",
    "compute_criticality_index",
    "compare_with_surrogates",
    "pool_avalanches_group",
    "sensitivity_analysis",
    "analyze_avalanches",
    # --- dfa ---
    "compute_dfa",
    "compute_dfa_per_roi",
    "compute_mfdfa",
    "analyze_dfa",
    # --- entropy ---
    "sample_entropy",
    "permutation_entropy",
    "lempel_ziv_complexity",
    "multiscale_entropy",
    "spectral_entropy",
    "analyze_entropy",
    # --- point_process ---
    "extract_point_process",
    "point_process_fc",
    "spatiotemporal_clustering",
    "compute_order_parameter",
    "compute_control_parameter",
    "compute_autocorrelation_decay",
    "compute_susceptibility_vs_threshold",
    "analyze_point_process",
]
