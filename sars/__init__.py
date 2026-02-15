# -*- coding: utf-8 -*-
"""
sars — SARS-CoV-2 Brain Connectivity Analysis Library
======================================================

A Python library for multimodal neuroimaging analysis of brain connectivity
in critically ill COVID-19 patients.  Processes resting-state fMRI and
diffusion MRI data across multiple brain atlases (SynthSeg 86, Schaefer 100,
AAL3 170, Brainnetome 246).

Subpackages
-----------
criticality
    Brain criticality analysis: neuronal avalanches, DFA, entropy measures,
    point-process framework.
graph_analysis
    Graph-theoretic network metrics, community detection, NBS.
viz
    Publication-quality matrix and brain visualizations.

Core modules
------------
config
    Centralized paths, parameters, and atlas definitions.
io
    Data loading (timeseries, FC, SC, labels) and saving.
data
    Data containers (ConnectivityData, GroupData), atlas management,
    matrix operations, quality control.
utils
    Statistical testing, surrogate generation, signal processing.

Quick start
-----------
    import sars

    # Load a subject
    from sars.data import ConnectivityData
    sub = ConnectivityData.from_subject("sub-01", "schaefer_100")

    # Criticality analysis
    from sars.criticality import analyze_avalanches, analyze_dfa
    aval = analyze_avalanches(sub.timeseries)
    dfa  = analyze_dfa(sub.timeseries)

    # Graph analysis
    from sars.graph_analysis import analyze_network
    net = analyze_network(sub.fc["correlation"])

    # Visualization
    from sars.viz import plot_connectivity_matrix
    fig, ax = plot_connectivity_matrix(sub.fc["correlation"], atlas="schaefer_100")
"""

__version__ = "0.1.0"
__author__ = "SARS-CoV-2 Neuroimaging Study"

from . import config
from . import io
