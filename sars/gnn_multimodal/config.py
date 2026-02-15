# -*- coding: utf-8 -*-
"""
sars.gnn_multimodal.config
==========================

Thin re-export shim.  All configuration dataclasses live in the
central ``sars.config`` module to avoid duplication and circular
imports.  This file exists so that intra-package imports such as::

    from .config import GATConfig, GNNMultimodalConfig

continue to work without modification.
"""

from sars.config import (          # noqa: F401
    GATConfig,
    ContrastiveConfig,
    GNNMultimodalConfig,
    # Convenience re-exports used by other gnn_multimodal modules
    ATLASES,
    ALL_SUBJECT_IDS,
    PROJECT_ROOT,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Backward-compatible aliases expected by __init__.py and other modules
# ---------------------------------------------------------------------------
ATLAS_REGISTRY = ATLASES
"""Full atlas metadata dict (same as sars.config.ATLASES)."""

SUBJECTS = ALL_SUBJECT_IDS
"""Legacy alias for the subject list."""
