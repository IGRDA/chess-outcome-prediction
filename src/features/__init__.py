"""Leakage-safe, family-organized feature layer.

Each feature family is a deep module exposing a simple ``add_*_features(df) -> df``
interface. :func:`build_feature_matrix` applies them in order to turn the
leak-safe base dataset into the modeling-ready feature matrix.
"""

from __future__ import annotations

from features.build import FEATURE_BUILDERS, build_feature_matrix

__all__ = ["FEATURE_BUILDERS", "build_feature_matrix"]
