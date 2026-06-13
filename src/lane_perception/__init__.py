"""Counterfactual stability analysis of classical lane perception.

A classical lane-perception pipeline (Canny -> ROI -> Hough -> fit ->
vanishing point) evaluated under counterfactual perturbations (synthetic
shadow, motion blur) across temporal stability metrics on KITTI sequences.
"""

from . import config, data, metrics, perturbations, pipeline, runner, viz

__all__ = [
    "config",
    "data",
    "metrics",
    "perturbations",
    "pipeline",
    "runner",
    "viz",
]
