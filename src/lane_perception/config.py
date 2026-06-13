"""Shared configuration: paths, figure styling, palette, and sequence metadata."""

from pathlib import Path

import numpy as np

# --- Paths -----------------------------------------------------------------
# Resolve relative to the repository root (two levels up from this file's
# package directory: src/lane_perception/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"


def ensure_dirs():
    """Create the data/ and outputs/ directories if they don't exist."""
    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)


# --- Figure styling --------------------------------------------------------
# Fixed palette mapping each condition to a color used throughout the report.
PALETTE = {
    "clean": "#1f4e79",
    "shadow": "#c44e52",
    "blur": "#5b8c5a",
}

# Per-sequence color pair (used in the cross-sequence comparison figure).
SEQUENCE_PALETTE = ["#1f4e79", "#7a4e7e"]


def set_plot_style():
    """Apply the consistent seaborn theme used for all report figures."""
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.0,
        rc={
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
        },
    )


# --- Perturbation parameters -----------------------------------------------
def make_shadow_poly(height, width):
    """Build the synthetic-shadow quadrilateral for an image of the given size.

    The same fractional coordinates are used for every sequence; only the
    absolute pixel positions change with image dimensions.
    """
    return np.array(
        [
            [int(0.30 * width), int(0.70 * height)],
            [int(0.85 * width), int(0.55 * height)],
            [int(0.90 * width), int(0.65 * height)],
            [int(0.35 * width), int(0.80 * height)],
        ],
        dtype=np.int32,
    )


# --- Sequence metadata -----------------------------------------------------
# `roi` selects which tuned ROI / detector variant to use (see pipeline.py).
SEQUENCES = {
    "drive_0002": {
        "seq_name": "2011_09_26_drive_0002",
        "calib_day": "2011_09_26",
        "roi": "v1",
        "n_frames": 77,  # informational; primary sequence
    },
    "drive_0026": {
        "seq_name": "2011_09_29_drive_0026",
        "calib_day": "2011_09_29",
        "roi": "v2",
        "n_frames": 158,  # informational; cross-sequence validation
    },
}

KITTI_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
