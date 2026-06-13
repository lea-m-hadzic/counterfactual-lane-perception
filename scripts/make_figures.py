"""Generate every report figure (and the demo GIFs) into outputs/.

Reads the per-frame signals (.npz) and scalar metrics (.json) produced by
run_analysis.py; re-renders the image-panel figures and GIFs directly from the
KITTI frames. Run `python scripts/run_analysis.py` first.

Usage:
    python scripts/make_figures.py              # all figures + GIFs
    python scripts/make_figures.py --skip-gifs  # PNG figures only (faster)
"""

import argparse
import json

import _bootstrap  # noqa: F401  (adds ../src to sys.path)

import numpy as np

from lane_perception import config, viz
from lane_perception.data import prepare_sequence
from lane_perception.perturbations import apply_motion_blur, apply_shadow
from lane_perception.pipeline import DETECTORS
from lane_perception.viz import _read_rgb

CONDITIONS = ["clean", "shadow", "blur"]


def _load_outputs(seq_key):
    """Load saved vp signals (npz) and metric results (json) for a sequence."""
    npz_path = config.OUT_DIR / f"{seq_key}_results.npz"
    json_path = config.OUT_DIR / f"{seq_key}_metrics.json"
    if not npz_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing outputs for {seq_key}. Run scripts/run_analysis.py first."
        )
    npz = np.load(npz_path)
    vp_x_by_cond = {c: npz[f"vp_x_{c}"] for c in CONDITIONS}
    with open(json_path) as f:
        results = json.load(f)
    return vp_x_by_cond, results


def _shadow_poly_for(frame_paths):
    H, W = _read_rgb(frame_paths[0]).shape[:2]
    return config.make_shadow_poly(H, W)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gifs", action="store_true",
                        help="skip the (slow) demo GIFs")
    args = parser.parse_args()

    config.ensure_dirs()

    # Primary sequence (drive_0002) drives figures 1-5, 7 and the two-metric chart.
    frames = prepare_sequence("drive_0002")
    detect_fn = DETECTORS[config.SEQUENCES["drive_0002"]["roi"]]
    shadow_poly = _shadow_poly_for(frames)
    vp_x_by_cond, results = _load_outputs("drive_0002")

    print("Rendering figures...")
    print(" ", viz.fig_perturbation_examples(frames, shadow_poly, detect_fn).name)
    print(" ", viz.fig_consecutive_frames(frames, detect_fn).name)
    print(" ", viz.fig_vp_timeseries(vp_x_by_cond).name)
    print(" ", viz.fig_cumulative_drift(vp_x_by_cond).name)
    print(" ", viz.fig_autocorrelation(vp_x_by_cond).name)
    print(" ", viz.fig_frame57_deepdive(frames, shadow_poly, detect_fn).name)
    print(" ", viz.fig_two_metrics(results).name)

    # Cross-sequence comparison needs the second sequence's metrics.
    _, results_2 = _load_outputs("drive_0026")
    print(" ", viz.fig_cross_sequence_metrics(results, results_2).name)

    if not args.skip_gifs:
        print("Rendering demo GIFs (drive_0002)...")
        transforms = {
            "clean": None,
            "shadow": lambda img: apply_shadow(img, shadow_poly),
            "blur": lambda img: apply_motion_blur(img, kernel_size=9),
        }
        for cond in CONDITIONS:
            print(" ", viz.make_gif(frames, cond, transforms[cond], detect_fn).name)


if __name__ == "__main__":
    main()
