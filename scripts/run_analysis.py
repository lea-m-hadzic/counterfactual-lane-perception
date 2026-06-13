"""Run the full counterfactual stability analysis end-to-end.

For each KITTI sequence: download frames, run the pipeline under the clean,
shadow, and motion-blur conditions, save the per-frame VP signals (.npz) and
the scalar stability metrics (.json), and print a metric table.

Usage:
    python scripts/run_analysis.py                 # all sequences
    python scripts/run_analysis.py drive_0002      # a single sequence
"""

import argparse
import json

import _bootstrap  # noqa: F401  (adds ../src to sys.path)

import cv2
import numpy as np

from lane_perception import config
from lane_perception.data import prepare_sequence
from lane_perception.metrics import METRIC_KEYS, compute_results
from lane_perception.perturbations import apply_motion_blur, apply_shadow
from lane_perception.pipeline import DETECTORS
from lane_perception.runner import extract_signals, run_pipeline

CONDITIONS = ["clean", "shadow", "blur"]


def analyze_sequence(seq_key):
    """Run all three conditions for one sequence; return (signals, results)."""
    meta = config.SEQUENCES[seq_key]
    detect_fn = DETECTORS[meta["roi"]]
    frame_paths = prepare_sequence(seq_key)

    # Shadow polygon scaled to this sequence's image dimensions.
    H, W = cv2.imread(str(frame_paths[0])).shape[:2]
    shadow_poly = config.make_shadow_poly(H, W)

    transforms = {
        "clean": None,
        "shadow": lambda img: apply_shadow(img, shadow_poly),
        "blur": lambda img: apply_motion_blur(img, kernel_size=9),
    }

    signals = {}
    for cond in CONDITIONS:
        records = run_pipeline(frame_paths, transform=transforms[cond], detect_fn=detect_fn)
        vp_x, vp_y, left_s, right_s = extract_signals(records)
        signals[cond] = {"vp_x": vp_x, "vp_y": vp_y, "left_s": left_s, "right_s": right_s}
        recovered = int(np.sum(~np.isnan(vp_x)))
        print(f"{cond:6s}: recovered {recovered:3d}/{len(frame_paths)}")

    clean_vp_x = signals["clean"]["vp_x"]
    results = {
        cond: compute_results(
            s["vp_x"], s["vp_y"], s["left_s"], s["right_s"],
            vp_x_clean=None if cond == "clean" else clean_vp_x,
        )
        for cond, s in signals.items()
    }

    _save_signals(seq_key, signals)
    _save_results(seq_key, results)
    _print_table(seq_key, results)
    return signals, results


def _save_signals(seq_key, signals):
    np.savez(
        config.OUT_DIR / f"{seq_key}_results.npz",
        **{
            f"{arr}_{cond}": signals[cond][arr]
            for cond in CONDITIONS
            for arr in ("vp_x", "vp_y")
        },
    )


def _save_results(seq_key, results):
    with open(config.OUT_DIR / f"{seq_key}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)


def _print_table(seq_key, results):
    print(f"\n=== {seq_key} stability metrics ===")
    header = f"{'metric':<14s}  {'clean':>10s}  {'shadow':>10s}  {'blur':>10s}"
    print(header)
    print("-" * len(header))
    for k in METRIC_KEYS:
        row = f"{k:<14s}"
        for cond in CONDITIONS:
            row += f"  {results[cond][k]:>10.3f}"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequences", nargs="*", default=None,
                        help="sequence keys to run (default: all)")
    args = parser.parse_args()

    config.ensure_dirs()
    seq_keys = args.sequences or list(config.SEQUENCES)
    for seq_key in seq_keys:
        analyze_sequence(seq_key)


if __name__ == "__main__":
    main()
