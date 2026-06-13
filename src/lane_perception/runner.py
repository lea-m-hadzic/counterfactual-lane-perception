"""Batch-run the pipeline over frames and extract per-frame signals."""

import cv2
import numpy as np

from .pipeline import detect_lanes, vanishing_point


def run_pipeline(frame_paths, transform=None, detect_fn=detect_lanes):
    """Run the full pipeline over a list of frames, optionally applying a
    perturbation transform first. Returns a list of per-frame records."""
    records = []
    for i, p in enumerate(frame_paths):
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if transform is not None:
            rgb = transform(rgb)

        left_seg, right_seg, _ = detect_fn(rgb)
        vp = vanishing_point(left_seg, right_seg)
        records.append({"frame": i, "left": left_seg, "right": right_seg, "vp": vp})

    return records


def extract_signals(records):
    """From records, extract per-frame vp_x, vp_y, left/right slope arrays."""
    N = len(records)
    vp_x = np.full(N, np.nan)
    vp_y = np.full(N, np.nan)
    left_s = np.full(N, np.nan)
    right_s = np.full(N, np.nan)

    for r in records:
        i = r["frame"]

        if r["vp"] is not None:
            vp_x[i], vp_y[i] = r["vp"]
        if r["left"] is not None:
            x1, y1, x2, y2 = r["left"]
            if x2 != x1:
                left_s[i] = (y2 - y1) / (x2 - x1)
        if r["right"] is not None:
            x1, y1, x2, y2 = r["right"]
            if x2 != x1:
                right_s[i] = (y2 - y1) / (x2 - x1)

    return vp_x, vp_y, left_s, right_s
