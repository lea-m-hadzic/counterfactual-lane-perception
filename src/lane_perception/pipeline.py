"""Classical lane-detection pipeline.

RGB image -> grayscale + Gaussian blur -> Canny -> trapezoidal ROI mask ->
probabilistic Hough -> slope-based grouping -> weighted least-squares lane
fit -> vanishing point (line intersection).
"""

import cv2
import numpy as np


def detect_edges(img_rgb, blur_ksize=5, canny_lo=50, canny_hi=150):
    """Grayscale + Gaussian blur + Canny."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return cv2.Canny(blurred, canny_lo, canny_hi)


def roi_mask(edges):
    """Trapezoidal region-of-interest mask tuned for drive_0002."""
    H, W = edges.shape
    mask = np.zeros_like(edges)
    poly = np.array(
        [
            [
                (int(0.08 * W), H),
                (int(0.44 * W), int(0.45 * H)),
                (int(0.48 * W), int(0.45 * H)),
                (int(0.66 * W), H),
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(edges, mask), poly


def roi_mask_2(edges):
    """Re-tuned ROI for drive_0026 (different scene framing)."""
    H, W = edges.shape
    mask = np.zeros_like(edges)
    poly = np.array(
        [
            [
                (int(0.00 * W), int(0.90 * H)),
                (int(0.42 * W), int(0.45 * H)),
                (int(0.55 * W), int(0.45 * H)),
                (int(0.68 * W), H),
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(edges, mask), poly


def hough_segments(
    masked_edges,
    rho=2,
    theta=np.pi / 180,
    threshold=50,
    min_line_len=40,
    max_line_gap=100,
    min_abs_slope=0.4,
):
    """Probabilistic Hough line detection + slope filtering."""
    lines = cv2.HoughLinesP(
        masked_edges,
        rho,
        theta,
        threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []

    out = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < min_abs_slope:
            continue
        out.append((x1, y1, x2, y2, slope))
    return out


def fit_lane_line(segments):
    """Weighted least-squares line fit to segment endpoints."""
    if not segments:
        return None

    xs, ys, ws = [], [], []
    for x1, y1, x2, y2, _ in segments:
        length = np.hypot(x2 - x1, y2 - y1)
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        ws.extend([length, length])

    m, b = np.polyfit(np.array(xs), np.array(ys), deg=1, w=np.array(ws))
    return m, b


def extrapolate(line, y_bot, y_top):
    """Endpoints of a fitted line between two y-values."""
    if line is None:
        return None
    m, b = line
    return (int((y_bot - b) / m), int(y_bot), int((y_top - b) / m), int(y_top))


def _detect_lanes(img_rgb, roi_fn, y_top_frac=0.48):
    """Full lane-detection pipeline. roi_fn selects which ROI to use."""
    edges = detect_edges(img_rgb)
    masked, _ = roi_fn(edges)
    segs = hough_segments(masked)

    left = [s for s in segs if s[4] < 0]
    right = [s for s in segs if s[4] > 0]

    left_fit = fit_lane_line(left)
    right_fit = fit_lane_line(right)

    H = img_rgb.shape[0]
    y_bot = H - 1
    y_top = int(y_top_frac * H)

    left_seg = extrapolate(left_fit, y_bot, y_top)
    right_seg = extrapolate(right_fit, y_bot, y_top)

    return left_seg, right_seg, {
        "n_segs": len(segs),
        "n_left": len(left),
        "n_right": len(right),
    }


def detect_lanes(img_rgb, y_top_frac=0.48):
    """Lane detection using the drive_0002 ROI."""
    return _detect_lanes(img_rgb, roi_mask, y_top_frac)


def detect_lanes_v2(img_rgb, y_top_frac=0.62):
    """Lane detection using the drive_0026 ROI."""
    return _detect_lanes(img_rgb, roi_mask_2, y_top_frac)


# Map a sequence's `roi` config key to its detector function.
DETECTORS = {
    "v1": detect_lanes,
    "v2": detect_lanes_v2,
}


def vanishing_point(left_seg, right_seg):
    """Intersection of two line segments (extended infinitely)."""
    if left_seg is None or right_seg is None:
        return None

    x1, y1, x2, y2 = left_seg
    x3, y3, x4, y4 = right_seg

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def draw_lanes(img_rgb, left_seg, right_seg, color=(0, 255, 0), thickness=4):
    """Overlay fitted lane lines on an image."""
    out = img_rgb.copy()
    for seg in (left_seg, right_seg):
        if seg is not None:
            cv2.line(out, (seg[0], seg[1]), (seg[2], seg[3]), color, thickness)
    return out
