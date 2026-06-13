"""Temporal stability metrics for the recovered vanishing-point signal."""

import numpy as np


def m_recovery(vp_x):
    """Number of frames with a successfully recovered vanishing point."""
    return int(np.sum(~np.isnan(vp_x)))


def m_mean_dvp_x(vp_x):
    """Mean frame-to-frame |delta vp_x| over recovered frames."""
    v = vp_x[~np.isnan(vp_x)]
    return float(np.mean(np.abs(np.diff(v))))


def m_slope_osc(left_s, right_s):
    """Average frame-to-frame |slope change| across left and right lanes."""
    dl = np.diff(left_s[~np.isnan(left_s)])
    dr = np.diff(right_s[~np.isnan(right_s)])
    return float((np.mean(np.abs(dl)) + np.mean(np.abs(dr))) / 2)


def m_vp_2d_drift(vp_x, vp_y):
    """Mean 2D Euclidean frame-to-frame VP movement."""
    valid = ~(np.isnan(vp_x) | np.isnan(vp_y))
    vx, vy = vp_x[valid], vp_y[valid]
    return float(np.mean(np.sqrt(np.diff(vx) ** 2 + np.diff(vy) ** 2)))


def m_heading_dev(vp_x_perturbed, vp_x_clean):
    """Mean per-frame deviation from clean baseline."""
    valid = ~(np.isnan(vp_x_perturbed) | np.isnan(vp_x_clean))
    return float(np.mean(np.abs(vp_x_perturbed[valid] - vp_x_clean[valid])))


def cumulative_drift_curve(vp_x):
    """Running cumulative |delta vp_x| indexed by frame number (NaN-safe)."""
    out = np.zeros(len(vp_x))
    prev_valid, cum = None, 0.0

    for i, v in enumerate(vp_x):
        if not np.isnan(v):
            if prev_valid is not None:
                cum += abs(v - prev_valid)
            prev_valid = v
        out[i] = cum

    return out


def autocorr_abs_diff(vp_x, max_lag=15):
    """Autocorrelation of |delta vp_x| at lags 0..max_lag."""
    d = np.abs(np.diff(vp_x[~np.isnan(vp_x)]))

    if d.std() == 0:
        return np.zeros(max_lag + 1)

    d = (d - d.mean()) / d.std()
    n = len(d)

    return np.array(
        [np.sum(d[: n - lag] * d[lag:]) / (n - lag) for lag in range(max_lag + 1)]
    )


def compute_results(vp_x, vp_y, left_s, right_s, vp_x_clean=None):
    """Bundle all scalar metrics for one condition into a dict."""
    return {
        "recovery": m_recovery(vp_x),
        "mean_dvp_x": m_mean_dvp_x(vp_x),
        "slope_osc": m_slope_osc(left_s, right_s),
        "vp_2d_drift": m_vp_2d_drift(vp_x, vp_y),
        "heading_dev": 0.0 if vp_x_clean is None else m_heading_dev(vp_x, vp_x_clean),
    }


# Metric keys in canonical display order, with human-readable labels.
METRIC_KEYS = ["recovery", "mean_dvp_x", "slope_osc", "vp_2d_drift", "heading_dev"]

METRIC_LABELS = [
    ("mean_dvp_x", "VP-x jitter (px/frame)"),
    ("slope_osc", "Slope osc. (slope/frame)"),
    ("vp_2d_drift", "2D VP drift (px/frame)"),
    ("heading_dev", "Heading dev. (px)"),
]
