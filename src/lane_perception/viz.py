"""Report figures and demo GIFs.

Every function saves a PNG (or GIF) into config.OUT_DIR and returns the path.
Filenames match the repository's outputs/ convention.
"""

import cv2
import numpy as np
import pandas as pd

from . import config
from .metrics import (
    METRIC_LABELS,
    autocorr_abs_diff,
    cumulative_drift_curve,
)
from .perturbations import apply_motion_blur, apply_shadow
from .pipeline import detect_lanes, draw_lanes, vanishing_point


def overlay_detection(img, detect_fn=detect_lanes):
    """Run detection and overlay lane lines + the vanishing point dot."""
    left_seg, right_seg, _ = detect_fn(img)
    vp = vanishing_point(left_seg, right_seg)
    vis = draw_lanes(img, left_seg, right_seg)
    if vp is not None:
        cv2.circle(vis, (int(vp[0]), int(vp[1])), 8, (255, 255, 0), -1)
    return vis


def _read_rgb(path):
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fig_perturbation_examples(frame_paths, shadow_poly, detect_fn=detect_lanes,
                              demo_idx=40):
    """Figure 1 - perturbation examples on a single frame."""
    import matplotlib.pyplot as plt

    rgb_demo = _read_rgb(frame_paths[demo_idx])
    panels = [
        (rgb_demo, "Clean baseline"),
        (apply_shadow(rgb_demo, shadow_poly), "With synthetic shadow ($\\alpha=0.4$)"),
        (apply_motion_blur(rgb_demo, kernel_size=9), "With horizontal motion blur (k=9)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(overlay_detection(img, detect_fn))
        ax.set_title(title, fontsize=11, loc="left")
        ax.axis("off")
    plt.tight_layout()

    out = config.OUT_DIR / "perturbation_examples.png"
    plt.savefig(out)
    plt.close(fig)
    return out


def fig_consecutive_frames(frame_paths, detect_fn=detect_lanes,
                           indices=(40, 50, 60, 70)):
    """Figure 2 - pipeline output on four consecutive frames."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(indices), 1, figsize=(8, 8))
    for ax, idx in zip(axes, indices):
        ax.imshow(overlay_detection(_read_rgb(frame_paths[idx]), detect_fn))
        ax.set_title(f"frame {idx}", fontsize=10, loc="left")
        ax.axis("off")
    plt.tight_layout()

    out = config.OUT_DIR / "consecutive_frames.png"
    plt.savefig(out)
    plt.close(fig)
    return out


def fig_vp_timeseries(vp_x_by_cond, seq_label="drive 0002"):
    """Figure 3 - VP x-coordinate over time."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    for cond, vx in vp_x_by_cond.items():
        ax.plot(vx, lw=1.5, color=config.PALETTE[cond], label=cond, alpha=0.9)
    ax.set_xlabel("frame")
    ax.set_ylabel("vanishing point $x$ (px)")
    ax.set_title(
        f"VP $x$-coordinate over time under perturbation ({seq_label})",
        fontsize=11, loc="left",
    )
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    out = config.OUT_DIR / "vp_timeseries.png"
    plt.savefig(out)
    plt.close(fig)
    return out


def fig_cumulative_drift(vp_x_by_cond, seq_label="drive 0002"):
    """Figure 4 - cumulative VP drift (compounding behavior)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    for cond, vx in vp_x_by_cond.items():
        ax.plot(cumulative_drift_curve(vx), lw=1.5,
                color=config.PALETTE[cond], label=cond, alpha=0.9)
    ax.set_xlabel("frame")
    ax.set_ylabel("cumulative $|\\Delta v_x|$ (px)")
    ax.set_title(
        f"Compounding behavior: accumulated VP drift ({seq_label})",
        fontsize=11, loc="left",
    )
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    out = config.OUT_DIR / "cumulative_drift.png"
    plt.savefig(out)
    plt.close(fig)
    return out


def fig_autocorrelation(vp_x_by_cond, seq_label="drive 0002"):
    """Figure 5 - autocorrelation of frame-to-frame instability."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    for cond, vx in vp_x_by_cond.items():
        acf = autocorr_abs_diff(vx)
        ax.plot(np.arange(len(acf)), acf, "o-",
                color=config.PALETTE[cond], label=cond, lw=1.5, ms=4, alpha=0.9)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("lag (frames)")
    ax.set_ylabel("autocorr of $|\\Delta v_x|$")
    ax.set_title(f"Temporal structure of instability ({seq_label})",
                 fontsize=11, loc="left")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    out = config.OUT_DIR / "autocorrelation.png"
    plt.savefig(out)
    plt.close(fig)
    return out


def fig_cross_sequence_metrics(results, results_2,
                               labels=("drive_0002", "drive_0026")):
    """Figure 6 - cross-sequence metric comparison."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    rows = []
    for seq_label, res in [(labels[0], results), (labels[1], results_2)]:
        for cond in ["clean", "shadow", "blur"]:
            for mkey, mlabel in METRIC_LABELS:
                rows.append({"sequence": seq_label, "condition": cond,
                             "metric": mlabel, "value": res[cond][mkey]})
    df = pd.DataFrame(rows)

    g = sns.catplot(
        data=df, kind="bar",
        x="condition", y="value",
        hue="sequence", col="metric",
        palette=config.SEQUENCE_PALETTE,
        col_wrap=4, height=2.8, aspect=0.95,
        sharey=False, legend=True,
    )
    g.set_titles("{col_name}", size=10)
    g.set_axis_labels("", "")
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.05),
                    ncol=2, frameon=False, title=None)

    out = config.OUT_DIR / "cross_sequence_metrics.png"
    g.savefig(out)
    plt.close(g.figure)
    return out


def fig_frame57_deepdive(frame_paths, shadow_poly, detect_fn=detect_lanes,
                         indices=(56, 57, 58)):
    """Figure 7 - per-frame detection succeeds while the VP fails."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(indices), 2, figsize=(11, 6.5),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})
    for row, idx in enumerate(indices):
        rgb = _read_rgb(frame_paths[idx])
        cols = [(rgb, "clean"), (apply_shadow(rgb, shadow_poly), "with shadow")]
        for col, (img, label) in enumerate(cols):
            axes[row, col].imshow(overlay_detection(img, detect_fn))
            if row == 0:
                axes[row, col].set_title(label, fontsize=11)
            if col == 0:
                axes[row, col].set_ylabel(f"frame {idx}", fontsize=10,
                                          rotation=0, labelpad=30, va="center")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    plt.suptitle(
        "Per-frame detection succeeds; vanishing point fails (drive 0002, shadow)",
        fontsize=11, y=1.02,
    )

    out = config.OUT_DIR / "frame57_deepdive.png"
    plt.savefig(out)
    plt.close(fig)
    return out


def fig_two_metrics(results):
    """Compact 2-metric chart for slide 6."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    rows = []
    for cond in ["clean", "shadow", "blur"]:
        for mkey, mlab in [("mean_dvp_x", "VP-x jitter (px/frame)"),
                           ("slope_osc", "Slope oscillation")]:
            rows.append({"condition": cond, "metric": mlab, "value": results[cond][mkey]})
    df = pd.DataFrame(rows)

    g = sns.catplot(
        data=df, kind="bar",
        x="condition", y="value",
        hue="condition", col="metric",
        palette=[config.PALETTE["clean"], config.PALETTE["shadow"], config.PALETTE["blur"]],
        saturation=1.0,
        legend=False,
        col_wrap=2, height=3.2, aspect=1.1,
        sharey=False,
    )
    g.set_titles("{col_name}", size=11)
    g.set_axis_labels("", "")

    for ax in g.axes.flat:
        for p in ax.patches:
            h = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2, h, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    out = config.OUT_DIR / "two_metrics.png"
    g.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(g.figure)
    return out


def make_gif(frame_paths, name, transform=None, detect_fn=detect_lanes, fps=10):
    """Render an annotated demo GIF for one condition into outputs/<name>.gif."""
    import imageio.v2 as imageio

    frames = []
    for p in frame_paths:
        rgb = _read_rgb(p)
        if transform is not None:
            rgb = transform(rgb)
        frames.append(overlay_detection(rgb, detect_fn))

    out = config.OUT_DIR / f"{name}.gif"
    imageio.mimsave(out, frames, fps=fps)
    return out
