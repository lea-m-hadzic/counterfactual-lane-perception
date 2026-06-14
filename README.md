<div align="center">

# Counterfactual Stability Analysis of Classical Lane Perception

</div>

*A classical lane perception pipeline (Canny edge detection → Hough transform → lane fit → vanishing point)
on KITTI driving sequences, evaluated under two counterfactual perturbations
(synthetic shadow and motion blur) across five temporal stability metrics.*

![](outputs/clean.gif)

**Headline finding:** under synthetic shadow, per-frame vanishing-point recovery
remains at 100% while frame-to-frame jitter doubles and a single-frame excursion
of >200 px occurs at frame 57 of `drive_0002` — a failure invisible to any
per-frame metric.

## Repository Structure

```
.
├── src/lane_perception/         # Importable package (the whole pipeline)
│   ├── config.py                # Paths, figure styling, sequence metadata
│   ├── pipeline.py              # Edges → ROI → Hough → fit → vanishing point
│   ├── perturbations.py         # Synthetic shadow + motion blur
│   ├── data.py                  # KITTI download / extract / frame listing
│   ├── runner.py                # Batch-run pipeline, extract per-frame signals
│   ├── metrics.py               # Temporal stability metrics
│   └── viz.py                   # Report figures + demo GIFs
│
├── scripts/
│   ├── run_analysis.py          # Download → run → save .npz/.json + tables
│   └── make_figures.py          # Render all figures + GIFs into outputs/
│
├── outputs/                     # Generated results & figures
│   ├── drive_0002_results.npz   # Per-frame VP signals (one .npz per sequence)
│   ├── drive_0026_results.npz
│   ├── *.png                    # Seven report figures
│   └── clean.gif / shadow.gif / blur.gif
│
├── paper.pdf                    # Write-up
├── requirements.txt             # Python dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python scripts/run_analysis.py      # download data, run pipeline, write .npz/.json + tables
python scripts/make_figures.py      # render all figures + GIFs into outputs/
```

`run_analysis.py` downloads two KITTI raw sequences (`2011_09_26_drive_0002`
and `2011_09_29_drive_0026`, ~770 MB total) into a local `data/` folder on first
run. Total runtime is ~3–5 minutes after download on a recent laptop. Pass a
sequence key (e.g. `python scripts/run_analysis.py drive_0002`) to run just one;
use `make_figures.py --skip-gifs` to skip the slower GIF rendering.

The package is plain modules under `src/` — no install step. The scripts add
`src/` to the path automatically; to use the code directly, point `PYTHONPATH`
at `src/` and `import lane_perception`.

## What the scripts produce

Everything lands in `outputs/`:

- Per-frame VP signals (`<sequence>_results.npz`) and scalar metrics
  (`<sequence>_metrics.json`), one of each per sequence
- Seven figures used in the final report
- Three demo GIFs (clean, shadow, motion blur) for the presentation
---
## Methodology summary

**Pipeline.** Grayscale → Gaussian blur → Canny edge detection → trapezoidal ROI
mask → probabilistic Hough → slope-based segment grouping → weighted least-squares
lane fit → vanishing point as line intersection.

**Perturbations.**

- *Synthetic shadow* — fixed quadrilateral on the road surface multiplicatively
  darkened by α = 0.4.
- *Motion blur* — horizontal box-filter convolution, kernel size 9 px.

**Stability metrics.**

- Per-frame VP recovery rate
- Mean frame-to-frame VP-x jitter
- Slope oscillation (frame-to-frame angular change)
- 2D VP drift
- Heading deviation from clean baseline

Compounding behavior analyzed via cumulative drift curve and lagged
autocorrelation.

**Sequences.**

- `2011_09_26_drive_0002` (77 frames, primary)
- `2011_09_29_drive_0026` (158 frames, cross-sequence validation)

### Results at a glance

| Metric                 |  Clean |        Shadow |          Blur |
|------------------------|-------:|--------------:|--------------:|
| Recovery rate          |  77/77 |         77/77 |         52/77 |
| VP-x jitter (px)       |   9.55 | 19.30 (2.02×) | 22.95 (2.40×) |
| Slope oscillation      |  0.030 | 0.054 (1.79×) | 0.193 (6.43×) |
| 2D VP drift (px)       |  11.61 |         22.22 |         27.97 |
| Heading deviation (px) |   0.00 |         18.50 |         23.93 |

Results replicate qualitatively on the second sequence, with magnitudes varying
by scene content.

## License

MIT — see [LICENSE](LICENSE).
