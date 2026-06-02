# outputs/

Generated artifacts from running `lane_baseline.ipynb`:

- `drive_0002_results.npz` — per-frame VP and slope arrays for drive_0002
- `drive_0026_results.npz` — same, for drive_0026
- `fig1_perturbation_examples.png` — clean / shadow / blur on a single frame
- `fig2_consecutive_frames.png` — four consecutive clean frames
- `fig3_vp_timeseries.png` — VP-x over time across all three conditions
- `fig4_cumulative_drift.png` — cumulative VP drift curves
- `fig5_autocorrelation.png` — lagged autocorrelation of jitter
- `fig6_cross_sequence_metrics.png` — drive_0002 vs drive_0026 bar chart
- `fig7_frame57_deepdive.png` — frames 56–58, clean vs shadow
- `gif_clean.gif`, `gif_shadow.gif`, `gif_blur.gif` — animated overlays for the demo

Run the notebook to (re)generate everything in this folder.
