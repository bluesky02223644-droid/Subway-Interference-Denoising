# Processed Data for Reproducing Figures and Tables

This folder contains processed data sufficient to reproduce all data-dependent
figures and tables in the manuscript:

> Huo, Q., Wang, X., Ma, X., & Zhang, S. (2026). Lightweight Deep Learning for
> Identification and Suppression of Subway Interference in Geomagnetic
> Observations. *Earth and Space Science*.

Raw geomagnetic one-second data are provided by the National Geomagnetic
Network Center, Institute of Geophysics, China Earthquake Administration.
The raw observations are subject to institutional access policies; access
may be granted upon formal written request to the National Geomagnetic
Network Center.

---

## File Descriptions

| File | Paper Element | Description |
|------|--------------|-------------|
| `figure4_training_log.csv` | Figure 4(a)(b), Table 2 | TinyCNN training log: 41 epochs × 10 metrics (Train_Loss, Val_Precision, Val_Recall, Val_F1, Val_AUC, etc.) |
| `figure4_confusion_matrix.npz` | Figure 4(c) | Validation set (28,479 samples): `y_true`, `y_pred`, `y_probs`, `confusion_matrix` (TN=14267, FP=4, FN=4, TP=14204) |
| `figure5_BMT_denoised_2024-07-13.npz` | Figure 5(a)(c)(e) | Beijing (BMT) station, 2024-07-13, 24-hour Z-component: `original` and `denoised` arrays (86,400 samples at 1 Hz) |
| `figure5_HZC_denoised_2025-02-19.npz` | Figure 5(b)(d)(f) | Hangzhou (HZC) station, 2025-02-19, 24-hour Z-component: `original` and `denoised` arrays (86,400 samples) |
| `figure6_TAY_detection_589days.csv` | Figure 6 | Taiyuan (TAY) 589-day detection summary (2023-12 to 2025-11): columns `file`, `date`, `n_detections`, `max_prob`, `mean_prob`, `has_interference` |
| `figure7_TAY_denoised_2025-11-05.npz` | Figure 7 | Taiyuan (TAY) station, 2025-11-05, 24-hour Z-component: `original` and `denoised` arrays (86,400 samples) |
| `table3_TAY_baseline_2024-02-01.npz` | Table 3, Figure 8 | Taiyuan baseline comparison, 2024-02-01: `original`, `vmd` (VMD-denoised), `dl` (ResidualCNN-denoised), `hierarchical` arrays (86,400 samples each). Band-stop filtered result can be computed from `original` using a 5th-order Butterworth filter at 0.004–0.032 Hz. |
| `first_order_diff_1387days.csv` | Section 3.3 (first-order difference analysis) | Per-day statistics for 1,387 station-days (BMT 612d + HZC 186d + TAY 589d): columns include `station`, `date`, `orig_std`, `den_std`, `orig_p95`, `den_p95`, `sif`, etc. |

## Data Formats

- **NPZ files**: NumPy compressed archive. Load with `numpy.load('file.npz')`.
  All geomagnetic arrays are 1-D float64, 86,400 points = 24 hours at 1 Hz sampling.
  Units: nT (nanotesla), Z-component.
- **CSV files**: UTF-8 encoded, comma-separated, with header row.

## Figures Without Data Dependencies

- **Figure 1** (TinyCNN architecture): programmatic diagram, no data needed.
- **Figure 2** (Subway interference characteristics): uses restricted CENC raw data.
- **Figure 3** (Method framework): programmatic diagram, no data needed.

## License

- Software (source code, model weights): MIT License
- Data (this folder): Creative Commons Attribution 4.0 International (CC BY 4.0)
