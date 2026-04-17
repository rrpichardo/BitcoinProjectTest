# Model Evaluation Summary — Selected-base Snapshot (Run B)

**Project:** BTC Volatility Spike Detector
**Artifact snapshot:** 2026-04-07 (submitted run)
**MLflow run_id:** `f4deb36327a74a4f9aa1f6433f169f28`

## Evaluation Setup

| Parameter | Value |
|---|---|
| Target | `vol_spike` |
| Label rule | `future_vol_60s >= 0.000048` |
| Features | `log_return`, `spread_bps`, `vol_60s`, `mean_return_60s`, `trade_intensity_60s`, `n_ticks_60s`, `spread_mean_60s` |
| Validation strategy | Strict chronological split (60% / 20% / 20%) |
| Primary comparison metric | PR-AUC |

## Dataset Context

| Split | Rows | Spike rate |
|---|---|---|
| Train (0–60%) | 368,311 | 15.4% |
| Validation (60–80%) | 122,771 | 14.1% |
| Test (80–100%) | 122,771 | 7.0% |

Test split time range: `2026-04-07 00:12:16 UTC` -> `2026-04-07 15:54:58 UTC`.
The evaluation is time-ordered with no shuffling. The label rate changes across windows, reflecting genuine market regime shifts within the ~65-hour collection period.

## Results

| Model | Validation PR-AUC | Test PR-AUC | Validation F1 (best) | Test F1 (best) |
|---|---|---|---|---|
| Z-score baseline (`vol_60s`, `z >= 2.0`) | 0.3270 | 0.1340 | 0.3868 | 0.1487 |
| Logistic Regression selected-base | 0.3580 | 0.1459 | 0.4463 | 0.1658 |

## Logistic Regression Operating Point

| Metric | Value |
|---|---|
| Threshold (`tau`) | 0.7015 |
| Validation F1 @ tau | 0.4463 |
| Test F1 @ tau | 0.1359 |
| Test predicted positive rate | 7.9% |
| Test actual positive rate | 7.0% |

## Conclusion

The Logistic Regression artifact is the selected-base model because it outperforms the z-score baseline on both validation and held-out test PR-AUC under the same chronological split. The val-to-test PR-AUC drop (0.3580 -> 0.1459) is driven by the test window landing on a quieter market regime (7.0% spike rate vs 14.1% in validation), not overfitting. The handoff package ships the matching model pickle, checksum, metadata, predictions file, and Evidently HTML report for this snapshot.
