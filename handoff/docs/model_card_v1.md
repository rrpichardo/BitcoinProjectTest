# Model Card — BTC Volatility Spike Detector v1

## Model Details

| Field | Value |
|---|---|
| Name | BTC Volatility Spike Detector |
| Version | v1 |
| Type | Binary classifier — Logistic Regression |
| Framework | scikit-learn pipeline |
| Artifact | `models/artifacts/lr_pipeline.pkl` |
| Metadata | `models/artifacts/metadata.json` |
| Trained / submitted | 2026-04-07 |
| MLflow run_id | `f4deb36327a74a4f9aa1f6433f169f28` |

**Architecture:** `StandardScaler -> LogisticRegression(C=0.1, class_weight="balanced", solver="lbfgs", max_iter=1000)`

## Intended Use

Predict whether BTC-USD volatility will spike over the next 60 seconds from the current tick-level market state. The model is intended for monitoring, alerting, and thin-slice inference demos rather than autonomous trading decisions.

## Data

| Field | Value |
|---|---|
| Source | Coinbase Advanced Trade WebSocket (`wss://advanced-trade-ws.coinbase.com`) |
| Pair | BTC-USD |
| Collection period | 2026-04-04 22:54 UTC -> 2026-04-07 15:54 UTC (~65 hours) |
| Labelled feature rows | 613,853 |
| Full-dataset spike rate | ~13.5% |
| Label definition | `vol_spike = 1` when `future_vol_60s >= 0.000048` |

Label threshold 0.000048 is the P85 of observed `future_vol_60s` across the training distribution.

## Features

The shipped model uses 7 engineered features (Variant B from the ablation study):

| Feature | Description |
|---|---|
| `log_return` | Instantaneous log-return |
| `spread_bps` | Bid-ask spread in basis points |
| `vol_60s` | Rolling volatility over 60 seconds |
| `mean_return_60s` | Rolling mean return over 60 seconds |
| `trade_intensity_60s` | Trades per second over 60 seconds |
| `n_ticks_60s` | Tick count over 60 seconds |
| `spread_mean_60s` | Mean spread over 60 seconds |

See `docs/feature_spec.md` for the full schema and feature notes.

## Training Split

The model is trained and evaluated with a strict chronological split — no shuffling, so temporal structure is preserved end-to-end.

| Split | Rows | Spike rate |
|---|---|---|
| Train (0–60%) | 368,311 | 15.4% |
| Validation (60–80%) | 122,771 | 14.1% |
| Test (80–100%) | 122,771 | 7.0% |

Test split time range: `2026-04-07 00:12:16 UTC` -> `2026-04-07 15:54:58 UTC`.
The variation in spike rate across splits reflects a genuine change in market regime during the collection window — the test period landed on a quieter stretch.

## Performance

### Selected-base Logistic Regression

| Metric | Value |
|---|---|
| Validation PR-AUC | 0.3580 |
| Test PR-AUC | 0.1459 |
| Operating threshold (`tau`) | 0.7015 |
| Validation F1 @ tau | 0.4463 |
| Test F1 @ tau | 0.1359 |

### Baseline Comparison

| Model | Validation PR-AUC | Test PR-AUC | Validation F1 (best) | Test F1 (best) |
|---|---|---|---|---|
| Z-score baseline (`vol_60s`, `z >= 2.0`) | 0.3270 | 0.1340 | 0.3868 | 0.1487 |
| Logistic Regression v1 | 0.3580 | 0.1459 | 0.4463 | 0.1658 |

The Logistic Regression model is the selected-base artifact because it outperforms the z-score baseline on both validation and held-out test PR-AUC under the same chronological split. The val-to-test drop (0.358 -> 0.1459) reflects the regime shift described above — the test window is a quieter period with a 7.0% spike rate vs 14.1% in validation — not overfitting.

## Feature Importance (Logistic Regression Coefficients)

| Feature | Coefficient | Direction |
|---|---|---|
| `vol_60s` | +0.4652 | Higher rolling volatility increases spike likelihood |
| `n_ticks_60s` | +0.1591 | More ticks increase spike likelihood |
| `trade_intensity_60s` | +0.1591 | Higher activity rate increases spike likelihood |
| `spread_mean_60s` | +0.1580 | Wider mean spread increases spike likelihood |
| `spread_bps` | +0.0492 | Wider instantaneous spread increases spike likelihood |
| `mean_return_60s` | -0.0363 | Negative drift slightly increases spike likelihood |
| `log_return` | +0.0119 | Positive instantaneous return has a small positive effect |

`vol_60s` is the dominant predictor; `spread_mean_60s` contributes as a smoothed liquidity signal (the differentiator from the 6-feature baseline).

## Monitoring

The handoff ships the primary Evidently comparison as `reports/train_vs_test.html`. The same source report lives in the project tree at `reports/evidently/train_vs_test.html`.

## Limitations

- The model is trained on a single market pair (`BTC-USD`).
- Chronological splits show strong regime drift, so absolute metrics depend heavily on the evaluation window.
- The feature set does not include depth-of-book information (`ob_imbalance` was unavailable from the Coinbase basic ticker feed).
- The shipped artifact is a pickle bundle and should be checksum-verified before loading.

## Responsible AI Considerations

- Uses public market data only; no PII is collected.
- Intended to support human monitoring and analysis, not replace judgment.
- Performance should be re-checked whenever the market regime shifts materially.

## Retraining Triggers

Retrain and re-evaluate when any of the following occur:

1. The feature set changes.
2. The held-out PR-AUC degrades meaningfully on a fresh chronological split.
3. A new market regime or new data source materially changes the label distribution.
