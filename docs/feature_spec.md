# Feature Specification — BTC Volatility Spike Detector

## Label

| Parameter | Value |
|---|---|
| Target horizon | 60 seconds |
| Volatility proxy | Rolling std of midprice log-returns over the next 60 seconds |
| Label definition | `vol_spike = 1` if `future_vol_60s >= 0.000048`; else `0` |
| Current full-dataset spike rate | ~16.4% of labelled rows |
| Notes | The `0.000048` threshold is the project-wide label definition retained by the current training pipeline |

## Feature Definitions

All features are computed per tick using a 60-second lookback window.

| Feature | Formula / description | Unit |
|---|---|---|
| `price` | Last traded price | USD |
| `midprice` | `(best_bid + best_ask) / 2` | USD |
| `log_return` | `ln(price_t / price_{t-1})` | dimensionless |
| `spread_abs` | `best_ask - best_bid` | USD |
| `spread_bps` | `spread_abs / midprice * 10,000` | basis points |
| `vol_60s` | Std of log-returns over the past 60 seconds | dimensionless |
| `mean_return_60s` | Mean log-return over the past 60 seconds | dimensionless |
| `n_ticks_60s` | Count of ticks observed in the past 60 seconds | integer |
| `trade_intensity_60s` | `n_ticks_60s / 60` | ticks/sec |
| `spread_mean_60s` | Mean absolute spread over the past 60 seconds | USD |
| `price_range_60s` | `max(price) - min(price)` over the past 60 seconds | USD |

## Current Deployed Feature Set

The current Logistic Regression training pipeline uses the exact `FEATURE_COLS` list in `models/train.py`:

| Feature | Role |
|---|---|
| `log_return` | Instantaneous momentum |
| `spread_bps` | Current liquidity in unit-free form |
| `vol_60s` | Rolling realized volatility |
| `mean_return_60s` | Rolling drift direction |
| `trade_intensity_60s` | Market activity rate |
| `n_ticks_60s` | Raw tick count in the lookback window |
| `spread_mean_60s` | Smoothed liquidity signal |

### Deployment Notes

- `spread_mean_60s` is the additional liquidity feature that distinguishes the shipped 7-feature model from the earlier 6-feature baseline.
- `price_range_60s` and `spread_abs` remain in the parquet dataset for analysis and diagnostics, but they are not part of the currently shipped LR model artifact.
- `n_ticks_60s` and `trade_intensity_60s` are intentionally both retained because the current artifact and metadata were trained with that exact pair.

## Parquet Schema (`data/processed/features.parquet`)

| Column | Type | Description |
|---|---|---|
| `product_id` | string | Example: `BTC-USD` |
| `timestamp` | string | ISO-8601 event timestamp |
| `price` | float64 | Last traded price |
| `midprice` | float64 | Midpoint of best bid / ask |
| `log_return` | float64 | Tick-to-tick log-return |
| `spread_abs` | float64 | Absolute bid-ask spread |
| `spread_bps` | float64 | Spread in basis points |
| `vol_60s` | float64 | Rolling volatility over 60 seconds |
| `mean_return_60s` | float64 | Rolling mean return over 60 seconds |
| `n_ticks_60s` | int64 | Tick count over 60 seconds |
| `trade_intensity_60s` | float64 | Ticks per second over 60 seconds |
| `spread_mean_60s` | float64 | Mean spread over 60 seconds |
| `price_range_60s` | float64 | Price range over 60 seconds |
| `future_vol_60s` | float64 | Realized future volatility over the next 60 seconds |
| `vol_spike` | int64 | Binary label derived from `future_vol_60s` |
