# Selected-base

This handoff uses the Logistic Regression pipeline in `handoff/models/artifacts/lr_pipeline.pkl` as the selected-base model.

## Exact Steps

1. Create and activate a Python 3.11 virtual environment, then install `handoff/requirements.txt`.
2. Copy `handoff/.env.example` to `.env` from the repo root.
3. Start the services with `docker compose -f handoff/docker/compose.yaml up -d`.
4. Load `handoff/models/artifacts/lr_pipeline.pkl`, read the saved threshold from `bundle["tau"]` (currently `0.7015`), and score `handoff/data_sample/features_slice.parquet` or another compatible Parquet features file.

## Why This Is The Base

- It is the same model family the current project runtime expects.
- It slightly outperforms the z-score baseline on both validation and held-out test PR-AUC under the current chronological split.
- The handoff package includes the matching artifact checksum, metadata, evaluation PDF, Evidently report, and predictions file.
