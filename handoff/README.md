# BTC Volatility Spike Detector — Handoff Package

## Model Selection: Selected-base

This handoff ships the current selected-base model: the Logistic Regression pipeline at `handoff/models/artifacts/lr_pipeline.pkl`. It matches the current project runtime and slightly outperforms the z-score baseline on both validation and held-out test PR-AUC under the latest time-ordered split.

## Exact Teammate Steps

Run the commands below from the repo root.

### 1. Install Requirements

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r handoff/requirements.txt
```

### 2. Set Up the `.env` File

```bash
cp handoff/.env.example .env
```

The defaults already match the local Docker stack. Edit `.env` only if you need local overrides.

### 3. Start Docker Compose

```bash
docker compose -f handoff/docker/compose.yaml up -d
```

This compose brings up Kafka, MLflow, and the FastAPI prediction service (`/health`, `/predict`, `/version`, `/metrics`). Add `--profile live-ingest` only if you also want the optional live ingestor container.

### 4. Load the Model from the Artifacts Folder

The bundle stores the decision threshold inside the pickle as `bundle["tau"]` (current value: `0.7015`). Use that saved value instead of hard-coding a threshold.

```python
import pickle
import pandas as pd

with open("handoff/models/artifacts/lr_pipeline.pkl", "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
feature_cols = bundle["feature_cols"]
tau = bundle["tau"]

df = pd.read_parquet("handoff/data_sample/features_slice.parquet")
scores = pipeline.predict_proba(df[feature_cols].values)[:, 1]
preds = (scores >= tau).astype(int)

print(f"tau={tau:.4f}, rows={len(df)}, predicted_spike_rate={preds.mean():.4f}")
```

## Package Notes

- `handoff/data_sample/` contains the required 10-minute raw slice and its matching feature slice.
- `handoff/reports/predictions.csv` is the full held-out inference output from the current verified run.
- `handoff/reports/train_vs_test.html` is the shipped Evidently report.
- `handoff/SELECTED_BASE_NOTE.md` is the short selection note requested by the assignment.

## Package Contents

```
handoff/
├── .env.example
├── docker/
│   ├── compose.yaml
│   └── Dockerfile.ingestor
├── docs/
│   ├── feature_spec.md
│   └── model_card_v1.md
├── models/
│   └── artifacts/
│       ├── lr_pipeline.pkl
│       ├── lr_pipeline.sha256
│       └── metadata.json
├── data_sample/
│   ├── raw_slice.ndjson
│   ├── raw_slice.parquet
│   ├── features_slice.csv
│   └── features_slice.parquet
├── reports/
│   ├── model_eval.pdf
│   ├── predictions.csv
│   └── train_vs_test.html
├── requirements.txt
├── README.md
└── SELECTED_BASE_NOTE.md
```
