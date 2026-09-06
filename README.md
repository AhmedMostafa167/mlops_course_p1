[![CI](https://github.com/AhmedMostafa167/mlops_course_p1/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmedMostafa167/mlops_course_p1/actions/workflows/ci.yml)
# NYC Green Taxi Duration Prediction Service

An MLOps pipeline built around one month of NYC TLC green taxi trip data, covering both mini projects of the course: a production-shaped FastAPI service (Module 1), and a self-gating training/promotion pipeline with MLflow, DVC, and CI (Module 2).

The service predicts taxi-trip duration in minutes from two engineered features:

- `PU_DO`: the pickup and drop-off location pair
- `trip_distance`: the trip distance in miles

Three model families are supported end to end — `lr` (scikit-learn `LinearRegression`), `xgboost` (`XGBRegressor`), and `mlp` (a small PyTorch network) — all sharing the same `DictVectorizer` preprocessing, persistence format, ONNX export path, and API.

## Current project status

| Area | Status |
| --- | --- |
| Packaged, tested, containerized FastAPI service | Completed |
| Structured JSON logging with correlation IDs | Completed |
| ONNX export + parity check, all three model types | Completed |
| Non-root multi-stage Docker image | Completed |
| MLflow experiment tracking (Postgres + MinIO) | Completed |
| Hyperparameter tuning (Optuna) + model registry promotion | Completed |
| Data + model versioning with DVC (remote: Backblaze B2) | Completed |
| CI: lint → test → coverage gate → MAE regression gate → build/push | Completed |
| Continuous training (manual, local-first — see below) | Completed |
| Terraform / cloud-hosted MLflow | **Not implemented — deferred by choice** |
| Cloud-triggered continuous training (`schedule`, `repository_dispatch`) | **Not implemented — requires a publicly reachable tracking server** |

## Repository structure

```
.
├── data/
│   └── green_tripdata_2024-01.parquet          # DVC-tracked
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml                      # postgres + minio, for local MLflow
├── metrics/
│   ├── new_run.json                             # written by train.py every run
│   └── production.json                          # current champion's metrics — the CI gate compares against this
├── models/
│   ├── model.pkl                                # DVC-tracked champion
│   └── model.onnx
├── notebooks/
│   └── 00-baseline.ipynb
├── reports/
│   ├── module-1.md
│   └── module-2.md
├── scripts/
│   ├── benchmark_serialization.py
│   └── check_mae_regression.py
├── src/prodml/
│   ├── config.py            # pydantic-settings — all paths/env, no hardcoding
│   ├── data.py               # download / load / train-validation split
│   ├── features.py           # PU_DO, target engineering, feature dicts
│   ├── model_types.py        # ModelType, MLPRegressor, infer/validate helpers
│   ├── persistence.py        # save_model / load_model (pickle bundle)
│   ├── evaluate.py           # RMSE / MAE / R²
│   ├── train.py              # fits lr/xgboost/mlp, logs to MLflow, writes metrics/new_run.json
│   ├── tuning.py             # Optuna sweep on xgboost, registers + promotes the winner
│   ├── tracking.py           # ExperimentTracker — params/metrics/artifacts/tags to MLflow
│   ├── registry.py           # promote_if_better() — None → Staging → Production
│   ├── predict.py            # DurationPredictor: .load(), .load_from_registry(), predict_one/batch
│   ├── export.py             # model-agnostic ONNX export + parity check
│   ├── serve.py              # launches `mlflow server` as a host process
│   ├── logging_config.py
│   └── api/
│       ├── main.py           # FastAPI app, lifespan model loading
│       └── schemas.py
├── tests/
├── .github/workflows/ci.yml
├── .dockerignore
├── .pre-commit-config.yaml
├── .env.example
├── pyproject.toml
└── uv.lock
```

## Environment configuration

The application reads its configuration from `.env` through `pydantic-settings`. Copy the example file before running anything:

```bash
cp .env.example .env
```

Only three settings are required with no default (`EXPERIMENT_NAME`, `REGISTERED_MODEL_NAME`, `PRODUCTION_MODEL_URI`); everything else — data paths, MLflow/Postgres/MinIO connection details — has a sane default matching `docker/docker-compose.yml`. Never commit `.env`; commit `.env.example` instead.

## Installation

```bash
uv sync --group dev
uv run pre-commit install
```

Dev dependencies live under `[dependency-groups]` in `pyproject.toml` — use `--group dev`, not `--extra dev` (there is no `dev` extra). Run all commands from the repository root.

## Local infrastructure

MLflow's backend store (Postgres) and artifact store (MinIO) run in Docker; the MLflow server itself runs as a host process so it can be reached at `localhost` from anywhere else on your machine.

```bash
docker compose -f docker/docker-compose.yml up -d
uv run python -m prodml.serve   # leave running in its own terminal
```

MLflow UI: `http://localhost:5000`. MinIO console: `http://localhost:9001`.

## Training

```bash
uv run python -m prodml.train --model lr        # or xgboost / mlp
```

Downloads the configured TLC Parquet file if missing, prepares data, fits the model, logs the run to MLflow, saves `models/model.pkl`, and writes `metrics/new_run.json` — the file CI's quality gate reads.

Current baseline numbers live in `reports/module-1.md`, not hardcoded here, since they change with whichever model was trained most recently.

## Hyperparameter tuning + promotion

```bash
uv run python -m prodml.tuning
```

Runs an Optuna sweep on `xgboost`, registers the best trial in the MLflow model registry, and calls `promote_if_better()`: if it beats the current Production model's MAE, it's promoted, `models/model.pkl` is overwritten to match, and `metrics/production.json` is refreshed. Otherwise it's left in `Staging`.

After a promotion, re-track the champion:
```bash
dvc add models/model.pkl
dvc push
git add models/model.pkl.dvc metrics/production.json
git commit -m "Promote new champion"
```

## Model registry — loading by stage

`predict.py` supports two loading paths:

```python
DurationPredictor.load(settings.model_path)      # baked-in pickle — what the Docker image uses
DurationPredictor.load_from_registry()            # live MLflow lookup, by Production stage
```

The API toggles between them via `LOAD_FROM_REGISTRY` (default `false`). Local demo of "swap the model with zero code change":

```bash
LOAD_FROM_REGISTRY=true uv run uvicorn prodml.api.main:app --port 8000
```
Promote a different version to Production in the MLflow UI, restart the same command — the served prediction changes without touching any code. The deployed Docker image always uses the baked-in pickle instead, since it has no network path to a local-only MLflow instance.

## Data and model versioning (DVC)

```bash
dvc pull        # fetch data/ and models/model.pkl from the B2 remote
dvc push        # after training or promoting
```

Every MLflow run is tagged with `data_version` — the DVC-tracked data file's md5 — so any registered model traces back to the exact bytes it was trained on.

## ONNX export and parity

```bash
uv run python -m prodml.export
```

Model-agnostic: dispatches to `skl2onnx` for `lr`/`xgboost` (via a registered `onnxmltools` converter for XGBoost's tree ops) and `torch.onnx.export` for `mlp`. Exports to `models/model.onnx`, validates the graph, and asserts pickle/ONNX predictions agree within `atol=1e-4` on 500 validation rows.

Latency comparison:
```bash
uv run python scripts/benchmark_serialization.py
```

Pickle remains the format actually served (see `reports/module-1.md` for the reasoning); ONNX is exported and verified as a parity-checked alternative.

## API

```bash
uv run uvicorn prodml.api.main:app --reload --port 8000
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Confirms the model is loaded in memory |
| `/metadata` | `GET` | Model version, training date, features, framework, artifact hash |
| `/predict` | `POST` | One prediction |
| `/predict/batch` | `POST` | Predictions for a list of inputs |

```json
// request
{"pu_do": "74_75", "trip_distance": 1.5}

// response
{"prediction": 9.60, "model_version": "linear-regression-2024-01", "correlation_id": "...", "latency_ms": 13.2}
```

The model loads once at startup via FastAPI's lifespan, not per request.

## Docker

Multi-stage build, runs as non-root (`appuser`, uid 1000):

```bash
docker build -f docker/Dockerfile -t prodml-api .
docker run --rm -p 8000:8000 prodml-api
docker exec <container_id> whoami   # appuser
```

## Testing and linting

```bash
uv run pytest                          # coverage gate: --cov-fail-under=70
uv run ruff check src tests
uv run black --check src tests
```

## CI/CD

`.github/workflows/ci.yml` — three jobs, each gating the next:

1. **lint** — ruff, black
2. **test** — pytest with the coverage gate, then `scripts/check_mae_regression.py`: blocks the merge if the new run's MAE regresses more than 5% against `metrics/production.json`
3. **build** (on `main` only) — `dvc pull`s the champion model, builds and pushes the Docker image

Requires these repo secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `B2_KEY_ID`, `B2_APP_KEY`.

## Continuous training

Training happens locally, where MLflow/Postgres actually live — a GitHub-hosted runner has no network path to reach them. The loop is: retrain locally → `dvc push` → `git push` → CI gates, builds, and deploys. Schedule- and webhook-triggered retraining inside CI would need a publicly reachable MLflow tracking server; deferred, see `reports/module-2.md` for the full trade-off.

## Known gaps

- Terraform / infrastructure-as-code — not implemented.
- Cloud-triggered continuous training (`schedule`, `repository_dispatch`) — not implemented.

Full detail on both in `reports/module-2.md`.

## Data source

[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
