# NYC Green Taxi Duration Prediction Service

This project is a Module 1 MLOps mini-project based on one month of NYC TLC green taxi trip data. The current implementation builds a reproducible baseline model, separates the workflow into reusable Python modules, adds environment-backed configuration and structured logging, exports the model to ONNX for comparison, and provides an initial FastAPI service.

The project predicts taxi-trip duration in minutes using two engineered features:

- `PU_DO`: the pickup and drop-off location pair.

- `trip_distance`: the trip distance in miles.

The baseline model uses a scikit-learn `DictVectorizer` followed by `LinearRegression`.

## Current project status

| Area | Status |
| --- | --- |
| NYC TLC monthly Parquet download | Completed |
| Duration target engineering | Completed |
| `PU_DO` and `trip_distance` features | Completed |
| Chronological train/validation split | Completed |
| `DictVectorizer` and `LinearRegression` baseline | Completed |
| RMSE and MAE evaluation | Completed |
| Model persistence to Pickle | Completed |
| Configuration with `pydantic-settings` | Completed |
| Decomposition into `src/prodml/` modules | Completed |
| Structured JSON logging | Initial implementation completed |
| ONNX export and prediction parity | Initial implementation completed |
| FastAPI application structure | Initial implementation completed |
| Refactoring tests from the old model | **Not completed; future work** |
| Dockerfile and container execution | **Not completed; future work** |

## Repository structure

```
.
├── data/
│   └── green_tripdata_2024-01.parquet
├── docs/
│   └── main_py_explained.md
├── models/
│   ├── model.pkl
│   └── model.onnx
├── notebooks/
│   └── 00-baseline.ipynb
├── reports/
│   └── module-1.md
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_config.py
│       ├── predict.py
│       ├── train.py
│       └── api/
│           ├── __init__.py
│           ├── main.py
│           └── schemas.py
├── tests/
├── .env.example
├── pyproject.toml
└── uv.lock
```

Each module has one primary responsibility:

| Module | Responsibility |
| --- | --- |
| `config.py` | Loads paths and settings from environment variables using Pydantic Settings |
| `data.py` | Downloads the monthly Parquet file, loads required columns, and creates the train/validation split |
| `features.py` | Creates the duration target, filters invalid rows, builds `PU_DO`, and prepares feature dictionaries |
| `train.py` | Fits the vectorizer and regression model, evaluates metrics, and saves the Pickle artifact |
| `predict.py` | Loads the saved model and exposes `predict_one()` and `predict_batch()` |
| `export.py` | Exports the fitted scikit-learn model to ONNX and compares predictions |
| `logging_config.py` | Configures structured JSON logging |
| `api/main.py` | Creates the FastAPI application and defines the service endpoints |
| `api/schemas.py` | Defines Pydantic request and response models |

## Environment configuration

The application reads its configuration from `.env` through `pydantic-settings`. Copy the example file before running the project:

```bash
cp .env.example .env
```

The environment file controls values such as the data month, data directory, model path, report path, and validation fraction. Do not commit private or machine-specific values from `.env`; commit `.env.example` instead.

## Installation

Install the project with its development dependencies:

```bash
uv sync --extra dev
```

Run commands from the repository root so that the local `prodml` package is imported correctly.

## Baseline workflow

Run the baseline from the repository root:

```bash
uv run -m prodml.train
```

The command downloads the configured TLC Parquet file if it is missing, prepares the data, fits the model, calculates validation metrics, and saves the model to:

```
models/model.pkl
```

Current baseline results:

| Metric | Result |
| --- | --- |
| Validation RMSE | 6.4647 minutes |
| Validation MAE | 3.8969 minutes |

The notebook `notebooks/00-baseline.ipynb` demonstrates the same workflow, while `reports/module-1.md` records the metrics.

## Structured logging

The application uses Structlog to produce JSON log records. The configuration is defined in `src/prodml/logging_config.py`.

The configuration is initialized by an executable entry point, while individual modules create named loggers:

```python
import structlog

logger = structlog.get_logger(__name__)
```

Logs from the training module are identified as `prodml.train`, and logs from the API module are identified as `prodml.api.main`.

Important events use structured fields rather than formatted strings. A record may look like this:

```json
{
  "message": "model_saved",
  "path": "models/model.pkl",
  "rmse": 6.4647,
  "mae": 3.8969,
  "logger": "prodml.train",
  "level": "info",
  "timestamp": "2026-08-21T10:00:00Z"
}
```

The API also attaches a correlation ID to request logs and returns it in the `X-Request-ID` response header.

## ONNX export and parity

The serialization workflow uses the fitted Pickle artifact as the reference model. The fitted `LinearRegression` estimator is exported to ONNX after the existing `DictVectorizer` transforms the feature dictionaries into a numeric matrix.

Run the export workflow with:

```bash
uv run -m prodml.export
```

The workflow should select 500 validation rows, export the model to `models/model.onnx`, validate the ONNX graph, compare Pickle and ONNX predictions, and assert that the maximum difference is within `atol=1e-4`.

Pickle remains the reference training artifact. ONNX is the preferred format for a portable serving implementation because it can run through ONNX Runtime without loading a Python Pickle object.

## API

The initial FastAPI implementation is under `src/prodml/api/`.

Start it from the repository root with:

```bash
uv run uvicorn prodml.api.main:app --reload --port 8000
```

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Confirms that the model is loaded in application memory |
| `/metadata` | `GET` | Returns model version, training date, features, framework, and artifact hash |
| `/predict` | `POST` | Returns one taxi-duration prediction |
| `/predict/batch` | `POST` | Returns predictions for a list of inputs |

A single request has this shape:

```json
{
  "pu_do": "74_75",
  "trip_distance": 1.5
}
```

A successful response contains the prediction, model version, correlation ID, and latency:

```json
{
  "prediction": 9.60,
  "model_version": "linear-regression-2024-01",
  "correlation_id": "example-request-id",
  "latency_ms": 13.2
}
```

The model is loaded once during FastAPI startup through the lifespan function. The request dependency retrieves the already-loaded predictor from `app.state`; it does not reload the model for every request.

## Future work

### Refactor testing

The test suite has **not yet been fully refactored from the old model to the new decomposed model and API**. This is future work. The tests still need to be reviewed and updated to cover the current interfaces, including model training and persistence, `DurationPredictor.predict_one()`, `predict_batch()`, API validation, health and metadata responses, single and batch predictions, error responses, correlation IDs, and Pickle/ONNX parity.

### Create the Dockerfile

A `Dockerfile` has **not yet been created**. Docker support is future work. It will eventually need to install the package and runtime dependencies, start Uvicorn, expose port `8000`, and include a healthcheck. The container workflow should be tested from a clean environment after the local workflow is stable.

### Other future work

Other later deliverables may include pre-commit hooks, CI checks, more complete logging tests, ONNX-backed serving, load testing, monitoring, model optimization, and deployment automation. These are intentionally outside the current completed stage.

## Current definition of done

At the current stage, the project is complete for the baseline, decomposition, initial logging, serialization, and initial API structure when the baseline and module workflow produce the same metrics, the model is saved to `models/model.pkl`, the ONNX export creates and validates `models/model.onnx`, Pickle and ONNX pass the 500-row parity check, structured logs identify the relevant module and event, and the API loads the model at startup and responds through the four endpoints.

The project is **not yet production-ready** because test refactoring and Docker support remain future work.

## Data source

The monthly data comes from the official [NYC TLC Trip Record Data page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).