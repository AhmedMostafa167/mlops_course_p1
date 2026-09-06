# Module 1 Report — From Notebook to Production-Ready Service

## Baseline
- Validation MAE: `3.896`
- Validation RMSE: `6.4647`

## Refactor summary
The baseline notebook was decomposed into a pip-installable package (`src/prodml/`):
- `data.py` — download / load / train-validation split
- `features.py` — feature engineering (`PU_DO` pair, `trip_distance`)
- `train.py` — model fitting (`lr`, `xgboost`, `mlp`) + persistence, writes `metrics/new_run.json` each run
- `predict.py` — `DurationPredictor` with `.load()`, `.load_from_registry()`, `.predict_one()`, `.predict_batch()`
- `config.py` — environment-backed settings via `pydantic-settings`, no hardcoded paths
- `api/main.py` — FastAPI service (4 endpoints)

`uv run python -m prodml.train --model lr` reproduces the baseline MAE within ±0.05 of the notebook.

## Structured logging
All modules use `structlog` with JSON output. Every log line carries a correlation ID, generated per-request in `api/main.py`'s middleware and returned as an `X-Request-ID` header — traceable end to end from request log → prediction log → response.

## Serialization
| Format   | Human-readable | Cross-language | Schema-enforced | Safe from untrusted source |
|----------|-----------------|------------------|-------------------|-------------------------------|
| JSON     | Yes | Yes | No  | Yes |
| Protobuf | No  | Yes | Yes | Yes |
| Pickle   | No  | No  | No  | **No — executes arbitrary code on load** |
| ONNX     | No  | Yes | Yes | Yes |

**Format served in production: pickle.** The FastAPI service loads a pickled `DurationPredictor` bundle (model + `DictVectorizer` + metadata as one object) rather than ONNX — it keeps serving code simple and avoids a separate preprocessing step at inference time. ONNX export is implemented and parity-verified (`src/prodml/export.py`) for all three model families (`lr`, `xgboost`, `mlp`), but not adopted for serving in this iteration.

Never load a `.pkl` you did not produce — pickle executes arbitrary code on load.

**Latency (500 validation rows):**
```
[FILL IN — run: uv run python -m prodml.export && uv run python scripts/benchmark_serialization.py]
Pickle: mean=___ms  p95=___ms
ONNX:   mean=___ms  p95=___ms
```
Parity check: max prediction difference `3.0517578125e-05`, tolerance 1e-4.

## Docker
- Multi-stage build, non-root user (`appuser`, uid 1000)
- `.dockerignore` excludes `.venv/`, `data/`, `notebooks/`, `tests/`
- Multi-stage image size: `162.71 MB` MB

```bash
docker exec <container_id> whoami   # appuser — confirmed non-root
```

## Test suite
- pytest with fixtures, mocks (`monkeypatch`), `@pytest.mark.parametrize`
- Coverage gate enforced in CI: `--cov-fail-under=70`
- Current coverage: `71.18`%



*"This repo currently sits at Level 2 of the MLOps maturity model — the training pipeline is packaged, tested, and containerized, but training itself is still triggered manually rather than by an automated pipeline. Module 2 (MLflow tracking, DVC-versioned data, CI-gated promotion) moves this toward Level 3."*
