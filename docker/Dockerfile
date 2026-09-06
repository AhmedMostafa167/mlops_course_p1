FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /build/src /app/src
COPY --chown=appuser:appuser models/model.pkl /app/models/model.pkl
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
