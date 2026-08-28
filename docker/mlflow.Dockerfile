FROM python:3.12-slim

ARG MLFLOW_VERSION=3.15.2

RUN pip install --no-cache-dir \
    "mlflow==${MLFLOW_VERSION}" \
    "psycopg2-binary>=2.9,<3"

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
