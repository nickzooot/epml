FROM python:3.12-slim

WORKDIR /app

# System deps (optional: build tools if needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    pyyaml==6.0.2 \
    mlflow==2.16.0 \
    wandb==0.23.1 \
    dvc==3.64.0

ENV PYTHONPATH=/app/src

CMD ["python", "-c", "print('Container ready: iris-mlops')"]
