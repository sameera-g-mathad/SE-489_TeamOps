# Base image
FROM python:3.11-slim

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements/ requirements/
COPY pyproject.toml pyproject.toml

COPY team_ops/ team_ops/
COPY data/ data/
COPY models/ models/

RUN pip install . --no-cache-dir

ENTRYPOINT ["python", "-u", "team_ops/predict_model.py"]