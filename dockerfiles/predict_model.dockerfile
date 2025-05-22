# Base image
FROM python:3.11-slim

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requiremnets/ requirements/
COPY pyproject.toml pyproject.toml

COPY team_ops/ team_ops/
COPY data/ data/

RUN pip install . --no-cache-dir
# WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir
# RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "team_ops/predict_model.py"]