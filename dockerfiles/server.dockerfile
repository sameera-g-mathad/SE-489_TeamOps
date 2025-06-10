# Base image
FROM python:3.11-slim

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements/ requirements/
COPY pyproject.toml pyproject.toml
COPY .git/ .git/
COPY .dvc/ .dvc/
COPY data.dvc data.dvc
COPY models.dvc models.dvc
COPY team_ops/ team_ops/
COPY server/ server/
COPY dockerfiles/entrypoint.sh entrypoint.sh
RUN pip install . --no-cache-dir
RUN chmod +x entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]
