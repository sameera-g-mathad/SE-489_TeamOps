#!/bin/bash
set -e

# Load environment variables and pull DVC data
echo "Pulling DVC data..."
dvc pull -r data-remote --force
dvc pull -r model-remote --force
dvc checkout

# Start the app
echo "Starting server..."
exec uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8080}
