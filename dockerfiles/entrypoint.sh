#!/bin/bash
set -e

# Start the app
echo "Starting server..."
exec uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8080}
