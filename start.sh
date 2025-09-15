#!/bin/bash
# make executable: chmod +x start.sh

echo "Starting Gunicorn server with low memory configuration..."
exec gunicorn wilddd:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 2 \
    --timeout 120 \
    --log-level debug
