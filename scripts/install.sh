#!/bin/bash
set -e

echo "Installing entity-memory CLI..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pipx install -e "$SCRIPT_DIR/.."

echo "Pulling Qdrant Docker image..."
docker pull qdrant/qdrant:latest

# Start Qdrant if not already running. Match either the current or the legacy
# container name, so upgrading a host that still runs the old openclaw-memory
# container doesn't try to start a second Qdrant on the same port.
if ! docker ps --format '{{.Names}}' | grep -qE '^(entity-memory|openclaw-memory)$'; then
    echo "Starting Qdrant container..."
    docker compose -f "$SCRIPT_DIR/../docker-compose.yml" up -d
fi

echo "Initializing collections..."
memory init

echo "Done. Entity memory is ready."
