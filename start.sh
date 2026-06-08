#!/bin/bash
# start.sh — run from RAG_METAML/ project root
# Sets PYTHONPATH so all packages resolve, then launches uvicorn.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting MetaML Taxonomy API..."
echo "Project root: $SCRIPT_DIR"
echo ""

# Local full install: pip install -r requirements-local.txt
PYTHONPATH="$SCRIPT_DIR" uvicorn index.api:app \
    --reload \
    --port 8000 \
    --host 127.0.0.1