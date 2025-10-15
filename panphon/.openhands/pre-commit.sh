#!/usr/bin/env bash
# Run Ruff and Mypy on all Python files in the project.
# Exits nonzero if any errors are found.

pip install ruff
pip install mypy

set -euo pipefail

# Find all Python files (excluding virtualenvs, build dirs, etc.)
PY_FILES=$(find . -type f -name "*.py" \
    -not -path "*/venv/*" \
    -not -path "*/.venv/*" \
    -not -path "*/__pycache__/*" \
    -not -path "*/build/*" \
    -not -path "*/dist/*")

if [ -z "$PY_FILES" ]; then
    echo "No Python files found."
    exit 0
fi

echo "🧹 Running Ruff..."
ruff check $PY_FILES

echo
echo "🔍 Running Mypy..."
mypy $PY_FILES

echo
echo "✅ All checks completed successfully."