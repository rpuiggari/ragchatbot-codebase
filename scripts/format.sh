#!/bin/bash

# Code Formatting Script
# Automatically format code using black and isort

set -e

echo "🎨 Formatting code..."
echo

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Install dependencies if needed
echo "📦 Ensuring dependencies are installed..."
uv sync
echo

# Run Black (code formatting)
echo "📋 Running Black (code formatting)..."
uv run black .
echo "✅ Black formatting completed"
echo

# Run isort (import sorting)
echo "📋 Running isort (import sorting)..."
uv run isort .
echo "✅ isort import sorting completed"
echo

echo "🎉 Code formatting completed!"