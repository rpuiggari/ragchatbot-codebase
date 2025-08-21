#!/bin/bash

# Code Quality Check Script
# Run all code quality tools for the RAG chatbot project

set -e

echo "🔍 Running code quality checks..."
echo

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Function to run a command and capture its exit code
run_check() {
    local tool_name="$1"
    local command="$2"
    echo "📋 Running $tool_name..."
    if eval "$command"; then
        echo "✅ $tool_name passed"
    else
        echo "❌ $tool_name failed"
        return 1
    fi
    echo
}

# Install dependencies if needed
echo "📦 Ensuring dependencies are installed..."
uv sync
echo

# Run Black (code formatting check)
run_check "Black (formatting check)" "uv run black --check --diff ."

# Run isort (import sorting check)
run_check "isort (import sorting check)" "uv run isort --check-only --diff ."

# Run flake8 (linting)
run_check "flake8 (linting)" "uv run flake8 ."

# Run mypy (type checking)
run_check "mypy (type checking)" "uv run mypy backend/ main.py"

# Run tests
run_check "pytest (tests)" "cd backend && uv run pytest"

echo "🎉 All code quality checks passed!"