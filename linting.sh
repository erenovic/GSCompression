#!/bin/bash

# Activate pipenv environment
pipenv shell

# Define folders to skip
SKIP_FOLDERS="/(\.direnv|\.eggs|\.git|\.hg|\.ipynb_checkpoints|"
SKIP_FOLDERS+="\.mypy_cache|\.nox|\.pytest_cache|\.ruff_cache|"
SKIP_FOLDERS+="\.tox|\.svn|\.venv|\.vscode|__pypackages__|_build|"
SKIP_FOLDERS+="buckout|build|dist|venv)|"
SKIP_FOLDERS+="SIBR_viewers|output|assets/"

# Run linting
echo "Running black..."
black . --line-length 100 --exclude $SKIP_FOLDERS
echo "Running isort..."
isort . --line-length 100 --skip $SKIP_FOLDERS
# echo "Running pylint..."
# pylint * --max-line-length=100 --ignore=$SKIP_FOLDERS

# Deactivate pipenv environment
# exit