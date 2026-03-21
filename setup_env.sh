#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: 'conda' command not found. Please install/configure conda first."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx oft; then
  echo "Conda env 'oft' already exists, updating from environment.yml..."
  conda env update -n oft -f environment.yml --prune
else
  echo "Creating conda env 'oft' from environment.yml..."
  conda env create -f environment.yml
fi

echo "Environment ready. Use: conda activate oft"
