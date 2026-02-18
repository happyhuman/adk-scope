#!/bin/bash
set -e

# Resolve the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <feature1.txtpb> <feature2.txtpb> [options]"
    exit 1
fi

python3 "${SCRIPT_DIR}/src/google/adk/scope/utils/score_features.py" "$@"
