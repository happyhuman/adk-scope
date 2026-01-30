#!/bin/bash
# extract_typescript.sh - Wrapper script to extract features from a TypeScript file, directory, or ADK repo.

set -e

# Determine the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add 'src' to PYTHONPATH so the python script can find modules
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Run the python extractor wrapper
# Pass all arguments to the python script
python3 -m google.adk.scope.extractors.extractor_ts "$@"
