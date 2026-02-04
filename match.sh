#!/bin/bash
# match.sh - Wrapper script to match features between two FeatureRegistries.

set -e

# Determine the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add 'src' to PYTHONPATH so the python script can find modules
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Run the python matcher
# Pass all arguments to the python script
python3 -m google.adk.scope.matcher.matcher "$@"
