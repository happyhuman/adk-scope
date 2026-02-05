#!/bin/bash
# match.sh - Wrapper script to match features between two FeatureRegistries.

set -e

# Determine the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add 'src' to PYTHONPATH so the python script can find modules
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Run the python matcher
# Pass all arguments to the python script
VERBOSE=""
if [[ "$1" == "-v" ]]; then
  VERBOSE="--verbose"
  shift
fi
python3 "${SCRIPT_DIR}/src/google/adk/scope/matcher/matcher.py" $VERBOSE "$@"
