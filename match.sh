#!/bin/bash
# match.sh - Wrapper script to match features between two FeatureRegistries.

set -e

# Default values
REPORT_TYPE="symmetric"
ALPHA="0.8"
VERBOSE=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --base)
            BASE_FILE="$2"
            shift 2
            ;;
        --target)
            TARGET_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --report-type)
            REPORT_TYPE="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Extract languages
BASE_LANG_RAW=$(head -n 1 "${BASE_FILE}" | grep -o 'language: "[A-Z]*"' | grep -o '"[A-Z]*"' | tr -d '"')
TARGET_LANG_RAW=$(head -n 1 "${TARGET_FILE}" | grep -o 'language: "[A-Z]*"' | grep -o '"[A-Z]*"' | tr -d '"')

# Function to map language to short code
get_lang_code() {
    case "$1" in
        PYTHON) echo "py" ;;
        TYPESCRIPT) echo "ts" ;;
        JAVA) echo "java" ;;
        GOLANG) echo "go" ;;
        *) echo "" ;;
    esac
}

BASE_LANG=$(get_lang_code "$BASE_LANG_RAW")
TARGET_LANG=$(get_lang_code "$TARGET_LANG_RAW")

# Construct filename
if [ "$REPORT_TYPE" == "raw" ]; then
    EXTENSION="csv"
else
    EXTENSION="md"
fi
OUTPUT_FILENAME="${BASE_LANG}_${TARGET_LANG}_${REPORT_TYPE}.${EXTENSION}"
FULL_OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_FILENAME}"

# Determine the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add 'src' to PYTHONPATH so the python script can find modules
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Run the python matcher
python3 "${SCRIPT_DIR}/src/google/adk/scope/matcher/matcher.py" \
    --base "${BASE_FILE}" \
    --target "${TARGET_FILE}" \
    --output "${FULL_OUTPUT_PATH}" \
    --report-type "${REPORT_TYPE}" \
    --alpha "${ALPHA}" \
    ${VERBOSE}
