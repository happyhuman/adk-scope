#!/bin/bash
# match.sh - Wrapper script to match features between two FeatureRegistries.

set -e

# Default values

VERBOSE=""
COMMON=""

# Parse arguments
REGISTRIES=()
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
        --registries)
            shift
            while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                REGISTRIES+=("$1")
                shift
            done
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;

        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --common)
            COMMON="--common"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to map language to short code
get_lang_code() {
    case "$1" in
        PYTHON) echo "py" ;;
        TYPESCRIPT) echo "ts" ;;
        JAVA) echo "java" ;;
        GO) echo "go" ;;
        *) echo "" ;;
    esac
}

if [[ ${#REGISTRIES[@]} -eq 0 && -n "$BASE_FILE" && -n "$TARGET_FILE" ]]; then
    REGISTRIES+=("$BASE_FILE" "$TARGET_FILE")
fi

if [[ ${#REGISTRIES[@]} -lt 2 ]]; then
    echo "Error: Must provide at least two registries via --registries or --base/--target"
    exit 1
fi

# Extract languages and construct filename
LANG_CODES=()
for REG_FILE in "${REGISTRIES[@]}"; do
    LANG_RAW=$(head -n 1 "${REG_FILE}" | grep -o 'language: "[A-Z]*"' | grep -o '"[A-Z]*"' | tr -d '"')
    LANG_CODES+=($(get_lang_code "$LANG_RAW"))
done

# Construct filename
# Default to markdown extension. The python script will generate CSV alongside it.
EXTENSION="md"

# Standard 2-way report
OUTPUT_FILENAME="${LANG_CODES[0]}_${LANG_CODES[1]}.${EXTENSION}"

# Check if we are running in matrix mode (CSV inputs)
if [[ "${REGISTRIES[0]}" == *.csv ]]; then
    OUTPUT_FILENAME="matrix_report.md"
fi

FULL_OUTPUT_PATH="${OUTPUT_DIR}/${OUTPUT_FILENAME}"

# Determine the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add 'src' to PYTHONPATH so the python script can find modules
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"

# Run the python reporter
python3 "${SCRIPT_DIR}/src/google/adk/scope/reporter/reporter.py" \
    --registries "${REGISTRIES[@]}" \
    --output "${FULL_OUTPUT_PATH}" \

    ${COMMON} \
    ${VERBOSE}

