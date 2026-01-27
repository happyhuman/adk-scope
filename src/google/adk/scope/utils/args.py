"""
Argument parsing utility for ADK Scope.
"""

import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the Python extractor.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Extract features from ADK Python repository.")
    parser.add_argument(
        "adk_repo",
        type=Path,
        help="Path to the ADK Python repository root",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to the output file",
    )
    return parser.parse_args()
