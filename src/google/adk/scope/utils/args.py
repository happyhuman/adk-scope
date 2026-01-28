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
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-file",
        type=Path,
        help="Single file to extract.",
    )
    group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory to extract (non-recursive).",
    )
    group.add_argument(
        "--input-repo",
        type=Path,
        help="Repository root to extract (recursive in src).",
    )
    
    parser.add_argument(
        "output",
        type=Path,
        help="Path to the output file",
    )
    return parser.parse_args()
