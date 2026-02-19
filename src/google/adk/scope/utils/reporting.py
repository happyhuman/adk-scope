"""
Reporting utilities for ADK Scope.
"""

from typing import Tuple

import pandas as pd


def get_match_icon(match: str, confidence: str) -> str:
    """Returns an icon representing the match status."""
    is_match = str(match).lower() == "true"
    if is_match:
        return "✅" if confidence == "high" else "⚠️"
    return "❌"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces NaN and empty strings with a placeholder."""
    # Create a copy to avoid side effects
    df_clean = df.copy()
    df_clean = df_clean.fillna("___")
    df_clean = df_clean.replace("", "___")
    # Replace stringified nan if they exist
    df_clean = df_clean.replace("nan", "___")
    df_clean = df_clean.replace("NaN", "___")
    return df_clean
