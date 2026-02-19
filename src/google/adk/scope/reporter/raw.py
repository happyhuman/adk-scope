from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from google.adk.scope import features_pb2
from google.adk.scope.utils import string
from google.adk.scope.utils.similarity import SimilarityScorer

# Global thresholds for match confidence
SIMILARITY_THRESHOLDS = {
    frozenset(["python", "go"]): {"high": 0.75, "avg": 0.70},
    frozenset(["python", "java"]): {"high": 0.8, "avg": 0.75},
    frozenset(["python", "typescript"]): {"high": 0.7, "avg": 0.55},
}

# Fallback thresholds if language pair not explicitly defined
DEFAULT_THRESHOLDS = {"high": 0.8, "avg": 0.6}


def get_type_display_name(f: features_pb2.Feature) -> str:
    FeatureType = features_pb2.Feature.Type
    if f.type == FeatureType.CONSTRUCTOR:
        return "constructor"
    elif f.type in (FeatureType.FUNCTION, FeatureType.CLASS_METHOD):
        return "function"
    elif f.type == features_pb2.Feature.Type.INSTANCE_METHOD:
        return "method"
    else:
        return "unknown"


class RawReportGenerator:
    def __init__(
        self,
        base_registry: features_pb2.FeatureRegistry,
        target_registry: features_pb2.FeatureRegistry,
    ):
        self.base_registry = base_registry
        self.target_registry = target_registry
        self.scorer = SimilarityScorer()

        # Pre-compute useful attributes
        self.base_name = string.get_language_name(self.base_registry.language)
        self.target_name = string.get_language_name(
            self.target_registry.language
        )
        self.base_code = self.base_name.lower()
        self.target_code = self.target_name.lower()
        self.thresholds = SIMILARITY_THRESHOLDS.get(
            frozenset([self.base_code, self.target_code]),
            DEFAULT_THRESHOLDS,
        )

        # Index target features by type
        self.target_by_type = defaultdict(list)
        for f in self.target_registry.features:
            self.target_by_type[f.type].append(f)

    def generate(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """Generates the raw report DataFrame and optionally saves it to CSV."""
        rows = []
        matched_target_ids = set()

        for f_base in self.base_registry.features:
            best_match, best_score = self._find_best_match(f_base)
            if best_match:
                matched_target_ids.add(id(best_match))
            row = self._create_row_data(f_base, best_match, best_score)
            rows.append(row)

    def generate(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """Generates the raw report DataFrame using global greedy assignment."""
        base_features = self.base_registry.features
        target_features = self.target_registry.features
        thresholds = self.thresholds

        # 1. Collect all candidate matches
        candidates = []
        for f_base in base_features:
            # Optimization: Only compare with features of compatible types to reduce N*M complexity
            # But earlier we decided to allow cross-type.
            
            for f_target in target_features:
                if "LlmAgent" in str(f_base):
                     print(f"DEBUG_RAW_BASE: name='{f_base.name}', orig='{f_base.original_name}', norm='{f_base.normalized_name}'")
                score, details = self.scorer.get_similarity_score(f_base, f_target)
                if score > 0.1:  # optimization: ignore very low scores
                    candidates.append((score, f_base, f_target, details))

        # 2. Sort by score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # 3. Greedy Assignment
        used_base = set()
        used_target = set()
        matches = []
        
        for score, f_base, f_target, details in candidates:
            if id(f_base) in used_base or id(f_target) in used_target:
                continue
            
            # This is a valid unique match
            used_base.add(id(f_base))
            used_target.add(id(f_target))
            
            # Determine validation status
            is_valid = score >= thresholds["avg"]
            confidence = "high" if score >= thresholds["high"] else "low"
            
            matches.append(
                self._create_match_row(
                    f_base, f_target, score, is_valid, confidence
                )
            )

        # 4. Add unmatched base features
        for f_base in base_features:
            if id(f_base) not in used_base:
                matches.append(
                    self._create_match_row(
                        f_base, None, 0.0, False, "low"
                    )
                )

        # 5. Sort output by base feature name for readability
        matches.sort(key=lambda x: (
            x.get(f"{self.base_code}_module", ""),
            x.get(f"{self.base_code}_container", ""),
            x.get(f"{self.base_code}_name", "")
        ))
        
        df = self._create_dataframe(matches)

        if output_path:
            self._save_csv(df, output_path)

        return df

    def _create_match_row(
        self,
        f_base: features_pb2.Feature,
        f_target: Optional[features_pb2.Feature],
        score: float,
        is_valid: bool = False,
        confidence: str = "low",
    ) -> Dict[str, Any]:
        """Constructs a dictionary representing a single row in the report."""
        row: Dict[str, Any] = {}

        # Base columns
        if f_base:
            self._fill_feature_cols(row, f_base, self.base_code)
        else:
            self._fill_empty_cols(row, self.base_code)

        # Target columns
        if f_target:
            self._fill_feature_cols(row, f_target, self.target_code)
        else:
            self._fill_empty_cols(row, self.target_code)

        # Metadata
        ref_feature = f_base if f_base else f_target
        if ref_feature:
            row["type"] = get_type_display_name(ref_feature)
        else:
            row["type"] = "unknown"

        row["score"] = score

        # Match status
        row["match"] = str(is_valid).lower()
        row["confidence"] = confidence

        return row

    def _fill_feature_cols(
        self, row: Dict[str, Any], f: features_pb2.Feature, prefix: str
    ):
        """Populates namespace, member_of, and name columns for a feature."""
        ns = f.namespace or f.normalized_namespace or ""
        mem = f.member_of or f.normalized_member_of or ""
        if str(mem).lower() == "null":
            mem = ""
        name = f.original_name or f.normalized_name or ""

        row[f"{prefix}_namespace"] = ns
        row[f"{prefix}_member_of"] = mem
        row[f"{prefix}_name"] = name

    def _fill_empty_cols(self, row: Dict[str, Any], prefix: str):
        """Fills feature columns with empty strings."""
        row[f"{prefix}_namespace"] = ""
        row[f"{prefix}_member_of"] = ""
        row[f"{prefix}_name"] = ""

    def _determine_match_status(self, score: float) -> Tuple[str, str]:
        """Determines match (true/false) and confidence (high/low)."""
        if score > self.thresholds["high"]:
            return "true", "high"
        elif score >= self.thresholds["avg"]:
            return "true", "low"
        else:
            return "false", "high"

    def _create_dataframe(self, rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """Creates and formats the pandas DataFrame."""
        cols_order = [
            f"{self.base_code}_namespace",
            f"{self.base_code}_member_of",
            f"{self.base_code}_name",
            f"{self.target_code}_namespace",
            f"{self.target_code}_member_of",
            f"{self.target_code}_name",
            "type",
            "score",
            "match",
            "confidence",
        ]

        if not rows:
            return pd.DataFrame(columns=cols_order)

        df = pd.DataFrame(rows)
        # Ensure correct column order and fill missing
        return df.reindex(columns=cols_order, fill_value="")

    def _save_csv(self, df: pd.DataFrame, output_path: str):
        """Saves DataFrame to CSV, creating directories if needed."""
        from pathlib import Path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
