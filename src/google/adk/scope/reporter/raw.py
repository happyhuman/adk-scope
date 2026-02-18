
import pandas as pd
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple
from google.adk.scope import features_pb2
from google.adk.scope.utils.similarity import SimilarityScorer

# Global thresholds for match confidence
SIMILARITY_THRESHOLDS = {
    frozenset(["py", "go"]): {"high": 0.6, "avg": 0.5},
    frozenset(["py", "java"]): {"high": 0.6, "avg": 0.58},
    frozenset(["py", "ts"]): {"high": 0.7, "avg": 0.55},
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


def _get_lang_code(language: str) -> str:
    """Returns a short code for the language (e.g. PYTHON -> py)."""
    name = language.upper()
    if name in {"PYTHON", "PY"}:
        return "py"
    elif name in {"TYPESCRIPT", "TS"}:
        return "ts"
    elif name == "JAVA":
        return "java"
    elif name in {"GOLANG", "GO"}:
        return "go"
    return name.lower()


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
        self.base_code = _get_lang_code(self.base_registry.language)
        self.target_code = _get_lang_code(self.target_registry.language)
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
        for f_base in self.base_registry.features:
            best_match, best_score = self._find_best_match(f_base)
            row = self._create_row_data(f_base, best_match, best_score)
            rows.append(row)

        df = self._create_dataframe(rows)

        if output_path:
            self._save_csv(df, output_path)

        return df

    def _find_best_match(
        self, f_base: features_pb2.Feature
    ) -> Tuple[Optional[features_pb2.Feature], float]:
        """Finds the best matching feature in the target registry."""
        candidates = self.target_by_type.get(f_base.type, [])
        if not candidates:
            return None, 0.0

        best_match = None
        best_score = -1.0

        for f_target in candidates:
            score = self.scorer.get_similarity_score(f_base, f_target)
            if score > best_score:
                best_score = score
                best_match = f_target

        return best_match, best_score

    def _create_row_data(
        self,
        f_base: features_pb2.Feature,
        f_target: Optional[features_pb2.Feature],
        score: float,
    ) -> Dict[str, Any]:
        """Constructs a dictionary representing a single row in the report."""
        row: Dict[str, Any] = {}

        # Base columns
        self._fill_feature_cols(row, f_base, self.base_code)

        # Target columns
        if f_target:
            self._fill_feature_cols(row, f_target, self.target_code)
        else:
            self._fill_empty_cols(row, self.target_code)

        # Metadata
        row["type"] = get_type_display_name(f_base)
        row["score"] = score

        # Match status
        match_str, confidence_str = self._determine_match_status(score)
        row["match"] = match_str
        row["confidence"] = confidence_str

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