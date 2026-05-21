import logging
import re
from collections import Counter
from typing import Optional, Set

import numpy as np
from jellyfish import jaro_winkler_similarity, levenshtein_distance
from scipy.optimize import linear_sum_assignment

from google.adk.scope import features_pb2 as features_pb

logger = logging.getLogger(__name__)

# Default weights for the similarity calculation.
DEFAULT_SIMILARITY_WEIGHTS = {
    "name": 0.35,
    "member_of": 0.25,
    "namespace": 0.10,
    "parameters": 0.15,
    "return_type": 0.15,
}


class SimilarityScorer:
    """Calculates a similarity score between two features."""

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
    ):
        self.weights = weights or DEFAULT_SIMILARITY_WEIGHTS
        logger.debug(
            f"Initializing SimilarityScorer with " f"weights={self.weights}"
        )
        assert "name" in self.weights
        assert "member_of" in self.weights
        assert "namespace" in self.weights
        assert "parameters" in self.weights
        assert "return_type" in self.weights

    def get_similarity(self, s1: str, s2: str) -> float:
        """Calculates similarity between two strings using the selected
        algorithm."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # 1. Levenshtein Distance (Character-based)
        dist = levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        lev_score = 1.0 - (dist / max_len)
        
        # 2. Jaccard Similarity (Multiset-based, handles reordering/partial match)
        jaccard_score = self._jaccard_similarity(s1, s2)
        
        # 3. Jaro-Winkler (Prefix bias, handles typos/short strings)
        jw_score = jaro_winkler_similarity(s1, s2)
        
        # Average the three scores
        return (lev_score + jaccard_score + jw_score) / 3.0

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Calculates Jaccard similarity based on character abundance (multisets)."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        c1 = Counter(s1.lower())
        c2 = Counter(s2.lower())
        
        # Union of keys
        all_chars = set(c1.keys()) | set(c2.keys())
        
        intersection = 0
        union = 0
        
        for char in all_chars:
            intersection += min(c1[char], c2[char])
            union += max(c1[char], c2[char])
            
        return intersection / union if union > 0 else 0.0

    def _fuzzy_type_match(self, types1: list, types2: list) -> float:
        """Calculates a fuzzy similarity score between two lists of types."""

        def _to_str_set(type_list):
            res = set()
            for t in type_list:
                if isinstance(t, int):
                    # It's a ParamType enum
                    try:
                        res.add(features_pb.ParamType.Name(t))
                    except ValueError:
                        res.add(str(t))
                else:
                    res.add(str(t).upper())
            return res

        set1 = _to_str_set(types1)
        set2 = _to_str_set(types2)

        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        if set1 == set2:
            logger.debug(f"Exact type match: {set1}")
            return 1.0

        # Check the best match between any pair of types
        best_score = 0.0

        logger.debug(f"Fuzzy type match between {set1} and {set2}")
        for t1 in set1:
            for t2 in set2:
                if t1 == t2:
                    score = 1.0
                elif {t1, t2} == {"MAP", "OBJECT"} or {t1, t2} == {
                    "MAP",
                    "ANY",
                }:
                    score = 0.4
                elif t1 in ("UNKNOWN", "ANY") or t2 in ("UNKNOWN", "ANY"):
                    score = 0.3
                elif t1 == "OBJECT" or t2 == "OBJECT":
                    score = 0.2
                else:
                    score = 0.0

                if score > best_score:
                    best_score = score

        return best_score

    def _calculate_param_similarity(
        self, param1: features_pb.Param, param2: features_pb.Param
    ) -> float:
        """Calculates the similarity score between two individual parameters."""
        s_p_name = self.get_similarity(
            param1.normalized_name, param2.normalized_name
        )
        s_p_type = self._fuzzy_type_match(
            list(param1.normalized_types), list(param2.normalized_types)
        )
        s_p_opt = 1.0 if param1.is_optional == param2.is_optional else 0.0

        # Weights for parameter components
        score = (0.5 * s_p_name) + (0.4 * s_p_type) + (0.1 * s_p_opt)
        logger.debug(
            f"Param '{param1.normalized_name}' vs '{param2.normalized_name}': "
            f"{score:.4f} (name:{s_p_name:.2f}, type:{s_p_type:.2f}, "
            f"opt:{s_p_opt:.2f})"
        )
        return score

    def _calculate_parameters_score(
        self, params1: list[features_pb.Param], params2: list[features_pb.Param]
    ) -> float:
        """Calculates aggregated similarity for two lists of parameters."""
        logger.debug(
            f"Calculating parameter score for {len(params1)} "
            f"vs {len(params2)} parameters"
        )
        if not params1 and not params2:
            logger.debug("Both parameter lists empty. Returning 1.0")
            return 1.0
        if not params1 or not params2:
            logger.debug(
                "One parameter list empty while other is not. Returning 0.0"
            )
            return 0.0

        similarity_matrix = np.zeros((len(params1), len(params2)))
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                similarity_matrix[i, j] = self._calculate_param_similarity(
                    p1, p2
                )

        row_ind, col_ind = linear_sum_assignment(
            similarity_matrix, maximize=True
        )
        total_match_score = similarity_matrix[row_ind, col_ind].sum()
        total_params = len(params1) + len(params2)

        if total_params == 0:
            return 1.0

        score = (2 * total_match_score) / total_params
        logger.debug(
            f"Matrix matched total score: {total_match_score:.4f}, "
            f"final parameter score: {score:.4f}"
        )
        # Log parameter matches
        for r, c in zip(row_ind, col_ind):
            if similarity_matrix[r, c] > 0:
                logger.debug(
                    f"  Matched param '{params1[r].normalized_name}' with "
                    f"'{params2[c].normalized_name}': "
                    f"{similarity_matrix[r, c]:.4f}"
                )
        return score

    def _calculate_return_type_score(
        self, f1: features_pb.Feature, f2: features_pb.Feature
    ) -> float:
        """Calculates the similarity score for the return types."""
        s_type_match = self._fuzzy_type_match(
            list(f1.normalized_return_types), list(f2.normalized_return_types)
        )
        s_async_match = (
            1.0 if getattr(f1, "async") == getattr(f2, "async") else 0.0
        )
        score = (0.7 * s_type_match) + (0.3 * s_async_match)
        logger.debug(
            f"Return type score: {score:.4f} (type match: "
            f"{s_type_match}, async match: {s_async_match})"
        )
        return score

    def _clean_identifier(self, name: str, namespace: str) -> str:
        """Strips namespace prefixes/suffixes from the identifier."""
        if not name or not namespace:
            return name

        # Strip generic packaging suffixes: e.g. 'adk_artifacts' -> 'artifact'
        clean_ns = namespace.lower()
        if clean_ns.startswith("adk_"):
            clean_ns = clean_ns[4:]

        # Singularize namespace for robust mapping: artifacts -> artifact
        clean_ns = clean_ns.rstrip("s")

        name_clean = name.lower()

        # Suffix and prefix clean patterns
        to_strip = [f"_{clean_ns}", clean_ns]
        for s in to_strip:
            if name_clean.endswith(s):
                name_clean = name_clean[: -len(s)].rstrip("_")
            elif name_clean.startswith(s):
                name_clean = name_clean[len(s) :].lstrip("_")

        return name_clean

    def _clean_member(self, member: str, namespace: str) -> str:
        """Strips package/namespace names from struct/class names."""
        if not member or not namespace:
            return member

        clean_ns = namespace.lower()
        if clean_ns.startswith("adk_"):
            clean_ns = clean_ns[4:]

        # e.g. 'artifacts' -> 'Artifact'
        clean_ns = clean_ns.rstrip("s").capitalize()

        # e.g. 'InMemoryArtifactService' -> 'InMemoryService'
        member_clean = member
        if clean_ns in member_clean:
            member_clean = member_clean.replace(clean_ns, "")

        return member_clean.lower()

    def get_similarity_score(
        self, feature1: features_pb.Feature, feature2: features_pb.Feature
    ) -> float:
        """Computes the overall similarity score between two features."""
        logger.debug(
            f"Comparing '{feature1.normalized_name}' and "
            f"'{feature2.normalized_name}'"
        )
        # 1. Type Compatibility and Dynamic Weights
        t1, t2 = feature1.type, feature2.type
        current_weights = self.weights.copy()

        FeatureType = features_pb.Feature.Type
        if t1 == FeatureType.CONSTRUCTOR and t2 == FeatureType.CONSTRUCTOR:
            # For constructors:
            # 1. Ignore Name (Python __init__ vs Go New)
            # 2. Ignore Return Type (Python None vs Go *T)
            # 3. Boost MemberOf (Class Match is the most important signal)
            current_weights["member_of"] += current_weights["name"] + current_weights["return_type"]
            current_weights["name"] = 0.0
            current_weights["return_type"] = 0.0
            logger.debug(
                "Both CONSTRUCTOR. " f"Adjusted weights: {current_weights}"
            )
        elif t1 in (FeatureType.FUNCTION, FeatureType.CLASS_METHOD) and t2 in (
            FeatureType.FUNCTION,
            FeatureType.CLASS_METHOD,
        ):
            current_weights["member_of"] /= 2.0
            current_weights["name"] += current_weights["member_of"]
            logger.debug(
                "Both FUNCTION/CLASS_METHOD. "
                f"Adjusted weights: {current_weights}"
            )
        elif (
            t1 == FeatureType.INSTANCE_METHOD
            and t2 == FeatureType.INSTANCE_METHOD
        ):
            logger.debug(
                "Both INSTANCE_METHOD. "
                f"Using default weights: {current_weights}"
            )
            pass  # Keep default weights

        elif {t1, t2} == {FeatureType.INSTANCE_METHOD, FeatureType.FUNCTION}:
            # Only allow if one is a generic method name (heuristic match)
            f1_name = feature1.normalized_name
            f2_name = feature2.normalized_name
            is_heuristic = (
                (t1 == FeatureType.INSTANCE_METHOD and f1_name in ("__call__", "invoke", "apply")) or
                (t2 == FeatureType.INSTANCE_METHOD and f2_name in ("__call__", "invoke", "apply"))
            )
            if not is_heuristic:
                 return 0.0, {}
            # If heuristic applies, fall through to calculations
            pass

        else:
            logger.debug(f"Incompatible types: {t1} vs {t2}. Returning 0.0")
            return 0.0, {}  # Fast out for incompatible types

        # 2. Clean naming redundancies
        f1_ns = feature1.normalized_namespace or ""
        f2_ns = feature2.normalized_namespace or ""

        f1_name_clean = self._clean_identifier(
            feature1.normalized_name, f1_ns
        )
        f2_name_clean = self._clean_identifier(
            feature2.normalized_name, f2_ns
        )

        f1_member_clean = self._clean_member(
            feature1.normalized_member_of, f1_ns
        )
        f2_member_clean = self._clean_member(
            feature2.normalized_member_of, f2_ns
        )

        # 3. Similarity Calculations
        scores = {
            "name": self.get_similarity(f1_name_clean, f2_name_clean),
            "member_of": self.get_similarity(
                f1_member_clean, f2_member_clean
            ),
            "namespace": self.get_similarity(
                feature1.normalized_namespace, feature2.normalized_namespace
            ),
        }

        # Heuristic: If comparing INSTANCE_METHOD vs FUNCTION, and method name is generic 
        # (e.g. __call__, invoke), use the Class Name (member_of) as the Name for comparison.
        if {t1, t2} == {FeatureType.INSTANCE_METHOD, FeatureType.FUNCTION}:
             f1_name = feature1.normalized_name
             f2_name = feature2.normalized_name
             heuristic_applied = False
             
             # Check f1 (Python side?)
             if t1 == FeatureType.INSTANCE_METHOD and f1_name in ("__call__", "invoke", "apply"):
                 f1_name = feature1.normalized_member_of
                 heuristic_applied = True
                 
             # Check f2 (if Python target?)
             if t2 == FeatureType.INSTANCE_METHOD and f2_name in ("__call__", "invoke", "apply"):
                 f2_name = feature2.normalized_member_of
                 heuristic_applied = True

             f1_name_clean_h = self._clean_identifier(f1_name, f1_ns)
             f2_name_clean_h = self._clean_identifier(f2_name, f2_ns)
             scores["name"] = self.get_similarity(
                 f1_name_clean_h, f2_name_clean_h
             )
             
             # NAME VETO: If names are too dissimilar, it's not a match.
             # Only apply if name is weighted (skips CONSTRUCTOR where name weight is 0.0)
             if current_weights["name"] > 0 and scores["name"] < 0.4:
                 logger.debug(
                     f"Name match {scores['name']:.2f} < 0.4. Vetoing match."
                 )
                 return 0.0, scores
             
             if heuristic_applied:
                 # Since we used member_of as name, we should not double count member_of matching
                 # (which will be poor anyway: ClassName vs "").
                 # Shift all member_of weight to name.
                 current_weights["name"] += current_weights["member_of"]
                 current_weights["member_of"] = 0.0
                 logger.debug(
                    "Applied __call__ heuristic. "
                    f"Adjusted weights: {current_weights}"
                )

        logger.debug(
            f"Comparison Details:\n"
            f"  Name: '{feature1.normalized_name}' vs "
            f"'{feature2.normalized_name}' -> {scores['name']:.4f}\n"
            f"  MemberOf: '{feature1.normalized_member_of}' vs "
            f"'{feature2.normalized_member_of}' -> {scores['member_of']:.4f}\n"
            f"  Namespace: '{feature1.normalized_namespace}' vs "
            f"'{feature2.normalized_namespace}' -> {scores['namespace']:.4f}"
        )
        logger.debug(f"Preliminary scores: {scores}")

        # 3. Early Exit Check (using dynamic weights)
        preliminary_score = (
            scores["name"] * current_weights["name"]
            + scores["member_of"] * current_weights["member_of"]
            + scores["namespace"] * current_weights["namespace"]
        )

        early_exit_threshold = 0.6 * (
            current_weights["name"]
            + current_weights["member_of"]
            + current_weights["namespace"]
        )
        logger.debug(
            f"Preliminary score: {preliminary_score:.4f}, "
            f"Early exit threshold: {early_exit_threshold:.4f}"
        )

        if preliminary_score < early_exit_threshold:
            logger.debug(
                f"Early exit triggered ({preliminary_score:.4f} < "
                f"{early_exit_threshold:.4f})"
            )
            return preliminary_score, scores

        scores["parameters"] = self._calculate_parameters_score(
            feature1.parameters, feature2.parameters
        )
        scores["return_type"] = self._calculate_return_type_score(
            feature1, feature2
        )

        final_score = sum(
            scores[key] * current_weights[key] for key in current_weights
        )
        logger.debug(f"Final scores including params & return: {scores}")

        # Log contributions
        logger.debug("Score Contributions:")
        for key in current_weights:
            contribution = scores[key] * current_weights[key]
            logger.debug(
                f"  {key}: {scores[key]:.4f} * {current_weights[key]:.4f} = "
                f"{contribution:.4f}"
            )

        logger.debug(f"Final weighted similarity score: {final_score:.4f}")
        return final_score, scores
