import logging
from typing import Optional

import numpy as np
from jellyfish import jaro_winkler_similarity
from scipy.optimize import linear_sum_assignment

from google.adk.scope import features_pb2 as features_pb

logger = logging.getLogger(__name__)

# Default weights for the similarity calculation.
DEFAULT_SIMILARITY_WEIGHTS = {
    "name": 0.30,
    "member_of": 0.30,
    "namespace": 0.15,
    "parameters": 0.15,
    "return_type": 0.10,
}


class SimilarityScorer:
    """Calculates a similarity score between two features."""

    def __init__(
        self, weights: Optional[dict[str, float]] = None, alpha: float = 0.8
    ):
        self.weights = weights or DEFAULT_SIMILARITY_WEIGHTS
        logger.debug(
            f"Initializing SimilarityScorer with alpha={alpha} and "
            f"weights={self.weights}"
        )
        assert "name" in self.weights
        assert "member_of" in self.weights
        assert "namespace" in self.weights
        assert "parameters" in self.weights
        assert "return_type" in self.weights

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
            return 1.0

        # Check the best match between any pair of types
        best_score = 0.0
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
        s_p_name = jaro_winkler_similarity(
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
            current_weights["member_of"] += current_weights["name"]
            current_weights["name"] = 0.0
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
        else:
            logger.debug(f"Incompatible types: {t1} vs {t2}. Returning 0.0")
            return 0.0  # Fast out for incompatible types

        # 2. Similarity Calculations
        scores = {
            "name": jaro_winkler_similarity(
                feature1.normalized_name, feature2.normalized_name
            ),
            "member_of": jaro_winkler_similarity(
                feature1.normalized_member_of, feature2.normalized_member_of
            ),
            "namespace": jaro_winkler_similarity(
                feature1.normalized_namespace, feature2.normalized_namespace
            ),
        }
        logger.debug(f"Preliminary scores: {scores}")

        # 3. Early Exit Check (using dynamic weights)
        preliminary_score = (
            scores["name"] * current_weights["name"]
            + scores["member_of"] * current_weights["member_of"]
            + scores["namespace"] * current_weights["namespace"]
        )

        early_exit_threshold = 0.8 * (
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
            return preliminary_score

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
        logger.debug(f"Final weighted similarity score: {final_score:.4f}")
        return final_score
