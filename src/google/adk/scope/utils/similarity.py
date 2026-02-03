"""Computes a similarity score between two ADK Features."""

"""Computes a similarity score between two ADK Features."""

from typing import Optional, Tuple

from jellyfish import jaro_winkler_similarity
import numpy as np
from scipy.optimize import linear_sum_assignment

from google.adk.scope import features_pb2 as features_pb

# Default weights for the similarity calculation.
DEFAULT_SIMILARITY_WEIGHTS = {
    'name': 0.30,
    'member_of': 0.25,
    'namespace': 0.15,
    'parameters': 0.20,
    'return_type': 0.10,
}

# If the preliminary score is below this, we skip expensive calculations.
EARLY_EXIT_THRESHOLD = 0.4

class SimilarityScorer:
    """Calculates a similarity score between two features."""

    def __init__(self, weights: Optional[dict[str, float]] = None):
        self.weights = weights or DEFAULT_SIMILARITY_WEIGHTS

    def _calculate_param_similarity(
        self, param1: features_pb.Param, param2: features_pb.Param
    ) -> float:
        """Calculates the similarity score between two individual parameters."""
        s_p_name = jaro_winkler_similarity(
            param1.normalized_name, param2.normalized_name
        )
        s_p_type = (
            1.0
            if param1.normalized_types == param2.normalized_types
            else 0.0
        )
        s_p_opt = 1.0 if param1.is_optional == param2.is_optional else 0.0

        # Weights for parameter components
        return (0.5 * s_p_name) + (0.4 * s_p_type) + (0.1 * s_p_opt)

    def _calculate_parameters_score(
        self, params1: list[features_pb.Param], params2: list[features_pb.Param]
    ) -> float:
        """Calculates the aggregated similarity score for two lists of parameters."""
        if not params1 and not params2:
            return 1.0
        if not params1 or not params2:
            return 0.0

        similarity_matrix = np.zeros((len(params1), len(params2)))
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                similarity_matrix[i, j] = self._calculate_param_similarity(p1, p2)

        row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)
        total_match_score = similarity_matrix[row_ind, col_ind].sum()
        total_params = len(params1) + len(params2)

        if total_params == 0:
            return 1.0

        return (2 * total_match_score) / total_params

    def _calculate_return_type_score(
        self, f1: features_pb.Feature, f2: features_pb.Feature
    ) -> float:
        """Calculates the similarity score for the return types."""
        s_type_match = (
            1.0
            if f1.normalized_return_types == f2.normalized_return_types
            else 0.0
        )
        s_async_match = 1.0 if getattr(f1, 'async') == getattr(f2, 'async') else 0.0
        return (0.7 * s_type_match) + (0.3 * s_async_match)

    def score(
        self, feature1: features_pb.Feature, feature2: features_pb.Feature
    ) -> Tuple[bool, float]:
        """Computes the overall similarity score between two features."""
        scores = {
            'name': jaro_winkler_similarity(
                feature1.normalized_name, feature2.normalized_name
            ),
            'member_of': jaro_winkler_similarity(
                feature1.normalized_member_of, feature2.normalized_member_of
            ),
            'namespace': jaro_winkler_similarity(
                feature1.normalized_namespace, feature2.normalized_namespace
            ),
        }

        preliminary_score = (
            scores['name'] * self.weights['name'] +
            scores['member_of'] * self.weights['member_of'] +
            scores['namespace'] * self.weights['namespace']
        )

        if preliminary_score < EARLY_EXIT_THRESHOLD:
            return preliminary_score, False

        scores['parameters'] = self._calculate_parameters_score(
            feature1.parameters, feature2.parameters
        )
        scores['return_type'] = self._calculate_return_type_score(feature1, feature2)

        final_score = sum(
            scores[key] * self.weights[key] for key in self.weights
        )
        return final_score, True
