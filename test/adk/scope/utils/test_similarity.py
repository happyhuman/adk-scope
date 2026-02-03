"""Unit tests for the SimilarityScorer class."""

import unittest

from google.adk.scope import features_pb2 as features_pb
from google.adk.scope.utils.similarity import SimilarityScorer


class TestSimilarityScorer(unittest.TestCase):
    """Test suite for the SimilarityScorer."""

    def setUp(self):
        """Set up a default scorer and a sample feature for the tests."""
        self.scorer = SimilarityScorer()
        feature_args = {
            'normalized_name': 'my_func',
            'normalized_member_of': 'my_class',
            'normalized_namespace': 'my_module',
            'parameters': [
                features_pb.Param(normalized_name='p1', normalized_types=[features_pb.ParamType.STRING]),
                features_pb.Param(normalized_name='p2', normalized_types=[features_pb.ParamType.NUMBER], is_optional=True),
            ],
            'normalized_return_types': ['STRING'],
            'async': True
        }
        self.feature1 = features_pb.Feature(**feature_args)

    def test_initialization(self):
        """Test that the scorer initializes with default and custom weights."""
        self.assertIsNotNone(self.scorer.weights)
        custom_weights = {'name': 1.0, 'member_of': 0.0, 'namespace': 0.0, 'parameters': 0.0, 'return_type': 0.0}
        custom_scorer = SimilarityScorer(weights=custom_weights)
        self.assertEqual(custom_scorer.weights, custom_weights)

    def test_identical_features(self):
        """Test that identical features yield a score of 1.0."""
        score, full_calc = self.scorer.score(self.feature1, self.feature1)
        self.assertTrue(full_calc)
        self.assertAlmostEqual(score, 1.0)

    def test_completely_dissimilar_features(self):
        """Test that dissimilar features trigger an early exit."""
        feature2 = features_pb.Feature(
            normalized_name='completely_different_function',
            normalized_member_of='another_world',
            normalized_namespace='a_galaxy_far_away'
        )
        score, full_calc = self.scorer.score(self.feature1, feature2)
        self.assertFalse(full_calc, f"Early exit failed; score was {score}")

    def test_partial_similarity(self):
        """Test a scenario with partially similar features."""
        feature2_args = {
            'normalized_name': 'my_func',  # Same name
            'normalized_member_of': 'my_class', # Same class
            'normalized_namespace': 'a_different_module', # Diff namespace
            'parameters': [
                features_pb.Param(normalized_name='p1', normalized_types=[features_pb.ParamType.STRING]),
            ],
            'normalized_return_types': ['NUMBER'], # Diff return
            'async': False # Diff async
        }
        feature2 = features_pb.Feature(**feature2_args)
        score, full_calc = self.scorer.score(self.feature1, feature2)
        self.assertTrue(full_calc)
        self.assertTrue(0 < score < 1.0, f"Score {score} was not in the expected range (0, 1)")

    def test_parameter_edge_cases(self):
        """Test scoring with different parameter list configurations."""
        feature_no_params = features_pb.Feature(normalized_name='func', normalized_member_of='class', normalized_namespace='ns')
        feature_one_param = features_pb.Feature(
            normalized_name='func', normalized_member_of='class', normalized_namespace='ns',
            parameters=[features_pb.Param(normalized_name='p1')]
        )
        
        # One empty, one not - should be an imperfect match
        score, _ = self.scorer.score(feature_no_params, feature_one_param)
        self.assertLess(score, 1.0, "Score should be less than 1.0 when one param list is empty")
        
    def test_return_type_edge_cases(self):
        """Test scoring with different return type configurations."""
        feature_no_return = features_pb.Feature(normalized_name='my_func', normalized_member_of='my_class', normalized_namespace='my_module')
        
        score, _ = self.scorer.score(self.feature1, feature_no_return)
        self.assertLess(score, 1.0, "Score should be less than 1.0 when return types differ")


    def test_run_async_integration(self):
        """Test the similarity of the TypeScript and Python 'run_async' features."""
        ts_feature_args = {
            'normalized_name': 'run_async',
            'normalized_member_of': 'runner',
            'normalized_namespace': 'runner',
            'parameters': [
                features_pb.Param(normalized_name='user_id', normalized_types=['STRING']),
                features_pb.Param(normalized_name='session_id', normalized_types=['STRING']),
                features_pb.Param(normalized_name='new_message', normalized_types=['OBJECT']),
                features_pb.Param(normalized_name='state_delta', normalized_types=['OBJECT'], is_optional=True),
                features_pb.Param(normalized_name='run_config', normalized_types=['OBJECT'], is_optional=True),
            ],
            'normalized_return_types': ['OBJECT'],
            'async': True,
        }
        ts_feature = features_pb.Feature(**ts_feature_args)

        py_feature_args = {
            'normalized_name': 'run_async',
            'normalized_member_of': 'runner',
            'normalized_namespace': 'runners',
            'parameters': [
                features_pb.Param(normalized_name='user_id', normalized_types=['STRING']),
                features_pb.Param(normalized_name='session_id', normalized_types=['STRING']),
                features_pb.Param(normalized_name='invocation_id', normalized_types=['STRING'], is_optional=True),
                features_pb.Param(normalized_name='new_message', normalized_types=['OBJECT'], is_optional=True),
                features_pb.Param(normalized_name='state_delta', normalized_types=['MAP'], is_optional=True),
                features_pb.Param(normalized_name='run_config', normalized_types=['OBJECT'], is_optional=True),
            ],
            'normalized_return_types': ['OBJECT', 'null'],
            'async': True,
        }
        py_feature = features_pb.Feature(**py_feature_args)

        score, full_calc = self.scorer.score(ts_feature, py_feature)

        self.assertTrue(full_calc)
        self.assertTrue(0.5 < score < 1.0, f"Score {score} was not in the expected range (0.5, 1.0)")


if __name__ == '__main__':
    unittest.main()
