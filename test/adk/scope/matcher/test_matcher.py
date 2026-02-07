import os
import tempfile
import unittest
from unittest.mock import patch
from google.adk.scope import features_pb2
from google.adk.scope.matcher import matcher


class TestMatcher(unittest.TestCase):


    def test_match_features(self):
        f1 = features_pb2.Feature(
            normalized_name="f_same",
            normalized_member_of="c_same",
            normalized_namespace="n_same",
            type=features_pb2.Feature.Type.INSTANCE_METHOD,
        )
        f2 = features_pb2.Feature(
            normalized_name="f_same",
            normalized_member_of="c_same",
            normalized_namespace="n_same",
            type=features_pb2.Feature.Type.INSTANCE_METHOD,
        )
        f3 = features_pb2.Feature(
            normalized_name="totally",
            normalized_member_of="different",
            normalized_namespace="stuff",
            type=features_pb2.Feature.Type.INSTANCE_METHOD,
        )
        f4 = features_pb2.Feature(
            normalized_name="entirely",
            normalized_member_of="unrelated",
            normalized_namespace="things",
            type=features_pb2.Feature.Type.INSTANCE_METHOD,
        )

        base_features = [f1, f3]
        target_features = [f4, f2]

        matches = matcher.match_features(base_features, target_features, 0.8)

        self.assertEqual(len(matches), 1)

        m_f1, m_f2, score = matches[0]
        self.assertEqual(m_f1.normalized_name, "f_same")
        self.assertEqual(m_f2.normalized_name, "f_same")
        self.assertGreater(score, 0.8)

        # Assert lists were mutated and matched elements removed
        self.assertEqual(len(base_features), 1)
        self.assertEqual(base_features[0].normalized_name, "totally")

        self.assertEqual(len(target_features), 1)
        self.assertEqual(target_features[0].normalized_name, "entirely")

    def test_fuzzy_match_namespaces(self):
        features_base = {"module.one": [], "module.two": []}
        features_target = {
            "module.one": [features_pb2.Feature(original_name="f1_target")],
            "module.ones": [features_pb2.Feature(original_name="f4")],
            "module.three": [features_pb2.Feature(original_name="f5")],
        }

        matcher.fuzzy_match_namespaces(features_base, features_target)

        self.assertIn("module.one", features_target)
        self.assertIn("module.two", features_target)
        self.assertNotIn("module.ones", features_target)
        self.assertNotIn("module.three", features_target)
        self.assertEqual(len(features_target["module.one"]), 3)
        self.assertEqual(len(features_target["module.two"]), 0)

    def test_fuzzy_match_namespaces_empty_base(self):
        features_base = {}
        features_target = {
            "module.one": [features_pb2.Feature(original_name="f1")]
        }

        matcher.fuzzy_match_namespaces(features_base, features_target)

        self.assertIn("module.one", features_target)
        self.assertEqual(len(features_target["module.one"]), 1)


if __name__ == "__main__":
    unittest.main()
