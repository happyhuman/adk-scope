import os
import tempfile
import unittest
from google.adk.scope import features_pb2
from google.adk.scope.matcher import matcher


class TestMatcher(unittest.TestCase):
    def test_read_feature_registry(self):
        content = """
    language: "PYTHON"
    version: "1.0.0"
    features {
      original_name: "test_feature"
      normalized_name: "test_feature"
      type: FUNCTION
    }
    """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txtpb", delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            registry = matcher.read_feature_registry(temp_path)
            self.assertEqual(registry.language, "PYTHON")
            self.assertEqual(registry.version, "1.0.0")
            self.assertEqual(len(registry.features), 1)
            self.assertEqual(registry.features[0].original_name, "test_feature")
            self.assertEqual(
                registry.features[0].type, features_pb2.Feature.Type.FUNCTION
            )
        finally:
            os.remove(temp_path)

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

    def test_match_registries(self):
        # f1 & f2 are a solid match (score ~ 1.0)
        f1 = features_pb2.Feature(
            original_name="fSameBase",
            normalized_name="f_same",
            member_of="BaseClass",
            namespace="google.adk.events",
            normalized_member_of="c_same",
            normalized_namespace="n_same",
            type=features_pb2.Feature.Type.INSTANCE_METHOD,
        )
        f2 = features_pb2.Feature(
            original_name="fSameTarget",
            normalized_name="f_same",
            member_of="TargetClass",
            namespace="adk.events",
            normalized_member_of="c_same",
            normalized_namespace="n_same",
            type=features_pb2.Feature.Type.INSTANCE_METHOD,
        )
        
        # f_near_base & f_near_target are a near miss (different names, same structural namespace/class)
        # Using different return types and different enough names to drop the score below 0.8
        f_near_base = features_pb2.Feature(
            original_name="base_name",
            normalized_name="base_name",
            member_of="base_member",
            namespace="google.adk.events",
            normalized_member_of="base_member",
            normalized_namespace="n_same",
            original_return_types=["string"],
            type=features_pb2.Feature.Type.INSTANCE_METHOD,
        )
        f_near_target = features_pb2.Feature(
            original_name="target_name",
            normalized_name="targ_name",
            member_of="target_member",
            namespace="adk.events",
            normalized_member_of="target_member",
            normalized_namespace="n_same",
            original_return_types=["int"],
            type=features_pb2.Feature.Type.INSTANCE_METHOD,
        )

        # f3 is a complete gap (base-exclusive)
        f3 = features_pb2.Feature(
            original_name="totally_diff",
            normalized_name="totally",
            member_of="null",
            namespace="google.adk.events",
            normalized_member_of="different",
            normalized_namespace="stuff",
            type=features_pb2.Feature.Type.INSTANCE_METHOD,
        )

        base_registry = features_pb2.FeatureRegistry(
            language="Python", version="1.0.0"
        )
        base_registry.features.extend([f1, f_near_base, f3])

        target_registry = features_pb2.FeatureRegistry(
            language="TypeScript", version="2.0.0"
        )
        target_registry.features.extend([f2, f_near_target])

        # Test Symmetric Report
        report_sym = matcher.match_registries(base_registry, target_registry, 0.9, report_type="symmetric")
        self.assertIn("# Cross-Language Feature Parity Report", report_sym)
        self.assertIn("**Base:** Python (1.0.0)", report_sym)
        self.assertIn("**Target:** TypeScript (2.0.0)", report_sym)
        self.assertIn("**Feature Parity Score (Jaccard Index):** 25.0%", report_sym)
        
        self.assertIn("## Module 'google.adk.events'", report_sym)
        
        # Solid Matches
        self.assertIn("### ✅ Solid Matches", report_sym)
        self.assertIn("| Type | Base Feature | Target Feature | Similarity Score |", report_sym)
        self.assertIn("| Method | `BaseClass.fSameBase` | `TargetClass.fSameTarget` |", report_sym)
        
        # Near Misses
        self.assertIn("### ⚠️ Near Misses", report_sym)
        self.assertIn("| Type | Base Feature | Closest Target Candidate | Similarity |", report_sym)
        self.assertIn("| Method | `base_member.base_name` | `target_member.target_name` |", report_sym)

        # Unmatched / Gaps
        self.assertIn("### ❌ Unmatched Features", report_sym)
        self.assertIn("| `totally_diff` | Target |", report_sym)

        # Test Directional Report
        report_dir = matcher.match_registries(base_registry, target_registry, 0.9, report_type="directional")
        self.assertIn("**Feature Parity Score (F1 Score):** 40.0%", report_dir)

        self.assertIn("## Module 'google.adk.events'", report_dir)

        # Solid Matches
        self.assertIn("### ✅ Matched Features", report_dir)
        self.assertIn("| Type | Base Feature | Target Feature | Similarity Score |", report_dir)
        self.assertIn("| Method | `BaseClass.fSameBase` | `TargetClass.fSameTarget` |", report_dir)

        # Near Misses
        self.assertIn("### ⚠️ Inconsistencies (Near Misses)", report_dir)
        self.assertIn("| Type | Base Feature | Closest Target Candidate | Similarity |", report_dir)
        self.assertIn("| Method | `base_member.base_name` | `target_member.target_name` |", report_dir)
        
        # Unmatched / Gaps
        self.assertIn("### ❌ Missing in Target (Base Exclusive)", report_dir)
        self.assertIn("| `totally_diff` |", report_dir)

if __name__ == "__main__":
    unittest.main()
