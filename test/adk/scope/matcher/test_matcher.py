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
        
        # f_near_base & f_near_target are a near miss
        # (different names, same structural namespace/class)
        # Using different return types and different enough names to
        # drop the score below 0.8
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
        result_sym = matcher.match_registries(
            base_registry, target_registry, 0.9, report_type="symmetric"
        )
        report_sym = result_sym.master_content
        
        # 1. Verify Master Report Structure
        self.assertIn("# Feature Matching Report: Symmetric", report_sym)
        self.assertIn("**Global Jaccard Index:** 25.00%", report_sym)
        self.assertIn("## Module Summary", report_sym)
        
        # Check for module entry in master summary
        self.assertIn("| `n_same` |", report_sym)
        self.assertIn("[View Details]({modules_dir}/n_same.md)", report_sym)

        # 2. Verify Module Content
        self.assertIn("n_same.md", result_sym.module_files)
        module_content = result_sym.module_files["n_same.md"]
        
        self.assertIn("# Module: `n_same`", module_content)
        self.assertIn("**Features:** 3", module_content)
        
        # Solid Matches
        self.assertIn("### ✅ Solid Matches", module_content)
        self.assertIn(
            "| Type | Base Feature | Target Feature | Similarity Score |",
            module_content
        )
        self.assertIn(
            "| method | `BaseClass.fSameBase` | `TargetClass.fSameTarget` |",
            module_content
        )
        
        # Potential Matches (formerly Near Misses)
        self.assertIn("### ⚠️ Potential Matches", module_content)
        self.assertIn(
            "| Type | Base Feature | Closest Target Candidate | Similarity |",
            module_content
        )
        self.assertIn(
            "| method | `base_member.base_name` | "
            "`target_member.target_name` |",
            module_content,
        )

        # Unmatched / Gaps (in 'stuff' module)
        self.assertIn("stuff.md", result_sym.module_files)
        stuff_content = result_sym.module_files["stuff.md"]
        self.assertIn("### ❌ Unmatched Features", stuff_content)
        self.assertIn("| `totally_diff` | Target |", stuff_content)
        self.assertIn("**Features:** 1", stuff_content)

        # Test Directional Report
        result_dir = matcher.match_registries(
            base_registry, target_registry, 0.9, report_type="directional"
        )
        report_dir = result_dir.master_content
        
        self.assertIn("| **Global F1 Score** | 40.00% |", report_dir)
        self.assertIn("n_same.md", result_dir.module_files)
        
        mod_dir_content = result_dir.module_files["n_same.md"]

        # Solid Matches
        self.assertIn("### ✅ Matched Features", mod_dir_content)
        self.assertIn(
            "| Type | Base Feature | Target Feature | Similarity Score |",
            mod_dir_content
        )
        self.assertIn(
            "| method | `BaseClass.fSameBase` | `TargetClass.fSameTarget` |",
            mod_dir_content
        )

        # Potential Matches
        self.assertIn("### ⚠️ Potential Matches", mod_dir_content)
        self.assertIn(
            "| Type | Base Feature | Closest Target Candidate | Similarity |",
            mod_dir_content
        )
        self.assertIn(
            "| method | `base_member.base_name` | "
            "`target_member.target_name` |",
            mod_dir_content,
        )

        
        # Unmatched / Gaps (in 'stuff' module)
        self.assertIn("stuff.md", result_dir.module_files)
        stuff_dir_content = result_dir.module_files["stuff.md"]
        self.assertIn("### ❌ Missing in Target", stuff_dir_content)
        self.assertIn("| `totally_diff` |", stuff_dir_content)

    def test_match_registries_raw(self):
        f1 = features_pb2.Feature(
            original_name="f_same",
            normalized_name="f_same",
            normalized_namespace="pkg",
            member_of="MyClass",
            normalized_member_of="myclass",
            type=features_pb2.Feature.Type.FUNCTION,
        )
        base = features_pb2.FeatureRegistry(language="Python", version="1")
        base.features.append(f1)
        target = features_pb2.FeatureRegistry(language="TS", version="2")
        target.features.append(f1)
        
        result = matcher.match_registries(base, target, 0.9, report_type="raw")
        csv_content = result.master_content
        
        expected_header = "base_namespace,base_member_of,base_name,target_namespace,target_member_of,target_name,type,score"
        self.assertIn(expected_header, csv_content)
        
        # Check for solid match line
        # f1 has: ns=pkg, mem=MyClass, name=f_same
        # Match should have same values for base and target
        expected_line = "pkg,MyClass,f_same,pkg,MyClass,f_same,function,1.0000"
        self.assertIn(expected_line, csv_content)
        self.assertFalse(result.module_files)

if __name__ == "__main__":
    unittest.main()
