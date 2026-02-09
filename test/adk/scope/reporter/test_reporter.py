import os
import tempfile
import unittest
from unittest.mock import patch

from google.protobuf import text_format

from google.adk.scope import features_pb2
from google.adk.scope.matcher import matcher
from google.adk.scope.reporter import reporter


class TestReporter(unittest.TestCase):
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
            registry = reporter._read_feature_registry(temp_path)
            self.assertEqual(registry.language, "PYTHON")
            self.assertEqual(registry.version, "1.0.0")
            self.assertEqual(len(registry.features), 1)
            self.assertEqual(registry.features[0].original_name, "test_feature")
            self.assertEqual(
                registry.features[0].type, features_pb2.Feature.Type.FUNCTION
            )
        finally:
            os.remove(temp_path)

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
        result_sym = reporter.match_registries(
            base_registry, target_registry, 0.9, report_type="symmetric"
        )
        report_sym = result_sym.master_content

        # 1. Verify Master Report Structure
        self.assertIn("# Feature Matching Report: Symmetric", report_sym)
        self.assertIn("**Jaccard Index:** 25.00%", report_sym)
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
        self.assertIn("### ✅ Solid Features", module_content)
        self.assertIn(
            "| Type | Base Feature | Target Feature | Similarity Score |",
            module_content,
        )
        self.assertIn(
            "| method | `BaseClass.fSameBase` | `TargetClass.fSameTarget` |",
            module_content,
        )

        # Potential Matches (formerly Near Misses)
        self.assertIn("### ⚠️ Potential Matches", module_content)
        self.assertIn(
            "| Type | Base Feature | Closest Target Candidate | Similarity |",
            module_content,
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
        result_dir = reporter.match_registries(
            base_registry, target_registry, 0.9, report_type="directional"
        )
        report_dir = result_dir.master_content

        self.assertIn("| **F1 Score** | 40.00% |", report_dir)
        self.assertIn("n_same.md", result_dir.module_files)

        mod_dir_content = result_dir.module_files["n_same.md"]

        # Solid Matches
        self.assertIn("### ✅ Matched Features", mod_dir_content)
        self.assertIn(
            "| Type | Base Feature | Target Feature | Similarity Score |",
            mod_dir_content,
        )
        self.assertIn(
            "| method | `BaseClass.fSameBase` | `TargetClass.fSameTarget` |",
            mod_dir_content,
        )

        # Potential Matches
        self.assertIn("### ⚠️ Potential Matches", mod_dir_content)
        self.assertIn(
            "| Type | Base Feature | Closest Target Candidate | Similarity |",
            mod_dir_content,
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

        result = reporter.match_registries(base, target, 0.9, report_type="raw")
        csv_content = result.master_content

        expected_header = (
            "python_namespace,python_member_of,python_name,ts_namespace,"
            "ts_member_of,ts_name,type,score"
        )
        self.assertIn(expected_header, csv_content)

        # Check for solid match line
        # f1 has: ns=pkg, mem=MyClass, name=f_same
        # Match should have same values for base and target
        expected_line = "pkg,MyClass,f_same,pkg,MyClass,f_same,function,1.0000"
        self.assertIn(expected_line, csv_content)
        self.assertFalse(result.module_files)

    def test_group_features_by_module(self):
        registry = features_pb2.FeatureRegistry()
        f1 = registry.features.add()
        f1.namespace = "module.one"
        f2 = registry.features.add()
        f2.namespace = "module.two"
        f3 = registry.features.add()
        f3.namespace = "module.one"

        result = reporter._group_features_by_module(registry)

        self.assertIn("module.one", result)
        self.assertIn("module.two", result)
        self.assertEqual(len(result["module.one"]), 2)
        self.assertEqual(len(result["module.two"]), 1)

    def test_process_module(self):
        """Tests the end-to-end processing of a single module."""
        f_base = features_pb2.Feature(
            original_name="f1_base",
            normalized_name="f1_base",
            normalized_namespace="n1",
            type=features_pb2.Feature.Type.FUNCTION,
        )
        f_target = features_pb2.Feature(
            original_name="f1_target",
            normalized_name="f1_target",
            normalized_namespace="n1",
            type=features_pb2.Feature.Type.FUNCTION,
        )

        with patch(
            "google.adk.scope.reporter.reporter.matcher.match_features"
        ) as mock_match:
            # Let's assume one solid match and no potential matches
            mock_match.side_effect = [
                [(f_base, f_target, 0.95)],  # Solid matches
                [],  # Potential matches
            ]

            result = matcher.process_module(
                module="n1",
                base_list=[f_base],
                target_list=[f_target],
                alpha=0.9,
                report_type="symmetric",
                base_lang_code="py",
                target_lang_code="ts",
            )

            self.assertEqual(result["solid_matches_count"], 1)
            self.assertEqual(result["score"], 1.0)
            self.assertIn("| py, ts |", result["row_content"])
            self.assertIn("# Module: `n1`", result["module_content"])
            self.assertIn("### ✅ Solid Features", result["module_content"])

    def test_generate_raw_report(self):
        """Tests the raw CSV report generation."""
        f_base = features_pb2.Feature(
            original_name="f1_base",
            normalized_name="f1_base",
            namespace="n1",
            member_of="c1",
            type=features_pb2.Feature.Type.FUNCTION,
        )

        base_registry = features_pb2.FeatureRegistry(
            language="Python", version="1.0.0"
        )
        base_registry.features.extend([f_base])
        target_registry = features_pb2.FeatureRegistry(
            language="TypeScript", version="2.0.0"
        )

        with patch(
            "google.adk.scope.reporter.reporter.matcher.match_features"
        ) as mock_match:
            mock_match.return_value = []  # No matches for simplicity

            result = reporter.ReportGenerator(
                base_registry, target_registry, 0.9
            ).generate_raw_report()

            self.assertIn(
                "python_namespace,python_member_of,python_name",
                result.master_content,
            )
            self.assertIn("n1,c1,f1_base", result.master_content)

    def test_generate_symmetric_report(self):
        """Tests the symmetric report generation."""
        base_registry = features_pb2.FeatureRegistry(
            language="Python", version="1.0.0"
        )
        f1 = base_registry.features.add()
        f1.namespace = "n1"
        target_registry = features_pb2.FeatureRegistry(
            language="TypeScript", version="2.0.0"
        )

        with patch(
            "google.adk.scope.reporter.reporter.matcher.process_module"
        ) as mock_process:
            mock_process.return_value = {
                "solid_matches_count": 1,
                "score": 1.0,
                "row_content": "| py, ts | `n1` | 1 | 100.00% | ✅ | n1.md |",
                "module_filename": "n1.md",
                "module_content": "# Module: `n1`",
            }

            result = reporter.ReportGenerator(
                base_registry, target_registry, 0.9
            ).generate_symmetric_report()

            self.assertIn(
                "# Feature Matching Report: Symmetric", result.master_content
            )
            self.assertIn("**Jaccard Index:**", result.master_content)
            self.assertIn("## Module Summary", result.master_content)
            self.assertIn("| `n1` |", result.master_content)
            self.assertIn("n1.md", result.module_files)

    def test_generate_directional_report(self):
        """Tests the directional report generation."""
        base_registry = features_pb2.FeatureRegistry(
            language="Python", version="1.0.0"
        )
        f1 = base_registry.features.add()
        f1.namespace = "n1"
        target_registry = features_pb2.FeatureRegistry(
            language="TypeScript", version="2.0.0"
        )

        with patch(
            "google.adk.scope.reporter.reporter.matcher.process_module"
        ) as mock_process:
            mock_process.return_value = {
                "solid_matches_count": 1,
                "score": 1.0,
                "row_content": "| `n1` | 1 | 100.00% | ✅ | n1.md |",
                "module_filename": "n1.md",
                "module_content": "# Module: `n1`",
            }

            result = reporter.ReportGenerator(
                base_registry, target_registry, 0.9
            ).generate_directional_report()

            self.assertIn(
                "# Feature Matching Report: Directional", result.master_content
            )
            self.assertIn("| **F1 Score** |", result.master_content)
            self.assertIn("## Module Summary", result.master_content)
            self.assertIn("| `n1` |", result.master_content)
            self.assertIn("n1.md", result.module_files)

    def test_raw_integration(self):
        """Tests the raw report generation end-to-end."""
        python_features_str = """
            language: "PYTHON"
            version: "1.23.0"
            features {
            original_name: "load_artifact"
            normalized_name: "load_artifact"
            description: "description"
            member_of: "InMemoryArtifactService"
            normalized_member_of: "in_memory_artifact_service"            
            type: INSTANCE_METHOD
            file_path: "adk/runners.py"
            namespace: "runners"
            normalized_namespace: "artifacts"
            parameters {
                original_name: "app_name"
                normalized_name: "app_name"
                original_types: "str"
                normalized_types: STRING
                description: "The app name."
            }
            parameters {
                original_name: "session_id"
                normalized_name: "session_id"
                original_types: "Optional[str]"
                normalized_types: STRING
                normalized_types: NULL
                description: "description"
                is_optional: true
            }
            original_return_types: "Optional[types.Part]"
            normalized_return_types: "OBJECT"
            normalized_return_types: "NULL"
            async: true            
        }
        """

        typescript_features_str = """
        language: "TYPESCRIPT"
        version: "0.3.0"
        features {
            original_name: "loadArtifact"
            normalized_name: "load_artifact"
            member_of: "InMemoryArtifactService"
            normalized_member_of: "in_memory_artifact_service"
            type: INSTANCE_METHOD
            file_path: "in_memory_artifact_service.ts"
            namespace: "artifacts"
            normalized_namespace: "artifacts"
            parameters {
                original_name: "request"
                normalized_name: "request"
                original_types: "LoadArtifactRequest"
                normalized_types: OBJECT
            }
            original_return_types: "Promise<Part | undefined>"
            normalized_return_types: "OBJECT"
            normalized_return_types: "NULL"
            async: true
        }
        """

        py_registry = text_format.Parse(
            python_features_str, features_pb2.FeatureRegistry()
        )
        ts_registry = text_format.Parse(
            typescript_features_str, features_pb2.FeatureRegistry()
        )

        result = reporter.ReportGenerator(
            py_registry, ts_registry, 0.8
        ).generate_raw_report()

        self.assertIn(
            "python_namespace,python_member_of,python_name,ts_namespace,ts_member_of,ts_name,type,score",
            result.master_content,
        )

        print(result.master_content)
        self.assertEqual(len(result.master_content.splitlines()), 2)
        # A known match


if __name__ == "__main__":
    unittest.main()
