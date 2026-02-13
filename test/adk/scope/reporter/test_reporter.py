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

        # Test Markdown Report
        result_md = reporter.match_registries(
            [base_registry, target_registry], 0.9, report_type="md"
        )
        report_md = result_md.master_content

        # 1. Verify Master Report Structure
        self.assertIn("# Feature Matching Parity Report", report_md)
        self.assertIn("## Summary", report_md)
        self.assertIn("| **✅ Common Shared** | **1** |", report_md)
        self.assertIn("| **📦 Exclusive to `Python`** | **2** |", report_md)
        self.assertIn("| **📦 Exclusive to `TypeScript`** | **1** |", report_md)
        self.assertIn("| **📊 Jaccard Score** | **25.00%** |", report_md)
        self.assertIn("## Module Summary", report_md)

        # Check for module entry in master summary
        self.assertIn("| ADK | Module | Features (Python) | Score | Status | Details |", report_md)
        self.assertIn("| `n_same` |", report_md)
        self.assertIn("[View Details]({modules_dir}/n_same.md)", report_md)

        # 2. Verify Module Content
        self.assertIn("n_same.md", result_md.module_files)
        module_content = result_md.module_files["n_same.md"]

        self.assertIn("# Module: `n_same`", module_content)
        self.assertIn("**Features:** 3", module_content)

        # Solid Matches
        self.assertIn("### ✅ Solid Features", module_content)
        self.assertIn(
            "| Type | Python Feature | TypeScript Feature | Similarity Score |",
            module_content,
        )
        self.assertIn(
            "| method | `BaseClass.fSameBase` | `TargetClass.fSameTarget` |",
            module_content,
        )

        # Potential Matches (formerly Near Misses)
        self.assertIn("### ⚠️ Potential Matches", module_content)
        self.assertIn(
            "| Type | Python Feature | Closest TypeScript Candidate | Similarity |",
            module_content,
        )
        self.assertIn(
            "| method | `base_member.base_name` | "
            "`target_member.target_name` |",
            module_content,
        )

        # Unmatched / Gaps (in 'stuff' module)
        self.assertIn("stuff.md", result_md.module_files)
        stuff_content = result_md.module_files["stuff.md"]
        self.assertIn("### ❌ Unmatched Features", stuff_content)
        self.assertIn("| `totally_diff` | TypeScript |", stuff_content)
        self.assertIn("**Features:** 1", stuff_content)

    def test_matrix_report(self):
        f_py = features_pb2.Feature(
            original_name="f",
            normalized_name="f",
            member_of="c",
            normalized_member_of="c",
            normalized_namespace="n",
            type=features_pb2.Feature.Type.FUNCTION,
        )
        f_ts = features_pb2.Feature(
            original_name="f",
            normalized_name="f",
            member_of="c",
            normalized_member_of="c",
            normalized_namespace="n",
            type=features_pb2.Feature.Type.FUNCTION,
        )
        # Go only matches partially (different name) or provides a new feature
        f_go1 = features_pb2.Feature(
            original_name="new_f",
            normalized_name="new_f",
            member_of="c",
            normalized_member_of="c",
            normalized_namespace="n",
            type=features_pb2.Feature.Type.FUNCTION,
        )

        r_py = features_pb2.FeatureRegistry(language="Python", version="1")
        r_py.features.append(f_py)
        
        r_ts = features_pb2.FeatureRegistry(language="TypeScript", version="2")
        r_ts.features.append(f_ts)
        
        r_go = features_pb2.FeatureRegistry(language="Go", version="3")
        r_go.features.append(f_go1)

        result_matrix = reporter.match_registries(
            [r_py, r_ts, r_go], 0.9, report_type="matrix"
        )
        
        report_md = result_matrix.master_content

        # 1. Check title & headers
        self.assertIn("# Multi-SDK Feature Matrix Report", report_md)
        self.assertIn("| **Anchor** | Python | 1 |", report_md)
        self.assertIn("| **Comparison 1** | TypeScript | 2 |", report_md)
        self.assertIn("| **Comparison 2** | Go | 3 |", report_md)

        # 2. Check Jaccard Matrix
        self.assertIn("## Global Parity Matrix", report_md)
        self.assertIn("| Language | Python | TypeScript | Go |", report_md)
        # Py vs TS should be 100% since they both only have 'f'
        self.assertIn("| **Python** | - | 100.00% | 0.00% |", report_md)
        # Py/TS vs Go should be 0% since Go has 'new_f' entirely disjoint
        self.assertIn("| **Go** | 0.00% | 0.00% | - |", report_md)

        # 3. Check Global Feature Matrix
        self.assertIn("## Global Feature Support", report_md)
        self.assertIn("### Module: `n`", report_md)
        self.assertIn("| Feature | Type | Python | TypeScript | Go |", report_md)
        
        # 'f' should be yes for Py/Ts, no for Go
        self.assertIn("| `c.f` | function | ✅ | ✅ | ❌ |", report_md)
        
        # 'new_f' should be no for Py/Ts, yes for Go
        self.assertIn("| `c.new_f` | function | ❌ | ❌ | ✅ |", report_md)

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

        result = reporter.match_registries([base, target], 0.9, report_type="raw")
        csv_content = result.master_content

        expected_header = (
            "py_namespace,py_member_of,py_name,ts_namespace,"
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
                base_lang_name="Python",
                target_lang_name="TypeScript",
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
                "py_namespace,py_member_of,py_name",
                result.master_content,
            )
            self.assertIn("n1,c1,f1_base", result.master_content)

    def test_generate_md_report(self):
        """Tests the md report generation."""
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
            ).generate_md_report()

            self.assertIn(
                "# Feature Matching Parity Report", result.master_content
            )
            self.assertIn("## Summary", result.master_content)
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
            "py_namespace,py_member_of,py_name,ts_namespace,ts_member_of,ts_name,type,score",
            result.master_content,
        )

        print(result.master_content)
        self.assertEqual(len(result.master_content.splitlines()), 2)
        # A known match


if __name__ == "__main__":
    unittest.main()
