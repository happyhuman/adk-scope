import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from google.protobuf import text_format

from google.adk.scope import features_pb2
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
            namespace="stuff",
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
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.md"
            result_md = reporter.generate_markdown_raw_reports(
                [base_registry, target_registry],
                report_type="md",
                output_path=output_path,
            )
            report_md = result_md.main_report_content

            # 1. Verify Master Report Structure
            self.assertIn("# Feature Matching Parity Report", report_md)
            self.assertIn("## Summary", report_md)
            # Check for High/Low confidence summaries
            self.assertIn(
                "| **✅ High Confidence Matches** | **1** |", report_md
            )
            self.assertIn("| **⚠️ Low Confidence Matches** | **1** |", report_md)
            self.assertIn("| **❌ Mismatches** | **1** |", report_md)
            self.assertIn("## Module Summary", report_md)

            # Check for module entry in master summary
            self.assertIn(
                "| Module | Features (Python) | Score | Status | Details |",
                report_md,
            )
            self.assertIn("| `google.adk.events` |", report_md)

            self.assertIn(
                "[View Details]({modules_dir}/google.adk.events.md)", report_md
            )

            # 2. Verify Module Content
            self.assertIn("google.adk.events.md", result_md.module_reports)
            module_content = result_md.module_reports["google.adk.events.md"]

            self.assertIn("# Module: `google.adk.events`", module_content)
            # New summary table in module
            self.assertIn("## Summary", module_content)
            self.assertIn("## Feature Details", module_content)

            # Solid Matches (High Confidence)
            self.assertIn("✅", module_content)
            self.assertIn("**High**", module_content)
            self.assertIn("`fSameBase`", module_content)
            self.assertIn("`fSameTarget`", module_content)

            # Potential Matches (Low Confidence)
            self.assertIn("⚠️", module_content)
            self.assertIn("Low", module_content)
            self.assertIn("`base_name`", module_content)
            self.assertIn("`target_name`", module_content)

            # Unmatched / Gaps (in 'stuff' module)
            self.assertIn("stuff.md", result_md.module_reports)
            stuff_content = result_md.module_reports["stuff.md"]
            self.assertIn("❌", stuff_content)
            self.assertIn("`totally_diff`", stuff_content)

    def test_generate_raw_report(self):
        """Tests the raw CSV report generation via RawReportGenerator."""
        f_base = features_pb2.Feature(
            original_name="f1_base",
            normalized_name="f1_base",
            namespace="n1",
            member_of="c1",
            type=features_pb2.Feature.Type.FUNCTION,
        )
        # f_target is a perfect match
        f_target = features_pb2.Feature(
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
        target_registry.features.extend([f_target])

        # Use RawReportGenerator directly
        generator = reporter.raw.RawReportGenerator(
            base_registry, target_registry
        )
        df = generator.generate()

        # Check columns
        self.assertIn("py_namespace", df.columns)
        self.assertIn("score", df.columns)

        # Check content
        row = df.iloc[0]
        self.assertEqual(row["py_name"], "f1_base")
        self.assertEqual(row["score"], 1.0)

    def test_global_best_match(self):
        """Tests that a feature matches best candidate globally, ignoring
        namespace."""
        # Base feature in namespace 'n1'
        f_base = features_pb2.Feature(
            original_name="my_feature",
            normalized_name="my_feature",
            namespace="n1",
            type=features_pb2.Feature.Type.FUNCTION,
        )

        # Target feature 1: Same namespace, but different name (low score)
        f_target_bad = features_pb2.Feature(
            original_name="other_feature",
            normalized_name="other_feature",
            namespace="n1",
            type=features_pb2.Feature.Type.FUNCTION,
        )

        # Target feature 2: Different namespace, but same name (high score)
        f_target_good = features_pb2.Feature(
            original_name="my_feature",
            normalized_name="my_feature",
            namespace="n2",
            type=features_pb2.Feature.Type.FUNCTION,
        )

        base_registry = features_pb2.FeatureRegistry(
            language="Python", version="1"
        )
        base_registry.features.append(f_base)

        target_registry = features_pb2.FeatureRegistry(
            language="Java", version="2"
        )
        target_registry.features.extend([f_target_bad, f_target_good])

        # RawReportGenerator logic
        generator = reporter.raw.RawReportGenerator(
            base_registry, target_registry
        )
        df = generator.generate()

        # Check that we found the match in n2
        row = df.iloc[0]
        self.assertEqual(row["java_namespace"], "n2")
        self.assertEqual(row["score"], 1.0)

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

        generator = reporter.raw.RawReportGenerator(py_registry, ts_registry)
        df = generator.generate()

        # Verify solid match (high score)
        row = df.iloc[0]
        self.assertEqual(row["py_name"], "load_artifact")
        self.assertEqual(row["ts_name"], "loadArtifact")
        self.assertGreater(row["score"], 0.8)

    def test_raw_report_match_confidence(self):
        """Tests match and confidence columns with various scores."""
        # 1. High match (score 0.9 > 0.6 for py/go)
        f_high = features_pb2.Feature(
            original_name="high",
            normalized_name="high",
            type=features_pb2.Feature.Type.FUNCTION,
        )
        # 2. Avg match (score 0.55 between 0.5 and 0.6 for py/go)
        f_avg = features_pb2.Feature(
            original_name="high",
            normalized_name="high_ish",
            type=features_pb2.Feature.Type.FUNCTION,
        )
        # 3. Low match (score 0.1 < 0.5 for py/go)
        f_low = features_pb2.Feature(
            original_name="high",
            normalized_name="completely_different",
            type=features_pb2.Feature.Type.FUNCTION,
        )

        base = features_pb2.FeatureRegistry(language="Python", version="1")
        base.features.append(f_high)

        target = features_pb2.FeatureRegistry(language="Go", version="1")
        # We need to craft targets that produce specific scores or mock the
        # scorer. It's easier to mock SimilarityScorer to return fixed scores.
        target.features.extend([f_high, f_avg, f_low])

        with patch(
            "google.adk.scope.reporter.raw.SimilarityScorer"
        ) as MockScorer:
            instance = MockScorer.return_value

            # Case 1: High match
            instance.get_similarity_score.return_value = 0.9
            gen = reporter.raw.RawReportGenerator(base, target)
            df = gen.generate()
            # Since generate iterates through base features, and we have 1 base
            # feature, it will run once. We need to test behavior for different
            # scores. But generate() does all at once.

            # Actually, `generate` iterates through base features.
            # If we want to test different outcomes, we should perhaps just
            # test the _get_confidence_level method or ensure our mock returns
            # different values for different calls if possible, or just run 3
            # separate gens.

            # Test High
            self.assertEqual(df.iloc[0]["match"], "true")
            self.assertEqual(df.iloc[0]["confidence"], "high")

            # Test Avg (Low Confidence)
            instance.get_similarity_score.return_value = 0.55
            gen = reporter.raw.RawReportGenerator(base, target)
            df = gen.generate()
            self.assertEqual(df.iloc[0]["match"], "true")
            self.assertEqual(df.iloc[0]["confidence"], "low")

            # Test Low (Mismatch)
            instance.get_similarity_score.return_value = 0.4
            gen = reporter.raw.RawReportGenerator(base, target)
            df = gen.generate()
            self.assertEqual(df.iloc[0]["match"], "false")
            self.assertEqual(
                df.iloc[0]["confidence"], "high"
            )  # Mismatches are high confidence if very low score?
            # Wait, raw.py logic:
            # if score > high_thresh: true, high
            # elif score > avg_thresh: true, low
            # else: match=false
            # if match=false, confidence depends on score?
            # Actually raw.py says:
            # if match: ...
            # else: row["match"] = "false"
            # And confidence is set to "high" by default for mismatches in
            # raw.py? Let's check raw.py.
            # "confidence": "high" is default init.
            # If match found, it might be updated to "low".
            # If no match found (score < avg), it remains "high" (High
            # confidence that it is NOT a match).

            self.assertEqual(df.iloc[0]["match"], "false")
            self.assertEqual(df.iloc[0]["confidence"], "high")


if __name__ == "__main__":
    unittest.main()
