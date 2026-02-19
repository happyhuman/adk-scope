import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from google.adk.scope import features_pb2
from google.adk.scope.reporter import matrix, reporter


class TestMatrixReport(unittest.TestCase):
    def setUp(self):
        self.base_registry = features_pb2.FeatureRegistry(
            language="PYTHON", version="1.0.0"
        )
        self.java_registry = features_pb2.FeatureRegistry(
            language="JAVA", version="1.0.0"
        )
        self.go_registry = features_pb2.FeatureRegistry(
            language="GO", version="1.0.0"
        )

        # Add some dummy features
        f1 = self.base_registry.features.add()
        f1.namespace = "google.cloud"
        f1.original_name = "Client"
        f1.type = features_pb2.Feature.Type.CLASS_METHOD

        f2 = self.java_registry.features.add()
        f2.namespace = "com.google.cloud"
        f2.original_name = "Client"
        f2.type = features_pb2.Feature.Type.CLASS_METHOD

    def test_matrix_generation_structure(self):
        # Create dummy dataframes
        df_java = pd.DataFrame(
            {
                "py_namespace": ["google.cloud"],
                "py_member_of": [""],
                "py_name": ["Client"],
                "java_namespace": ["com.google.cloud"],
                "java_member_of": [""],
                "java_name": ["Client"],
                "match": ["true"],
                "confidence": ["high"],
            }
        )

        match_dataframes = {"java": df_java}

        gen = matrix.MatrixReportGenerator(
            match_dataframes, self.base_registry, [self.java_registry]
        )
        report = gen.generate()

        self.assertIn("# Feature Matrix Report", report.content)
        self.assertIn("**Base Language**: Python", report.content)
        self.assertIn(
            "| Module (Python) | Container | Name | Java |", report.content
        )
        self.assertIn(
            "| `google.cloud` | `___` | `Client` | ✅ |", report.content
        )

    def test_csv_integration(self):
        # Create temporary CSV files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv1 = temp_path / "py_java.csv"
            csv2 = temp_path / "py_go.csv"
            # Output to a directory to test automatic filename appending
            # NOTE: generate_matrix_report_from_csvs expects a FILE path.
            # The directory logic is in main(); here we test the generator logic directly.
            output_file = temp_path / "matrix_report.md"

            # Use empty string for namespace to test replacement with ___
            df1 = pd.DataFrame(
                {
                    "py_namespace": [""],
                    "py_member_of": ["c1"],
                    "py_name": ["n1"],
                    "java_namespace": [""],
                    "java_member_of": ["c1"],
                    "java_name": ["n1"],
                    "match": ["true"],
                    "confidence": ["high"],
                }
            )
            df2 = pd.DataFrame(
                {
                    "py_namespace": [""],
                    "py_member_of": ["c1"],
                    "py_name": ["n1"],
                    "go_namespace": [""],
                    "go_member_of": ["c1"],
                    "go_name": ["n1"],
                    "match": ["true"],
                    "confidence": ["low"],
                }
            )

            df1.to_csv(csv1, index=False)
            df2.to_csv(csv2, index=False)

            reporter.generate_matrix_report_from_csvs(
                [str(csv1), str(csv2)], output_file
            )

            self.assertTrue(output_file.exists())
            content = output_file.read_text()
            self.assertIn("___", content)
            self.assertIn("# Feature Matrix Report", content)
            self.assertTrue(
                "| Java | Go |" in content or "| Go | Java |" in content
            )
            if "| Java | Go |" in content:
                # Java is high, Go is low
                self.assertIn("| ✅ | ⚠️ |", content)
            else:
                self.assertIn("| ⚠️ | ✅ |", content)


if __name__ == "__main__":
    unittest.main()
