import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from google.adk.scope.features_pb2 import Feature
from google.adk.scope.extractors.extractor_java import (
    extract_features,
    find_files,
)


class TestExtractor(unittest.TestCase):
    def test_find_files(self):
        # Mock Path.rglob
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.rglob") as mock_rglob,
        ):
            p1 = Path("src/main/java/A.java")
            p2 = Path("src/test/java/TestA.java")  # Should be excluded
            p3 = Path("build/classes/B.java")  # Should be excluded
            p4 = Path("package-info.java")  # Should be excluded
            p5 = Path("src/main/java/subdir/C.java")
            p6 = Path("node_modules/lib.java")  # Should be excluded

            mock_rglob.return_value = [p1, p2, p3, p4, p5, p6]

            results = list(find_files(Path("src")))

            self.assertIn(p1, results)
            self.assertNotIn(p2, results)  # test excluded
            self.assertNotIn(p3, results)  # build excluded
            self.assertNotIn(p4, results)  # package-info excluded
            self.assertIn(p5, results)
            self.assertNotIn(p6, results)  # node_modules excluded

    @patch("google.adk.scope.extractors.extractor_java.QueryCursor")
    @patch("google.adk.scope.extractors.extractor_java.Query")
    @patch("google.adk.scope.extractors.extractor_java.PARSER")
    def test_extract_features(
        self, mock_parser, mock_query_cls, mock_cursor_cls
    ):
        mock_path = MagicMock(spec=Path)
        mock_path.read_bytes.return_value = b"class A { void foo() {} }"

        mock_tree = MagicMock()
        mock_parser.parse.return_value = mock_tree
        mock_tree.root_node = MagicMock()

        mock_cursor_instance = mock_cursor_cls.return_value

        mock_node = MagicMock()
        mock_node.id = 123
        mock_cursor_instance.captures.return_value = {"method": [mock_node]}

        with patch(
            "google.adk.scope.extractors.extractor_java.NodeProcessor"
        ) as MockProcessor:
            processor_instance = MockProcessor.return_value
            expected_feature = Feature(
                original_name="foo", normalized_name="foo"
            )
            processor_instance.process.return_value = expected_feature

            features = extract_features(mock_path, Path("/repo"))

            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], expected_feature)

            processor_instance.process.assert_called_once()

    def test_find_files_not_exists(self):
        with patch("pathlib.Path.exists", return_value=False):
            results = list(find_files(Path("bad_path")))
            self.assertEqual(results, [])

    def test_extract_features_read_error(self):
        mock_path = MagicMock(spec=Path)
        mock_path.read_bytes.side_effect = IOError("Read error")
        features = extract_features(mock_path, Path("/repo"))
        self.assertEqual(features, [])


if __name__ == "__main__":
    unittest.main()
