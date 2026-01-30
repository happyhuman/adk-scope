import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
from google.adk.scope.extractors.extractor_py import (
    find_files,
    extract_features,
)
from google.adk.scope.features_pb2 import Feature

# Mock tree_sitter modules BEFORE importing extractor
mock_ts = MagicMock()
mock_ts_py = MagicMock()
sys.modules["tree_sitter"] = mock_ts
sys.modules["tree_sitter_python"] = mock_ts_py


class TestExtractor(unittest.TestCase):
    def test_find_files(self):
        # Mock Path.rglob
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.rglob") as mock_rglob,
        ):
            p1 = Path("src/a.py")
            p2 = Path("src/__init__.py")  # Should be excluded
            p3 = Path("src/.hidden.py")  # Should be excluded
            p4 = Path("src/b.py")
            p5 = Path("src/subdir/c.py")
            p6 = Path("src/.venv/lib.py")  # Should be excluded because of .venv

            mock_rglob.return_value = [p1, p2, p3, p4, p5, p6]

            results = list(find_files(Path("src")))

            self.assertIn(p1, results)
            self.assertIn(p4, results)
            self.assertIn(p5, results)
            self.assertNotIn(p2, results)  # __init__ excluded
            self.assertNotIn(p3, results)  # hidden file excluded
            self.assertNotIn(p6, results)  # hidden dir excluded

    @patch("google.adk.scope.extractors.extractor_py.QueryCursor")
    @patch("google.adk.scope.extractors.extractor_py.Query")
    @patch("google.adk.scope.extractors.extractor_py.PARSER")
    def test_extract_features(
        self, mock_parser, mock_query_cls, mock_cursor_cls
    ):
        # Mock file read
        mock_path = MagicMock(spec=Path)
        mock_path.read_bytes.return_value = b"def foo(): pass"

        # Mock tree
        mock_tree = MagicMock()
        mock_parser.parse.return_value = mock_tree
        mock_tree.root_node = MagicMock()

        # Mock query and cursor
        mock_cursor_instance = mock_cursor_cls.return_value

        # Mock captures
        # capture returns dict of {capture_name: [nodes]}
        mock_node = MagicMock()
        mock_node.type = "function_definition"
        mock_cursor_instance.captures.return_value = {"func": [mock_node]}

        # We need to mock NodeProcessor.process to avoid complex node
        # mocking if we just want to test flow
        with patch(
            "google.adk.scope.extractors.extractor_py.NodeProcessor"
        ) as MockProcessor:
            processor_instance = MockProcessor.return_value
            expected_feature = Feature(
                original_name="foo", normalized_name="foo"
            )
            processor_instance.process.return_value = expected_feature

            features = extract_features(mock_path, Path("/repo"))

            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], expected_feature)

            # Verify process was called
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
