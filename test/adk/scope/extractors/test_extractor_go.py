import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock tree_sitter modules BEFORE importing extractor
mock_ts = MagicMock()
mock_ts_go = MagicMock()
sys.modules["tree_sitter"] = mock_ts
sys.modules["tree_sitter_go"] = mock_ts_go

from google.adk.scope.extractors.extractor_go import (  # noqa: E402
    extract_features,
    find_files,
    get_version,
)
from google.adk.scope.features_pb2 import Feature  # noqa: E402


class TestExtractor(unittest.TestCase):
    @patch("google.adk.scope.extractors.extractor_go.QueryCursor")
    @patch("google.adk.scope.extractors.extractor_go.Query")
    @patch("google.adk.scope.extractors.extractor_go.PARSER")
    def test_extract_features(
        self, mock_parser, mock_query_cls, mock_cursor_cls
    ):
        mock_path = MagicMock(spec=Path)
        mock_path.name = "agent.go"
        mock_path.read_bytes.return_value = b"func MyFunc() {}"

        mock_tree = MagicMock()
        mock_parser.parse.return_value = mock_tree
        mock_tree.root_node = MagicMock()

        mock_cursor_instance = mock_cursor_cls.return_value

        mock_func_node = MagicMock()
        mock_func_body = MagicMock()
        mock_func_stmt_list = MagicMock()
        mock_func_stmt_list.type = "statement_list"
        mock_func_stmt_list.named_child_count = 2
        mock_func_body.children = [mock_func_stmt_list]
        mock_func_node.child_by_field_name.return_value = mock_func_body

        mock_method_node = MagicMock()
        mock_method_body = MagicMock()
        mock_method_stmt_list = MagicMock()
        mock_method_stmt_list.type = "statement_list"
        mock_method_stmt_list.named_child_count = 2
        mock_method_body.children = [mock_method_stmt_list]
        mock_method_node.child_by_field_name.return_value = mock_method_body

        mock_cursor_instance.captures.return_value = {
            "func": [mock_func_node],
            "method": [mock_method_node],
        }

        with patch(
            "google.adk.scope.extractors.extractor_go.NodeProcessor"
        ) as MockProcessor:
            processor_instance = MockProcessor.return_value
            expected_feature = Feature(
                original_name="MyFunc", normalized_name="my_func"
            )
            processor_instance.process.return_value = expected_feature

            features = extract_features(mock_path, Path("/repo"), ".")

            self.assertEqual(len(features), 2)
            self.assertEqual(features[0], expected_feature)
            self.assertEqual(features[1], expected_feature)

    def test_get_version(self):
        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "pathlib.Path.read_text",
                return_value="module github.com/my/module",
            ):
                version = get_version(Path("/repo"))
                self.assertEqual(version, "github.com/my/module")

    def test_find_files(self):
        # Mock Path.rglob
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.rglob") as mock_rglob,
        ):
            p1 = Path("src/a.go")
            p2 = Path("src/.hidden.go")  # Should be excluded
            p3 = Path("src/b.go")
            p4 = Path("src/subdir/c.go")
            p5 = Path("src/.venv/lib.go")  # Should be excluded

            mock_rglob.return_value = [p1, p2, p3, p4, p5]

            results = list(find_files(Path("src")))

            self.assertIn(p1, results)
            self.assertIn(p3, results)
            self.assertIn(p4, results)
            self.assertNotIn(p2, results)  # hidden file excluded
            self.assertNotIn(p5, results)  # hidden dir excluded


if __name__ == "__main__":
    unittest.main()
