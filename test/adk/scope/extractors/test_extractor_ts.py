import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from google.adk.scope import features_pb2

# Import the module under test
# We need to make sure the src path is in PYTHONPATH which is handled
# by test runner usually
from google.adk.scope.extractors import extractor_ts as extractor


class TestExtractor(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for file system tests
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_find_files(self):
        # Structure:
        # /src/a.ts
        # /src/b.ts
        # /src/sub/c.ts
        # /src/d.d.ts (should be ignored)
        # /src/node_modules/e.ts (should be ignored)

        src_dir = self.test_dir / "src"
        src_dir.mkdir()
        (src_dir / "sub").mkdir()
        (src_dir / "node_modules").mkdir()

        (src_dir / "a.ts").touch()
        (src_dir / "b.ts").touch()
        (src_dir / "sub" / "c.ts").touch()
        (src_dir / "d.d.ts").touch()
        (src_dir / "node_modules" / "e.ts").touch()

        files = list(extractor.find_files(src_dir, recursive=True))
        filenames = sorted([f.name for f in files])
        self.assertEqual(filenames, ["a.ts", "b.ts", "c.ts"])

        # Non-recursive
        files_nr = list(extractor.find_files(src_dir, recursive=False))
        filenames_nr = sorted([f.name for f in files_nr])
        self.assertEqual(filenames_nr, ["a.ts", "b.ts"])

    @patch("google.adk.scope.extractors.extractor_ts.PARSER")
    def test_extract_features(self, mock_parser):
        # Mock file read
        p = self.test_dir / "test.ts"
        p.write_text("function foo() {}", encoding="utf-8")

        # Mock parser return
        mock_tree = Mock()
        mock_root = Mock()
        mock_tree.root_node = mock_root
        mock_parser.parse.return_value = mock_tree

        # Mock Query and QueryCursor
        with (
            patch("google.adk.scope.extractors.extractor_ts._build_global_type_map"),
            patch("google.adk.scope.extractors.extractor_ts.Query"),
            patch(
                "google.adk.scope.extractors.extractor_ts.QueryCursor"
            ) as MockCursor,
        ):
            mock_cursor_instance = MockCursor.return_value
            # Captures return a dict { capture_name: [nodes] }
            mock_node = Mock()
            mock_node.id = 1
            mock_cursor_instance.captures.return_value = {"func": [mock_node]}

            # Mock NodeProcessor
            with patch(
                "google.adk.scope.extractors.extractor_ts.NodeProcessor"
            ) as MockProcessor:
                mock_proc_instance = MockProcessor.return_value
                mock_proc_instance.process.return_value = features_pb2.Feature(
                    original_name="foo"
                )

                features = extractor.extract_features(p, self.test_dir, ".")

                self.assertEqual(len(features), 1)
                self.assertEqual(features[0].original_name, "foo")
                mock_parser.parse.assert_called_once()
                mock_proc_instance.process.assert_called_with(
                    mock_node, p, self.test_dir
                )


if __name__ == "__main__":
    unittest.main()
