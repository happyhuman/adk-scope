
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil

# Import the module under test
# We need to make sure the src path is in PYTHONPATH which is handled by test runner usually
from google.adk.scope.extractors.typescript import extractor
from google.adk.scope import features_pb2

class TestExtractor(unittest.TestCase):
    def setUp(self):
        # Create a temp directory for file system tests
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_find_ts_files(self):
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
        
        files = list(extractor.find_ts_files(src_dir, recursive=True))
        filenames = sorted([f.name for f in files])
        self.assertEqual(filenames, ["a.ts", "b.ts", "c.ts"])
        
        # Non-recursive
        files_nr = list(extractor.find_ts_files(src_dir, recursive=False))
        filenames_nr = sorted([f.name for f in files_nr])
        self.assertEqual(filenames_nr, ["a.ts", "b.ts"])

    @patch('google.adk.scope.extractors.typescript.extractor.PARSER')
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
        with patch('google.adk.scope.extractors.typescript.extractor.Query') as MockQuery, \
             patch('google.adk.scope.extractors.typescript.extractor.QueryCursor') as MockCursor:
            
            mock_cursor_instance = MockCursor.return_value
            # Captures return a dict { capture_name: [nodes] }
            mock_node = Mock()
            mock_node.id = 1
            mock_cursor_instance.captures.return_value = {'func': [mock_node]}
            
            # Mock NodeProcessor
            with patch('google.adk.scope.extractors.typescript.extractor.NodeProcessor') as MockProcessor:
                mock_proc_instance = MockProcessor.return_value
                mock_proc_instance.process.return_value = features_pb2.Feature(original_name="foo")
                
                features = extractor.extract_features(p, self.test_dir)
                
                self.assertEqual(len(features), 1)
                self.assertEqual(features[0].original_name, "foo")
                mock_parser.parse.assert_called_once()
                mock_proc_instance.process.assert_called_with(mock_node, p, self.test_dir)

    @patch('google.adk.scope.extractors.typescript.extractor.parse_args')
    @patch('google.adk.scope.extractors.typescript.extractor.extract_features')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_main_input_file(self, mock_file, mock_extract, mock_args):
        # Setup args
        args = Mock()
        args.input_file = self.test_dir / "test.ts"
        args.input_dir = None
        args.input_repo = None
        args.output = str(self.test_dir / "output.json")
        mock_args.return_value = args
        
        # Setup file existence
        (self.test_dir / "test.ts").touch()
        
        # Mock features
        mock_extract.return_value = [features_pb2.Feature(original_name="foo")]
        
        extractor.main()
        
        # Verify extract called
        mock_extract.assert_called()
        # Verify output written
        mock_file.assert_called_with(str(self.test_dir / "output.json"), "w")
        handle = mock_file()
        handle.write.assert_called()

    @patch('google.adk.scope.extractors.typescript.extractor.parse_args')
    @patch('google.adk.scope.extractors.typescript.extractor.extract_features')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_main_input_repo(self, mock_file, mock_extract, mock_args):
        # Setup args
        args = Mock()
        args.input_file = None
        args.input_dir = None
        args.input_repo = self.test_dir / "repo"
        args.output = str(self.test_dir / "output.json")
        mock_args.return_value = args
        
        # Setup repo structure
        repo = self.test_dir / "repo"
        repo.mkdir()
        (repo / "core" / "src").mkdir(parents=True)
        (repo / "core" / "src" / "a.ts").touch()
        
        # Mock features
        mock_extract.return_value = []
        
        extractor.main()
        
        # Should search in repo/core/src
        # extract_features called for a.ts
        mock_extract.assert_called()
        args_call = mock_extract.call_args[0]
        self.assertEqual(args_call[0].name, "a.ts")

if __name__ == '__main__':
    unittest.main()
