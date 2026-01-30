
import unittest
from unittest.mock import patch, MagicMock
import sys
import argparse

# Mock tree_sitter modules BEFORE importing extractor
mock_ts = MagicMock()
mock_ts_py = MagicMock()
sys.modules['tree_sitter'] = mock_ts
sys.modules['tree_sitter_python'] = mock_ts_py

from pathlib import Path  # noqa: E402
from google.adk.scope.extractors.extractor_py import find_python_files, extract_features, main  # noqa: E402
from google.adk.scope.features_pb2 import Feature  # noqa: E402

class TestExtractor(unittest.TestCase):
    def test_find_python_files(self):
        # Mock Path.rglob
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.rglob') as mock_rglob:
            
            p1 = Path("src/a.py")
            p2 = Path("src/__init__.py") # Should be excluded
            p3 = Path("src/.hidden.py") # Should be excluded
            p4 = Path("src/b.py")
            p5 = Path("src/subdir/c.py")
            p6 = Path("src/.venv/lib.py") # Should be excluded because of .venv
            
            mock_rglob.return_value = [p1, p2, p3, p4, p5, p6]
            
            results = list(find_python_files(Path("src")))
            
            self.assertIn(p1, results)
            self.assertIn(p4, results)
            self.assertIn(p5, results)
            self.assertNotIn(p2, results) # __init__ excluded
            self.assertNotIn(p3, results) # hidden file excluded
            self.assertNotIn(p6, results) # hidden dir excluded
            
    @patch('google.adk.scope.extractors.extractor_py.QueryCursor')
    @patch('google.adk.scope.extractors.extractor_py.Query')
    @patch('google.adk.scope.extractors.extractor_py.PARSER')
    def test_extract_features(self, mock_parser, mock_query_cls, mock_cursor_cls):
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
        mock_node.type = 'function_definition'
        mock_cursor_instance.captures.return_value = {"func": [mock_node]}
    
        # We need to mock NodeProcessor.process to avoid complex node mocking if we just want to test flow
        with patch('google.adk.scope.extractors.extractor_py.NodeProcessor') as MockProcessor:
            processor_instance = MockProcessor.return_value
            expected_feature = Feature(original_name="foo", normalized_name="foo")
            processor_instance.process.return_value = expected_feature
    
            features = extract_features(mock_path, Path("/repo"))
    
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0], expected_feature)
            
            # Verify process was called
            processor_instance.process.assert_called_once()

    @patch('google.adk.scope.extractors.extractor_py.parse_args')
    @patch('google.adk.scope.extractors.extractor_py.extract_features')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('google.adk.scope.extractors.extractor_py.pathlib.Path.exists')
    def test_main_single_file(self, mock_exists, mock_open, mock_extract, mock_parse_args):
        # Setup args
        args = argparse.Namespace(
            input_file=Path("test.py"),
            input_dir=None,
            input_repo=None,
            output=Path("out.json")
        )
        mock_parse_args.return_value = args
        mock_exists.return_value = True
        
        # Setup extract return
        mock_extract.return_value = [Feature(original_name="f")]
        
        # Run main
        main()
        
        # Verify extract called with correct paths
        # Logic tries to find repo_root from input_file parents. 
        # Since we mocked path, parents might be empty or default.
        mock_extract.assert_called_once()
        args, _ = mock_extract.call_args
        self.assertEqual(args[0], Path("test.py"))
        
        # Verify file write
        mock_open.assert_called_with(Path("out.json"), "w")

    @patch('google.adk.scope.extractors.extractor_py.parse_args')
    @patch('google.adk.scope.extractors.extractor_py.extract_features')
    @patch('google.adk.scope.extractors.extractor_py.find_python_files')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('google.adk.scope.extractors.extractor_py.pathlib.Path.exists')
    def test_main_directory(self, mock_exists, mock_open, mock_find, mock_extract, mock_parse_args):
        # Setup args
        args = argparse.Namespace(
            input_file=None,
            input_dir=Path("src"),
            input_repo=None,
            output=Path("out.json")
        )
        mock_parse_args.return_value = args
        mock_exists.return_value = True
        
        # Setup finding files
        p1 = Path("src/a.py")
        mock_find.return_value = iter([p1])
        
        mock_extract.return_value = []
        
        main()
        
        mock_find.assert_called_once_with(Path("src"), recursive=False)
        mock_extract.assert_called_once()

        mock_find.assert_called_once_with(Path("src"), recursive=False)
        mock_extract.assert_called_once()

    @patch('google.adk.scope.extractors.extractor_py.parse_args')
    @patch('google.adk.scope.extractors.extractor_py.extract_features')
    @patch('google.adk.scope.extractors.extractor_py.find_python_files')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('google.adk.scope.extractors.extractor_py.pathlib.Path.exists')
    def test_main_directory_logging_error(self, mock_exists, mock_open, mock_find, mock_extract, mock_parse_args):
        # Setup args
        args = argparse.Namespace(
            input_file=None,
            input_dir=Path("src"),
            input_repo=None,
            output=Path("out.json")
        )
        mock_parse_args.return_value = args
        mock_exists.return_value = True
        
        # Return a file that is NOT relative to src to trigger ValueError
        p1 = Path("/other/a.py")
        mock_find.return_value = iter([p1])
        
        # Return features so we enter the logging block
        mock_extract.return_value = [Feature(original_name="f")]
        
        main()
        
        # Should execute without error and hit ValueError catch block
        mock_extract.assert_called_once()

    @patch('google.adk.scope.extractors.extractor_py.parse_args')
    @patch('google.adk.scope.extractors.extractor_py.extract_features')
    @patch('google.adk.scope.extractors.extractor_py.find_python_files')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('google.adk.scope.extractors.extractor_py.pathlib.Path.exists')
    def test_main_repo(self, mock_exists, mock_open, mock_find, mock_extract, mock_parse_args):
        # Setup args
        args = argparse.Namespace(
            input_file=None,
            input_dir=None,
            input_repo=Path("/repo"),
            output=Path("out.json")
        )
        mock_parse_args.return_value = args
        mock_exists.return_value = True
        
        # Setup finding files
        p1 = Path("/repo/src/a.py")
        mock_find.return_value = iter([p1])
        
        # Return features so we enter logging block
        mock_extract.return_value = [Feature(original_name="f")]
        
        main()
        
        # Should look in src
        mock_find.assert_called_once_with(Path("/repo/src"), recursive=True)
        mock_extract.assert_called_once()

    def test_find_python_files_not_exists(self):
        with patch('pathlib.Path.exists', return_value=False):
            results = list(find_python_files(Path("bad_path")))
            self.assertEqual(results, [])

    @patch('google.adk.scope.extractors.extractor_py.parse_args')
    @patch('sys.exit')
    @patch('google.adk.scope.extractors.extractor_py.pathlib.Path.exists')
    def test_main_not_found_errors(self, mock_exists, mock_exit, mock_parse_args):
        # Case 1: input_file not found
        mock_exists.return_value = False
        mock_parse_args.return_value = argparse.Namespace(
            input_file=Path("missing.py"), input_dir=None, input_repo=None, output=Path("out.json")
        )
        main()
        mock_exit.assert_called_with(1)
        
        # Case 2: input_dir not found
        mock_exit.reset_mock()
        mock_parse_args.return_value = argparse.Namespace(
            input_file=None, input_dir=Path("missing_dir"), input_repo=None, output=Path("out.json")
        )
        main()
        mock_exit.assert_called_with(1)

        # Case 3: input_repo not found
        mock_exit.reset_mock()
        mock_parse_args.return_value = argparse.Namespace(
            input_file=None, input_dir=None, input_repo=Path("missing_repo"), output=Path("out.json")
        )
        main()
        mock_exit.assert_called_with(1)

    @patch('google.adk.scope.extractors.extractor_py.parse_args')
    @patch('google.adk.scope.extractors.extractor_py.pathlib.Path.exists')
    def test_main_repo_no_src(self, mock_exists, mock_parse_args):
        # We need flexible side_effect because new logical checks might exist (like version file)
        def side_effect(*args, **kwargs):
            # args[0] should be self (the path instance) if called as method
            # But wait, patch('pathlib.Path.exists') replaces the method on the class.
            # When `input_path.exists()` is called, it translates to `Path.exists(input_path)`.
            # So yes, args[0] is the path.
            if not args:
                return True # Fallback
            path_str = str(args[0])
            if path_str.endswith('src'):
                return False
            # Allow others
            return True
            
        mock_exists.side_effect = side_effect
        
        mock_parse_args.return_value = argparse.Namespace(
            input_file=None, input_dir=None, input_repo=Path("/repo"), output=Path("out.json")
        )
        
        main()
        # Should just log warning and exit/return without crashing
        # Verify no extraction happened? (Hard to verify absence of side effect without mocking find/extract, 
        # but coverage will show we hit the warning line)

    @patch('google.adk.scope.extractors.extractor_py.parse_args')
    @patch('google.adk.scope.extractors.extractor_py.extract_features', return_value=[])
    @patch('builtins.open', new_callable=MagicMock)
    @patch('google.adk.scope.extractors.extractor_py.pathlib.Path.exists', return_value=True)
    def test_main_write_error(self, mock_exists, mock_open, mock_extract, mock_parse_args):
        mock_parse_args.return_value = argparse.Namespace(
            input_file=Path("test.py"), input_dir=None, input_repo=None, output=Path("out.json")
        )
        mock_open.side_effect = IOError("Permissions")
        main()
        # Should log error and not crash

    def test_extract_features_read_error(self):
        mock_path = MagicMock(spec=Path)
        mock_path.read_bytes.side_effect = IOError("Read error")
        features = extract_features(mock_path, Path("/repo"))
        self.assertEqual(features, [])

    @patch('google.adk.scope.extractors.extractor_py.parse_args')
    @patch('sys.exit')
    @patch('google.adk.scope.extractors.extractor_py.pathlib.Path.exists', return_value=True)
    def test_main_no_args(self, mock_exists, mock_exit, mock_parse_args):
        # Simulate argparse failing to enforce mutually exclusive group (e.g. if we changed code)
        # or just manual call with all Nones
        mock_parse_args.return_value = argparse.Namespace(
            input_file=None, input_dir=None, input_repo=None, output=Path("out.json")
        )
        main()
        mock_exit.assert_called_with(1)

    @patch('google.adk.scope.extractors.extractor_py.parse_args')
    @patch('google.adk.scope.extractors.extractor_py.extract_features', return_value=[Feature(original_name="f")])
    @patch('builtins.open', new_callable=MagicMock)
    @patch('google.adk.scope.extractors.extractor_py.pathlib.Path.exists', return_value=True)
    def test_main_input_file_logging_error(self, mock_exists, mock_open, mock_extract, mock_parse_args):
        # Case where input_file is not relative to repo_root
        # repo_root determined by parents with 'src'. 
        
        # We need to construct a path where parent / "src" exists but path is relative to it?
        # Actually logic is:
        # repo_root = input_path.parent
        # for parent in input_path.parents: if (parent/"src").exists: repo_root = parent; break
        
        # If we define input_path such that repo_root becomes something that input_path is NOT relative to?
        # Impossible if repo_root is one of the parents.
        # UNLESS we patch relative_to to raise ValueError specifically.
        
        input_file = MagicMock(spec=Path)
        input_file.exists.return_value = True
        input_file.name = "test.py"
        # Setup parents to NOT have src, so repo_root = input_file.parent
        input_file.parents = [Path("/dir")]
        input_file.parent = Path("/dir")
        
        # We need relative_to to raise ValueError
        input_file.relative_to.side_effect = ValueError("Not relative")
        
        mock_parse_args.return_value = argparse.Namespace(
            input_file=input_file, input_dir=None, input_repo=None, output=Path("out.json")
        )
        
        main()
        
        # Should verify we didn't crash
        mock_extract.assert_called()

    @patch('google.adk.scope.extractors.extractor_py.parse_args')
    @patch('google.adk.scope.extractors.extractor_py.extract_features', return_value=[Feature(original_name="f")])
    @patch('google.adk.scope.extractors.extractor_py.find_python_files')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('google.adk.scope.extractors.extractor_py.pathlib.Path.exists', return_value=True)
    def test_main_repo_logging_error(self, mock_exists, mock_open, mock_find, mock_extract, mock_parse_args):
        mock_parse_args.return_value = argparse.Namespace(
            input_file=None, input_dir=None, input_repo=Path("/repo"), output=Path("out.json")
        )
        
        # Return a path object that mocks relative_to
        p1 = MagicMock(spec=Path)
        p1.name = "a.py"
        p1.relative_to.side_effect = ValueError("Not relative")
        
        mock_find.return_value = iter([p1])
        
        main()
        
        # Should catch ValueError and log filename
        mock_extract.assert_called()

if __name__ == '__main__':
    unittest.main()
