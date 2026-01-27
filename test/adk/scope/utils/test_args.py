
import unittest
import argparse
from unittest.mock import patch
from pathlib import Path
from google.adk.scope.utils.args import parse_args

class TestArgs(unittest.TestCase):
    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args(self, mock_parse):
        # Setup mock return value
        mock_args = argparse.Namespace(adk_repo=Path('/tmp/repo'), output=Path('/tmp/out.json'))
        mock_parse.return_value = mock_args
        
        # Call the function (arguments are parsed from sys.argv by default, but we mocked parse_args)
        args = parse_args()
        
        self.assertEqual(args.adk_repo, Path('/tmp/repo'))
        self.assertEqual(args.output, Path('/tmp/out.json'))
        
    def test_arg_definitions(self):
        # Verify that the parser is set up with correct arguments
        # We can inspect the parser by creating one manually or patching ArgumentParser
        with patch('argparse.ArgumentParser') as mock_parser_cls:
            mock_parser = mock_parser_cls.return_value
            parse_args()
            
            # Verify calls
            calls = mock_parser.add_argument.call_args_list
            
            # We expect 2 calls: one for 'adk-repo' and one for 'output'
            self.assertEqual(len(calls), 2)
            
            # Check first arg (adk-repo)
            args1, kwargs1 = calls[0]
            self.assertEqual(args1[0], 'adk_repo') # Expect underscore? No, user defined "adk-repo" in string but wait.
            # In my implementation of args.py I used "adk_repo" in the python code?
            # Let's check args.py content.
            # Step 209: I implemented it with `parser.add_argument("adk_repo", ...)`
            # So it should be 'adk_repo'.
            
            self.assertEqual(args1[0], 'adk_repo')
            self.assertEqual(kwargs1['type'], Path)
            
            # Check second arg (output)
            args2, kwargs2 = calls[1]
            self.assertEqual(args2[0], 'output')
            self.assertEqual(kwargs2['type'], Path)

if __name__ == '__main__':
    unittest.main()
