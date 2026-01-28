
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
        with patch('argparse.ArgumentParser') as mock_parser_cls:
            mock_parser = mock_parser_cls.return_value
            # We also need to mock the group returned by add_mutually_exclusive_group
            mock_group = mock_parser.add_mutually_exclusive_group.return_value
            
            parse_args()
            
            # Verify mutual exclusive group creation
            mock_parser.add_mutually_exclusive_group.assert_called_once_with(required=True)
            
            # Verify group arguments
            group_calls = mock_group.add_argument.call_args_list
            self.assertEqual(len(group_calls), 3)
            
            # --input-file
            self.assertEqual(group_calls[0][0][0], '--input-file')
            # --input-dir
            self.assertEqual(group_calls[1][0][0], '--input-dir')
            # --input-repo
            self.assertEqual(group_calls[2][0][0], '--input-repo')

            # Verify parser arguments (output)
            # add_argument is called once for 'output' on the parser itself
            parser_calls = mock_parser.add_argument.call_args_list
            self.assertEqual(len(parser_calls), 1) 
            self.assertEqual(parser_calls[0][0][0], 'output')

if __name__ == '__main__':
    unittest.main()
