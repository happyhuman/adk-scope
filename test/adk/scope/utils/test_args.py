import argparse
import logging
import unittest
from unittest import mock

from google.adk.scope.utils import args as adk_args


class ArgsTest(unittest.TestCase):

    def test_add_verbose_argument(self):
        parser = argparse.ArgumentParser()
        adk_args.add_verbose_argument(parser)
        args = parser.parse_args(['--verbose'])
        self.assertTrue(args.verbose)

    @mock.patch('logging.basicConfig')
    def test_configure_logging_verbose(self, mock_basic_config):
        args = argparse.Namespace(verbose=True)
        adk_args.configure_logging(args)
        mock_basic_config.assert_called_once_with(level=logging.DEBUG)

    @mock.patch('logging.basicConfig')
    def test_configure_logging_default(self, mock_basic_config):
        args = argparse.Namespace(verbose=False)
        adk_args.configure_logging(args)
        mock_basic_config.assert_called_once_with(level=logging.INFO)

    @mock.patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_python(self, mock_parse_args):
        mock_parse_args.return_value = argparse.Namespace(
            language='py',
            input_file='test.py',
            output='out',
            verbose=False
        )
        parsed_args = adk_args.parse_args()
        self.assertEqual(parsed_args.language, 'python')

    @mock.patch('argparse.ArgumentParser.parse_args')
    def test_parse_args_typescript(self, mock_parse_args):
        mock_parse_args.return_value = argparse.Namespace(
            language='ts',
            input_file='test.ts',
            output='out',
            verbose=False
        )
        parsed_args = adk_args.parse_args()
        self.assertEqual(parsed_args.language, 'typescript')


    def test_arg_definitions(self):
        # Verify that the parser is set up with correct arguments
        with mock.patch("argparse.ArgumentParser") as mock_parser_cls:
            mock_parser = mock_parser_cls.return_value
            # We also need to mock the group returned by
            # add_mutually_exclusive_group
            mock_group = mock_parser.add_mutually_exclusive_group.return_value

            adk_args.parse_args()

            # Verify mutual exclusive group creation
            mock_parser.add_mutually_exclusive_group.assert_called_once_with(
                required=True
            )

            # Verify group arguments
            group_calls = mock_group.add_argument.call_args_list
            self.assertEqual(len(group_calls), 3)

            # --input-file
            self.assertEqual(group_calls[0][0][0], "--input-file")
            # --input-dir
            self.assertEqual(group_calls[1][0][0], "--input-dir")
            # --input-repo
            self.assertEqual(group_calls[2][0][0], "--input-repo")

            # Verify parser arguments (--language, output, --verbose)
            # add_argument called 3 times: '--language', 'output', '--verbose'
            parser_calls = mock_parser.add_argument.call_args_list
            self.assertEqual(len(parser_calls), 3)

            # Check first call (language)
            args, _ = parser_calls[0]
            self.assertEqual(args[0], "--language")

            # Check second call (output)
            args, _ = parser_calls[1]
            self.assertEqual(args[0], "output")

            # Check third call (verbose)
            args, _ = parser_calls[2]
            self.assertEqual(args[0], "--verbose")

if __name__ == '__main__':
    unittest.main()
