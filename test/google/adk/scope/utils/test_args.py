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


if __name__ == '__main__':
    unittest.main()
