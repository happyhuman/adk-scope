import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from google.adk.scope.extractors import extract


class TestExtractHelpers(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_get_repo_root_python(self):
        # Create structure: /root/src/pkg/file.py
        src_dir = self.root / "src"
        src_dir.mkdir()
        (src_dir / "pkg").mkdir()
        input_path = src_dir / "pkg" / "file.py"

        # Should find 'src' marker in 'src_dir' parent?
        # Markers for python: ["src"]
        # Logic: for parent in input_path.parents: if (parent/marker).exists()
        # if input_path is .../src/pkg/file.py
        # parents: .../src/pkg, .../src, .../root, ...
        # if parent is .../root, (parent/"src") exists. -> returns .../root

        root = extract.get_repo_root(input_path, "python")
        self.assertEqual(root, self.root)

    def test_get_repo_root_typescript(self):
        # Markers: package.json, tsconfig.json
        # Structure: /root/package.json, /root/core/src/file.ts
        (self.root / "package.json").touch()
        core_src = self.root / "core" / "src"
        core_src.mkdir(parents=True)
        input_path = core_src / "file.ts"

        root = extract.get_repo_root(input_path, "typescript")
        self.assertEqual(root, self.root)

    def test_get_repo_root_none(self):
        input_path = self.root / "some" / "file.py"
        root = extract.get_repo_root(input_path, "python")
        self.assertIsNone(root)

    def test_get_search_dir_python(self):
        # Subdirs: src
        # Structure: /root/src
        src = self.root / "src"
        src.mkdir()

        search_dir = extract.get_search_dir(self.root, "python")
        self.assertEqual(search_dir, src)

    def test_get_search_dir_fallback(self):
        # No src dir
        search_dir = extract.get_search_dir(self.root, "python")
        self.assertEqual(search_dir, self.root)


class TestExtractMain(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.root = Path(self.test_dir)
        self.output_file = self.root / "output.json"

        # Patches
        self.mock_args_patcher = patch(
            "google.adk.scope.extractors.extract.parse_args"
        )
        self.mock_args = self.mock_args_patcher.start()

        self.mock_extractors_patcher = patch.dict(
            "google.adk.scope.extractors.extract.EXTRACTORS", clear=True
        )
        self.mock_extractors = self.mock_extractors_patcher.start()

        self.mock_py_extractor = MagicMock()
        self.mock_ts_extractor = MagicMock()
        self.mock_extractors["python"] = self.mock_py_extractor
        self.mock_extractors["typescript"] = self.mock_ts_extractor

        # FeatureRegistry mock/patch ?
        # Actual FeatureRegistry is fine if we mock return of extract_features
        # But we write to file using MessageToJson

    def tearDown(self):
        self.mock_args_patcher.stop()
        self.mock_extractors_patcher.stop()
        shutil.rmtree(self.test_dir)

    def configure_args(
        self,
        lang="python",
        input_file=None,
        input_dir=None,
        input_repo=None,
        output=None,
    ):
        mock_args = MagicMock()
        mock_args.language = lang
        mock_args.input_file = Path(input_file) if input_file else None
        mock_args.input_dir = Path(input_dir) if input_dir else None
        mock_args.input_repo = Path(input_repo) if input_repo else None
        mock_args.output = output or str(self.output_file)
        self.mock_args.return_value = mock_args

    def test_unsupported_language(self):
        self.configure_args(lang="rust")
        with self.assertRaises(SystemExit) as cm:
            extract.main()
        self.assertEqual(cm.exception.code, 1)

    def test_no_input_mode(self):
        self.configure_args(lang="python")  # all inputs None
        with self.assertRaises(SystemExit) as cm:
            extract.main()
        self.assertEqual(cm.exception.code, 1)

    def test_input_file_mode(self):
        f = self.root / "test.py"
        f.touch()
        self.configure_args(lang="python", input_file=str(f))

        self.mock_py_extractor.extract_features.return_value = []
        self.mock_py_extractor.get_version.return_value = "1.0"

        extract.main()

        self.mock_py_extractor.extract_features.assert_called()
        self.assertTrue(self.output_file.exists())

    def test_input_file_not_found(self):
        self.configure_args(lang="python", input_file="/non/existent.py")
        with self.assertRaises(SystemExit) as cm:
            extract.main()
        self.assertEqual(cm.exception.code, 1)

    def test_input_dir_mode(self):
        d = self.root / "pkg"
        d.mkdir()
        self.configure_args(lang="python", input_dir=str(d))

        self.mock_py_extractor.find_files.return_value = [d / "a.py"]
        self.mock_py_extractor.extract_features.return_value = []
        self.mock_py_extractor.get_version.return_value = "1.0"

        extract.main()

        self.mock_py_extractor.find_files.assert_called_with(d, recursive=False)
        self.assertTrue(self.output_file.exists())

    def test_input_repo_mode(self):
        r = self.root
        (r / "src").mkdir()
        self.configure_args(lang="python", input_repo=str(r))

        self.mock_py_extractor.find_files.return_value = []
        self.mock_py_extractor.get_version.return_value = "1.0"

        extract.main()

        # python searches in src if exists
        self.mock_py_extractor.find_files.assert_called_with(
            r / "src", recursive=True
        )
        self.assertTrue(self.output_file.exists())


if __name__ == "__main__":
    unittest.main()
