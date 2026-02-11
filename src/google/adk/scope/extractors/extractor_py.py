import logging
import pathlib
from typing import Iterator, List

import tree_sitter_python as tspy
from tree_sitter import Language, Parser, Query, QueryCursor

from google.adk.scope.extractors.converter_py import NodeProcessor
from google.adk.scope.features_pb2 import Feature
from google.adk.scope.utils.normalizer import normalize_namespace

# Initialize Tree-sitter
PY_LANGUAGE = Language(tspy.language())
PARSER = Parser()
PARSER.language = PY_LANGUAGE

logger = logging.getLogger(__name__)


def find_files(
    root: pathlib.Path, recursive: bool = True
) -> Iterator[pathlib.Path]:
    """Find Python files in the given directory.

    Args:
        root (Path): The root directory to search.
        recursive (bool): Whether to search recursively.

    Yields:
        Iterator[Path]: An iterator of Paths to Python files.
    """
    if not root.exists():
        logger.warning("Directory %s does not exist. Skipping traversal.", root)
        return

    # Files to exclude from extraction
    excluded_files = {
        "__init__.py",
        "version.py",
        "setup.py",
        "conftest.py",
    }

    iterator = root.rglob("*.py") if recursive else root.glob("*.py")

    for path in iterator:
        if path.name in excluded_files:
            logger.debug("Skipping excluded file: %s", path)
            continue
        # Also exclude hidden files or directories starting with .
        if any(
            part.startswith(".") and part not in (".", "..")
            for part in path.parts
        ):
            logger.debug("Skipping hidden path: %s", path)
            continue

        yield path


def extract_features(
    file_path: pathlib.Path, repo_root: pathlib.Path, source_root: str
) -> List[Feature]:
    """Extract Feature objects from a Python file.

    Args:
        file_path (Path): Path to the Python file.
        repo_root (Path): Path to the repository root.

    Returns:
        List[feature_pb2.Feature]: A list of extracted Features.
    """
    try:
        content = file_path.read_bytes()
    except IOError as e:
        logger.error("Failed to read %s: %s", file_path, e)
        return []

    tree = PARSER.parse(content)
    root_node = tree.root_node

    processor = NodeProcessor()
    features = []

    # Query for functions and methods
    query = Query(
        PY_LANGUAGE,
        """
    (function_definition
      name: (identifier) @name
    ) @func
    """,
    )

    cursor = QueryCursor(query)
    captures = cursor.captures(root_node)

    # captures is a dict {capture_name: [nodes]}
    func_nodes = captures.get("func", [])
    logger.debug(
        "Found %d potential function nodes in %s", len(func_nodes), file_path
    )

    for node in func_nodes:
        # The node is a function_definition
        feature = processor.process(node, file_path, repo_root)
        if feature:
            feature.normalized_namespace = normalize_namespace(
                str(file_path), str(repo_root / source_root)
            )
            features.append(feature)
            logger.debug("Extracted feature: %s", feature.original_name)
        else:
            # Maybe log why it was skipped if process returns None
            # But process usually logs internal reasons if we add logs there
            pass

    return features


def get_version(repo_root: pathlib.Path) -> str:
    version = "0.0.0"
    version_file = repo_root / "src" / "google" / "adk" / "version.py"
    if version_file.exists():
        try:
            content = version_file.read_text()
            for line in content.splitlines():
                if line.startswith("__version__"):
                    # __version__ = "1.22.0"
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    break
        except Exception as e:
            logger.warning("Failed to read version file: %s", e)
    return version
