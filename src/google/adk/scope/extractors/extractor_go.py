import logging
import pathlib
from typing import Iterator, List

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser, Query, QueryCursor

from google.adk.scope.extractors.converter_go import NodeProcessor
from google.adk.scope.features_pb2 import Feature
from google.adk.scope.utils.normalizer import normalize_namespace

# Initialize Tree-sitter
GO_LANGUAGE = Language(tsgo.language())
PARSER = Parser()
PARSER.language = GO_LANGUAGE

logger = logging.getLogger(__name__)


def find_files(
    root: pathlib.Path, recursive: bool = True
) -> Iterator[pathlib.Path]:
    """Find Go files in the given directory."""
    if not root.exists():
        logger.warning("Directory %s does not exist. Skipping traversal.", root)
        return

    iterator = root.rglob("*.go") if recursive else root.glob("*.go")

    for path in iterator:
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
    """Extract Feature objects from a Go file."""
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
        GO_LANGUAGE,
        '''
        (function_declaration
          name: (identifier) @name) @func
        (method_declaration
          name: (identifier) @name) @method
        ''',
    )

    cursor = QueryCursor(query)
    captures = cursor.captures(root_node)

    all_nodes = captures.get("func", [])
    all_nodes.extend(captures.get("method", []))
    logger.debug(
        "Found %d potential function nodes in %s", len(all_nodes), file_path
    )
    print(f"root_node: {root_node}")
    print(f"query: {query}")
    logger.debug("root_node: %s", root_node)
    logger.debug("query: %s", query)

    for node in all_nodes:
        feature = processor.process(node, file_path, repo_root)
        if feature:
            feature.normalized_namespace = normalize_namespace(
                str(file_path), str(repo_root / source_root)
            )
            features.append(feature)
            logger.debug("Extracted feature: %s", feature.original_name)
        else:
            pass

    return features


def get_version(repo_root: pathlib.Path) -> str:
    """Get the module path from a go.mod file."""
    go_mod_path = repo_root / "go.mod"
    if go_mod_path.exists():
        try:
            content = go_mod_path.read_text()
            for line in content.splitlines():
                if line.startswith("module"):
                    return line.split()[1]
        except Exception as e:
            logger.warning("Failed to read go.mod file: %s", e)
    return ""
