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


def find_files(root: pathlib.Path, recursive: bool = True) -> Iterator[pathlib.Path]:
    """Find Go files in the given directory."""
    if not root.exists():
        logger.warning("Directory %s does not exist. Skipping traversal.", root)
        return

    iterator = root.rglob("*.go") if recursive else root.glob("*.go")

    for path in iterator:
        # Check if any part of the path starts with '.' (excluding '.' and '..')
        if any(part.startswith(".") and part not in (".", "..") for part in path.parts):
            continue
        yield path


def extract_features(
    file_path: pathlib.Path, repo_root: pathlib.Path, source_root: str
) -> List[Feature]:
    """Extract Feature objects from a Go file."""
    try:
        # Resolve paths to absolute to prevent normalization issues
        file_path = file_path.resolve()
        repo_root = repo_root.resolve()
        content = file_path.read_bytes()
    except Exception as e:
        logger.error("Failed to read %s: %s", file_path, e)
        return []

    tree = PARSER.parse(content)
    root_node = tree.root_node

    # DEBUG: Ensure the parser is actually working
    if root_node.type == "ERROR" or root_node.child_count == 0:
        logger.error("Tree-sitter failed to parse %s (Root type: %s)", file_path, root_node.type)
        return []

    processor = NodeProcessor()
    features = []

    # REVISED QUERY: Matches the declaration nodes.
    # We tag them specifically so the processor knows what it's looking at.
    query_text = '''
        (function_declaration) @func
        (method_declaration) @method
    '''
    query = Query(GO_LANGUAGE, query_text)
    cursor = QueryCursor(query)
    captures = cursor.captures(root_node)

    all_nodes = []
    for capture in captures:
        node = capture[0]
        tag = capture[1]

        # Robustly handle tag name (handles both index and string)
        if isinstance(tag, str):
            tag_name = tag
        else:
            tag_name = query.capture_names[tag]

        if tag_name in ("func", "method"):
            all_nodes.append(node)

    # Log results for debugging
    if not all_nodes:
        logger.warning("Query found 0 functions/methods in %s", file_path)
    else:
        logger.info("Found %d potential nodes in %s", len(all_nodes), file_path)

    for node in all_nodes:
        # Ensure the processor gets the node and context
        feature = processor.process(node, file_path, repo_root)
        
        if feature:
            # Source root needs to be relative to repo root for normalize_namespace
            feature.normalized_namespace = normalize_namespace(
                str(file_path), str(repo_root / source_root)
            )
            features.append(feature)
            logger.debug("Extracted feature: %s", feature.original_name)
        else:
            # If nodes are found but features are None, the NodeProcessor is filtering them out.
            # This often happens if the function is not exported (starts with lowercase).
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