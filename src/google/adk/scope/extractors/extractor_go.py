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
        # Check if any part of the path starts with '.' (excluding '.' and '..')
        if any(
            part.startswith(".") and part not in (".", "..")
            for part in path.parts
        ):
            continue
        yield path


def extract_features(
    file_path: pathlib.Path, repo_root: pathlib.Path, source_root: str
) -> List[Feature]:
    """Extract Feature objects from a Go file."""
    if file_path.name.endswith("_test.go"):
        return []

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
        logger.error(
            "Tree-sitter failed to parse %s (Root type: %s)",
            file_path,
            root_node.type,
        )
        return []

    processor = NodeProcessor()
    features = []

    # REVISED QUERY: Matches the declaration nodes.
    # We tag them specifically so the processor knows what it's looking at.
    query_text = """
        (function_declaration) @func
        (method_declaration) @method
    """
    query = Query(GO_LANGUAGE, query_text)
    cursor = QueryCursor(query)
    captures = cursor.captures(root_node)

    all_nodes = []
    for node_list in captures.values():
        all_nodes.extend(node_list)

    # Log results for debugging
    logger.debug("Found %d potential nodes in %s", len(all_nodes), file_path)

    for node in all_nodes:
        # Filter out simple functions (e.g., getters, setters) by checking
        # the body. Note: In Go AST, the function 'body' is a 'block' which
        # contains a 'statement_list'. We need to check the size of the
        # 'statement_list' to know the actual number of statements.
        body_node = node.child_by_field_name("body")
        if body_node:
            stmt_list = next(
                (
                    child
                    for child in body_node.children
                    if child.type == "statement_list"
                ),
                None,
            )
            # If there is no statement list, or it has 1 or fewer statements,
            # consider it simple.
            if stmt_list is None or stmt_list.named_child_count <= 1:
                function_name_node = node.child_by_field_name("name")
                if function_name_node:
                    logger.debug(
                        "Skipping simple function: %s",
                        function_name_node.text.decode("utf8"),
                    )
                continue

        # Prepare namespace and normalized namespace
        try:
            rel_path = file_path.relative_to(repo_root)
            parts = list(rel_path.parent.parts)
            # Remove hidden dirs or known roots if needed (Go usually relies
            # on dir path or go.mod, we'll use the relative directory path as
            # base).
            parts = [p for p in parts if p and p not in (".", "..", "src")]
            namespace = ".".join(parts)
        except ValueError:
            namespace = ""

        # Using the same normalization logic as earlier for parity
        normalized_namespace = normalize_namespace(
            str(file_path), str(repo_root / source_root)
        )

        # Ensure the processor gets the node and context (including namespace)
        feature = processor.process(
            node, file_path, repo_root, namespace, normalized_namespace
        )

        if feature:
            features.append(feature)
            logger.debug("Extracted feature: %s", feature.original_name)
        else:
            # If nodes are found but features are None, the NodeProcessor is
            # filtering them out. This often happens if the function is not
            # exported (starts with lowercase).
            pass

    return features


def get_version(repo_root: pathlib.Path) -> str:
    """Get the version of the ADK from internal/version/version.go."""
    version_path = repo_root / "internal" / "version" / "version.go"
    if version_path.exists():
        try:
            content = version_path.read_text()
            for line in content.splitlines():
                if "const Version string =" in line:
                    # e.g., const Version string = "0.3.0"
                    parts = line.split('"')
                    if len(parts) >= 3:
                        return parts[1]
        except Exception as e:
            logger.warning("Failed to read version.go file: %s", e)
    
    # Fallback to reading go.mod module path if version isn't found
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
