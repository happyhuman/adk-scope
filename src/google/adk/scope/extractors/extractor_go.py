import logging
import pathlib
import re
import subprocess
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


# Repository-level cache to keep parsed package structures.
# key: Path (repo root directory), value: struct definitions dictionary
_REPO_STRUCTS_CACHE = {}


def _pre_parse_repo_structs(
    repo_root: pathlib.Path,
) -> dict[str, list[tuple[str, str, bool]]]:
    """Scans all Go files under repo_root and extracts struct definitions."""
    global _REPO_STRUCTS_CACHE
    repo_root = repo_root.resolve()
    if repo_root in _REPO_STRUCTS_CACHE:
        return _REPO_STRUCTS_CACHE[repo_root]

    logger.debug("Pre-parsing repository for Go structs: %s", repo_root)

    # Scan ALL Go files recursively under repo_root!
    # Reuse find_files helper to exclude tests, hidden directories, etc.
    go_files = list(find_files(repo_root, recursive=True))

    # Simple struct-only Tree-sitter query
    struct_query = Query(
        GO_LANGUAGE,
        """
        (type_declaration
          (type_spec
            name: (type_identifier) @struct_name
            type: (struct_type) @struct_body
          )
        )
        """,
    )

    # Simple NodeProcessor with just register_struct capability
    temp_processor = NodeProcessor()

    for file_path in go_files:
        try:
            content = file_path.read_bytes()
            tree = PARSER.parse(content)
            cursor = QueryCursor(struct_query)
            captures = cursor.captures(tree.root_node)

            struct_bodies = captures.get("struct_body", [])
            for body in struct_bodies:
                temp_processor.register_struct(body)
        except Exception as e:
            logger.debug("Failed to pre-parse structs in %s: %s", file_path, e)

    # Save to global cache
    _REPO_STRUCTS_CACHE[repo_root] = temp_processor._struct_definitions
    return temp_processor._struct_definitions


def find_files(
    root: pathlib.Path, recursive: bool = True
) -> Iterator[pathlib.Path]:
    """Find Go files in the given directory."""
    if not root.exists():
        logger.warning("Directory %s does not exist. Skipping traversal.", root)
        return

    iterator = root.rglob("*.go") if recursive else root.glob("*.go")

    for path in iterator:
        if path.name.endswith("_test.go"):
            continue

        # Exclude hidden directories, files, and common testing directories
        if any(
            (part.startswith(".") and part not in (".", ".."))
            or part in ("tests", "testutil", "testing", "testdata")
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

    # Pre-parse repository-wide structs to enable cross-package struct matching
    repo_structs = _pre_parse_repo_structs(repo_root)
    processor = NodeProcessor(struct_definitions=repo_structs)

    # Pre-process structs to build the definition map
    # We need to re-query or process struct nodes specifically.
    # To keep it simple, let's just use the query we have.
    pass
    features = []

    # REVISED QUERY: Matches the declaration nodes.
    # We tag them specifically so the processor knows what it's looking at.
    query_text = """
        (function_declaration) @func
        (method_declaration) @method
        (type_declaration
          (type_spec
            name: (type_identifier) @interface_name
            type: (interface_type
              (method_elem) @interface_method
            )
          )
        )
        (type_declaration
          (type_spec
            name: (type_identifier) @struct_name
            type: (struct_type) @struct_body
          )
        )
        (type_declaration
          (type_spec
            name: (type_identifier) @func_type_name
            type: (function_type) @func_type_body
          )
        )
    """
    query = Query(GO_LANGUAGE, query_text)
    cursor = QueryCursor(query)
    captures = cursor.captures(root_node)

    all_nodes = []
    struct_nodes = []
    # We only want to process the actual function/method nodes, not the
    # interface names which are captured just for context by the processor
    # (via tree traversal).
    for capture_name, node_list in captures.items():
        if capture_name in ("func", "method", "interface_method"):
            all_nodes.extend(node_list)
        elif capture_name == "func_type_body":
            # For function types, we want to process the parent type_spec to get the name
            # node_list contains the func_type nodes.
            for node in node_list:
                # Parent is type_spec
                if node.parent and node.parent.type == "type_spec":
                    all_nodes.append(node.parent)
        elif capture_name == "struct_body":
            # We need to associate the struct body with its name.
            # The query captures @struct_name and @struct_body separately but
            # in order.
            # However, 'captures' is a dict of lists, so order might be tricky
            # if we rely on index alignment across lists.
            # Better strategy: Capture the parent type_spec and process it?
            # Or iterate the captures list (which we can't easily do with the
            # dict output).
            # Let's rely on NodeProcessor to find the name from the struct_body
            # node's parent.
            struct_nodes.extend(node_list)

    # Log results for debugging
    logger.debug("Found %d potential nodes in %s", len(all_nodes), file_path)

    # Build struct definitions map first
    for node in struct_nodes:
        processor.register_struct(node)

    for node in all_nodes:
        # Prevent filtering out abstract interface methods which have no body
        if node.type == "method_elem" or node.type == "type_spec":
            pass
        else:
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
                # If there is no statement list, or it has 1 or fewer
                # statements, consider it simple.
                if stmt_list is None or stmt_list.named_child_count <= 1:
                    # Also check physical line span to prevent skipping large
                    # single-statement functions (e.g. methods returning a large
                    # anonymous function).
                    start_row = body_node.start_point[0]
                    end_row = body_node.end_point[0]
                    line_span = end_row - start_row + 1

                    if line_span <= 4:
                        function_name_node = node.child_by_field_name("name")
                        if function_name_node:
                            logger.debug(
                                "Skipping simple function: %s (span: %d lines)",
                                function_name_node.text.decode("utf8"),
                                line_span,
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
    """Get the version of the ADK.

    Args:
        repo_root: The root directory of the repository.

    Returns:
        The extracted version string, or an empty string if not found.
    """
    # 1. Try git describe to get the tag version (e.g., v1.3.0)
    try:
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if version:
            return version
    except Exception:
        pass

    # 2. Fallback: parse internal/version/version.go
    version_path = repo_root / "internal" / "version" / "version.go"
    if version_path.exists():
        try:
            content = version_path.read_text()
            # Match both `const Version string = "..."` and `const Version = "..."`
            match = re.search(
                r'const\s+Version\s+(?:string\s+)?=\s*"([^"]+)"', content
            )
            if match:
                return match.group(1)
        except Exception as e:
            logger.warning("Failed to read version.go file: %s", e)

    # 3. Fallback to reading go.mod module path
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
