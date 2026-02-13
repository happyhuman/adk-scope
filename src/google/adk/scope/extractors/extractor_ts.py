import logging
import pathlib
from typing import Iterator, List

import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser, Query, QueryCursor

from google.adk.scope.extractors.converter_ts import NodeProcessor
from google.adk.scope.features_pb2 import Feature
from google.adk.scope.utils.normalizer import normalize_namespace

# Initialize Tree-sitter
try:
    TS_LANGUAGE = Language(tsts.language_typescript())
except AttributeError:
    TS_LANGUAGE = Language(tsts.language())

PARSER = Parser()
PARSER.language = TS_LANGUAGE

logger = logging.getLogger(__name__)

GLOBAL_TYPE_MAP = {}
_GLOBAL_TYPE_MAP_INITIALIZED = False


def _build_global_type_map(repo_root: pathlib.Path):
    global _GLOBAL_TYPE_MAP_INITIALIZED, GLOBAL_TYPE_MAP
    if _GLOBAL_TYPE_MAP_INITIALIZED:
        return

    _GLOBAL_TYPE_MAP_INITIALIZED = True

    query = Query(
        TS_LANGUAGE,
        """
        (interface_declaration) @interface
        (type_alias_declaration) @alias
        """,
    )
    cursor = QueryCursor(query)

    for file_path in find_files(repo_root):
        try:
            content = file_path.read_bytes()
            tree = PARSER.parse(content)
            captures = cursor.captures(tree.root_node)
            nodes = captures.get("interface", []) + captures.get("alias", [])

            for node in nodes:
                name_node = node.child_by_field_name("name")
                if node.type == "interface_declaration":
                    body_node = node.child_by_field_name("body")
                else:
                    body_node = node.child_by_field_name("value")

                if name_node and body_node:
                    name = name_node.text.decode("utf-8")
                    type_map = {}
                    for child in body_node.children:
                        if child.type == "property_signature":
                            prop_name_node = child.child_by_field_name("name")
                            prop_type_node = child.child_by_field_name("type")
                            if prop_name_node:
                                p_name = prop_name_node.text.decode("utf-8")
                                p_type = ""
                                if prop_type_node:
                                    p_type = prop_type_node.text.decode("utf-8")
                                    if p_type.startswith(":"):
                                        p_type = p_type[1:].strip()
                                p_optional = False
                                for sub in child.children:
                                    if (
                                        sub.type == "?"
                                        or sub.text.decode("utf-8") == "?"
                                    ):
                                        p_optional = True
                                        break
                                type_map[p_name] = (p_type, p_optional)
                    GLOBAL_TYPE_MAP[name] = type_map
        except Exception as e:
            logger.debug(
                "Failed to read %s for global type map: %s", file_path, e
            )


def find_files(
    root: pathlib.Path, recursive: bool = True
) -> Iterator[pathlib.Path]:
    if not root.exists():
        logger.warning("Directory %s does not exist. Skipping traversal.", root)
        return

    if recursive:
        iterator = root.rglob("*.ts")
    else:
        iterator = root.glob("*.ts")

    for path in iterator:
        # Exclude .d.ts
        if path.name.endswith(".d.ts"):
            logger.debug("Skipping .d.ts file: %s", path)
            continue

        # Exclude node_modules, etc.
        if any(
            part == "node_modules" or part.startswith(".")
            for part in path.parts
            if part not in (".", "..")
        ):
            logger.debug("Skipping hidden/node_modules path: %s", path)
            continue

        yield path


def extract_features(
    file_path: pathlib.Path, repo_root: pathlib.Path, source_root: str
) -> List[Feature]:
    try:
        content = file_path.read_bytes()
    except IOError as e:
        logger.error("Failed to read %s: %s", file_path, e)
        return []

    tree = PARSER.parse(content)
    root_node = tree.root_node

    _build_global_type_map(repo_root)

    processor = NodeProcessor(GLOBAL_TYPE_MAP)
    features = []

    # Query for Class Declarations, Method Definitions, Function Declarations
    query = Query(
        TS_LANGUAGE,
        """
    (function_declaration) @func
    (method_definition) @method
  """,
    )

    cursor = QueryCursor(query)
    captures = cursor.captures(root_node)

    all_nodes = []
    if "func" in captures:
        all_nodes.extend(captures["func"])
    if "method" in captures:
        all_nodes.extend(captures["method"])

    logger.debug("Found %d potential nodes in %s", len(all_nodes), file_path)

    for node in all_nodes:

        feature = processor.process(node, file_path, repo_root)
        if feature:
            feature.normalized_namespace = normalize_namespace(
                str(file_path), str(repo_root / source_root)
            )
            features.append(feature)
            logger.debug("Extracted feature: %s", feature.original_name)

    return features


def get_version(repo_root: pathlib.Path) -> str:
    version = "0.0.0"
    pkg_json = repo_root / "package.json"
    if pkg_json.exists():
        try:
            import json

            data = json.loads(pkg_json.read_text())
            version = data.get("version", "0.0.0")
        except Exception:
            pass
    return version
