import logging
import pathlib
from typing import Iterator, List

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Query, QueryCursor

from google.adk.scope.extractors.converter_java import NodeProcessor
from google.adk.scope.features_pb2 import Feature
from google.adk.scope.utils.normalizer import normalize_namespace

# Initialize Tree-sitter
try:
    JAVA_LANGUAGE = Language(tsjava.language())
except AttributeError:
    # Some older versions have .language_java()
    JAVA_LANGUAGE = Language(tsjava.language_java())

PARSER = Parser()
PARSER.language = JAVA_LANGUAGE

logger = logging.getLogger(__name__)


def find_files(
    root: pathlib.Path, recursive: bool = True
) -> Iterator[pathlib.Path]:
    """Find Java files in the given directory.

    Args:
        root (Path): The root directory to search.
        recursive (bool): Whether to search recursively.

    Yields:
        Iterator[Path]: An iterator of Paths to Java files.
    """
    if not root.exists():
        logger.warning("Directory %s does not exist. Skipping traversal.", root)
        return

    # Files to exclude from extraction
    excluded_files = {
        "module-info.java",
        "package-info.java",
    }

    iterator = root.rglob("*.java") if recursive else root.glob("*.java")

    for path in iterator:
        if path.name in excluded_files:
            logger.debug("Skipping excluded file: %s", path)
            continue

        # exclude node_modules, build, etc.
        if any(
            part in ("node_modules", "build", "target", "out", "bin")
            or part.startswith(".")
            for part in path.parts
            if part not in (".", "..")
        ):
            logger.debug("Skipping hidden/build path: %s", path)
            continue

        # Also exclude commonly known test directories
        if any(part == "test" for part in path.parts):
            logger.debug("Skipping test file: %s", path)
            continue

        yield path


def extract_features(
    file_path: pathlib.Path, repo_root: pathlib.Path, source_root: str
) -> List[Feature]:
    """Extract Feature objects from a Java file.

    Args:
        file_path (Path): Path to the Java file.
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

    # Query for methods and constructors
    query = Query(
        JAVA_LANGUAGE,
        """
        (method_declaration) @method
        (constructor_declaration) @constructor
        """,
    )

    cursor = QueryCursor(query)
    captures = cursor.captures(root_node)

    processed_ids = set()

    all_nodes = []
    if "method" in captures:
        all_nodes.extend(captures["method"])
    if "constructor" in captures:
        all_nodes.extend(captures["constructor"])

    logger.debug("Found %d potential nodes in %s", len(all_nodes), file_path)

    for node in all_nodes:
        if node.id in processed_ids:
            continue
        processed_ids.add(node.id)

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

    # 1. Try to get version from Version.java
    version_file = (
        repo_root / "core" / "src" / "main" / "java" / "com" / "google" / "adk" / "Version.java"
    )
    if version_file.exists():
        try:
            content = version_file.read_text()
            import re

            match = re.search(
                r'JAVA_ADK_VERSION\s*=\s*"([^"]+)"', content
            )
            if match:
                return match.group(1)
        except Exception as e:
            logger.warning("Failed to read or parse Version.java: %s", e)

    # 2. Fallback to pom.xml
    pom_xml = repo_root / "pom.xml"
    if pom_xml.exists():
        import xml.etree.ElementTree as ET

        try:
            tree = ET.parse(pom_xml)
            root = tree.getroot()
            # Handle XML namespace usually present in Maven POMs
            ns = {"mvn": "http://maven.apache.org/POM/4.0.0"}
            version_node = root.find("mvn:version", ns)
            if version_node is None:
                # Check parent version
                parent = root.find("mvn:parent", ns)
                if parent is not None:
                    version_node = parent.find("mvn:version", ns)

            if version_node is None:
                # Try without namespace
                version_node = root.find("version")

            if version_node is not None and version_node.text:
                version = version_node.text.strip()
                return version  # Return as soon as we find it
        except Exception as e:
            logger.warning("Failed to parse pom.xml for version: %s", e)

    # 3. Fallback to build.gradle / build.gradle.kts
    for gradle_file in ("build.gradle", "build.gradle.kts"):
        path = repo_root / gradle_file
        if path.exists():
            try:
                content = path.read_text()
                for line in content.splitlines():
                    if line.strip().startswith("version"):
                        import re

                        match = re.search(
                            r"""version\s*=?\s*['"]([^'"]+)['"]""", line
                        )
                        if match:
                            version = match.group(1)
                            return version  # Return as soon as we find it
            except Exception:
                pass

    return version
