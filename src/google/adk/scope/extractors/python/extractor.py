import logging
import pathlib
import sys
from typing import Iterator, List

from google.adk.scope.extractors.python.converter import NodeProcessor
from google.adk.scope.types import Feature
from google.adk.scope.types import FeatureRegistry
from google.adk.scope.types import to_json
from google.adk.scope.utils.args import parse_args
from tree_sitter import Language
from tree_sitter import Parser
import tree_sitter_python


SRC_DIR = "src"

# Initialize Tree-sitter
PY_LANGUAGE = Language(tree_sitter_python.language(), "python")
PARSER = Parser()
PARSER.set_language(PY_LANGUAGE)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_python_files(root: pathlib.Path) -> Iterator[pathlib.Path]:
  """Recursively find all Python files in the given directory.

  Args:
      root (Path): The root directory to search.

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

  for path in root.rglob("*.py"):
    if path.name in excluded_files:
      continue
    # Also exclude hidden files or directories starting with .
    if any(part.startswith(".") for part in path.parts):
      continue

    yield path


def extract_features(
    file_path: pathlib.Path, repo_root: pathlib.Path
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
  query = PY_LANGUAGE.query("""
    (function_definition
      name: (identifier) @name
    ) @func
    """)

  captures = query.captures(root_node)

  for node, tag in captures:
    if tag != "func":
      continue

    # The node is a function_definition
    feature = processor.process(node, file_path, repo_root)
    if feature:
      features.append(feature)

  return features


def main() -> None:
  """Main entry point for the extractor."""
  args = parse_args()

  if not args.adk_repo.exists():
    logger.error("Repository path '%s' does not exist.", args.adk_repo)
    sys.exit(1)

  adk_src_dir = args.adk_repo / SRC_DIR

  logger.info("Analyzing repo: %s", args.adk_repo)
  logger.info("Output will be saved to: %s", args.output)

  python_files = list(find_python_files(adk_src_dir))
  logger.info("Found %d Python files.", len(python_files))

  all_features = []
  for p in python_files:
    features = extract_features(p, args.adk_repo)
    all_features.extend(features)
    if features:
      logger.info(
          "File: %s - Found %d features",
          p.relative_to(args.adk_repo),
          len(features),
      )

  logger.info("Total features found: %d", len(all_features))

  # Create Registry
  registry = FeatureRegistry(
      language="PYTHON",
      version="0.0.0",  # TODO: Extract from version file?
      schema_version="1.0.0",
      features=all_features,
  )

  # Output to JSON using protobuf's json_format
  try:
    with open(args.output, "w") as f:
      f.write(to_json(registry, indent=2))
    logger.info("Successfully wrote output to %s", args.output)
  except IOError as e:
    logger.error("Failed to write output: %s", e)


if __name__ == "__main__":
  main()
