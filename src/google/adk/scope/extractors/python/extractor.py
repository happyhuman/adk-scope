import logging
import pathlib
import sys
from typing import Iterator, List

from google.protobuf.json_format import MessageToJson
from google.adk.scope.extractors.python.converter import NodeProcessor
from google.adk.scope.features_pb2 import Feature, FeatureRegistry
from google.adk.scope.utils.args import parse_args
from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_python


SRC_DIR = "src"

# Initialize Tree-sitter
PY_LANGUAGE = Language(tree_sitter_python.language())
PARSER = Parser()
PARSER.language = PY_LANGUAGE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_python_files(root: pathlib.Path, recursive: bool = True) -> Iterator[pathlib.Path]:
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
      continue
    # Also exclude hidden files or directories starting with .
    # For recursive, we manually check parts. For glob, it shouldn't match .files unless explicitly capable?
    # pathlib glob might match .files if * matches them? usually * doesn't match .
    if any(part.startswith(".") and part not in (".", "..") for part in path.parts):
      continue

    yield path


def main():  # Output to JSON using protobuf's json_format
  """Main entry point for the extractor."""
  args = parse_args()

  logger.info("Output will be saved to: %s", args.output)

  all_features = []
  
  if args.input_file:
      input_path = args.input_file
      if not input_path.exists():
          logger.error("Input file '%s' does not exist.", input_path)
          sys.exit(1)
          
      logger.info("Mode: Single file extraction: %s", input_path)
      
      # Determine repo root (heuristic: look for 'src' in parents, or use parent dir)
      repo_root = input_path.parent
      for parent in input_path.parents:
          if (parent / "src").exists():
              repo_root = parent
              break
              
      features = extract_features(input_path, repo_root)
      all_features.extend(features)
      
      try:
          rel_path = input_path.relative_to(repo_root)
          logger.info("File: %s - Found %d features", rel_path, len(features))
      except ValueError:
          logger.info("File: %s - Found %d features", input_path.name, len(features))

  elif args.input_dir:
      input_path = args.input_dir
      if not input_path.exists():
          logger.error("Input directory '%s' does not exist.", input_path)
          sys.exit(1)
          
      logger.info("Mode: Directory extraction (non-recursive): %s", input_path)
      search_dir = input_path
      repo_root = input_path # Using dir as root since no other context known

      python_files = list(find_python_files(search_dir, recursive=False))
      logger.info("Found %d Python files in %s.", len(python_files), search_dir)

      for p in python_files:
        features = extract_features(p, repo_root)
        all_features.extend(features)
        if features:
          try:
              display_path = p.relative_to(input_path)
          except ValueError:
              display_path = p.name
          logger.info(
              "File: %s - Found %d features",
              display_path,
              len(features),
          )

  elif args.input_repo:
      input_path = args.input_repo
      if not input_path.exists():
          logger.error("Input repo '%s' does not exist.", input_path)
          sys.exit(1)

      logger.info("Mode: ADK Repo Root extraction (recursive in src): %s", input_path)
      src_dir = input_path / "src"
      if not src_dir.exists():
           logger.warning("'src' directory not found in %s. Skipping extraction.", input_path)
           # Or should we error out? User explicitly said "extract for any python file it finds in src"
           # Implies if no src, no extraction.
      else:
          search_dir = src_dir
          repo_root = input_path

          python_files = list(find_python_files(search_dir, recursive=True))
          logger.info("Found %d Python files in %s.", len(python_files), search_dir)

          for p in python_files:
            features = extract_features(p, repo_root)
            all_features.extend(features)
            if features:
              try:
                  display_path = p.relative_to(input_path)
              except ValueError:
                  display_path = p.name
              logger.info(
                  "File: %s - Found %d features",
                  display_path,
                  len(features),
              )
  else:
      # Should be unreachable due to required=True in argparse
      logger.error("No input mode specified.")
      sys.exit(1)

  logger.info("Total features found: %d", len(all_features))

  # Extract version if repo root can be determined
  version = "0.0.0"
  if args.input_repo:
      version_file = args.input_repo / "src" / "google" / "adk" / "version.py"
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

  # Create Registry
  registry = FeatureRegistry(
      language="PYTHON",
      version=version,
      features=all_features,
  )

  # Output to JSON using protobuf's json_format
  try:
    with open(args.output, "w") as f:
      f.write(MessageToJson(registry, indent=2, preserving_proto_field_name=True, always_print_fields_with_no_presence=True))
    logger.info("Successfully wrote output to %s", args.output)
  except IOError as e:
    logger.error("Failed to write output: %s", e)


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
  query = Query(PY_LANGUAGE, """
    (function_definition
      name: (identifier) @name
    ) @func
    """)
  
  cursor = QueryCursor(query)
  captures = cursor.captures(root_node)

  # captures is a dict {capture_name: [nodes]}
  for node in captures.get("func", []):
    # The node is a function_definition
    feature = processor.process(node, file_path, repo_root)
    if feature:
      features.append(feature)

  return features





if __name__ == "__main__":
  main()
