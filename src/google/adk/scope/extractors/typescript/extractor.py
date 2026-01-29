
import logging
import pathlib
import sys
from typing import Iterator, List

from google.protobuf.json_format import MessageToJson
from google.adk.scope.features_pb2 import Feature, FeatureRegistry
from google.adk.scope.utils.args import parse_args
from google.adk.scope.extractors.typescript.converter import NodeProcessor

from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_typescript

# Initialize Tree-sitter
# tree-sitter-typescript has 'typescript' and 'tsx' dictionaries usually?
# Inspecting tree_sitter_typescript module commonly provides language_typescript()
try:
    TS_LANGUAGE = Language(tree_sitter_typescript.language_typescript())
except AttributeError:
    # Fallback or different version? 
    # newer bindings: tree_sitter_typescript.language() might be pure TS
    TS_LANGUAGE = Language(tree_sitter_typescript.language())

PARSER = Parser()
PARSER.language = TS_LANGUAGE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def find_ts_files(root: pathlib.Path, recursive: bool = True) -> Iterator[pathlib.Path]:
  if not root.exists():
    logger.warning("Directory %s does not exist. Skipping traversal.", root)
    return

  # Exclude d.ts? Maybe, usually definitions, not implementation.
  # Prompt says "target TypeScript files ... src ... .ts"
  
  if recursive:
      iterator = root.rglob("*.ts")
  else:
      iterator = root.glob("*.ts")

  for path in iterator:
    # Exclude .d.ts
    if path.name.endswith('.d.ts'):
        continue
        
    # Exclude node_modules, etc.
    if any(part == 'node_modules' or part.startswith('.') for part in path.parts if part not in ('.', '..')):
        continue

    yield path

def main():
  args = parse_args()

  logger.info("TS Extractor - Output: %s", args.output)

  all_features = []
  
  if args.input_file:
      input_path = args.input_file
      if not input_path.exists():
          logger.error("Input file '%s' does not exist.", input_path)
          sys.exit(1)
          
      logger.info("Mode: Single file extraction: %s", input_path)
      
      # Determine repo root
      repo_root = input_path.parent
      for parent in input_path.parents:
          # Check for package.json or tsconfig.json as marker
          if (parent / "package.json").exists() or (parent / "tsconfig.json").exists():
              repo_root = parent
              break
              
      features = extract_features(input_path, repo_root)
      all_features.extend(features)
      
      logger.info("File: %s - Found %d features", input_path.name, len(features))

  elif args.input_dir:
      input_path = args.input_dir
      if not input_path.exists():
          logger.error("Input directory '%s' does not exist.", input_path)
          sys.exit(1)
          
      logger.info("Mode: Directory extraction: %s", input_path)
      repo_root = input_path

      files = list(find_ts_files(input_path, recursive=False))
      logger.info("Found %d TypeScript files.", len(files))

      for p in files:
        features = extract_features(p, repo_root)
        all_features.extend(features)
        
  elif args.input_repo:
      input_path = args.input_repo
      if not input_path.exists():
          logger.error("Input repo '%s' does not exist.", input_path)
          sys.exit(1)

      logger.info("Mode: Repo extraction: %s", input_path)
      # User specified: repo path implies repo/core/src
      src_dir = input_path / "core" / "src"
      if not src_dir.exists():
           logger.warning("'core/src' directory not found in %s. Checking 'src' as fallback.", input_path)
           src_dir = input_path / "src"
      
      if not src_dir.exists():
           # Fallback to root or error?
           # Prompt said: "if the user uses ... XYZ, so the code should append /core/src"
           # I'll enforce it but allow fallback or just warn?
           # Let's try core/src, then src, then root? 
           # The user was specific: "should append /core/src".
           logger.warning("Could not find core/src or src. Scanning root %s", input_path)
           src_dir = input_path

      # We'll search everything under src_dir
      repo_root = input_path
      
      files = list(find_ts_files(src_dir, recursive=True))
      logger.info("Found %d TypeScript files in %s.", len(files), src_dir)

      for p in files:
        features = extract_features(p, repo_root)
        all_features.extend(features)
        
  else:
      logger.error("No input mode specified.")
      sys.exit(1)

  logger.info("Total features found: %d", len(all_features))

  # Version extraction? (Read package.json)
  version = "0.0.0"
  if args.input_repo or (args.input_file and repo_root):
      # Try to find package.json in repo_root
      rr = args.input_repo if args.input_repo else repo_root
      pkg_json = rr / "package.json"
      if pkg_json.exists():
          try:
              import json
              data = json.loads(pkg_json.read_text())
              version = data.get("version", "0.0.0")
          except Exception:
              pass

  registry = FeatureRegistry(
      language="TYPESCRIPT",
      version=version,
      features=all_features,
  )

  try:
    with open(args.output, "w") as f:
      f.write(MessageToJson(registry, indent=2, preserving_proto_field_name=True, always_print_fields_with_no_presence=True))
    logger.info("Successfully wrote output to %s", args.output)
  except IOError as e:
    logger.error("Failed to write output: %s", e)


def extract_features(
    file_path: pathlib.Path, repo_root: pathlib.Path
) -> List[Feature]:
  try:
    content = file_path.read_bytes()
  except IOError as e:
    logger.error("Failed to read %s: %s", file_path, e)
    return []

  tree = PARSER.parse(content)
  root_node = tree.root_node

  processor = NodeProcessor()
  features = []

  # Query for Class Declarations, Method Definitions, Function Declarations
  # We can traverse or use query.
  # Let's use recursive traversal or query for top-level + nested
  
  query = Query(TS_LANGUAGE, """
    (function_declaration) @func
    (method_definition) @method
  """)
  
  cursor = QueryCursor(query)
  captures = cursor.captures(root_node)

  # Process captures
  # Captures is { name: [nodes] }
  # We just iterate all nodes in order of appearance usually?
  # Or merge lists
  
  processed_ids = set()
  
  all_nodes = []
  if 'func' in captures:
      all_nodes.extend(captures['func'])
  if 'method' in captures:
      all_nodes.extend(captures['method'])
  
  # Sort by start byte to process in order? Not strictly necessary for output list
  
  for node in all_nodes:
      if node.id in processed_ids:
          continue
      processed_ids.add(node.id)
      
      feature = processor.process(node, file_path, repo_root)
      if feature:
          features.append(feature)

  return features

if __name__ == "__main__":
  main()
