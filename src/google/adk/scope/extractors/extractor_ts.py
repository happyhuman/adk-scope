
import logging
import pathlib
from typing import Iterator, List

from google.adk.scope.features_pb2 import Feature
from google.adk.scope.extractors.converter_ts import NodeProcessor

from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_typescript as tsts

# Initialize Tree-sitter
try:
    TS_LANGUAGE = Language(tsts.language_typescript())
except AttributeError:
    TS_LANGUAGE = Language(tsts.language())

PARSER = Parser()
PARSER.language = TS_LANGUAGE

logger = logging.getLogger(__name__)

def find_files(root: pathlib.Path, recursive: bool = True) -> Iterator[pathlib.Path]:
  if not root.exists():
    logger.warning("Directory %s does not exist. Skipping traversal.", root)
    return

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
  query = Query(TS_LANGUAGE, """
    (function_declaration) @func
    (method_definition) @method
  """)
  
  cursor = QueryCursor(query)
  captures = cursor.captures(root_node)

  processed_ids = set()
  
  all_nodes = []
  if 'func' in captures:
      all_nodes.extend(captures['func'])
  if 'method' in captures:
      all_nodes.extend(captures['method'])
  
  for node in all_nodes:
      if node.id in processed_ids:
          continue
      processed_ids.add(node.id)
      
      feature = processor.process(node, file_path, repo_root)
      if feature:
          features.append(feature)

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
