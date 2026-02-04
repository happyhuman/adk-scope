from google.protobuf import text_format
from google.adk.scope import features_pb2

def read_feature_registry(file_path: str) -> features_pb2.FeatureRegistry:
  """Reads a FeatureRegistry from a text proto file.

  Args:
    file_path: Path to the .txtpb file.

  Returns:
    A FeatureRegistry instance.
  """
  registry = features_pb2.FeatureRegistry()
  with open(file_path, 'rb') as f:
    text_format.Parse(f.read(), registry)
  return registry