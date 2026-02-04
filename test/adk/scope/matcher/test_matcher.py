
import os
import tempfile
import unittest
from google.adk.scope import features_pb2
from google.adk.scope.matcher import matcher

class TestMatcher(unittest.TestCase):

  def test_read_feature_registry(self):
    content = """
    language: "PYTHON"
    version: "1.0.0"
    features {
      original_name: "test_feature"
      normalized_name: "test_feature"
      type: FUNCTION
    }
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txtpb', delete=False) as f:
      f.write(content)
      temp_path = f.name
    
    try:
      registry = matcher.read_feature_registry(temp_path)
      self.assertEqual(registry.language, "PYTHON")
      self.assertEqual(registry.version, "1.0.0")
      self.assertEqual(len(registry.features), 1)
      self.assertEqual(registry.features[0].original_name, "test_feature")
      self.assertEqual(registry.features[0].type, features_pb2.Feature.Type.FUNCTION)
    finally:
      os.remove(temp_path)

if __name__ == '__main__':
  unittest.main()
