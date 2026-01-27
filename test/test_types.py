
import json
import unittest

from google.adk.scope.types import Feature
from google.adk.scope.types import FeatureRegistry
from google.adk.scope.types import Maturity
from google.adk.scope.types import Param
from google.adk.scope.types import ParamType
from google.adk.scope.types import to_json
from google.adk.scope.types import Type


class JsonConversionTest(unittest.TestCase):

  def test_to_json_with_nested_dataclasses_and_enums(self):
    """Tests conversion of nested dataclasses and enums to JSON."""
    registry = FeatureRegistry(
        language="Python",
        version="1.1",
        features=[
            Feature(
                original_name="my_func",
                normalized_name="my_func",
                description="Does something",
                member_of="MyClass",
                normalized_member_of="myclass",
                maturity=Maturity.STABLE,
                type=Type.INSTANCE_METHOD,
                file_path="a/b.py",
                namespace="a.b",
                normalized_namespace="a.b",
                parameters=[
                    Param(
                        original_name="arg1",
                        normalized_name="arg1",
                        original_types=["int", "float"],
                        normalized_types=[ParamType.INT, ParamType.FLOAT],
                        description="An important arg",
                        is_optional=False,
                    ),
                    Param(
                        original_name="opt",
                        normalized_name="opt",
                        original_types=["str"],
                        normalized_types=[ParamType.STRING],
                        description="",
                        is_optional=True,
                    ),
                ],
                original_return_types=["bool"],
                normalized_return_types=[ParamType.BOOL],
                blocking=False,
            )
        ],
    )

    expected_dict = {
        "language": "Python",
        "version": "1.1",
        "features": [{
            "original_name": "my_func",
            "normalized_name": "my_func",
            "description": "Does something",
            "member_of": "MyClass",
            "normalized_member_of": "myclass",
            "maturity": "STABLE",
            "type": "INSTANCE_METHOD",
            "file_path": "a/b.py",
            "namespace": "a.b",
            "normalized_namespace": "a.b",
            "parameters": [
                {
                    "original_name": "arg1",
                    "normalized_name": "arg1",
                    "original_types": ["int", "float"],
                    "normalized_types": ["INT", "FLOAT"],
                    "description": "An important arg",
                    "is_optional": False,
                },
                {
                    "original_name": "opt",
                    "normalized_name": "opt",
                    "original_types": ["str"],
                    "normalized_types": ["STRING"],
                    "description": "",
                    "is_optional": True,
                },
            ],
            "original_return_types": ["bool"],
            "normalized_return_types": ["BOOL"],
            "blocking": False,
        }],
    }

    # Test without indentation
    json_output_no_indent = to_json(registry)
    self.assertEqual(json.loads(json_output_no_indent), expected_dict)

    # Test with indentation
    json_output_indent = to_json(registry, indent=2)
    self.assertEqual(json.loads(json_output_indent), expected_dict)
    # Check if indentation actually happened (simple check)
    self.assertIn("\n", json_output_indent)
    self.assertIn("  ", json_output_indent)


if __name__ == "__main__":
  unittest.main()
