import unittest
from pathlib import Path
from unittest.mock import Mock

from google.adk.scope.extractors.converter_go import NodeProcessor
from google.adk.scope.features_pb2 import Feature


class TestNodeProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = NodeProcessor()
        self.repo_root = Path("/repo")
        self.file_path = Path("/repo/src/google/adk/agent.go")

    def create_mock_node(
        self, node_type, children=None, text=None, parent=None
    ):
        node = Mock()
        node.type = node_type
        node.text = text.encode("utf-8") if text else b""
        node.children = children or []
        node.parent = parent

        # Mock child_by_field_name
        def child_by_field_name_side_effect(name):
            if children:
                for child in children:
                    if getattr(child, "field_name", None) == name:
                        return child
            return None

        node.child_by_field_name.side_effect = child_by_field_name_side_effect
        return node

    def test_process_simple_function(self):
        # Construct a simple function node: func my_func() {}
        name_node = self.create_mock_node("identifier", text="my_func")
        name_node.field_name = "name"

        node = self.create_mock_node(
            "function_declaration",
            children=[name_node],
            text="func my_func() {}",
        )

        result = self.processor.process(node, self.file_path, self.repo_root)

        self.assertIsNotNone(result)
        self.assertEqual(result.original_name, "my_func")
        self.assertEqual(result.normalized_name, "my_func")
        self.assertEqual(result.type, Feature.Type.FUNCTION)

    def test_parameters(self):
        # func my_func(a int, b string) {}
        p1_name = self.create_mock_node("identifier", text="a")
        p1_name.field_name = "name"
        p1_type = self.create_mock_node("type_identifier", text="int")
        p1_type.field_name = "type"
        p1_node = self.create_mock_node(
            "parameter_declaration", children=[p1_name, p1_type]
        )

        p2_name = self.create_mock_node("identifier", text="b")
        p2_name.field_name = "name"
        p2_type = self.create_mock_node("type_identifier", text="string")
        p2_type.field_name = "type"
        p2_node = self.create_mock_node(
            "parameter_declaration", children=[p2_name, p2_type]
        )

        params_node = self.create_mock_node(
            "parameter_list", children=[p1_node, p2_node]
        )
        params_node.field_name = "parameters"

        name_node = self.create_mock_node("identifier", text="my_func")
        name_node.field_name = "name"

        node = self.create_mock_node(
            "function_declaration",
            children=[name_node, params_node],
        )

        result = self.processor.process(node, self.file_path, self.repo_root)

        self.assertEqual(len(result.parameters), 2)
        self.assertEqual(result.parameters[0].original_name, "a")
        self.assertEqual(result.parameters[0].original_types, ["int"])
        self.assertEqual(result.parameters[1].original_name, "b")
        self.assertEqual(result.parameters[1].original_types, ["string"])

    def test_return_types(self):
        # func my_func() string {}
        name_node = self.create_mock_node("identifier", text="my_func")
        name_node.field_name = "name"

        return_type_node = self.create_mock_node("type_identifier", text="string")
        return_type_node.field_name = "result"

        node = self.create_mock_node(
            "function_declaration",
            children=[name_node, return_type_node],
        )

        result = self.processor.process(node, self.file_path, self.repo_root)

        self.assertEqual(result.original_return_types, ["string"])


if __name__ == "__main__":
    unittest.main()
