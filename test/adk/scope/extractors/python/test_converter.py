
import unittest
from unittest.mock import Mock, MagicMock
from pathlib import Path
from google.adk.scope.extractors.python.converter import NodeProcessor
from google.adk.scope.extractors.python.types import Feature, Type, Maturity

class TestNodeProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = NodeProcessor()
        self.repo_root = Path("/repo")
        self.file_path = Path("/repo/src/google/adk/agent.py")
        
    def create_mock_node(self, node_type, children=None, text=None, parent=None):
        node = Mock()
        node.type = node_type
        node.text = text.encode('utf-8') if text else b""
        node.children = children or []
        node.parent = parent
        
        # Mock child_by_field_name
        def child_by_field_name_side_effect(name):
            if children:
                for child in children:
                    if getattr(child, 'field_name', None) == name:
                        return child
            return None
        node.child_by_field_name.side_effect = child_by_field_name_side_effect
        return node

    def test_process_non_function(self):
        node = self.create_mock_node("class_definition")
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertIsNone(result)

    def test_process_simple_function(self):
        # Construct a simple function node: def my_func(): pass
        name_node = self.create_mock_node("identifier", text="my_func")
        name_node.field_name = "name"
        
        node = self.create_mock_node(
            "function_definition",
            children=[name_node],
            text="def my_func(): pass"
        )
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.original_name, "my_func")
        self.assertEqual(result.normalized_name, "my_func")
        self.assertEqual(result.type, Type.FUNCTION)
        self.assertEqual(result.namespace, "google.adk") # Based on /repo/src/google/adk/agent.py (src stripped)
        
    def test_process_method(self):
        # Class definition -> Function definition
        class_name = self.create_mock_node("identifier", text="MyClass")
        class_name.field_name = "name"
        class_def = self.create_mock_node("class_definition", children=[class_name])
        
        method_name = self.create_mock_node("identifier", text="my_method")
        method_name.field_name = "name"
        
        # Method node
        node = self.create_mock_node(
            "function_definition",
            children=[method_name],
            parent=class_def
        )
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertIsNotNone(result)
        self.assertEqual(result.member_of, "MyClass")
        self.assertEqual(result.type, Type.INSTANCE_METHOD)

    def test_process_constructor(self):
        # Need to allow name extraction for class
        class_name = self.create_mock_node("identifier", text="MyClass")
        class_name.field_name = "name"
        class_def = self.create_mock_node("class_definition", children=[class_name])
        
        name_node = self.create_mock_node("identifier", text="__init__")
        name_node.field_name = "name"
        node = self.create_mock_node("function_definition", children=[name_node], parent=class_def)
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.original_name, "__init__")
        self.assertEqual(result.type, Type.CONSTRUCTOR)

    def test_parameters(self):
        # def func(a: int, b=2): ...
        
        # Param 1: a: int
        p1_name = self.create_mock_node("identifier", text="a")
        p1_type = self.create_mock_node("type", text="int")
        p1_node = self.create_mock_node("typed_parameter")
        # Mock child_by_field_name for p1
        def p1_child(name):
            if name == 'name': return p1_name
            if name == 'type': return p1_type
            return None
        p1_node.child_by_field_name.side_effect = p1_child
        p1_node.children = [p1_name, p1_type] # Fallback if logic uses children

        # Param 2: b=2
        p2_name = self.create_mock_node("identifier", text="b")
        p2_node = self.create_mock_node("default_parameter")
        def p2_child(name):
            if name == 'name': return p2_name
            return None
        p2_node.child_by_field_name.side_effect = p2_child
        p2_node.children = [p2_name]

        params_node = self.create_mock_node("parameters", children=[p1_node, p2_node])
        params_node.field_name = "parameters"
        
        name_node = self.create_mock_node("identifier", text="func")
        
        node = self.create_mock_node("function_definition", children=[params_node])
        def node_child(name):
            if name == 'name': return name_node
            if name == 'parameters': return params_node
            return None
        node.child_by_field_name.side_effect = node_child
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(len(result.parameters), 2)
        
        self.assertEqual(result.parameters[0].original_name, "a")
        self.assertEqual(result.parameters[0].normalized_types, ["INT"])
        self.assertFalse(result.parameters[0].is_optional)
        
        self.assertEqual(result.parameters[1].original_name, "b")
        self.assertTrue(result.parameters[1].is_optional)

if __name__ == '__main__':
    unittest.main()
