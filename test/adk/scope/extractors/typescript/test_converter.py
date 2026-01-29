
import unittest
from unittest.mock import Mock
from pathlib import Path
from google.adk.scope.extractors.typescript.converter import NodeProcessor
from google.adk.scope import features_pb2 as feature_pb2

class TestNodeProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = NodeProcessor()
        self.repo_root = Path("/repo")
        self.file_path = Path("/repo/src/core/src/agent.ts")

    def create_mock_node(self, node_type, children=None, text=None, parent=None, prev_sibling=None):
        node = Mock()
        node.type = node_type
        node.text = text.encode('utf-8') if text else b""
        node.children = children or []
        node.parent = parent
        node.prev_sibling = prev_sibling
        
        def child_by_field_name_side_effect(name):
            if children:
                for child in children:
                    if getattr(child, 'field_name', None) == name:
                        return child
            return None
        node.child_by_field_name.side_effect = child_by_field_name_side_effect
        return node

    def test_process_jsdoc(self):
        # /**
        #  * My description.
        #  * @param p1 Param 1 description
        #  */
        # function foo(p1: string) {}
        
        jsdoc_text = "/**\n * My description.\n * @param p1 Param 1 description\n */"
        comment_node = self.create_mock_node("comment", text=jsdoc_text)
        
        name_node = self.create_mock_node("identifier", text="foo")
        name_node.field_name = "name"
        
        # Param
        p1_name = self.create_mock_node("identifier", text="p1")
        p1_type = self.create_mock_node("type_identifier", text="string")
        # In actual structure it might be more complex, but let's mock essential
        # required_parameter -> identifier (pattern), type_annotation
        # But extractor uses pattern logic.
        
        # Mocking process_param_node required children
        # required_parameter
        #   pattern: identifier (p1)
        #   type: type_annotation (: string)
        
        p1_node = self.create_mock_node("required_parameter")
        
        p1_patt = self.create_mock_node("identifier", text="p1")
        p1_patt.field_name = "pattern"
        
        p1_t_ann = self.create_mock_node("type_annotation", text=": string")
        p1_t_ann.field_name = "type"
        
        p1_node.children = [p1_patt, p1_t_ann]
        
        # Extractor logic for param extraction calls _process_param_node
        # which looks for child with field name pattern or type=identifier
        
        def p1_child(name):
            if name == 'pattern': return p1_patt
            if name == 'type': return p1_t_ann
            return None
        p1_node.child_by_field_name.side_effect = p1_child

        params_node = self.create_mock_node("parameters", children=[p1_node])
        params_node.field_name = "parameters"
        
        node = self.create_mock_node("function_declaration", children=[name_node, params_node], prev_sibling=comment_node)
        def node_child(name):
            if name == 'name': return name_node
            if name == 'parameters': return params_node
            return None
        node.child_by_field_name.side_effect = node_child
        
        # We need to successfully call process
        # We also need to mock _extract_member_of walking up (returns None -> member_of="")
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.description, "My description.")
        self.assertEqual(len(result.parameters), 1)
        self.assertEqual(result.parameters[0].original_name, "p1")
        self.assertEqual(result.parameters[0].description, "Param 1 description")

if __name__ == '__main__':
    unittest.main()
