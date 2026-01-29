
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

    def test_process_class_methods(self):
        # class MyClass {
        #   constructor() {}
        #   myMethod() {}
        #   static myStatic() {}
        #   private myPrivate() {}
        #   get myProp() {}
        # }
        
        # We need to test one by one or mock walking up parent for member_of
        
        # 1. Instance Method
        class_name = self.create_mock_node("identifier", text="MyClass")
        class_name.field_name = "name"
        class_decl = self.create_mock_node("class_declaration", children=[class_name])
        
        method_name = self.create_mock_node("property_identifier", text="myMethod")
        method_name.field_name = "name"
        
        node = self.create_mock_node("method_definition", children=[method_name], parent=class_decl)
        def node_child(field_name):
            if field_name == 'name': return method_name
            return None
        node.child_by_field_name.side_effect = node_child
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertIsNotNone(result)
        self.assertEqual(result.member_of, "MyClass")
        self.assertEqual(result.type, feature_pb2.Feature.INSTANCE_METHOD)
        
        # 2. Static Method
        # (method_definition (modifiers (static)) name: ...)
        static_mod = self.create_mock_node("static", text="static")
        node_static = self.create_mock_node("method_definition", children=[static_mod, method_name], parent=class_decl)
        def node_static_child(field_name):
            if field_name == 'name': return method_name
            return None
        node_static.child_by_field_name.side_effect = node_static_child
        
        result = self.processor.process(node_static, self.file_path, self.repo_root)
        self.assertEqual(result.type, feature_pb2.Feature.CLASS_METHOD)
        
        # 3. Constructor
        # name is 'constructor'
        ctor_name = self.create_mock_node("property_identifier", text="constructor")
        ctor_name.field_name = "name"
        node_ctor = self.create_mock_node("method_definition", children=[ctor_name], parent=class_decl)
        def ctor_child(field_name):
            if field_name == 'name': return ctor_name
            return None
        node_ctor.child_by_field_name.side_effect = ctor_child
        
        result = self.processor.process(node_ctor, self.file_path, self.repo_root)
        self.assertEqual(result.type, feature_pb2.Feature.CONSTRUCTOR)

    def test_skip_private_and_accessors(self):
        # Private method by name
        name_p = self.create_mock_node("property_identifier", text="_private")
        name_p.field_name = "name"
        node_p = self.create_mock_node("method_definition", children=[name_p])
        def node_p_child(field_name):
            if field_name == 'name': return name_p
            return None
        node_p.child_by_field_name.side_effect = node_p_child
        
        self.assertIsNone(self.processor.process(node_p, self.file_path, self.repo_root))
        
        # Private by modifier
        name = self.create_mock_node("property_identifier", text="myPriv")
        name.field_name = "name"
        mod = self.create_mock_node("accessibility_modifier", text="private")
        node_mod = self.create_mock_node("method_definition", children=[mod, name])
        def node_mod_child(field_name):
            if field_name == 'name': return name
            return None
        node_mod.child_by_field_name.side_effect = node_mod_child
        
        self.assertIsNone(self.processor.process(node_mod, self.file_path, self.repo_root))
        
        # Getter
        # (method_definition "get" name: ...)
        get_kw = self.create_mock_node("get", text="get")
        node_get = self.create_mock_node("method_definition", children=[get_kw, name])
        node_get.child_by_field_name.side_effect = node_mod_child
        self.assertIsNone(self.processor.process(node_get, self.file_path, self.repo_root))

    def test_async_extraction(self):
        # async method
        name = self.create_mock_node("property_identifier", text="func")
        name.field_name = "name"
        
        # async keyword child
        async_kw = self.create_mock_node("async", text="async")
        
        node = self.create_mock_node("method_definition", children=[async_kw, name])
        def node_child(field_name):
            if field_name == 'name': return name
            return None
        node.child_by_field_name.side_effect = node_child
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertTrue(result.HasField("async"))
        self.assertTrue(getattr(result, "async"))

        # Promise return type -> async
        # (return_type (type_reference (identifier Promise) ...)) - simplified mock
        ret_type = self.create_mock_node("type_annotation", text="Promise<void>")
        ret_type.field_name = "return_type"
        node_prom = self.create_mock_node("method_definition", children=[name, ret_type])
        def node_prom_child(field_name):
            if field_name == 'name': return name
            if field_name == 'return_type': return ret_type
            return None
        node_prom.child_by_field_name.side_effect = node_prom_child
        
        result = self.processor.process(node_prom, self.file_path, self.repo_root)
        self.assertTrue(getattr(result, "async"))

    def test_parameter_destructuring_priority(self):
        # function f({ a }: MyType)
        # Should derive name from "MyType" -> "myType"
        
        # param node
        p_pattern = self.create_mock_node("object_pattern", text="{ a }")
        p_pattern.field_name = "pattern"
        
        p_type = self.create_mock_node("type_annotation", text=": MyRequest")
        p_type.field_name = "type"
        
        p_node = self.create_mock_node("required_parameter", children=[p_pattern, p_type])
        def p_child(field_name):
            if field_name == 'pattern': return p_pattern
            if field_name == 'type': return p_type
            return None
        p_node.child_by_field_name.side_effect = p_child
        
        params_node = self.create_mock_node("parameters", children=[p_node])
        params_node.field_name = "parameters"
        
        name = self.create_mock_node("identifier", text="f")
        name.field_name = "name"
        node = self.create_mock_node("function_declaration", children=[name, params_node])
        def node_child(field_name):
            if field_name == 'name': return name
            if field_name == 'parameters': return params_node
            return None
        node.child_by_field_name.side_effect = node_child
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.parameters[0].original_name, "request") # Derived from MyRequest

    def test_parameter_destructuring_fallback(self):
        # function f({ a, b }) 
        # No type -> flatten "{ a, b }"
        
        p_pattern = self.create_mock_node("object_pattern", text="{ a, b }")
        p_pattern.field_name = "pattern"
        
        p_node = self.create_mock_node("required_parameter", children=[p_pattern])
        def p_child(field_name):
            if field_name == 'pattern': return p_pattern
            return None
        p_node.child_by_field_name.side_effect = p_child
        
        params_node = self.create_mock_node("parameters", children=[p_node])
        params_node.field_name = "parameters"
        
        name = self.create_mock_node("identifier", text="f")
        name.field_name = "name"
        node = self.create_mock_node("function_declaration", children=[name, params_node])
        def node_child(field_name):
            if field_name == 'name': return name
            if field_name == 'parameters': return params_node
            return None
        node.child_by_field_name.side_effect = node_child
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.parameters[0].original_name, "{ a, b }")

    def test_complex_types_and_unwrapping(self):
        # Return type: Promise<string> -> STRING
        # Param type: string | number -> FLOAT (mapped from INT/FLOAT, unique? actually normalized list)
        
        name = self.create_mock_node("identifier", text="f")
        name.field_name = "name"
        
        ret_type = self.create_mock_node("type_annotation", text=": Promise<string>")
        ret_type.field_name = "return_type"
        
        # Param: p1: string | number
        p1_pattern = self.create_mock_node("identifier", text="p1")
        p1_pattern.field_name = "pattern"
        p1_type = self.create_mock_node("type_annotation", text=": string | number")
        p1_type.field_name = "type"
        
        p1_node = self.create_mock_node("required_parameter", children=[p1_pattern, p1_type])
        def p1_child(field_name):
            if field_name == 'pattern': return p1_pattern
            if field_name == 'type': return p1_type
            return None
        p1_node.child_by_field_name.side_effect = p1_child
        
        params_node = self.create_mock_node("parameters", children=[p1_node])
        params_node.field_name = "parameters"
        
        node = self.create_mock_node("function_declaration", children=[name, params_node, ret_type])
        def node_child(field_name):
            if field_name == 'name': return name
            if field_name == 'parameters': return params_node
            if field_name == 'return_type': return ret_type
            return None
        node.child_by_field_name.side_effect = node_child
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        
        # Verify return type unwrapping
        self.assertEqual(result.normalized_return_types, ["STRING"])
        
        # Verify param type union
        # string -> STRING, number -> FLOAT/INT (usually FLOAT/INT logic maps number to both? or one?)
        # _normalize_ts_type('number') -> ['FLOAT', 'INT']
        # _normalize_ts_type('string') -> ['STRING']
        # Combined -> STRING, FLOAT, INT
        self.assertIn(feature_pb2.ParamType.STRING, result.parameters[0].normalized_types)
        self.assertIn(feature_pb2.ParamType.INT, result.parameters[0].normalized_types)
        self.assertIn(feature_pb2.ParamType.FLOAT, result.parameters[0].normalized_types)


    def test_namespace_extraction(self):
        # Test _extract_namespace isolated
        # Case 1: core/src/foo/bar.ts -> foo.bar
        p1 = Path("/repo/src/core/src/foo/bar/file.ts")
        ns, norm = self.processor._extract_namespace(p1, Path("/repo/src"))
        self.assertEqual(ns, "foo.bar")
        self.assertEqual(norm, "foo_bar")
        
        # Case 2: src/foo.ts -> foo (if file is in src) OR if file is in src/foo?
        # Logic: rel_path.parent.parts
        # src/foo.ts -> parent is src -> parts=['src'] -> mapped to empty?
        # If parts=['src'] -> matches elif parts[0]=='src' -> parts=[] -> empty??
        # Let's check logic:
        # if parts[0] == 'src': parts = parts[1:]
        # So src/foo.ts -> ns=""?
        # Usually namespace is package structure.
        p2 = Path("/repo/src/foo.ts")
        ns, norm = self.processor._extract_namespace(p2, Path("/repo"))
        # rel: src/foo.ts. parent: src. parts ('src',).
        # parts[0]=='src' -> parts[1:] -> [] -> ""
        self.assertEqual(ns, "")
        
        # Case 3: lib/utils.ts
        p3 = Path("/repo/lib/utils/helper.ts")
        ns, norm = self.processor._extract_namespace(p3, Path("/repo"))
        # rel: lib/utils/helper.ts. parent: lib/utils.
        # parts[0]=='lib' -> parts[1:] -> ['utils']
        self.assertEqual(ns, "utils")

    def test_maturity_decorators(self):
        # @beta
        deco = self.create_mock_node("decorator", text="@beta")
        name = self.create_mock_node("identifier", text="f")
        name.field_name = "name"
        node = self.create_mock_node("function_declaration", children=[deco, name])
        def node_child(field_name):
            if field_name == 'name': return name
            return None
        node.child_by_field_name.side_effect = node_child
        
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertTrue(result.HasField("maturity"))
        self.assertEqual(result.maturity, feature_pb2.Feature.Maturity.BETA)

if __name__ == '__main__':
    unittest.main()
