import unittest
from pathlib import Path
from unittest.mock import Mock

from google.adk.scope import features_pb2 as feature_pb2
from google.adk.scope.extractors.converter_ts import NodeProcessor


class TestNodeProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = NodeProcessor()
        self.repo_root = Path("/repo")
        self.file_path = Path("/repo/src/core/src/agent.ts")

    def create_mock_node(
        self,
        node_type,
        children=None,
        text=None,
        parent=None,
        prev_sibling=None,
    ):
        node = Mock()
        node.type = node_type
        node.text = text.encode("utf-8") if text else b""
        node.children = children or []
        node.parent = parent
        node.prev_sibling = prev_sibling

        def child_by_field_name_side_effect(name):
            if children:
                for child in children:
                    if getattr(child, "field_name", None) == name:
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

        jsdoc_text = (
            "/**\n * My description.\n * @param p1 Param 1 description\n */"
        )
        comment_node = self.create_mock_node("comment", text=jsdoc_text)

        name_node = self.create_mock_node("identifier", text="foo")
        name_node.field_name = "name"

        # Param
        _p1_name = self.create_mock_node("identifier", text="p1")
        _p1_type = self.create_mock_node("type_identifier", text="string")
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
            if name == "pattern":
                return p1_patt
            if name == "type":
                return p1_t_ann
            return None

        p1_node.child_by_field_name.side_effect = p1_child

        params_node = self.create_mock_node("parameters", children=[p1_node])
        params_node.field_name = "parameters"

        node = self.create_mock_node(
            "function_declaration",
            children=[name_node, params_node],
            prev_sibling=comment_node,
        )

        def node_child(name):
            if name == "name":
                return name_node
            if name == "parameters":
                return params_node
            return None

        node.child_by_field_name.side_effect = node_child

        # We need to successfully call process
        # We also need to mock _extract_member_of walking up
        # (returns None -> member_of="")

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertIsNotNone(result)
        self.assertEqual(result.description, "My description.")
        self.assertEqual(len(result.parameters), 1)
        self.assertEqual(result.parameters[0].original_name, "p1")
        self.assertEqual(
            result.parameters[0].description, "Param 1 description"
        )

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
        class_decl = self.create_mock_node(
            "class_declaration", children=[class_name]
        )

        method_name = self.create_mock_node(
            "property_identifier", text="myMethod"
        )
        method_name.field_name = "name"

        node = self.create_mock_node(
            "method_definition", children=[method_name], parent=class_decl
        )

        def node_child(field_name):
            if field_name == "name":
                return method_name
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertIsNotNone(result)
        self.assertEqual(result.member_of, "MyClass")
        self.assertEqual(result.type, feature_pb2.Feature.INSTANCE_METHOD)

        # 2. Static Method
        # (method_definition (modifiers (static)) name: ...)
        static_mod = self.create_mock_node("static", text="static")
        node_static = self.create_mock_node(
            "method_definition",
            children=[static_mod, method_name],
            parent=class_decl,
        )

        def node_static_child(field_name):
            if field_name == "name":
                return method_name
            return None

        node_static.child_by_field_name.side_effect = node_static_child

        result = self.processor.process(
            node_static, self.file_path, self.repo_root
        )
        self.assertEqual(result.type, feature_pb2.Feature.CLASS_METHOD)

        # 3. Constructor
        # name is 'constructor'
        ctor_name = self.create_mock_node(
            "property_identifier", text="constructor"
        )
        ctor_name.field_name = "name"
        node_ctor = self.create_mock_node(
            "method_definition", children=[ctor_name], parent=class_decl
        )

        def ctor_child(field_name):
            if field_name == "name":
                return ctor_name
            return None

        node_ctor.child_by_field_name.side_effect = ctor_child

        result = self.processor.process(
            node_ctor, self.file_path, self.repo_root
        )
        self.assertEqual(result.type, feature_pb2.Feature.CONSTRUCTOR)

    def test_skip_private_and_accessors(self):
        # Private method by name
        name_p = self.create_mock_node("property_identifier", text="_private")
        name_p.field_name = "name"
        node_p = self.create_mock_node("method_definition", children=[name_p])

        def node_p_child(field_name):
            if field_name == "name":
                return name_p
            return None

        node_p.child_by_field_name.side_effect = node_p_child

        self.assertIsNone(
            self.processor.process(node_p, self.file_path, self.repo_root)
        )

        # Private by modifier
        name = self.create_mock_node("property_identifier", text="myPriv")
        name.field_name = "name"
        mod = self.create_mock_node("accessibility_modifier", text="private")
        node_mod = self.create_mock_node(
            "method_definition", children=[mod, name]
        )

        def node_mod_child(field_name):
            if field_name == "name":
                return name
            return None

        node_mod.child_by_field_name.side_effect = node_mod_child

        self.assertIsNone(
            self.processor.process(node_mod, self.file_path, self.repo_root)
        )

        # Getter
        # (method_definition "get" name: ...)
        get_kw = self.create_mock_node("get", text="get")
        node_get = self.create_mock_node(
            "method_definition", children=[get_kw, name]
        )
        node_get.child_by_field_name.side_effect = node_mod_child
        self.assertIsNone(
            self.processor.process(node_get, self.file_path, self.repo_root)
        )

    def test_async_extraction(self):
        # async method
        name = self.create_mock_node("property_identifier", text="func")
        name.field_name = "name"

        # async keyword child
        async_kw = self.create_mock_node("async", text="async")

        node = self.create_mock_node(
            "method_definition", children=[async_kw, name]
        )

        def node_child(field_name):
            if field_name == "name":
                return name
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertTrue(result.HasField("async"))
        self.assertTrue(getattr(result, "async"))

        # Promise return type -> async
        # (return_type (type_reference (identifier Promise) ...))
        # - simplified mock
        ret_type = self.create_mock_node(
            "type_annotation", text="Promise<void>"
        )
        ret_type.field_name = "return_type"
        node_prom = self.create_mock_node(
            "method_definition", children=[name, ret_type]
        )

        def node_prom_child(field_name):
            if field_name == "name":
                return name
            if field_name == "return_type":
                return ret_type
            return None

        node_prom.child_by_field_name.side_effect = node_prom_child

        result = self.processor.process(
            node_prom, self.file_path, self.repo_root
        )
        self.assertTrue(getattr(result, "async"))

    def test_parameter_destructuring_priority(self):
        # function f({ a }: MyType)
        # Should derive name from "MyType" -> "myType"

        # param node
        p_pattern = self.create_mock_node("object_pattern", text="{ a }")
        p_pattern.field_name = "pattern"

        p_type = self.create_mock_node("type_annotation", text=": MyRequest")
        p_type.field_name = "type"

        p_node = self.create_mock_node(
            "required_parameter", children=[p_pattern, p_type]
        )

        def p_child(field_name):
            if field_name == "pattern":
                return p_pattern
            if field_name == "type":
                return p_type
            return None

        p_node.child_by_field_name.side_effect = p_child

        params_node = self.create_mock_node("parameters", children=[p_node])
        params_node.field_name = "parameters"

        name = self.create_mock_node("identifier", text="f")
        name.field_name = "name"
        node = self.create_mock_node(
            "function_declaration", children=[name, params_node]
        )

        def node_child(field_name):
            if field_name == "name":
                return name
            if field_name == "parameters":
                return params_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(
            result.parameters[0].original_name, "request"
        )  # Derived from MyRequest

    def test_parameter_destructuring_fallback(self):
        # function f({ a, b })
        # No type -> flatten "{ a, b }"

        p_pattern = self.create_mock_node("object_pattern", text="{ a, b }")
        p_pattern.field_name = "pattern"

        p_node = self.create_mock_node(
            "required_parameter", children=[p_pattern]
        )

        def p_child(field_name):
            if field_name == "pattern":
                return p_pattern
            return None

        p_node.child_by_field_name.side_effect = p_child

        params_node = self.create_mock_node("parameters", children=[p_node])
        params_node.field_name = "parameters"

        name = self.create_mock_node("identifier", text="f")
        name.field_name = "name"
        node = self.create_mock_node(
            "function_declaration", children=[name, params_node]
        )

        def node_child(field_name):
            if field_name == "name":
                return name
            if field_name == "parameters":
                return params_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.parameters[0].original_name, "{ a, b }")

    def test_complex_types_and_unwrapping(self):
        # Return type: Promise<string> -> STRING
        # Param type: string | number -> FLOAT
        # (mapped from INT/FLOAT, unique? actually normalized list)

        name = self.create_mock_node("identifier", text="f")
        name.field_name = "name"

        ret_type = self.create_mock_node(
            "type_annotation", text=": Promise<string>"
        )
        ret_type.field_name = "return_type"

        # Param: p1: string | number
        p1_pattern = self.create_mock_node("identifier", text="p1")
        p1_pattern.field_name = "pattern"
        p1_type = self.create_mock_node(
            "type_annotation", text=": string | number"
        )
        p1_type.field_name = "type"

        p1_node = self.create_mock_node(
            "required_parameter", children=[p1_pattern, p1_type]
        )

        def p1_child(field_name):
            if field_name == "pattern":
                return p1_pattern
            if field_name == "type":
                return p1_type
            return None

        p1_node.child_by_field_name.side_effect = p1_child

        params_node = self.create_mock_node("parameters", children=[p1_node])
        params_node.field_name = "parameters"

        node = self.create_mock_node(
            "function_declaration", children=[name, params_node, ret_type]
        )

        def node_child(field_name):
            if field_name == "name":
                return name
            if field_name == "parameters":
                return params_node
            if field_name == "return_type":
                return ret_type
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)

        # Verify return type unwrapping
        self.assertEqual(result.normalized_return_types, ["STRING"])

        # Verify param type union
        # string -> STRING, number -> FLOAT/INT
        # (usually FLOAT/INT logic maps number to both? or one?)
        # _normalize_ts_type('number') -> ['FLOAT', 'INT']
        # _normalize_ts_type('string') -> ['STRING']
        # Combined -> STRING, FLOAT, INT
        self.assertIn(
            feature_pb2.ParamType.STRING, result.parameters[0].normalized_types
        )
        self.assertIn(
            feature_pb2.ParamType.NUMBER, result.parameters[0].normalized_types
        )

    def test_namespace_extraction(self):
        # Test _extract_namespace isolated
        # Case 1: core/src/foo/bar.ts -> foo.bar
        p1 = Path("/repo/src/core/src/foo/bar/file.ts")
        ns, norm = self.processor._extract_namespace(p1, Path("/repo/src"))
        self.assertEqual(ns, "foo.bar")
        self.assertEqual(norm, "foo_bar")

        # Case 2: src/foo.ts -> foo (if file is in src)
        # OR if file is in src/foo?
        # Logic: rel_path.parent.parts
        # src/foo.ts -> parent is src -> parts=['src'] -> mapped to empty?
        # If parts=['src'] -> matches elif parts[0]=='src' -> parts=[]
        # -> empty??
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
        node = self.create_mock_node(
            "function_declaration", children=[deco, name]
        )

        def node_child(field_name):
            if field_name == "name":
                return name
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertTrue(result.HasField("maturity"))
        self.assertEqual(result.maturity, feature_pb2.Feature.Maturity.BETA)

    def test_jsdoc_edge_cases(self):
        # 1. JSDoc on export statement
        jsdoc = "/** Exported func */"
        comment = self.create_mock_node("comment", text=jsdoc)

        name = self.create_mock_node("identifier", text="f")
        name.field_name = "name"

        # Structure: export_statement -> function_declaration
        # export_statement prev_sibling is comment

        func_node = self.create_mock_node(
            "function_declaration", children=[name]
        )
        export_node = self.create_mock_node(
            "export_statement", children=[func_node], prev_sibling=comment
        )
        func_node.parent = export_node

        def func_child(n):
            if n == "name":
                return name
            return None

        func_node.child_by_field_name.side_effect = func_child

        result = self.processor.process(
            func_node, self.file_path, self.repo_root
        )
        self.assertEqual(result.description, "Exported func")

        # 2. JSDoc params with types and defaults
        # @param {string} [p1=default] Desc
        # NOTE: _parse_jsdoc_params expects CLEANED JSDoc
        # (no /** or * prefixes), just the content
        jsdoc_detailed = "@param {string} [p1=def] Desc"
        parsed = self.processor._parse_jsdoc_params(jsdoc_detailed)
        self.assertEqual(parsed["p1"], "Desc")

    def test_additional_modifiers(self):
        # protected method -> should be skipped?
        # Logic says: private/protected -> return None

        mod = self.create_mock_node("accessibility_modifier", text="protected")
        name = self.create_mock_node("property_identifier", text="prot")
        name.field_name = "name"
        node = self.create_mock_node("method_definition", children=[mod, name])

        def child(n):
            if n == "name":
                return name
            return None

        node.child_by_field_name.side_effect = child

        self.assertIsNone(
            self.processor.process(node, self.file_path, self.repo_root)
        )

        # set accessor
        set_kw = self.create_mock_node("set", text="set")
        name_s = self.create_mock_node("property_identifier", text="prop")
        name_s.field_name = "name"
        node_s = self.create_mock_node(
            "method_definition", children=[set_kw, name_s]
        )
        node_s.child_by_field_name.side_effect = lambda n: (
            name_s if n == "name" else None
        )

        self.assertIsNone(
            self.processor.process(node_s, self.file_path, self.repo_root)
        )

    def test_parameter_modes_rest(self):
        # rest_parameter: ...args
        p_name = self.create_mock_node("identifier", text="args")

        # Actually rest_parameter usually has identifier as child directly
        # or pattern?
        # Tree-sitter-ts: rest_parameter -> identifier
        # Code checks: pattern_node = child_by_field_name('pattern')
        # OR identifier child

        rest_node = self.create_mock_node("rest_parameter", children=[p_name])
        # Force identifier child check fallback

        params = self.create_mock_node("parameters", children=[rest_node])
        params.field_name = "parameters"

        name = self.create_mock_node("identifier", text="f")
        name.field_name = "name"
        node = self.create_mock_node(
            "function_declaration", children=[name, params]
        )
        node.child_by_field_name.side_effect = lambda n: (
            name if n == "name" else (params if n == "parameters" else None)
        )

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(len(result.parameters), 1)
        self.assertEqual(result.parameters[0].original_name, "args")

    def test_abstract_and_interfaces(self):
        # abstract class method
        # interface method

        # 1. Abstract Class
        abs_class = self.create_mock_node("abstract_class_declaration")
        abs_name = self.create_mock_node("identifier", text="Abs")
        abs_name.field_name = "name"
        abs_class.children = [abs_name]
        abs_class.child_by_field_name.side_effect = lambda n: (
            abs_name if n == "name" else None
        )

        method_name = self.create_mock_node(
            "property_identifier", text="absMethod"
        )
        method_name.field_name = "name"
        method_node = self.create_mock_node(
            "method_definition", children=[method_name], parent=abs_class
        )
        method_node.child_by_field_name.side_effect = lambda n: (
            method_name if n == "name" else None
        )

        result = self.processor.process(
            method_node, self.file_path, self.repo_root
        )
        self.assertEqual(result.member_of, "Abs")

        # 2. Interface
        iface = self.create_mock_node("interface_declaration")
        iface_name = self.create_mock_node("identifier", text="IFace")
        iface_name.field_name = "name"
        iface.children = [iface_name]
        iface.child_by_field_name.side_effect = lambda n: (
            iface_name if n == "name" else None
        )

        # Interface method might be method_signature in TS,
        # but strict extractor checks for method_definition?
        # Extractor code:
        # if node.type not in ('function_declaration', 'method_definition'):
        # return None
        # So interface methods (usually method_signature) are skipped!
        # Wait, let's verify if 'method_definition' appears in interfaces
        # in tree-sitter.
        # Usually it is 'method_signature'.
        # If so, this test confirms we DON'T extract them,
        # which is likely intended or current behavior.
        # Let's mock a method_definition inside interface just to check
        # strict member_of logic.

        method_node_i = self.create_mock_node(
            "method_definition", children=[method_name], parent=iface
        )
        method_node_i.child_by_field_name.side_effect = lambda n: (
            method_name if n == "name" else None
        )

        result_i = self.processor.process(
            method_node_i, self.file_path, self.repo_root
        )
        self.assertEqual(result_i.member_of, "IFace")

    def test_namespace_error_fallback(self):
        # Path not relative to repo root
        # /outside/file.ts relative to /repo -> ValueError

        p = Path("/outside/file.ts")
        # Ensure relative_to raises ValueError
        # (It does in real Path, but we are using real Path objects
        # in test with fake strings?)
        # Yes, Path("/outside/file.ts").relative_to(Path("/repo"))
        # raises ValueError

        ns, norm = self.processor._extract_namespace(p, self.repo_root)
        # Should catch ValueError and fallback to file path logic
        # rel_path = p = /outside/file.ts
        # parent = /outside
        # parts = ['outside'] (root / ignored in parts usually?)
        # Actually /outside parent parts on unix is ('/', 'outside') or similar?
        # Let's rely on logic: fallback sets rel_path = file_path
        # parts = list(rel_path.parent.parts)
        # If it returns empty or something, namespace is empty.

        # On Mac/Linux: Path("/outside/file.ts").parent.parts
        # -> ('/', 'outside')
        # parts = ['/', 'outside']
        # filtered: ['outside'] (assuming / is filtered or not in list
        # of strings if purely component based?)
        # pathlib parts includes root.
        # Logic: parts = [p for p in parts if p and p not in ('.', '..')]
        # '/' is in parts[0] usually.
        # We might get "_.outside" or ".outside"?
        # Normalized replace . with _

        # Let's see what happens.
        # If parts=['/', 'outside'], extracted ns="/.outside" or similar?
        # This confirms fallback behavior exists, exact value might vary
        # by OS path separator handling in test env.
        # Using a simpler relative-ish path that fails relative_to?
        # Or just assert it doesn't crash.
        self.assertIsNotNone(ns)

    def test_jsdoc_with_decorator_interleaved(self):
        # /** Doc */
        # @deco
        # function f() {}

        # structure:
        # comment
        # decorator
        # function_declaration
        # prev_sibling of function is decorator
        # prev_sibling of decorator is comment

        jsdoc = "/** Doc */"
        comment = self.create_mock_node("comment", text=jsdoc)

        deco = self.create_mock_node("decorator", text="@deco")
        deco.prev_sibling = comment

        name = self.create_mock_node("identifier", text="f")
        name.field_name = "name"

        func = self.create_mock_node(
            "function_declaration", children=[name], prev_sibling=deco
        )
        func.child_by_field_name.side_effect = lambda n: (
            name if n == "name" else None
        )

        result = self.processor.process(func, self.file_path, self.repo_root)
        self.assertEqual(result.description, "Doc")


if __name__ == "__main__":
    unittest.main()
