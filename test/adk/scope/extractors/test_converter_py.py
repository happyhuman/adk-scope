import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from google.adk.scope import features_pb2 as feature_pb2
from google.adk.scope.extractors.converter_py import NodeProcessor
from google.adk.scope.features_pb2 import Feature


class TestNodeProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = NodeProcessor()
        self.repo_root = Path("/repo")
        self.file_path = Path("/repo/src/google/adk/agent.py")

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
            text="def my_func(): pass",
        )

        result = self.processor.process(node, self.file_path, self.repo_root)

        self.assertIsNotNone(result)
        self.assertEqual(result.original_name, "my_func")
        self.assertEqual(result.normalized_name, "my_func")
        self.assertEqual(result.type, Feature.Type.FUNCTION)
        self.assertEqual(
            result.namespace, "agent"
        )  # Based on /repo/src/google/adk/agent.py
        # (src and google.adk stripped, fallback to agent)

    def test_process_init_file(self):
        # __init__.py should have empty namespace if at root of adk
        init_path = Path("/repo/src/google/adk/__init__.py")

        name_node = self.create_mock_node("identifier", text="my_func")
        name_node.field_name = "name"
        node = self.create_mock_node(
            "function_definition",
            children=[name_node],
            text="def my_func(): pass",
        )

        result = self.processor.process(node, init_path, self.repo_root)
        self.assertIsNotNone(result)
        self.assertEqual(result.namespace, "")

    def test_process_method(self):
        # Class definition -> Function definition
        class_name = self.create_mock_node("identifier", text="MyClass")
        class_name.field_name = "name"
        class_def = self.create_mock_node(
            "class_definition", children=[class_name]
        )

        method_name = self.create_mock_node("identifier", text="my_method")
        method_name.field_name = "name"

        # Method node
        node = self.create_mock_node(
            "function_definition", children=[method_name], parent=class_def
        )

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertIsNotNone(result)
        self.assertEqual(result.member_of, "MyClass")
        self.assertEqual(result.type, Feature.Type.INSTANCE_METHOD)

    def test_process_constructor(self):
        # Need to allow name extraction for class
        class_name = self.create_mock_node("identifier", text="MyClass")
        class_name.field_name = "name"
        class_def = self.create_mock_node(
            "class_definition", children=[class_name]
        )

        name_node = self.create_mock_node("identifier", text="__init__")
        name_node.field_name = "name"
        node = self.create_mock_node(
            "function_definition", children=[name_node], parent=class_def
        )

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.original_name, "__init__")
        self.assertEqual(result.type, Feature.Type.CONSTRUCTOR)

    def test_parameters(self):
        # def func(a: int, b=2): ...

        # Param 1: a: int
        p1_name = self.create_mock_node("identifier", text="a")
        p1_type = self.create_mock_node("type", text="int")
        p1_node = self.create_mock_node("typed_parameter")

        # Mock child_by_field_name for p1
        def p1_child(name):
            if name == "name":
                return p1_name
            if name == "type":
                return p1_type
            return None

        p1_node.child_by_field_name.side_effect = p1_child
        p1_node.children = [p1_name, p1_type]  # Fallback if logic uses children

        # Param 2: b=2
        p2_name = self.create_mock_node("identifier", text="b")
        p2_node = self.create_mock_node("default_parameter")

        def p2_child(name):
            if name == "name":
                return p2_name
            return None

        p2_node.child_by_field_name.side_effect = p2_child
        p2_node.children = [p2_name]

        params_node = self.create_mock_node(
            "parameters", children=[p1_node, p2_node]
        )
        params_node.field_name = "parameters"

        name_node = self.create_mock_node("identifier", text="func")

        node = self.create_mock_node(
            "function_definition", children=[params_node]
        )

        def node_child(name):
            if name == "name":
                return name_node
            if name == "parameters":
                return params_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(len(result.parameters), 2)

        self.assertEqual(result.parameters[0].original_name, "a")
        self.assertEqual(
            result.parameters[0].normalized_types,
            [feature_pb2.ParamType.NUMBER],
        )
        self.assertFalse(result.parameters[0].is_optional)

        self.assertEqual(result.parameters[1].original_name, "b")
        self.assertTrue(result.parameters[1].is_optional)

    def test_return_types(self):
        # def func() -> str: ...
        name_node = self.create_mock_node("identifier", text="func")
        name_node.field_name = "name"

        return_type = self.create_mock_node("type", text="str")
        return_type.field_name = "return_type"

        node = self.create_mock_node(
            "function_definition", children=[name_node, return_type]
        )

        def node_child(name):
            if name == "name":
                return name_node
            if name == "return_type":
                return return_type
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.original_return_types, ["str"])
        self.assertEqual(result.normalized_return_types, ["STRING"])

    def test_decorators_classmethod(self):
        # @classmethod
        # def func(cls): ...

        # decorated_definition -> decorator, function_definition

        class_name = self.create_mock_node("identifier", text="MyClass")
        class_name.field_name = "name"
        class_def = self.create_mock_node(
            "class_definition", children=[class_name]
        )

        decorator_node = self.create_mock_node("decorator", text="@classmethod")

        func_name = self.create_mock_node("identifier", text="func")
        func_name.field_name = "name"

        func_node = self.create_mock_node(
            "function_definition", children=[func_name]
        )
        # IMPORTANT: parent of func_node is decorated_def

        decorated_def = self.create_mock_node(
            "decorated_definition",
            children=[decorator_node, func_node],
            parent=class_def,
        )
        func_node.parent = decorated_def

        def func_node_child(name):
            if name == "name":
                return func_name
            return None

        func_node.child_by_field_name.side_effect = func_node_child

        result = self.processor.process(
            func_node, self.file_path, self.repo_root
        )

        self.assertEqual(result.member_of, "MyClass")
        self.assertEqual(result.type, Feature.Type.CLASS_METHOD)

    def test_decorators_maturity(self):
        # @deprecated
        # def func(): ...

        # Helper to create decorated node
        def create_decorated(deco_text):
            decorator_node = self.create_mock_node("decorator", text=deco_text)
            func_name = self.create_mock_node("identifier", text="func")
            func_name.field_name = "name"
            func_node = self.create_mock_node(
                "function_definition", children=[func_name]
            )
            decorated_def = self.create_mock_node(
                "decorated_definition", children=[decorator_node, func_node]
            )
            func_node.parent = decorated_def

            def func_node_child(name):
                if name == "name":
                    return func_name
                return None

            func_node.child_by_field_name.side_effect = func_node_child
            return func_node

        # Deprecated
        node = create_decorated("@deprecated")
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.maturity, Feature.Maturity.DEPRECATED)

        # Experimental
        node = create_decorated("@experimental")
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.maturity, Feature.Maturity.EXPERIMENTAL)

        # Beta
        node = create_decorated("@beta")
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.maturity, Feature.Maturity.BETA)

        # Stable (no decorator) -> No maturity set
        # Re-create simple func
        name_node = self.create_mock_node("identifier", text="stable_func")
        name_node.field_name = "name"
        node = self.create_mock_node(
            "function_definition", children=[name_node]
        )

        def node_child(name):
            if name == "name":
                return name_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        # Should not have maturity field set
        self.assertFalse(result.HasField("maturity"))

    def test_private_method(self):
        name_node = self.create_mock_node("identifier", text="_private")
        name_node.field_name = "name"
        node = self.create_mock_node(
            "function_definition", children=[name_node]
        )

        def node_child(name):
            if name == "name":
                return name_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertIsNone(result)

    def test_name_extraction_failure(self):
        # Node with no name child
        node = self.create_mock_node("function_definition", children=[])
        node.child_by_field_name.return_value = None

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertIsNone(result)

    def test_complex_parameters(self):
        # def func(plain, typed: int=1, self=None): ...

        # 1. plain identifier
        p1 = self.create_mock_node("identifier", text="plain")

        # 2. typed_default_parameter
        p2_name = self.create_mock_node("identifier", text="typed")
        p2_type = self.create_mock_node("type", text="int")
        p2 = self.create_mock_node("typed_default_parameter")

        def p2_child(name):
            if name == "name":
                return p2_name
            if name == "type":
                return p2_type
            return None

        p2.child_by_field_name.side_effect = p2_child
        p2.children = [p2_name, p2_type]

        # 3. self (should be ignored)
        p3_name = self.create_mock_node("identifier", text="self")
        p3 = self.create_mock_node("default_parameter")

        def p3_child(name):
            if name == "name":
                return p3_name
            return None

        p3.child_by_field_name.side_effect = p3_child
        p3.children = [p3_name]

        params_node = self.create_mock_node("parameters", children=[p1, p2, p3])
        params_node.field_name = "parameters"

        name_node = self.create_mock_node("identifier", text="func")
        node = self.create_mock_node(
            "function_definition", children=[params_node, name_node]
        )

        def node_child(name):
            if name == "name":
                return name_node
            if name == "parameters":
                return params_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)

        self.assertEqual(len(result.parameters), 2)
        # Verify plain
        self.assertEqual(result.parameters[0].original_name, "plain")
        self.assertEqual(
            result.parameters[0].normalized_types,
            [feature_pb2.ParamType.OBJECT],
        )

        # Verify typed default
        self.assertEqual(result.parameters[1].original_name, "typed")
        self.assertEqual(
            result.parameters[1].normalized_types,
            [feature_pb2.ParamType.NUMBER],
        )
        self.assertTrue(result.parameters[1].is_optional)

    def test_unknown_param_type_and_null(self):
        # def func(a: UnknownType, b: Optional[int]): ...
        # Optional[int] maps to [INT, null], null should be skipped in enum list

        p1_name = self.create_mock_node("identifier", text="a")
        p1_type = self.create_mock_node("type", text="UnknownType")
        p1 = self.create_mock_node("typed_parameter")

        def p1_child(name):
            if name == "name":
                return p1_name
            if name == "type":
                return p1_type
            return None

        p1.child_by_field_name.side_effect = p1_child

        p2_name = self.create_mock_node("identifier", text="b")
        p2_type = self.create_mock_node("type", text="Optional[int]")
        p2 = self.create_mock_node("typed_parameter")

        def p2_child(name):
            if name == "name":
                return p2_name
            if name == "type":
                return p2_type
            return None

        p2.child_by_field_name.side_effect = p2_child

        params_node = self.create_mock_node("parameters", children=[p1, p2])
        params_node.field_name = "parameters"

        name_node = self.create_mock_node("identifier", text="func")
        node = self.create_mock_node(
            "function_definition", children=[params_node, name_node]
        )

        def node_child(name):
            if name == "name":
                return name_node
            if name == "parameters":
                return params_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)

        # a: UnknownType -> OBJECT (fallback catch AttributeError/Enum check)
        # Actually logic: default normalized_types is [OBJECT] if empty.
        # strings.py likely returns ["UNKNOWN_TYPE"] or similar?
        # strings.py returns "OBJECT" for unknowns if they don't match list?
        # Let's check strings.py logic.
        # strings.py: _simple_normalize returns 'OBJECT' for anything not
        # in list.
        # So it returns 'OBJECT'. Enum OBJECT is 0.
        self.assertEqual(
            result.parameters[0].normalized_types,
            [feature_pb2.ParamType.OBJECT],
        )

        # b: Optional[int] -> [int, null] -> [INT, null] -> INT (null skipped)
        self.assertEqual(
            result.parameters[1].normalized_types,
            [feature_pb2.ParamType.NUMBER],
        )

    def test_param_empty_name(self):
        # Param with empty name
        p1 = self.create_mock_node("identifier", text="")
        params_node = self.create_mock_node("parameters", children=[p1])
        params_node.field_name = "parameters"

        func_name = self.create_mock_node("identifier", text="func")
        node = self.create_mock_node(
            "function_definition", children=[params_node, func_name]
        )

        def node_child(name):
            if name == "name":
                return func_name
            if name == "parameters":
                return params_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(len(result.parameters), 0)

    @patch("google.adk.scope.extractors.converter_py.TypeNormalizer.normalize")
    def test_param_enum_attribute_error(self, mock_normalize):
        # Force normalize to return a value not in keys
        mock_normalize.return_value = ["INVALID_TYPE_NAME"]

        p1 = self.create_mock_node("identifier", text="a")
        params_node = self.create_mock_node("parameters", children=[p1])
        params_node.field_name = "parameters"

        func_name = self.create_mock_node("identifier", text="func")
        node = self.create_mock_node(
            "function_definition", children=[params_node, func_name]
        )

        def node_child(name):
            if name == "name":
                return func_name
            if name == "parameters":
                return params_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)

        # Should fallback to OBJECT
        self.assertEqual(
            result.parameters[0].normalized_types,
            [feature_pb2.ParamType.OBJECT],
        )

    def test_process_docstrings(self):
        # def func(a, b):
        #     """My docstring.
        #
        #     Args:
        #         a (str): Description for a.
        #         b: Description for b.
        #     """
        #     pass

        name_node = self.create_mock_node("identifier", text="func")
        name_node.field_name = "name"

        # Params
        p1_name = self.create_mock_node("identifier", text="a")
        p1 = self.create_mock_node("typed_parameter")

        def p1_child(name):
            if name == "name":
                return p1_name
            return None

        p1.child_by_field_name.side_effect = p1_child

        _p2_name = self.create_mock_node("identifier", text="b")
        p2 = self.create_mock_node(
            "identifier", text="b"
        )  # simple identifier param

        params_node = self.create_mock_node("parameters", children=[p1, p2])
        params_node.field_name = "parameters"

        # Docstring
        docstring_text = '''"""My docstring.

    Args:
        a (str): Description for a.
        b: Description for b.
    """'''
        string_node = self.create_mock_node("string", text=docstring_text)
        expr_stmt = self.create_mock_node(
            "expression_statement", children=[string_node]
        )

        body_node = self.create_mock_node("block", children=[expr_stmt])
        body_node.field_name = "body"

        node = self.create_mock_node(
            "function_definition", children=[name_node, params_node, body_node]
        )

        def node_child(name):
            if name == "name":
                return name_node
            if name == "parameters":
                return params_node
            if name == "body":
                return body_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)

        self.assertIsNotNone(result)
        # Check description (should exclude Args)
        self.assertEqual(result.description, "My docstring.")

        # Check params
        self.assertEqual(len(result.parameters), 2)
        self.assertEqual(result.parameters[0].original_name, "a")
        self.assertEqual(result.parameters[0].description, "Description for a.")

        self.assertEqual(result.parameters[1].original_name, "b")
        self.assertEqual(result.parameters[1].description, "Description for b.")

    def test_async_method(self):
        # async def func(): ...
        name_node = self.create_mock_node("identifier", text="func")
        name_node.field_name = "name"

        # Async modifier
        async_node = self.create_mock_node("async")

        # In tree-sitter, async is usually a child of function_definition
        node = self.create_mock_node(
            "function_definition", children=[async_node, name_node]
        )

        def node_child(name):
            if name == "name":
                return name_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)

        # Should have async=True
        # Use kwargs access or getattr
        # Protobuf field name is likely 'async' in some contexts but access
        # might need handling
        # Since we verified 'async' is in descriptor, let's try
        # getattr(result, "async") or similar?
        # Actually in Python it might be result.async if Python version
        # allows (3.7+ prevents it as attribute)
        # So we expect result.async_ usually?
        # Wait, my inspection script said "Field 'async' found". It did NOT
        # say 'async_'.
        # But accessing it: getattr(result, 'async') works?
        # Let's check logic:
        # If I used feature_kwargs["async"] = True, it should correspond to
        # 'async' field.
        # But if 'async' is keyword, accessing it as attribute is syntax error.

        # We can use HasField("async")?
        # HasField works with string name.
        self.assertTrue(result.HasField("async"))
        # Get value
        # self.assertTrue(getattr(result, "async")) might fail syntax
        # parsing if I try result.async
        # But getattr(result, "async") is runtime safe.

    def test_sync_method(self):
        name_node = self.create_mock_node("identifier", text="func")
        name_node.field_name = "name"
        node = self.create_mock_node(
            "function_definition", children=[name_node]
        )

        def node_child(name):
            if name == "name":
                return name_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        # Should NOT have async set
        self.assertFalse(result.HasField("async"))

    def test_docstring_edge_cases(self):
        # Alternative quotes and sections
        def create_doc_test(doc_text, expected_desc, has_params=False):
            name_node = self.create_mock_node("identifier", text="func")
            name_node.field_name = "name"
            string_node = self.create_mock_node("string", text=doc_text)
            expr_stmt = self.create_mock_node(
                "expression_statement", children=[string_node]
            )
            body_node = self.create_mock_node("block", children=[expr_stmt])
            body_node.field_name = "body"

            children = [name_node, body_node]
            params_node = None
            if has_params:
                # Add default parameter node for 'x'
                p_name = self.create_mock_node("identifier", text="x")
                p_node = self.create_mock_node(
                    "identifier", children=[p_name], text="x"
                )  # Simple identifier param
                # Mock child_by_field_name for p_node?
                # _process_param_node for identifier checks name from text

                params_node = self.create_mock_node(
                    "parameters", children=[p_node]
                )
                params_node.field_name = "parameters"
                children.append(params_node)

            node = self.create_mock_node(
                "function_definition", children=children
            )

            def node_child(name):
                if name == "name":
                    return name_node
                if name == "body":
                    return body_node
                if name == "parameters":
                    return params_node
                return None

            node.child_by_field_name.side_effect = node_child
            return self.processor.process(node, self.file_path, self.repo_root)

        # 1. Single quotes
        res = create_doc_test("'Single quoted string'", "Single quoted string")
        self.assertEqual(res.description, "Single quoted string")

        # 2. Double quotes
        res = create_doc_test('"Double quoted string"', "Double quoted string")
        self.assertEqual(res.description, "Double quoted string")

        # 3. Arguments: instead of Args:
        res = create_doc_test(
            '"""Desc.\n\nArguments:\n    x: X desc.\n"""',
            "Desc.",
            has_params=True,
        )
        self.assertEqual(res.description, "Desc.")
        self.assertEqual(res.parameters[0].description, "X desc.")

    def test_parse_docstring_params_detailed(self):
        # Test private method for edge cases logic
        doc = """
        Desc.
        
        Args:
           p1 (int): Desc 1.
           p2: Desc 2
               continued.
           p3 (str): Desc 3.
        
        Yields:
           Something.
        """
        parsed = self.processor._parse_docstring_params(doc)
        self.assertEqual(parsed["p1"], "Desc 1.")
        self.assertEqual(parsed["p2"], "Desc 2 continued.")
        self.assertEqual(parsed["p3"], "Desc 3.")

    def test_static_method(self):
        class_name = self.create_mock_node("identifier", text="C")
        class_name.field_name = "name"  # FIXED: added field_name
        class_def = self.create_mock_node(
            "class_definition", children=[class_name]
        )

        deco = self.create_mock_node("decorator", text="@staticmethod")
        func_name = self.create_mock_node("identifier", text="stat")
        func_name.field_name = "name"
        func_node = self.create_mock_node(
            "function_definition", children=[func_name]
        )

        # decorated_definition -> decorator, function_definition
        # In tree-sitter, decorated_definition children are decorators then
        # the definition
        decorated = self.create_mock_node(
            "decorated_definition", children=[deco, func_node], parent=class_def
        )
        func_node.parent = decorated

        def func_child(name):
            if name == "name":
                return func_name
            return None

        func_node.child_by_field_name.side_effect = func_child

    def test_namespace_variations(self):
        # 1. Deep path
        p1 = Path("/repo/src/a/b/c/d.py")
        ns1, norm1 = self.processor._extract_namespace(p1, self.repo_root)
        self.assertEqual(ns1, "a.b.c")

        # 2. Non-src path
        p2 = Path("/repo/custom/lib/file.py")
        ns2, norm2 = self.processor._extract_namespace(p2, self.repo_root)
        # relative: custom/lib/file.py -> parent custom/lib -> parts
        # ['custom', 'lib']
        self.assertEqual(ns2, "custom.lib")

        # 3. Path without google/adk structure
        p3 = Path("/repo/src/other/pkg/file.py")
        ns3, norm3 = self.processor._extract_namespace(p3, self.repo_root)
        # relative: src/other/pkg/file.py -> parent src/other/pkg
        # -> parts [src, other, pkg] -> [other, pkg]
        self.assertEqual(ns3, "other.pkg")

        # 4. __init__.py in non-root
        p4 = Path("/repo/src/google/adk/subpkg/__init__.py")
        ns4, norm4 = self.processor._extract_namespace(p4, self.repo_root)
        # relative: src/google/adk/subpkg/__init__.py ->
        # parent src/google/adk/subpkg
        # parts: [src, google, adk, subpkg] -> [subpkg]
        # (after stripping src, google, adk)
        # self.assertEqual(ns4, "subpkg")
        # WAIT: logic in _extract_namespace:
        # if len(parts) >= 2 and parts[0] == 'google' and parts[1] == 'adk':
        # parts=parts[2:]
        # So "subpkg" remains.
        # Then join.
        self.assertEqual(ns4, "subpkg")

    def test_docstring_extended(self):
        # 1. Triple single quotes
        # '''Doc'''
        ds = "'''Doc'''"
        string_node = self.create_mock_node("string", text=ds)
        expr = self.create_mock_node(
            "expression_statement", children=[string_node]
        )
        body = self.create_mock_node("block", children=[expr])
        body.field_name = "body"

        name = self.create_mock_node("identifier", text="f")
        node = self.create_mock_node(
            "function_definition", children=[name, body]
        )
        node.child_by_field_name.side_effect = lambda n: (
            name if n == "name" else (body if n == "body" else None)
        )

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.description, "Doc")

        # 2. Empty string (not docstring per se, or empty docstring)
        # ""
        ds_empty = '""'
        string_node.text = ds_empty.encode("utf-8")
        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.description, "")

        # 3. Non-string statement first (optimization, should break loop)
        # x = 1
        assign = self.create_mock_node(
            "assignment"
        )  # not expression_statement with string
        body.children = [
            assign,
            expr,
        ]  # expression statement after assignment is NOT docstring

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.description, "")  # Should be empty

    def test_filter_cls_parameter(self):
        # def func(cls): ...

        p_node = self.create_mock_node("identifier")  # simluating simple param
        # Actually `identifier` type param means text is the name
        p_node.text = b"cls"
        p_node.type = "identifier"

        params_node = self.create_mock_node("parameters", children=[p_node])
        params_node.field_name = "parameters"

        func_name = self.create_mock_node("identifier", text="func")
        node = self.create_mock_node(
            "function_definition", children=[params_node, func_name]
        )

        def node_child(name):
            if name == "name":
                return func_name
            if name == "parameters":
                return params_node
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(len(result.parameters), 0)

    def test_comments_before_docstring(self):
        # Body starts with comment then docstring
        comment = self.create_mock_node("comment", text="# some comment")

        doc_str = '"""Real doc."""'
        string_node = self.create_mock_node("string", text=doc_str)
        expr_stmt = self.create_mock_node(
            "expression_statement", children=[string_node]
        )

        body = self.create_mock_node("block", children=[comment, expr_stmt])
        body.field_name = "body"

        name = self.create_mock_node("identifier", text="f")
        node = self.create_mock_node(
            "function_definition", children=[name, body]
        )

        def node_child(n):
            if n == "name":
                return name
            if n == "body":
                return body
            return None

        node.child_by_field_name.side_effect = node_child

        result = self.processor.process(node, self.file_path, self.repo_root)
        self.assertEqual(result.description, "Real doc.")
