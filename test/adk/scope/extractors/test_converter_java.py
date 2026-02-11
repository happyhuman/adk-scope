import unittest
from pathlib import Path

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from google.adk.scope.extractors.converter_java import NodeProcessor
from google.adk.scope.features_pb2 import Feature, ParamType


class TestNodeProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = NodeProcessor()
        self.language = Language(tsjava.language())
        self.parser = Parser()
        self.parser.language = self.language
        self.file_path = Path("/mock/repo/src/main/java/com/example/Test.java")
        self.repo_root = Path("/mock/repo")

    def test_extract_method_basic(self):
        code = b"""
        package com.example;
        public class Test {
            /**
             * @return the result
             */
            public String doSomething(int count) { return "test"; }
        }
        """
        tree = self.parser.parse(code)

        # Manually find the method_declaration
        root = tree.root_node
        method_node = None
        for child in root.children:
            if child.type == "class_declaration":
                body = child.child_by_field_name("body")
                for member in body.children:
                    if member.type == "method_declaration":
                        method_node = member
                        break

        feature = self.processor.process(
            method_node, self.file_path, self.repo_root
        )

        self.assertIsNotNone(feature)
        self.assertEqual(feature.original_name, "doSomething")
        self.assertEqual(feature.normalized_name, "do_something")
        self.assertEqual(feature.member_of, "Test")
        self.assertEqual(feature.normalized_member_of, "test")
        self.assertEqual(feature.namespace, "com.example")
        self.assertEqual(feature.type, Feature.Type.INSTANCE_METHOD)
        self.assertEqual(feature.description, "@return the result")

        self.assertEqual(len(feature.parameters), 1)
        self.assertEqual(feature.parameters[0].original_name, "count")
        self.assertEqual(feature.parameters[0].normalized_name, "count")
        self.assertEqual(
            feature.parameters[0].normalized_types, [ParamType.NUMBER]
        )

        self.assertEqual(list(feature.original_return_types), ["String"])
        self.assertEqual(list(feature.normalized_return_types), ["STRING"])
        self.assertFalse(getattr(feature, "async"))

    def test_extract_constructor(self):
        code = b"""
        package com.example;
        public class Test {
            public Test(String name) {}
        }
        """
        tree = self.parser.parse(code)

        # find the constructor_declaration
        root = tree.root_node
        constructor_node = None
        for child in root.children:
            if child.type == "class_declaration":
                body = child.child_by_field_name("body")
                for member in body.children:
                    if member.type == "constructor_declaration":
                        constructor_node = member
                        break

        feature = self.processor.process(
            constructor_node, self.file_path, self.repo_root
        )

        self.assertIsNotNone(feature)
        self.assertEqual(feature.original_name, "Test")
        self.assertEqual(feature.normalized_name, "test")
        self.assertEqual(feature.type, Feature.Type.CONSTRUCTOR)

    def test_extract_static_async(self):
        code = b"""
        package com.example;
        import java.util.concurrent.CompletableFuture;
        public class Test {
            @Beta
            public static CompletableFuture<Integer> runAsync() { return null; }
        }
        """
        tree = self.parser.parse(code)

        root = tree.root_node
        method_node = None
        for child in root.children:
            if child.type == "class_declaration":
                body = child.child_by_field_name("body")
                for member in body.children:
                    if member.type == "method_declaration":
                        method_node = member
                        break

        feature = self.processor.process(
            method_node, self.file_path, self.repo_root
        )

        self.assertIsNotNone(feature)
        self.assertEqual(feature.original_name, "runAsync")
        self.assertEqual(feature.type, Feature.Type.CLASS_METHOD)
        self.assertEqual(list(feature.normalized_return_types), ["NUMBER"])
        self.assertTrue(getattr(feature, "async"))
        self.assertEqual(feature.maturity, Feature.Maturity.EXPERIMENTAL)

    def test_extract_namespace_rewriting(self):
        code = b"""
        package com.google.adk.agents;
        public class Test {
            public void agentMethod() {}
        }
        """
        tree = self.parser.parse(code)

        root = tree.root_node
        method_node = None
        for child in root.children:
            if child.type == "class_declaration":
                body = child.child_by_field_name("body")
                for member in body.children:
                    if member.type == "method_declaration":
                        method_node = member
                        break

        feature = self.processor.process(
            method_node, self.file_path, self.repo_root
        )

        self.assertIsNotNone(feature)
        self.assertEqual(feature.namespace, "agents")
        self.assertEqual(feature.normalized_namespace, "agents")

    def test_extract_boilerplate_filter(self):
        code = b"""
        package com.google.adk;
        public class Test {
            public String getName() { return ""; }
            public void setName(String name) {}
            public boolean isValid() { return true; }
            public boolean equals(Object o) { return false; }
            public int hashCode() { return 0; }
            public String toString() { return ""; }
            public void normalMethod() {}
        }
        """
        tree = self.parser.parse(code)

        root = tree.root_node
        methods = []
        for child in root.children:
            if child.type == "class_declaration":
                body = child.child_by_field_name("body")
                for member in body.children:
                    if member.type == "method_declaration":
                        methods.append(member)

        features = [
            self.processor.process(m, self.file_path, self.repo_root)
            for m in methods
        ]

        # Only 'normalMethod' should not be filtered out
        valid_features = [f for f in features if f is not None]
        self.assertEqual(len(valid_features), 1)
        self.assertEqual(valid_features[0].original_name, "normalMethod")
        # Ensure 'com.google.adk' was completely stripped to empty string
        self.assertEqual(valid_features[0].namespace, "")


if __name__ == "__main__":
    unittest.main()
