"""
Converter to transform Tree-sitter Java nodes into Feature objects.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from tree_sitter import Node

from google.adk.scope import features_pb2 as feature_pb2
from google.adk.scope.utils.normalizer import TypeNormalizer, normalize_name

logger = logging.getLogger(__name__)


class NodeProcessor:
    """Process Tree-sitter nodes into Feature objects for Java."""

    def __init__(self):
        self.normalizer = TypeNormalizer()

    def process(
        self, node: Node, file_path: Path, repo_root: Path
    ) -> Optional[feature_pb2.Feature]:
        """Convert a Tree-sitter node into a Feature.

        Args:
            node: The method_declaration or constructor_declaration node.
            file_path: Absolute path to the file.
            repo_root: Root of the repository.
        """
        if node.type not in ("method_declaration", "constructor_declaration"):
            return None

        # 1. Identity
        original_name = self._extract_name(node)
        if not original_name:
            return None

        normalized_name = normalize_name(original_name)

        # Skip testing methods if they happen to sneak in
        if original_name.startswith("test"):
            # A simplistic heuristic, could be improved
            logger.debug("Skipping test method: %s", original_name)
            return None

        # Exclude boilerplate methods
        if original_name in (
            "equals",
            "hashCode",
            "toString",
            "canEqual",
            "clone",
        ):
            logger.debug("Skipping boilerplate method: %s", original_name)
            return None

        # Exclude getters and setters
        if node.type == "method_declaration":
            if (
                (
                    original_name.startswith("get")
                    and len(original_name) > 3
                    and original_name[3].isupper()
                )
                or (
                    original_name.startswith("set")
                    and len(original_name) > 3
                    and original_name[3].isupper()
                )
                or (
                    original_name.startswith("is")
                    and len(original_name) > 2
                    and original_name[2].isupper()
                )
            ):
                logger.debug("Skipping getter/setter: %s", original_name)
                return None

        member_of, normalized_member_of = self._extract_class(node)

        # If it's a constructor, the name is typically the class name
        if node.type == "constructor_declaration" and not original_name:
            original_name = member_of
            normalized_name = normalized_member_of

        if not member_of:
            member_of = "null"

        namespace, normalized_namespace = self._extract_namespace(
            file_path, repo_root, node
        )

        # 3. Contract
        jsdoc = self._extract_javadoc(node)
        description = jsdoc if jsdoc else None

        parameters = self._extract_params(node)

        feature_type = feature_pb2.Feature.Type.INSTANCE_METHOD
        if node.type == "constructor_declaration":
            feature_type = feature_pb2.Feature.Type.CONSTRUCTOR
        elif self._is_static(node):
            feature_type = feature_pb2.Feature.Type.CLASS_METHOD

        original_returns, normalized_returns = self._extract_return_types(node)

        is_async = self._is_async(node, original_returns)

        maturity = self._extract_maturity(node)

        feature_kwargs = {
            "original_name": original_name,
            "normalized_name": normalized_name,
            "member_of": member_of,
            "normalized_member_of": normalized_member_of,
            "file_path": str(file_path.resolve()),
            "namespace": namespace,
            "normalized_namespace": normalized_namespace,
            "type": feature_type,
            "parameters": parameters,
            "original_return_types": original_returns,
            "normalized_return_types": normalized_returns,
        }

        if is_async:
            feature_kwargs["async"] = True

        feature = feature_pb2.Feature(**feature_kwargs)

        if description:
            feature.description = description

        if maturity is not None:
            feature.maturity = maturity

        return feature

    def _extract_name(self, node: Node) -> str:
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf-8")
        return ""

    def _extract_class(self, node: Node) -> Tuple[str, str]:
        parent = node.parent
        while parent:
            if parent.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            ):
                name_node = parent.child_by_field_name("name")
                if name_node:
                    original = name_node.text.decode("utf-8")
                    return original, normalize_name(original)
            parent = parent.parent
        return "", ""

    def _extract_namespace(
        self, file_path: Path, repo_root: Path, node: Node
    ) -> Tuple[str, str]:
        # Try to find package_declaration in the file
        root = node
        while root.parent:
            root = root.parent

        namespace = ""
        for child in root.children:
            if child.type == "package_declaration":
                # Find scoped_identifier or identifier
                for sub in child.children:
                    if sub.type in ("scoped_identifier", "identifier"):
                        namespace = sub.text.decode("utf-8")
                        break
            if namespace:
                break

        if not namespace:
            # Fallback to directory structure
            try:
                rel_path = file_path.relative_to(repo_root)
                parts = list(rel_path.parent.parts)
                # Try to strip common java roots like src/main/java
                if "src" in parts:
                    idx = parts.index("src")
                    if (
                        len(parts) > idx + 2
                        and parts[idx + 1] == "main"
                        and parts[idx + 2] == "java"
                    ):
                        parts = parts[idx + 3 :]
                    elif len(parts) > idx + 1:
                        parts = parts[idx + 1 :]
            except ValueError:
                parts = list(file_path.parent.parts)[-3:]

            parts = [p for p in parts if p and p not in (".", "..")]

            if not parts:
                return "", ""

            namespace = ".".join(parts)

        if namespace == "com.google.adk":
            namespace = ""
        elif namespace.startswith("com.google.adk."):
            namespace = namespace[len("com.google.adk.") :]

        normalized = namespace.replace(".", "_")
        return namespace, normalized

    def _extract_params(self, node: Node) -> List[feature_pb2.Param]:
        params = []
        parameters_node = node.child_by_field_name("parameters")
        if not parameters_node:
            return params

        for p_node in parameters_node.children:
            if p_node.type == "formal_parameter":
                p_name_node = p_node.child_by_field_name("name")
                p_type_node = p_node.child_by_field_name("type")

                name = p_name_node.text.decode("utf-8") if p_name_node else ""
                type_str = (
                    p_type_node.text.decode("utf-8")
                    if p_type_node
                    else "Object"
                )

                normalized_types = self.normalizer.normalize(type_str, "java")

                param = feature_pb2.Param(
                    original_name=name,
                    normalized_name=normalize_name(name),
                    original_types=[type_str],
                    normalized_types=[
                        getattr(feature_pb2.ParamType, nt)
                        for nt in normalized_types
                    ],
                    is_optional=False,  # Java params aren't optional by default
                )
                params.append(param)

        return params

    def _extract_return_types(
        self, node: Node
    ) -> Tuple[List[str], List[feature_pb2.ParamType]]:
        if node.type == "constructor_declaration":
            return [], []

        type_node = node.child_by_field_name("type")
        if type_node:
            raw = type_node.text.decode("utf-8")
            normalized = self.normalizer.normalize(raw, "java")
            return [raw], normalized
        return [], []

    def _is_static(self, node: Node) -> bool:
        modifiers = node.child_by_field_name("modifiers")
        if modifiers:
            for child in modifiers.children:
                if child.text.decode("utf-8") == "static":
                    return True
        # also check node children if modifiers node not direct
        for child in node.children:
            if child.type == "modifiers":
                for m_child in child.children:
                    if m_child.text.decode("utf-8") == "static":
                        return True
        return False

    def _is_async(self, node: Node, return_types: List[str]) -> bool:
        # Check return types for CompletableFuture, Mono, Flux, etc.
        for rt in return_types:
            if any(
                rt.startswith(async_type)
                for async_type in (
                    "CompletableFuture",
                    "Future",
                    "Mono",
                    "Flux",
                )
            ):
                return True

        # Check for @Async annotation
        modifiers = node.child_by_field_name("modifiers")
        if modifiers:
            for child in modifiers.children:
                if child.type == "marker_annotation":
                    name = child.child_by_field_name("name")
                    if name and name.text.decode("utf-8") == "Async":
                        return True
        for child in node.children:
            if child.type == "modifiers":
                for m_child in child.children:
                    if m_child.type == "marker_annotation":
                        name = m_child.child_by_field_name("name")
                        if name and name.text.decode("utf-8") == "Async":
                            return True
        return False

    def _extract_maturity(self, node: Node) -> Optional[int]:
        modifiers = node.child_by_field_name("modifiers")

        def _check_annotations(mods_node):
            for child in mods_node.children:
                if child.type == "marker_annotation":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        anno = name_node.text.decode("utf-8")
                        if anno in ("Experimental", "Beta"):
                            return feature_pb2.Feature.Maturity.EXPERIMENTAL
                        if anno == "Deprecated":
                            return feature_pb2.Feature.Maturity.DEPRECATED
            return None

        if modifiers:
            res = _check_annotations(modifiers)
            if res is not None:
                return res

        for child in node.children:
            if child.type == "modifiers":
                res = _check_annotations(child)
                if res is not None:
                    return res

        return None

    def _extract_javadoc(self, node: Node) -> str:
        prev = node.prev_sibling
        while prev:
            if prev.type == "block_comment":
                text = prev.text.decode("utf-8")
                if text.startswith("/**"):
                    lines = text.split("\n")
                    clean_lines = []
                    for line in lines:
                        line = line.strip()
                        if line.startswith("/**"):
                            line = line[3:]
                        if line.endswith("*/"):
                            line = line[:-2]
                        if line.startswith("*"):
                            line = line[1:]
                        clean_lines.append(line.strip())
                    return "\n".join(clean_lines).strip()
            # If we hit modifiers or annotations, we keep going up
            elif prev.type == "modifiers" or prev.type == "marker_annotation":
                pass
            else:
                break
            prev = prev.prev_sibling

        # Also check if it's placed inside modifiers by some AST quirks
        modifiers = node.child_by_field_name("modifiers")
        if modifiers:
            for child in modifiers.children:
                if child.type == "block_comment":
                    text = child.text.decode("utf-8")
                    if text.startswith("/**"):
                        lines = text.split("\n")
                        clean_lines = []
                        for line in lines:
                            line = line.strip()
                            if line.startswith("/**"):
                                line = line[3:]
                            if line.endswith("*/"):
                                line = line[:-2]
                            if line.startswith("*"):
                                line = line[1:]
                            clean_lines.append(line.strip())
                        return "\n".join(clean_lines).strip()

        return ""
