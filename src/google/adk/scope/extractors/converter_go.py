"""
Converter to transform Tree-sitter nodes into Feature objects for Go.
"""

import logging
from pathlib import Path
from typing import Optional

from tree_sitter import Node

from google.adk.scope import features_pb2 as feature_pb2
from google.adk.scope.utils.normalizer import TypeNormalizer, normalize_name

logger = logging.getLogger(__name__)


class NodeProcessor:
    """Process Tree-sitter nodes into Feature objects for Go."""

    def __init__(self):
        self.normalizer = TypeNormalizer()

    def process(
        self,
        node: Node,
        file_path: Path,
        repo_root: Path,
        namespace: str,
        normalized_namespace: str,
    ) -> Optional[feature_pb2.Feature]:
        """Convert a Tree-sitter node into a Feature."""
        if node.type not in ("function_declaration", "method_declaration"):
            return None

        original_name = self._extract_name(node)
        if not original_name:
            return None

        # Exclude unexported functions/methods (lowercase first letter)
        if original_name and original_name[0].islower():
            return None

        feature_type = feature_pb2.Feature.Type.FUNCTION
        member_of = ""
        normalized_member_of = ""

        if node.type == "method_declaration":
            feature_type = feature_pb2.Feature.Type.INSTANCE_METHOD
            member_of = self._extract_receiver_type(node)
            if member_of and member_of[0].islower():
                member_of = member_of[0].upper() + member_of[1:]
            normalized_member_of = (
                normalize_name(member_of) if member_of else ""
            )
        elif node.type == "function_declaration" and original_name.startswith(
            "New"
        ):
            feature_type = feature_pb2.Feature.Type.CONSTRUCTOR

        parameters = self._extract_params(node)

        original_returns, normalized_returns = self._extract_return_types(node)

        feature = feature_pb2.Feature(
            original_name=original_name,
            normalized_name=normalize_name(original_name),
            member_of=member_of,
            normalized_member_of=normalized_member_of,
            file_path=str(file_path.resolve()),
            namespace=namespace,
            normalized_namespace=normalized_namespace,
            type=feature_type,
            parameters=parameters,
            original_return_types=original_returns,
            normalized_return_types=normalized_returns,
        )

        return feature

    def _extract_receiver_type(self, node: Node) -> str:
        """Extract the receiver type from a method_declaration."""
        receiver_node = node.child_by_field_name("receiver")
        if not receiver_node:
            return ""

        for child in receiver_node.children:
            if child.type == "parameter_declaration":
                type_node = child.child_by_field_name("type")
                if type_node:
                    return type_node.text.decode("utf-8").lstrip("*")
        return ""

    def _extract_return_types(
        self, node: Node
    ) -> tuple[list[str], list[feature_pb2.ParamType]]:
        """Extract return types from a function_declaration node, ignoring
        'error'.
        """
        return_node = node.child_by_field_name("result")
        if not return_node:
            return [], []

        raw_types = []

        # If the return is a single type identifier or pointer
        if return_node.type in (
            "type_identifier",
            "pointer_type",
            "qualified_type",
            "slice_type",
            "map_type",
        ):
            raw_types.append(return_node.text.decode("utf-8"))
        # If it returns multiple types, they are wrapped in a parameter_list
        elif return_node.type == "parameter_list":
            for child in return_node.children:
                if child.type == "parameter_declaration":
                    type_node = child.child_by_field_name("type")
                    if type_node:
                        raw_types.append(type_node.text.decode("utf-8"))

        original_returns = []
        normalized_returns = []

        for raw in raw_types:
            if raw == "error":
                continue
            original_returns.append(raw)
            norm_types = self.normalizer.normalize(raw, "go")
            normalized_returns.extend(norm_types)

        return original_returns, normalized_returns

    def _extract_params(self, node: Node) -> list[feature_pb2.Param]:
        """Extract parameters from a function_declaration node."""
        params = []
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return []

        for child in params_node.children:
            if child.type == "parameter_declaration":
                name_node = child.child_by_field_name("name")
                type_node = child.child_by_field_name("type")

                if name_node and type_node:
                    param_name = name_node.text.decode("utf-8")
                    param_type = type_node.text.decode("utf-8")

                    # Skip Go context.Context parameters to align with other
                    # languages
                    if param_type == "context.Context":
                        continue

                    norm_types = self.normalizer.normalize(param_type, "go")
                    norm_enums = [getattr(feature_pb2, nt) for nt in norm_types]

                    p = feature_pb2.Param(
                        original_name=param_name,
                        normalized_name=normalize_name(param_name),
                        original_types=[param_type],
                        normalized_types=norm_enums,
                    )
                    params.append(p)
        return params

    def _extract_name(self, node: Node) -> str:
        """Extract the name from a function_declaration node."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf-8")
        return ""
