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
        self, node: Node, file_path: Path, repo_root: Path
    ) -> Optional[feature_pb2.Feature]:
        """Convert a Tree-sitter node into a Feature."""
        if node.type != "function_declaration":
            return None

        original_name = self._extract_name(node)
        if not original_name:
            return None

        parameters = self._extract_params(node)

        feature = feature_pb2.Feature(
            original_name=original_name,
            normalized_name=normalize_name(original_name),
            file_path=str(file_path.resolve()),
            type=feature_pb2.Feature.FUNCTION,  # Default to FUNCTION for now
            parameters=parameters,
            original_return_types=self._extract_return_types(node),
        )

        return feature

    def _extract_return_types(self, node: Node) -> list[str]:
        """Extract return types from a function_declaration node."""
        return_node = node.child_by_field_name("result")
        if return_node:
            return [return_node.text.decode("utf-8")]
        return []

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

                    p = feature_pb2.Param(
                        original_name=param_name,
                        normalized_name=normalize_name(param_name),
                        original_types=[param_type],
                    )
                    params.append(p)
        return params

    def _extract_name(self, node: Node) -> str:
        """Extract the name from a function_declaration node."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf-8")
        return ""
