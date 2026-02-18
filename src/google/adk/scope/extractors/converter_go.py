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
    def __init__(self):
        self.normalizer = TypeNormalizer()
        # Mapping from struct name to list of (field_name, field_type, is_optional)
        self._struct_definitions: dict[str, list[tuple[str, str, bool]]] = {}

    def process(
        self,
        node: Node,
        file_path: Path,
        repo_root: Path,
        namespace: str,
        normalized_namespace: str,
    ) -> Optional[feature_pb2.Feature]:
        """Convert a Tree-sitter node into a Feature."""
        valid_nodes = ("function_declaration", "method_declaration", "method_elem")
        if node.type not in valid_nodes:
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
            # For constructors, try to infer member_of from the return type
            # e.g. func NewAgent() *Agent -> member_of = Agent
            original_returns, _ = self._extract_return_types(node)
            if original_returns:
                # Typically the first return value is the struct
                ret_type = original_returns[0]
                # access the struct name, e.g. *Agent -> Agent, mypkg.Agent -> Agent
                # Similar logic to parameter flattening type extraction
                clean_ret = ret_type.lstrip("*").split(".")[-1]
                if clean_ret:
                    member_of = clean_ret
                    normalized_member_of = normalize_name(member_of)
        elif node.type == "method_elem":
            feature_type = feature_pb2.Feature.Type.INSTANCE_METHOD
            member_of = self._extract_interface_name(node)
            normalized_member_of = (
                normalize_name(member_of) if member_of else ""
            )

        parameters, is_async = self._extract_params(node)

        original_returns, normalized_returns = self._extract_return_types(node)
        
        docstring = self._extract_docstring(node)

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
        if is_async:
            setattr(feature, "async", True)

        if docstring:
            feature.description = docstring

        if docstring:
            feature.description = docstring

        return feature

    def register_struct(self, node: Node) -> None:
        """Register a struct definition to allow parameter flattening."""
        # Find struct name from parent type_spec
        parent = node.parent
        # The query capture is on (type_spec name: ... type: (struct_type) @struct_body)
        # So node is the struct_type node. Parent should be type_spec.
        if not parent or parent.type != "type_spec":
            return
            
        name_node = parent.child_by_field_name("name")
        if not name_node:
            return
            
        struct_name = name_node.text.decode("utf-8")
        
        # Parse fields
        fields = []
        
        # Iterating children to find field_declaration_list because child_by_field_name 
        # might be failing or the field name is different in this version of tree-sitter-go
        field_list = None
        for child in node.children:
            if child.type == "field_declaration_list":
                field_list = child
                break
                
        if field_list:
            for child in field_list.children:
                if child.type == "field_declaration":
                    # Handle multiple names for same type e.g. A, B int
                    type_node = child.child_by_field_name("type")
                    if not type_node:
                        continue
                        
                    type_str = type_node.text.decode("utf-8")
                    
                    # Determine if optional
                    is_optional = False
                    if type_node.type == "pointer_type":
                        is_optional = True
                    
                    # field_declaration children names
                    # Loop through children to find all field_identifier nodes
                    field_names = []
                    for subchild in child.children:
                        if subchild.type == "field_identifier":
                            field_names.append(subchild.text.decode("utf-8"))
                            
                    for fname in field_names:
                        fields.append((fname, type_str, is_optional))
        
        self._struct_definitions[struct_name] = fields

    def _extract_docstring(self, node: Node) -> str:
        """Extract comments immediately preceding the declaration."""
        comments = []
        prev = node.prev_sibling
        while prev and prev.type == "comment":
            clean_comment = prev.text.decode("utf-8").lstrip("//").strip()
            comments.insert(0, clean_comment)
            prev = prev.prev_sibling
        return "\n".join(comments)

    def _extract_interface_name(self, node: Node) -> str:
        """Walk up the AST from a method_spec to find the interface type name."""
        parent = node.parent
        while parent:
            if parent.type == "type_spec":
                name_node = parent.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode("utf-8")
            parent = parent.parent
        return ""

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

    def _extract_params(self, node: Node) -> tuple[list[feature_pb2.Param], bool]:
        """Extract parameters from a function_declaration node."""
        params = []
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            return [], False

        is_async = False
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
                        is_async = True
                        continue
                    
                    # Check if this parameter type should be flattened
                    # We strip pointer and module prefix to find the struct name
                    # e.g. *Config -> Config, mypkg.Config -> Config
                    # Simple heuristic: take the last part after dot, strip *
                    clean_type_name = param_type.lstrip("*").split(".")[-1]
                    
                    if clean_type_name in self._struct_definitions:
                        # FLATTEN: Add all fields of the struct as parameters
                        for field_name, field_type, is_optional in self._struct_definitions[clean_type_name]:
                            # Recursively normalize the field type
                            norm_types = self.normalizer.normalize(field_type, "go")
                            norm_enums = [getattr(feature_pb2, nt) for nt in norm_types]
                            
                            p = feature_pb2.Param(
                                original_name=field_name,
                                normalized_name=normalize_name(field_name),
                                original_types=[field_type],
                                normalized_types=norm_enums,
                            )
                            if is_optional:
                                p.is_optional = True
                            params.append(p)
                    else:
                        # Normal processing
                        norm_types = self.normalizer.normalize(param_type, "go")
                        norm_enums = [getattr(feature_pb2, nt) for nt in norm_types]

                        p = feature_pb2.Param(
                            original_name=param_name,
                            normalized_name=normalize_name(param_name),
                            original_types=[param_type],
                            normalized_types=norm_enums,
                        )
                        params.append(p)
        return params, is_async

    def _extract_name(self, node: Node) -> str:
        """Extract the name from a function_declaration node."""
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode("utf-8")
        return ""
