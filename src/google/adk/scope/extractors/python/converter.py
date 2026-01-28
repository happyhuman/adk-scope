"""
Converter to transform Tree-sitter nodes into Feature objects.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Set

from tree_sitter import Node

from google.adk.scope.utils.strings import normalize_name, normalize_type_complex
from google.adk.scope import features_pb2 as feature_pb2

class NodeProcessor:
    """Process Tree-sitter nodes into Feature objects."""
    
    def process(self, node: Node, file_path: Path, repo_root: Path) -> Optional[feature_pb2.Feature]:
        """Convert a Tree-sitter node into a Feature.
        
        Args:
            node: The function_definition node.
            file_path: Absolute path to the file.
            repo_root: Root of the repository.
        """
        if node.type != 'function_definition':
            return None

        # 1. Identity
        original_name = self._extract_name(node)
        if not original_name:
            return None

        # Filter private methods
        if original_name.startswith('_') and original_name not in ('__init__', '__call__'):
            return None
            
        normalized_name = normalize_name(original_name)
        
        # 2. Context
        member_of, normalized_member_of = self._extract_member_of(node)
        
        feature_type = self._determine_type(node, original_name, bool(member_of))
        
        namespace, normalized_namespace = self._extract_namespace(file_path, repo_root)
        
        # 3. Contract
        parameters = self._extract_params(node)
        original_returns, normalized_returns = self._extract_return_types(node)
        blocking = self._is_blocking(node)
        # Force non-blocking if async
        if not blocking:
            # Maybe redundant check, but ensures property consistency
            pass
            
        maturity = self._extract_maturity(node)
        
        return feature_pb2.Feature(
            original_name=original_name,
            normalized_name=normalized_name,
            member_of=member_of,
            normalized_member_of=normalized_member_of,
            file_path=str(file_path.resolve()),
            namespace=namespace,
            normalized_namespace=normalized_namespace,
            maturity=maturity,
            type=feature_type,
            parameters=parameters,
            original_return_types=original_returns,
            normalized_return_types=normalized_returns,
            blocking=blocking
        )

    def _extract_name(self, node: Node) -> str:
        name_node = node.child_by_field_name('name')
        if name_node:
            return name_node.text.decode('utf-8')
        return ""

    def _determine_type(self, node: Node, name: str, is_member: bool) -> feature_pb2.Feature.Type:
        if is_member:
            if name == '__init__':
                return feature_pb2.Feature.CONSTRUCTOR
            
            # Check decorators for @classmethod or @staticmethod
            decorators = self._get_decorators(node)
            if 'classmethod' in decorators or 'staticmethod' in decorators:
                return feature_pb2.Feature.CLASS_METHOD
                
            return feature_pb2.Feature.INSTANCE_METHOD
        return feature_pb2.Feature.FUNCTION

    def _get_decorators(self, node: Node) -> Set[str]:
        decorators = set()

        # In tree-sitter-python, decorators are usually siblings just before the function_definition
        # wrapped in a 'decorated_definition' node if present?
        # Actually structure:
        # decorated_definition
        #   decorator
        #   function_definition
        
        # If the 'node' passed to process is 'function_definition', we need to check its parent
        if node.parent and node.parent.type == 'decorated_definition':
            # Iterate children of decorated_definition
            for child in node.parent.children:
                if child.type == 'decorator':
                    # Content is usually @staticmethod
                    text = child.text.decode('utf-8').lstrip('@')
                    decorators.add(text)
        return decorators

    def _extract_member_of(self, node: Node) -> Tuple[str, str]:
        # Walk up to find class_definition
        parent = node.parent
        while parent:
            if parent.type == 'class_definition':
                name_node = parent.child_by_field_name('name')
                if name_node:
                    original = name_node.text.decode('utf-8')
                    return original, normalize_name(original)
            parent = parent.parent
        return "", ""

    def _extract_namespace(self, file_path: Path, repo_root: Path) -> Tuple[str, str]:
        # file path: src/google/adk/events/loop_agent.py
        # relative: google/adk/events/loop_agent.py
        # namespace: google.adk.events
        
        rel_path = file_path.relative_to(repo_root)
        # Assuming 'src' is possibly in the path, usually namespace is implied by directory structure
        # If src/google/adk, namespace is google.adk
        
        parts = list(rel_path.parent.parts)
        if parts and parts[0] == 'src':
            parts = parts[1:]
            
        namespace = ".".join(parts)
        normalized = namespace.replace(".", "_")
        return namespace, normalized

    def _extract_params(self, node: Node) -> List[feature_pb2.Param]:
        params = []
        parameters_node = node.child_by_field_name('parameters')
        if not parameters_node:
            return []
            
        # Iterate over parameters
        # Structure: (parameters (typed_parameter (identifier) (type)) ...)
        for child in parameters_node.children:
            if child.type in ('identifier', 'typed_parameter', 'default_parameter', 'typed_default_parameter'):
                p = self._process_param_node(child)
                if p:
                    # Filter 'self' and 'cls'
                    if p.original_name in ('self', 'cls'):
                        continue
                    params.append(p)
        return params

    def _process_param_node(self, node: Node) -> Optional[feature_pb2.Param]:
        name = ""
        types = []
        optional = False
        
        if node.type == 'identifier':
            name = node.text.decode('utf-8')
            
        elif node.type == 'typed_parameter':
            # (typed_parameter (identifier) (type))
             name_node = node.child_by_field_name('name') or node.children[0]
             name = name_node.text.decode('utf-8')
             
             type_node = node.child_by_field_name('type')
             if type_node:
                 types.append(type_node.text.decode('utf-8'))
                 
        elif node.type == 'default_parameter':
            # (default_parameter (identifier) (value))
            name_node = node.child_by_field_name('name') or node.children[0]
            name = name_node.text.decode('utf-8')
            optional = True
            
        elif node.type == 'typed_default_parameter':
             name_node = node.child_by_field_name('name')
             if name_node:
                 name = name_node.text.decode('utf-8')
             type_node = node.child_by_field_name('type')
             if type_node:
                 types.append(type_node.text.decode('utf-8'))
             optional = True

        if not name:
            return None
        
        normalized_strings = []
        for t in types:
            normalized_strings.extend(normalize_type_complex(t))
        # Unique
        normalized_strings = sorted(list(set(normalized_strings)))
        if not normalized_strings:
            normalized_strings = ["OBJECT"]

        # Map to enum
        normalized_enums = []
        for s in normalized_strings:
            # s is like "INT", "STRING", "null"
            # ParamType has OBJECT=0, STRING=1, ..., UNKNOWN=9
            # "null" -> ?
            # If s is "null", we usually skip it in types enum list? 
            # Or map to UNKNOWN?
            # Or better: "null" in types usually implies is_optional=True. 
            # But we already set is_optional based on default value.
            # Optional[T] -> [T, null].
            # If we just depend on is_optional, maybe we can drop "null" from types list if checking ParamType?
            # Protocol Buffer enums don't have NULL usually.
            # Let's drop "null" from the enum list for now, or map to UNKNOWN if forced.
            if s == "null":
                continue
                
            try:
                enum_val = getattr(feature_pb2.ParamType, s)
                normalized_enums.append(enum_val)
            except AttributeError:
                # Fallback to OBJECT or UNKNOWN?
                # User defined OBJECT=0.
                normalized_enums.append(feature_pb2.ParamType.OBJECT)

        return feature_pb2.Param(
            original_name=name,
            normalized_name=normalize_name(name),
            original_types=types,
            normalized_types=normalized_enums,
            is_optional=optional
        )

    def _extract_return_types(self, node: Node) -> Tuple[List[str], List[str]]:
        # Function definition has return_type field? 
        # (function_definition ... -> (type) ...)
        return_type_node = node.child_by_field_name('return_type')
        if return_type_node:
            raw = return_type_node.text.decode('utf-8')
            normalized = normalize_type_complex(raw)
            return [raw], normalized
        return [], []

    def _is_blocking(self, node: Node) -> bool:
        # Check for 'async' keyword
        # async function_definition
        # parent of function_definition might be decorated_definition, which is fine
        # BUT if it is async, the type found by query is usually still function_definition ?
        # Wait, tree-sitter-python has 'async' keyword as a child of function_definition?
        # Or creates a different node type?
        # Usually: (function_definition async: (async) ...) or node.text starts with async
        
        # Check leading 'async'
        # Safest is to check if 'async' token exists in children
        for child in node.children:
            if child.type == 'async':
                return False
        return True

    def _extract_maturity(self, node: Node) -> feature_pb2.Feature.Maturity:
        decorators = self._get_decorators(node)
        if 'deprecated' in decorators:
            return feature_pb2.Feature.DEPRECATED
        if 'experimental' in decorators:
            return feature_pb2.Feature.EXPERIMENTAL
        if 'beta' in decorators:
            return feature_pb2.Feature.BETA
        return feature_pb2.Feature.STABLE
