"""
Converter to transform TypeScript Tree-sitter nodes into Feature objects.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple, Set

from tree_sitter import Node

from google.adk.scope.utils.strings import normalize_name, normalize_type_complex
from google.adk.scope import features_pb2 as feature_pb2

class NodeProcessor:
    """Process Tree-sitter nodes into Feature objects for TypeScript."""
    
    def process(self, node: Node, file_path: Path, repo_root: Path) -> Optional[feature_pb2.Feature]:
        """Convert a Tree-sitter node into a Feature.
        
        Args:
            node: The function_declaration, method_definition, or class_declaration node.
            file_path: Absolute path to the file.
            repo_root: Root of the repository.
        """
        # Node types we care about:
        # function_declaration: function foo() {}
        # method_definition: class X { foo() {} }
        # constructor: class X { constructor() {} } -> usually method_definition with name 'constructor' in some grammars, or checks
        # In tree-sitter-typescript:
        # - function_declaration
        # - method_definition
        # - public_field_definition (if arrow function property? maybe skip for now per prompt "functions defined") 
        
        if node.type not in ('function_declaration', 'method_definition'):
            return None

        # 1. Identity
        original_name = self._extract_name(node)
        if not original_name:
            return None

        # Filter private methods & setters/getters
        # private keyword or #name (private fields)
        # get/set keywords
        if self._is_private_or_accessor(node, original_name):
            return None
            
        normalized_name = normalize_name(original_name)
        
        # 2. Context
        member_of, normalized_member_of = self._extract_member_of(node)
        
        feature_type = self._determine_type(node, original_name, bool(member_of))
        
        if not member_of:
            member_of = "null"
        
        namespace, normalized_namespace = self._extract_namespace(file_path, repo_root)
        
        # 3. Contract
        jsdoc = self._extract_jsdoc(node)
        param_docs = self._parse_jsdoc_params(jsdoc)
        
        # Main description is everything before the first tag (@)
        description = jsdoc
        if '@' in jsdoc:
             lines = jsdoc.split('\n')
             desc_lines = []
             for line in lines:
                 if line.strip().startswith('@'):
                     break
                 desc_lines.append(line)
             description = "\n".join(desc_lines).strip()

        parameters = self._extract_params(node, param_docs)
        original_returns, normalized_returns = self._extract_return_types(node)
        original_returns, normalized_returns = self._extract_return_types(node)
        is_async = not self._is_blocking(node, original_returns)
        
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
        name_node = node.child_by_field_name('name')
        if name_node:
            return name_node.text.decode('utf-8')
        return ""

    def _extract_jsdoc(self, node: Node) -> str:
        # Check immediate previous sibling
        prev = node.prev_sibling
        while prev:
             if prev.type == 'comment':
                 text = prev.text.decode('utf-8')
                 if text.startswith('/**'):
                     # Clean up JSDoc
                     lines = text.split('\n')
                     clean_lines = []
                     for line in lines:
                         line = line.strip()
                         if line.startswith('/**'):
                             line = line[3:]
                         if line.endswith('*/'):
                             line = line[:-2]
                         if line.startswith('*'):
                             line = line[1:]
                         clean_lines.append(line.strip())
                     return "\n".join(clean_lines).strip()
             elif prev.type == 'decorator':
                 # Skip decorators (they are attached to method/class but might be siblings in some grammars)
                 pass
             elif prev.type == 'export_statement':
                 # If we are strictly finding previous sibling, export might be wrapping us?
                 # But if we are a function_declaration, and valid JSDoc is outside export...
                 pass
             elif not prev.text.strip():
                 # whitespace
                 pass
             else:
                 # Found something else, stop
                 break
             prev = prev.prev_sibling
             
        # If parent is export_statement (common in TS), check its previous sibling
        if node.parent and node.parent.type == 'export_statement':
             prev = node.parent.prev_sibling
             while prev:
                 if prev.type == 'comment':
                     text = prev.text.decode('utf-8')
                     if text.startswith('/**'):
                         lines = text.split('\n')
                         clean_lines = []
                         for line in lines:
                             line = line.strip()
                             if line.startswith('/**'):
                                 line = line[3:]
                             if line.endswith('*/'):
                                 line = line[:-2]
                             if line.startswith('*'):
                                 line = line[1:]
                             clean_lines.append(line.strip())
                         return "\n".join(clean_lines).strip()
                 elif not prev.text.strip():
                     pass
                 else:
                     break
                 prev = prev.prev_sibling
                 
        return ""

    def _parse_jsdoc_params(self, jsdoc: str) -> dict:
        if not jsdoc:
             return {}
        
        param_docs = {}
        # Look for @param {type} name description
        # or @param name description
        
        lines = jsdoc.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('@param'):
                # @param ...
                content = line[6:].strip()
                # Check for optional type {Type}
                if content.startswith('{'):
                    # Find closing }
                    end_brace = content.find('}')
                    if end_brace != -1:
                        content = content[end_brace+1:].strip()
                
                # Now content should be "name description"
                if ' ' in content:
                    name, desc = content.split(' ', 1)
                    # Handle name enclosed in [] (optional)
                    if name.startswith('[') and name.endswith(']'):
                         name = name[1:-1]
                         # strict optional syntax might be [name=default]
                         if '=' in name:
                             name = name.split('=')[0]
                             
                    param_docs[name] = desc.strip()
                else:
                    # just name?
                    param_docs[content] = ""
                    
        return param_docs

    def _is_private_or_accessor(self, node: Node, name: str) -> bool:
        # 1. Check name
        if name.startswith('_') or name.startswith('#'):
            return True
        
        # 2. Check accessibility modifier
        # method_definition usually has 'accessibility' field or modifier children
        # (method_definition (accessibility_modifier) ...)
        # accessibility_modifier text can be "private", "protected", "public"
        for child in node.children:
            if child.type == 'accessibility_modifier':
                text = child.text.decode('utf-8')
                if text in ('private', 'protected'):
                    return True
            # getters/setters: (method_definition "get" ... )?
            # actually usually: (method_definition name: (property_identifier) ...)
            # Checking if it is a getter/setter
            # In tree-sitter-typescript, it might be separate 'get' 'set' token
            if child.type in ('get', 'set'):
                return True
                
        return False

    def _determine_type(self, node: Node, name: str, is_member: bool) -> feature_pb2.Feature.Type:
        if is_member:
            if name == 'constructor':
                return feature_pb2.Feature.CONSTRUCTOR
            
            # Check for static
            if self._is_static(node):
                return feature_pb2.Feature.CLASS_METHOD
                
            return feature_pb2.Feature.INSTANCE_METHOD
        return feature_pb2.Feature.FUNCTION

    def _is_static(self, node: Node) -> bool:
        for child in node.children:
            if child.text.decode('utf-8') == 'static':
                return True
        return False

    def _get_decorators(self, node: Node) -> Set[str]:
        decorators = set()
        # In TS, decorators are usually siblings or children
        # (decorator (call_expression (identifier)))
        # They usually appear before the method_definition or function_declaration
        # or inside it?
        # Tree-sitter grammar usually puts them as children or siblings
        # Check children for 'decorator'
        for child in node.children:
            if child.type == 'decorator':
                # decorator -> expression call_expression or identifier
                # @deprecated or @deprecated()
                text = child.text.decode('utf-8').lstrip('@')
                # remove parens if any
                if '(' in text:
                    text = text.split('(')[0]
                decorators.add(text)
        return decorators

    def _extract_member_of(self, node: Node) -> Tuple[str, str]:
        # Walk up to find class_declaration
        parent = node.parent
        while parent:
            if parent.type in ('class_declaration', 'abstract_class_declaration', 'interface_declaration'):
                name_node = parent.child_by_field_name('name')
                if name_node:
                    original = name_node.text.decode('utf-8')
                    return original, normalize_name(original)
            parent = parent.parent
        return "", ""

    def _extract_namespace(self, file_path: Path, repo_root: Path) -> Tuple[str, str]:
        # Similar logic to Python: derived from file path
        try:
            rel_path = file_path.relative_to(repo_root)
        except ValueError:
             # Fallback if not relative
             rel_path = file_path
             
        parts = list(rel_path.parent.parts)
        if len(parts) >= 2 and parts[0] == 'core' and parts[1] == 'src':
            parts = parts[2:]
        elif parts and parts[0] == 'src':
            parts = parts[1:]
        elif parts and parts[0] == 'lib': # common in JS/TS
             parts = parts[1:]
             
        # Remove empty parts or dots
        parts = [p for p in parts if p and p not in ('.', '..')]
        
        if not parts:
            return "", ""

        namespace = ".".join(parts)
        normalized = namespace.replace(".", "_")
        return namespace, normalized

    def _extract_params(self, node: Node, param_docs: dict = None) -> List[feature_pb2.Param]:
        if param_docs is None:
            param_docs = {}
        params = []
        parameters_node = node.child_by_field_name('parameters')
        if not parameters_node:
            return []
            
        # Structure: (formal_parameters (required_parameter (identifier) (type_annotation)?) ...)
        for child in parameters_node.children:
            # Types: required_parameter, optional_parameter, rest_parameter
            if child.type in ('required_parameter', 'optional_parameter', 'rest_parameter'):
                p = self._process_param_node(child)
                if p:
                    if p.original_name in param_docs and param_docs[p.original_name]:
                        p.description = param_docs[p.original_name]
                    params.append(p)
        return params

    def _process_param_node(self, node: Node) -> Optional[feature_pb2.Param]:
        name = ""
        types = []
        optional = False
        
        # 1. Type extraction (prioritized for naming)
        raw_type_str = ""
        type_node = node.child_by_field_name('type')
        if type_node:
            raw_type = type_node.text.decode('utf-8')
            if raw_type.startswith(':'):
                raw_type = raw_type[1:].strip()
            types.append(raw_type)
            raw_type_str = raw_type
            
        # 2. Name extraction
        pattern_node = node.child_by_field_name('pattern')
        if not pattern_node:
            # sometimes just (identifier) if no pattern field?
            # tree-sitter-typescript: required_parameter -> pattern, type
            # pattern is usually identifier
             for child in node.children:
                 if child.type == 'identifier':
                     pattern_node = child
                     break

        if pattern_node:
            if pattern_node.type == 'object_pattern':
                if raw_type_str:
                    # Priority 1: Derive from type
                    name = self._derive_name_from_type(raw_type_str)
                else:
                    # Priority 2: Flatten source
                    text = pattern_node.text.decode('utf-8')
                    name = re.sub(r'\s+', ' ', text).strip()
            else:
                name = pattern_node.text.decode('utf-8')
        
        if not name:
            return None

        if node.type == 'optional_parameter':
            optional = True
        
        normalized_strings = []
        for t in types:
            # normalize_type_complex is Python specific (Union[A,B])
            # We might need a TS specific one, or reuse it if syntax matches
            # TS uses A | B, not Union[A, B] usually (unless explicit)
            # We'll normalize manually here for basic handling
            normalized_strings.extend(self._normalize_ts_type(t))
            
        normalized_strings = sorted(list(set(normalized_strings)))
        if not normalized_strings:
            normalized_strings = ["OBJECT"]

        normalized_enums = []
        for s in normalized_strings:
            try:
                enum_val = getattr(feature_pb2.ParamType, s)
                normalized_enums.append(enum_val)
            except AttributeError:
                normalized_enums.append(feature_pb2.ParamType.OBJECT)

        return feature_pb2.Param(
            original_name=name,
            normalized_name=normalize_name(name),
            original_types=types,
            normalized_types=normalized_enums,
            is_optional=optional
        )

    def _derive_name_from_type(self, type_name: str) -> str:
        # Strip generics if present
        base_type = re.sub(r'<.*>', '', type_name).strip()
        
        # Common suffixes
        if base_type.endswith('Request'):
            return 'request'
        if base_type.endswith('Response'):
            return 'response'
        if base_type.endswith('Options'):
            return 'options'
        if base_type.endswith('Config'):
            return 'config'
        if base_type.endswith('Context'):
            return 'context'
            
        # Fallback: camelCase
        if base_type:
            return base_type[0].lower() + base_type[1:]
        
        return "obj"

    def _normalize_ts_type(self, t: str) -> List[str]:
        # Handle fundamental TS types
        t = t.strip()
        if not t:
            return ["OBJECT"]
            
        # A | B
        if '|' in t:
            parts = t.split('|')
            res = []
            for p in parts:
                res.extend(self._normalize_ts_type(p))
            return res
            
        # Generics: Promise<T>, Array<T>
        if '<' in t and t.endswith('>'):
            base = t.split('<', 1)[0].strip()
            # Find matching closing bracket or assumue last
            inner = t[t.find('<')+1:-1].strip()
            
            if base == 'Promise':
                return self._normalize_ts_type(inner)
            if base in ('Array', 'ReadonlyArray'):
                return ['LIST']
            if base == 'Map':
                return ['MAP']
            if base == 'Set':
                return ['SET']
            # Fallback for others
            return ['OBJECT']

        t_lower = t.lower()
        if t_lower in ('string', 'formattedstring', 'path'):
            return ['STRING']
        if t_lower in ('number', 'int', 'float', 'integer', 'double'):
            return ['FLOAT', 'INT'] 
        if t_lower in ('boolean', 'bool'):
            return ['BOOLEAN']
        if t_lower in ('any', 'unknown', 'object'):
            return ['OBJECT']
        if t_lower.endswith('[]'):
            return ['LIST']
        if t_lower.startswith('map') or t_lower.startswith('record') or '{' in t:
             return ['MAP']
        if t_lower.startswith('set'):
             return ['SET']
        if t_lower == 'void':
            return [] 
            
        return ['OBJECT']

    def _extract_return_types(self, node: Node) -> Tuple[List[str], List[str]]:
        return_type_node = node.child_by_field_name('return_type')
        if return_type_node:
            # (type_annotation ...)
            raw = return_type_node.text.decode('utf-8')
            if raw.startswith(':'):
                 raw = raw[1:].strip()
            
            # If Promise<T>, the return type effectively is Promise<T>, but logically T for async?
            # Schema says "original_return_types".
            # normalized usually unwrap?
            return [raw], self._normalize_ts_type(raw)
        return [], []

    def _is_blocking(self, node: Node, return_types: List[str]) -> bool:
        # Check for 'async' modifier or keyword
        for child in node.children:
            text = child.text.decode('utf-8')
            if text == 'async':
                return False
                
        # Check return types for Promise
        for rt in return_types:
            if rt.startswith('Promise<') or rt == 'Promise':
                return False
        
        return True
        # Check for 'async' modifier or keyword
        for child in node.children:
            text = child.text.decode('utf-8')
            if text == 'async':
                return False
            # Sometimes modifiers are wrapped?
            # But usually async is a direct child in TS grammar for method_definition
        
        return True

    def _extract_maturity(self, node: Node) -> feature_pb2.Feature.Maturity:
        decorators = self._get_decorators(node)
        # Also check JSDoc comments if accessible?
        # Tree-sitter might expose JSDoc as comment nodes?
        # For now, rely on decorators or comments if easy
        if 'deprecated' in decorators:
            return feature_pb2.Feature.DEPRECATED
        if 'experimental' in decorators:
            return feature_pb2.Feature.EXPERIMENTAL
        if 'beta' in decorators:
            return feature_pb2.Feature.BETA
        return None
