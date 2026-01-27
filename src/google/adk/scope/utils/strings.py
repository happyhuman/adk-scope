"""
String utility functions for ADK Scope.
"""
import re

def normalize_name(name: str) -> str:
    """Convert name to snake_case (e.g. CamelCase -> camel_case)."""
    name = name.replace('-', '_')
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def normalize_type(type_name: str) -> str:
    """Map raw type to canonical type."""
    t = type_name.lower()
    if t in ('list', 'array', 'slice', 'vector'): return 'LIST'
    if t in ('set',): return 'SET'
    if t in ('map', 'dictionary', 'dict', 'record', 'hash'): return 'MAP'
    if t in ('int', 'integer', 'long', 'int64'): return 'INT'
    if t in ('float', 'double'): return 'FLOAT'
    if t in ('str', 'string'): return 'STRING'
    if t in ('bool', 'boolean'): return 'BOOLEAN'
    return 'OBJECT'
