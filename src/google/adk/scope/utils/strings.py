"""
String utility functions for ADK Scope.
"""

import re


def normalize_name(name: str) -> str:
    """Convert name to snake_case (e.g. CamelCase -> camel_case)."""
    name = name.replace("-", "_")
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def normalize_type(type_name: str) -> str:
    """Legacy wrapper for simple normalization (returns single string)."""
    # For backward compatibility if needed, else we can migrate all callers
    # But signature in Features expects lists now for normalized_types if we
    # change converter
    # Let's keep this as simple for now or deprecate?
    # The new converter will use normalize_type_complex which returns List[str]
    # But wait, existing calls might expect single str?
    # Actually, converter.py calls: [normalize_type(t) for t in types]
    # We should change converter to use new function.
    return _simple_normalize(type_name)


def normalize_type_complex(type_name: str) -> list[str]:
    """Map raw type to canonical types, handling generics."""
    type_name = type_name.strip()
    if not type_name:
        return ["OBJECT"]

    # Handle Union[A, B]
    if type_name.startswith("Union[") and type_name.endswith("]"):
        inner = type_name[6:-1]
        parts = _split_generics(inner)
        result = []
        for p in parts:
            result.extend(normalize_type_complex(p))
        return _unique(result)

    # Handle Optional[T] -> T | None
    if type_name.startswith("Optional[") and type_name.endswith("]"):
        inner = type_name[9:-1]
        result = normalize_type_complex(inner)
        if "null" not in result:
            result.append("null")
        return result

    # Handle AsyncGenerator[A, B] -> A | B
    if type_name.startswith("AsyncGenerator[") and type_name.endswith("]"):
        inner = type_name[15:-1]
        parts = _split_generics(inner)
        result = []
        for p in parts:
            result.extend(normalize_type_complex(p))
        return _unique(result)

    # Handle tuple[A, B] -> [A, B]
    if type_name.lower().startswith("tuple[") and type_name.endswith("]"):
        # tuple[...] or Tuple[...]
        inner = type_name[6:-1]
        parts = _split_generics(inner)
        result = []
        for p in parts:
            result.extend(normalize_type_complex(p))
        return _unique(result)

    # Handle other generics like List[int] -> LIST
    if "[" in type_name and type_name.endswith("]"):
        base = type_name.split("[", 1)[0]
        return [_simple_normalize(base)]

    return [_simple_normalize(type_name)]


def _simple_normalize(t: str) -> str:
    t = t.lower().strip()
    if t == "none":
        return "null"
    if t in ("list", "array", "slice", "vector"):
        return "LIST"
    if t in ("set",):
        return "SET"
    if t in ("map", "dictionary", "dict", "record", "hash"):
        return "MAP"
    if t in ("int", "integer", "long", "int64", "float", "double"):
        return "NUMBER"
    if t in ("str", "string"):
        return "STRING"
    if t in ("bool", "boolean"):
        return "BOOLEAN"
    if t == "any":
        return "OBJECT"
    if not t:
        return "OBJECT"
    return "OBJECT"


def _split_generics(s: str) -> list[str]:
    """Split string by comma, ignoring nested brackets."""
    parts = []
    balance = 0
    current = []
    for char in s:
        if char == "[":
            balance += 1
            current.append(char)
        elif char == "]":
            balance -= 1
            current.append(char)
        elif char == "," and balance == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def _unique(lst: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
