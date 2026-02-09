"""
Unified type normalization for ADK Scope.
"""

import re
from typing import List


def normalize_name(name: str) -> str:
    """Convert name to snake_case (e.g. CamelCase -> camel_case)."""
    name = name.replace("-", "_")
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class TypeNormalizer:
    """A language-aware type normalizer."""

    def normalize(self, type_name: str, language: str) -> List[str]:
        """Normalize a type name for a given language."""
        if language == "python":
            return self._normalize_py_type(type_name)
        elif language == "typescript":
            return self._normalize_ts_type(type_name)
        # Add placeholders for future languages like 'java' and 'go'
        # Fallback for unknown languages: only normalize if it's a known simple
        # type, otherwise OBJECT
        normalized = self._simple_normalize(type_name)
        return [normalized]

    def _normalize_py_type(self, type_name: str) -> list[str]:
        """Map raw type to canonical types, handling generics."""
        type_name = type_name.strip()
        if not type_name:
            return ["OBJECT"]

        # Handle Union[A, B]
        if type_name.startswith("Union[") and type_name.endswith("]"):
            inner = type_name[6:-1]
            parts = self._split_generics(inner)
            result = []
            for p in parts:
                result.extend(self._normalize_py_type(p))
            return self._unique(result)

        # Handle Optional[T] -> T | None
        if type_name.startswith("Optional[") and type_name.endswith("]"):
            inner = type_name[9:-1]
            result = self._normalize_py_type(inner)
            if "NULL" not in result:
                result.append("NULL")
            return result

        # Handle tuple[A, B] -> [A, B]
        if type_name.lower().startswith("tuple[") and type_name.endswith("]"):
            inner = type_name[6:-1]
            parts = self._split_generics(inner)
            result = []
            for p in parts:
                result.extend(self._normalize_py_type(p))
            return self._unique(result)

        # Handle str | list[str]
        if "|" in type_name:
            parts = type_name.split("|")
            result = []
            for p in parts:
                result.extend(self._normalize_py_type(p.strip()))
            return self._unique(result)

        # Handle other generics like List[int] -> LIST
        if "[" in type_name and type_name.endswith("]"):
            base = type_name.split("[", 1)[0]
            return [self._simple_normalize(base)]

        return [self._simple_normalize(type_name)]

    def _normalize_ts_type(self, t: str) -> List[str]:
        # Handle fundamental TS types
        t = t.strip()
        if not t:
            return ["OBJECT"]

        if t in ("null", "undefined", "void"):
            return ["NULL"]

        # A | B
        if "|" in t:
            # Split by '|' only at the top level, respecting generics
            parts = self._split_unions(t)
            if len(parts) > 1:
                res = []
                for p in parts:
                    res.extend(self._normalize_ts_type(p.strip()))
                return self._unique(res)

        # Generics: Promise<T>, Array<T>
        match = re.match(r"([a-zA-Z0-9_]+)<(.+)>$", t)
        if match:
            base, inner = match.groups()

            if base == "Promise":
                return self._normalize_ts_type(inner)
            if base in (
                "Array",
                "ReadonlyArray",
                "Generator",
                "AsyncGenerator",
                "Iterable",
                "Iterator",
                "AsyncIterable",
                "AsyncIterator",
            ):
                return ["LIST"]
            if base == "Map":
                return ["MAP"]
            if base == "Set":
                return ["SET"]
            # Fallback for others
            return ["OBJECT"]

        t_lower = t.lower()

        if t_lower in ("string", "formattedstring", "path"):
            return ["STRING"]
        if t_lower in ("number", "int", "float", "integer", "double"):
            return ["NUMBER"]
        if t_lower in ("boolean", "bool"):
            return ["BOOLEAN"]
        if t_lower == "unknown":
            return ["UNKNOWN"]
        if t_lower in ("any", "object"):
            return ["OBJECT"]
        if t_lower.endswith("[]"):
            return ["LIST"]
        if (
            t_lower.startswith("map")
            or t_lower.startswith("record")
            or "{" in t
        ):
            return ["MAP"]
        if t_lower.startswith("set"):
            return ["SET"]

        return ["OBJECT"]

    def _simple_normalize(self, t: str) -> str:
        t = t.lower().strip()
        if t == "none":
            return "NULL"
        if t in (
            "list",
            "array",
            "slice",
            "vector",
            "generator",
            "asyncgenerator",
            "iterable",
            "iterator",
            "asynciterable",
            "asynciterator",
        ):
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

    def _split_generics(self, s: str) -> list[str]:
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

    def _unique(self, lst: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _split_unions(self, s: str) -> list[str]:
        """Split string by |, ignoring nested generics."""
        parts = []
        balance = 0
        current = []
        for char in s:
            if char == "<":
                balance += 1
                current.append(char)
            elif char == ">":
                balance -= 1
                current.append(char)
            elif char == "|" and balance == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            parts.append("".join(current).strip())
        return parts
