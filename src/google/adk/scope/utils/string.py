"""
Language normalization utilities for ADK Scope.
"""

def get_language_name(language_name: str) -> str:
    """Returns a properly capitalized display name for the language."""
    name = language_name.upper()
    if name in {"PYTHON", "PY"}:
        return "Python"
    elif name in {"TYPESCRIPT", "TS"}:
        return "TypeScript"
    elif name == "JAVA":
        return "Java"
    elif name in {"GOLANG", "GO"}:
        return "Go"
    else:
        return language_name.title()