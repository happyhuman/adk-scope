def calculate_precision(matches: int, total_target: int) -> float:
    """Calculates precision: matches / total_target."""
    if total_target > 0:
        return matches / total_target
    return 1.0


def calculate_recall(matches: int, total_base: int) -> float:
    """Calculates recall: matches / total_base."""
    if total_base > 0:
        return matches / total_base
    return 1.0


def calculate_f1(precision: float, recall: float) -> float:
    """Calculates F1 score: 2 * (P * R) / (P + R)."""
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0.0
