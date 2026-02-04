import argparse
import sys
from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from google.protobuf import text_format
from google.adk.scope import features_pb2
from google.adk.scope.utils.similarity import SimilarityScorer


def read_feature_registry(file_path: str) -> features_pb2.FeatureRegistry:
    """Reads a FeatureRegistry from a text proto file.

    Args:
      file_path: Path to the .txtpb file.

    Returns:
      A FeatureRegistry instance.
    """
    registry = features_pb2.FeatureRegistry()
    with open(file_path, "rb") as f:
        text_format.Parse(f.read(), registry)
    return registry


def match_features(
    base_features: List[features_pb2.Feature],
    target_features: List[features_pb2.Feature],
    alpha: float,
) -> List[Tuple[features_pb2.Feature, features_pb2.Feature, float]]:
    """Matches features between two lists based on a similarity threshold.

    Features that score higher than `alpha` are considered matches, added to
    the result list, and removed from both input lists to avoid duplicate
    matching. Uses the Hungarian algorithm for global optimization.

    Args:
      base_features: The first list of features. Modified in-place.
      target_features: The second list of features. Modified in-place.
      alpha: The similarity threshold (0.0 to 1.0) for a match.

    Returns:
      A list of tuples (feature_from_base, feature_from_target,
                        similarity_score).
    """
    if not base_features or not target_features:
        return []

    scorer = SimilarityScorer(alpha=alpha)
    matches = []

    # Build Cost Matrix (Rows=Base, Cols=Target)
    n_base = len(base_features)
    n_target = len(target_features)
    similarity_matrix = np.zeros((n_base, n_target))

    for i, f1 in enumerate(base_features):
        for j, f2 in enumerate(target_features):
            similarity_matrix[i, j] = scorer.get_similarity_score(f1, f2)

    # Run Hungarian Algorithm (Global Optimization)
    row_ind, col_ind = linear_sum_assignment(similarity_matrix, maximize=True)

    matched_base_indices = set()
    matched_target_indices = set()

    # Filter Optimal Assignments by Alpha Threshold
    for r, c in zip(row_ind, col_ind):
        score = similarity_matrix[r, c]
        if score > alpha:
            matches.append((base_features[r], target_features[c], score))
            matched_base_indices.add(r)
            matched_target_indices.add(c)

    # Update the input lists in-place (Remove matched items)
    base_features[:] = [
        f for i, f in enumerate(base_features)
        if i not in matched_base_indices
    ]
    target_features[:] = [
        f for i, f in enumerate(target_features)
        if i not in matched_target_indices
    ]

    return matches

    return matches


def match_registries(
    base_registry: features_pb2.FeatureRegistry,
    target_registry: features_pb2.FeatureRegistry,
    alpha: float,
    report_type: str = "symmetric",
) -> str:
    """Matches features between two FeatureRegistries and generates a report.

    This delegates to `match_features` and constructs a human-readable
    Markdown string listing the matched original feature names.

    Args:
      base_registry: The first FeatureRegistry.
      target_registry: The second FeatureRegistry.
      alpha: The similarity threshold (0.0 to 1.0) for a match.
      report_type: 'symmetric' or 'directional' reporting style.

    Returns:
      A Markdown string documenting the matched features.
    """
    from collections import defaultdict

    base_features = list(base_registry.features)
    target_features = list(target_registry.features)

    total_base = len(base_features)
    total_target = len(target_features)

    # Pass 1: Solid Matches (mutates lists)
    solid_matches = match_features(base_features, target_features, alpha)

    # Pass 2: Near Misses (mutates lists)
    beta = max(0.0, alpha - 0.15)
    near_misses = match_features(base_features, target_features, beta)

    # Leftovers
    unmatched_base = base_features
    unmatched_target = target_features

    if report_type == "symmetric":
        union_size = total_base + total_target - len(solid_matches)
        parity_score = (
            len(solid_matches) / union_size if union_size > 0 else 1.0
        )
        score_name = "Jaccard Index"
    else:  # directional
        precision = (
            len(solid_matches) / total_target if total_target > 0 else 1.0
        )
        recall = len(solid_matches) / total_base if total_base > 0 else 1.0
        if precision + recall == 0:
            parity_score = 0.0
        else:
            parity_score = 2 * (precision * recall) / (precision + recall)
        score_name = "F1 Score"

    lines = [
        "# Cross-Language Feature Parity Report",
        f"**Base:** {base_registry.language} ({base_registry.version})",
        f"**Target:** {target_registry.language} ({target_registry.version})",
        f"**Feature Parity Score ({score_name}):** {parity_score:.1%}",
        "",
    ]

    modules = defaultdict(lambda: {
        'solid': [],
        'near': [],
        'unmatched_base': [],
        'unmatched_target': []
    })

    for f_base, f_target, score in solid_matches:
        ns = f_base.namespace or "Unknown Module"
        modules[ns]['solid'].append((f_base, f_target, score))

    for f_base, f_target, score in near_misses:
        ns = f_base.namespace or "Unknown Module"
        modules[ns]['near'].append((f_base, f_target, score))

    for f_base in unmatched_base:
        ns = f_base.namespace or "Unknown Module"
        modules[ns]['unmatched_base'].append(f_base)

    for f_target in unmatched_target:
        ns = f_target.namespace or "Unknown Module"
        modules[ns]['unmatched_target'].append(f_target)

    def format_feature(f: features_pb2.Feature) -> str:
        name = f.original_name or f.normalized_name
        member = f.member_of
        if member and member.lower() != "null":
            return f"{member}.{name}"
        return name

    def get_type_display_name(f: features_pb2.Feature) -> str:
        """Map Feature Type enum to a human-readable Type string."""
        FeatureType = features_pb2.Feature.Type
        if f.type == FeatureType.CONSTRUCTOR:
            return "Constructor"
        elif f.type in (FeatureType.FUNCTION, FeatureType.CLASS_METHOD):
            return "Function"
        elif f.type == features_pb2.Feature.Type.INSTANCE_METHOD:
            return "Method"
        else:
            return "Unknown"

    for ns in sorted(modules.keys()):
        lines.append(f"## Module '{ns}'")
        lines.append("")
        mod_data = modules[ns]

        if report_type == "symmetric":
            if mod_data['solid']:
                lines.append("### ✅ Solid Matches")
                lines.append(
                    "| Type | Base Feature | Target Feature | "
                    "Similarity Score |"
                )
                lines.append("|---|---|---|---|")
                for f_base, f_target, score in mod_data['solid']:
                    f_type = get_type_display_name(f_base)
                    lines.append(
                        f"| {f_type} | `{format_feature(f_base)}` | "
                        f"`{format_feature(f_target)}` | {score:.2f} |"
                    )
                lines.append("")

            if mod_data['near']:
                lines.append("### \u26A0\uFE0F Near Misses")
                lines.append(
                    "| Type | Base Feature | Closest Target Candidate | "
                    "Similarity |"
                )
                lines.append("|---|---|---|---|")
                for f_base, f_target, score in mod_data['near']:
                    f_type = get_type_display_name(f_base)
                    lines.append(
                        f"| {f_type} | `{format_feature(f_base)}` | "
                        f"`{format_feature(f_target)}` | {score:.2f} |"
                    )
                lines.append("")

            if mod_data['unmatched_base'] or mod_data['unmatched_target']:
                lines.append("### \u274C Unmatched Features")
                lines.append("| Missing Feature | Missing In |")
                lines.append("|---|---|")
                for f_base in mod_data['unmatched_base']:
                    lines.append(f"| `{format_feature(f_base)}` | Target |")
                for f_target in mod_data['unmatched_target']:
                    lines.append(f"| `{format_feature(f_target)}` | Base |")
                lines.append("")
        else:
            if mod_data['solid']:
                lines.append("### ✅ Matched Features")
                lines.append(
                    "| Type | Base Feature | Target Feature | "
                    "Similarity Score |"
                )
                lines.append("|---|---|---|---|")
                for f_base, f_target, score in mod_data['solid']:
                    f_type = get_type_display_name(f_base)
                    lines.append(
                        f"| {f_type} | `{format_feature(f_base)}` | "
                        f"`{format_feature(f_target)}` | {score:.2f} |"
                    )
                lines.append("")

            if mod_data['near']:
                lines.append("### \u26A0\uFE0F Inconsistencies (Near Misses)")
                lines.append(
                    "| Type | Base Feature | Closest Target Candidate | "
                    "Similarity |"
                )
                lines.append("|---|---|---|---|")
                for f_base, f_target, score in mod_data['near']:
                    f_type = get_type_display_name(f_base)
                    lines.append(
                        f"| {f_type} | `{format_feature(f_base)}` | "
                        f"`{format_feature(f_target)}` | {score:.2f} |"
                    )
                lines.append("")

            if mod_data['unmatched_base']:
                lines.append("### \u274C Missing in Target (Base Exclusive)")
                lines.append("| Missing Feature |")
                lines.append("|---|")
                for f_base in mod_data['unmatched_base']:
                    lines.append(f"| `{format_feature(f_base)}` |")
                lines.append("")

            if mod_data['unmatched_target']:
                lines.append("### \u274C Target Exclusives")
                lines.append("| Extra Target Feature |")
                lines.append("|---|")
                for f_target in mod_data['unmatched_target']:
                    lines.append(f"| `{format_feature(f_target)}` |")
                lines.append("")

    return "\n".join(lines).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Match ADK features between two languages."
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Path to the base FeatureRegistry .txtpb file.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to the target FeatureRegistry .txtpb file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the Markdown report.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Similarity threshold (0.0 to 1.0) defaults to 0.7.",
    )
    parser.add_argument(
        "--report-type",
        choices=["symmetric", "directional"],
        default="symmetric",
        help="Type of gap report to generate (symmetric or directional).",
    )
    args = parser.parse_args()

    try:
        base_registry = read_feature_registry(args.base)
        target_registry = read_feature_registry(args.target)
    except Exception as e:
        print(f"Error reading feature registries: {e}", file=sys.stderr)
        sys.exit(1)

    report = match_registries(
        base_registry, target_registry, args.alpha, args.report_type
    )

    try:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Successfully wrote match report to {args.output}")
    except Exception as e:
        print(f"Error writing report to {args.output}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
