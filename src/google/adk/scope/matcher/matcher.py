from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from jellyfish import jaro_winkler_similarity
from scipy.optimize import linear_sum_assignment

from google.adk.scope import features_pb2
from google.adk.scope.utils import stats
from google.adk.scope.utils.similarity import SimilarityScorer

_NEAR_MISS_THRESHOLD = 0.15


def _format_feature(f: features_pb2.Feature) -> str:
    name = f.original_name or f.normalized_name
    member = f.member_of
    if member and member.lower() != "null":
        return f"{member}.{name}"
    return name


def get_type_display_name(f: features_pb2.Feature) -> str:
    FeatureType = features_pb2.Feature.Type
    if f.type == FeatureType.CONSTRUCTOR:
        return "constructor"
    elif f.type in (FeatureType.FUNCTION, FeatureType.CLASS_METHOD):
        return "function"
    elif f.type == features_pb2.Feature.Type.INSTANCE_METHOD:
        return "method"
    else:
        return "unknown"


def _get_type_priority(f: features_pb2.Feature) -> int:
    """Returns priority: constructor < function < method < unknown."""
    type_name = get_type_display_name(f)
    priorities = {
        "constructor": 0,
        "function": 1,
        "method": 2,
        "unknown": 3,
    }
    return priorities.get(type_name, 99)


def match_features(
    base_features: List[features_pb2.Feature],
    target_features: List[features_pb2.Feature],
    alpha: float,
) -> List[Tuple[features_pb2.Feature, features_pb2.Feature, float]]:
    """Matches features between two lists using Hungarian algorithm."""
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
        f for i, f in enumerate(base_features) if i not in matched_base_indices
    ]
    target_features[:] = [
        f
        for i, f in enumerate(target_features)
        if i not in matched_target_indices
    ]

    return matches


def fuzzy_match_namespaces(
    features_base: Dict[str, List[features_pb2.Feature]],
    features_target: Dict[str, List[features_pb2.Feature]],
) -> None:
    """Remaps target namespaces to base namespaces using fuzzy matching."""

    base_namespaces = sorted(list(features_base.keys()))
    remapped_features = defaultdict(list, {k: [] for k in features_base})

    for t_ns, features in features_target.items():
        if t_ns in base_namespaces:
            remapped_features[t_ns].extend(features)
            continue

        if not base_namespaces:
            # No base to match against, so keep original target namespace
            remapped_features[t_ns].extend(features)
            continue

        best_match, best_score = max(
            (
                (b_ns, jaro_winkler_similarity(t_ns, b_ns))
                for b_ns in base_namespaces
            ),
            key=lambda item: item[1],
            default=(None, 0.0),
        )

        if best_score > 0.8 and best_match:
            remapped_features[best_match].extend(features)

    features_target.clear()
    features_target.update(remapped_features)


def process_module(
    module: str,
    base_list: List[features_pb2.Feature],
    target_list: List[features_pb2.Feature],
    alpha: float,
    base_lang_name: str,
    target_lang_name: str,
    base_lang_code: str,
    target_lang_code: str,
) -> Dict:
    """Analyzes a single module and generates its report content."""
    mod_base_count = len(base_list)
    mod_target_count = len(target_list)

    solid_matches = match_features(base_list, target_list, alpha)
    mod_solid_count = len(solid_matches)

    beta = max(0.0, alpha - _NEAR_MISS_THRESHOLD)
    potential_matches = match_features(base_list, target_list, beta)

    unmatched_base = base_list
    unmatched_target = target_list

    union_size = mod_base_count + mod_target_count - mod_solid_count
    mod_score = mod_solid_count / union_size if union_size > 0 else 1.0

    status_icon = (
        "✅" if mod_score == 1.0 else "⚠️" if mod_score >= 0.8 else "❌"
    )
    module_safe_name = module.replace(".", "_")
    module_filename = f"{module_safe_name}.md"

    details_link = f"[View Details]({{modules_dir}}/{module_filename})"
    adk_parts = []
    if mod_base_count > 0:
        adk_parts.append(base_lang_code)
    if mod_target_count > 0:
        adk_parts.append(target_lang_code)
    adk_value = ", ".join(adk_parts)
    row_content = (
        f"| {adk_value} | `{module}` | {mod_base_count} | {mod_score:.2%} |"
        f" {status_icon} | {details_link} |"
    )

    # Module Content
    mod_lines = [
        f"# Module: `{module}`",
        "[⬅️ Back to Master Report](../{master_report})",
        "",
        f"**Score:** {mod_score:.2%} ({status_icon})",
    ]
    mod_total_features = mod_base_count + mod_target_count - mod_solid_count
    mod_lines.extend(["", f"**Features:** {mod_total_features}", ""])

    solid_matches.sort(
        key=lambda x: (_get_type_priority(x[0]), x[0].normalized_name)
    )
    potential_matches.sort(
        key=lambda x: (_get_type_priority(x[0]), x[0].normalized_name)
    )

    if solid_matches:
        mod_lines.append(
            "### ✅ Solid Features"
        )
        mod_lines.extend(
            [
                f"| Type | {base_lang_name} Feature | {target_lang_name} Feature | Similarity Score |",
                "|---|---|---|---|",
            ]
        )
        mod_lines.extend(
            [
                f"| {get_type_display_name(f_base)} |"
                f" `{_format_feature(f_base)}`"
                f" | `{_format_feature(f_target)}` | {score:.2f} |"
                for f_base, f_target, score in solid_matches
            ]
        )
        mod_lines.append("")

    if potential_matches:
        mod_lines.extend(
            [
                "### ⚠️ Potential Matches",
                f"| Type | {base_lang_name} Feature | Closest {target_lang_name} Candidate"
                " | Similarity |",
                "|---|---|---|---|",
            ]
        )
        mod_lines.extend(
            [
                f"| {get_type_display_name(f_base)} |"
                f" `{_format_feature(f_base)}`"
                f" | `{_format_feature(f_target)}` | {score:.2f} |"
                for f_base, f_target, score in potential_matches
            ]
        )
        mod_lines.append("")

    if unmatched_base or unmatched_target:
        mod_lines.extend(
            [
                "### ❌ Unmatched Features",
                "\n| Missing Feature | Missing In |",
                "|---|---|",
            ]
        )
        mod_lines.extend(
            [f"| `{_format_feature(f)}` | {target_lang_name} |" for f in unmatched_base]
        )
        mod_lines.extend(
            [f"| `{_format_feature(f)}` | {base_lang_name} |" for f in unmatched_target]
        )
        mod_lines.append("")

    return {
        "solid_matches_count": mod_solid_count,
        "score": mod_score,
        "row_content": row_content,
        "module_filename": module_filename,
        "module_content": "\n".join(mod_lines).strip(),
    }
