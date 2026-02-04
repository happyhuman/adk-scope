import argparse
import dataclasses
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from google.protobuf import text_format
from google.adk.scope import features_pb2
from google.adk.scope.utils.similarity import SimilarityScorer
from google.adk.scope.utils import stats


_NEAR_MISS_THRESHOLD = 0.15


@dataclasses.dataclass
class MatchResult:
    master_content: str
    module_files: Dict[str, str]  # filename -> content


def format_feature(f: features_pb2.Feature) -> str:
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


def get_type_priority(f: features_pb2.Feature) -> int:
    """Returns priority: constructor < function < method < unknown."""
    type_name = get_type_display_name(f)
    priorities = {
        "constructor": 0,
        "function": 1,
        "method": 2,
        "unknown": 3,
    }
    return priorities.get(type_name, 99)


def read_feature_registry(file_path: str) -> features_pb2.FeatureRegistry:
    """Reads a FeatureRegistry from a text proto file."""
    registry = features_pb2.FeatureRegistry()
    with open(file_path, "rb") as f:
        text_format.Parse(f.read(), registry)
    return registry


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


def get_language_code(language_name: str) -> str:
    """Returns a short code for the language."""
    name = language_name.upper()
    if name == "PYTHON":
        return "py"
    elif name == "TYPESCRIPT":
        return "ts"
    elif name == "JAVA":
        return "java"
    elif name == "GOLANG":
        return "go"
    else:
        return name[:2].lower()


def match_registries(
    base_registry: features_pb2.FeatureRegistry,
    target_registry: features_pb2.FeatureRegistry,
    alpha: float,
    report_type: str = "symmetric",
) -> MatchResult:
    """Matches features and generates a master report + module sub-reports."""
    
    # 1. Group by Module (Normalized Namespace)
    features_base = defaultdict(list)
    for f in base_registry.features:
        key = f.normalized_namespace or f.namespace or "Unknown Module"
        features_base[key].append(f)

    features_target = defaultdict(list)
    for f in target_registry.features:
        key = f.normalized_namespace or f.namespace or "Unknown Module"
        features_target[key].append(f)

    if report_type == "directional":
        all_modules = sorted(features_base.keys())
    else:
        all_modules = sorted(
            set(features_base.keys()) | set(features_target.keys())
        )

    # Global Stats using Set logic for Jaccard/F1
    # We will accumulate counts as we process modules
    total_solid_matches = 0
    total_base_features = len(base_registry.features)
    total_target_features = len(target_registry.features)

    # Master Report Header
    from datetime import datetime
    
    master_lines = []
    
    if report_type == "raw":
        # Raw CSV Report
        # Columns: base_namespace,base_member_of,base_name,target_namespace,
        #          target_member_of,target_name,type,score
        csv_header = (
            "base_namespace,base_member_of,base_name,target_namespace,"
            "target_member_of,target_name,type,score"
        )
        csv_lines = [csv_header]
        
        def get_feature_cols(f: features_pb2.Feature) -> tuple[str, str, str]:
            ns = f.namespace or ""
            if not ns and f.normalized_namespace:
                ns = f.normalized_namespace
            
            # member_of
            mem = f.member_of or ""
            if not mem and f.normalized_member_of:
                mem = f.normalized_member_of
            if mem.lower() == "null":
                mem = ""
                
            # name
            name = f.original_name or f.normalized_name or ""
            return ns, mem, name

        def escape_csv(s):
            if s is None:
                return ""
            if ',' in s or '"' in s or '\n' in s:
                escaped = s.replace('"', '""')
                return f'"{escaped}"'
            return s

        for module in all_modules:
            base_list = features_base[module]
            target_list = features_target[module]
            
            # Pass 1: Solid Matches
            solid_matches = match_features(base_list, target_list, alpha)
            
            # Pass 2: Potential Matches (formerly Near Misses)
            beta = max(0.0, alpha - _NEAR_MISS_THRESHOLD)
            potential_matches = match_features(base_list, target_list, beta)
            
            # Leftovers
            unmatched_base = base_list
            unmatched_target = target_list
            
            for f_base, f_target, score in solid_matches:
                b_ns, b_mem, b_name = get_feature_cols(f_base)
                t_ns, t_mem, t_name = get_feature_cols(f_target)
                f_type = get_type_display_name(f_base)
                
                line = (
                    f"{escape_csv(b_ns)},{escape_csv(b_mem)},"
                    f"{escape_csv(b_name)},"
                    f"{escape_csv(t_ns)},{escape_csv(t_mem)},"
                    f"{escape_csv(t_name)},"
                    f"{escape_csv(f_type)},{score:.4f}"
                )
                csv_lines.append(line)

            for f_base, f_target, score in potential_matches:
                b_ns, b_mem, b_name = get_feature_cols(f_base)
                t_ns, t_mem, t_name = get_feature_cols(f_target)
                f_type = get_type_display_name(f_base)
                
                line = (
                    f"{escape_csv(b_ns)},{escape_csv(b_mem)},"
                    f"{escape_csv(b_name)},"
                    f"{escape_csv(t_ns)},{escape_csv(t_mem)},"
                    f"{escape_csv(t_name)},"
                    f"{escape_csv(f_type)},{score:.4f}"
                )
                csv_lines.append(line)

            for f_base in unmatched_base:
                b_ns, b_mem, b_name = get_feature_cols(f_base)
                f_type = get_type_display_name(f_base)
                
                line = (
                    f"{escape_csv(b_ns)},{escape_csv(b_mem)},"
                    f"{escape_csv(b_name)},"
                    f",,,"
                    f"{escape_csv(f_type)},0.0000"
                )
                csv_lines.append(line)

            for f_target in unmatched_target:
                t_ns, t_mem, t_name = get_feature_cols(f_target)
                f_type = get_type_display_name(f_target)
                
                line = (
                    f",,,"
                    f"{escape_csv(t_ns)},{escape_csv(t_mem)},"
                    f"{escape_csv(t_name)},"
                    f"{escape_csv(f_type)},0.0000"
                )
                csv_lines.append(line)

        return MatchResult(
            master_content="\n".join(csv_lines),
            module_files={}
        )

    title_suffix = "Symmetric" if report_type == "symmetric" else "Directional"
    master_lines.append(f"# Feature Matching Report: {title_suffix}")
    master_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    master_lines.append("")
    master_lines.append(
        f"**Base:** {base_registry.language} ({base_registry.version})"
    )
    master_lines.append(
        f"**Target:** {target_registry.language} ({target_registry.version})"
    )
    
    # Placeholder for Global Score (calculated at end)
    global_score_idx = len(master_lines)
    master_lines.append("GLOBAL_SCORE_PLACEHOLDER") 
    master_lines.append("")

    master_lines.append("## Module Summary")
    header = "| Module | Features (Base) | Score | Status | Details |"
    divider = "|---|---|---|---|---|"
    if report_type == "symmetric":
        header = "| ADK | Module | Features (Base) | Score | Status | Details |"
        divider = "|---|---|---|---|---|---|"
    
    master_lines.append(header)
    master_lines.append(divider)

    module_files = {}
    module_rows = []

    base_code = get_language_code(base_registry.language)
    target_code = get_language_code(target_registry.language)

    for module in all_modules:
        base_list = features_base[module]
        target_list = features_target[module]
        
        mod_base_count = len(base_list)
        mod_target_count = len(target_list)

        # Pass 1: Solid Matches
        solid_matches = match_features(base_list, target_list, alpha)
        mod_solid_count = len(solid_matches)
        total_solid_matches += mod_solid_count

        # Pass 2: Potential Matches (formerly Near Misses)
        beta = max(0.0, alpha - _NEAR_MISS_THRESHOLD)
        potential_matches = match_features(base_list, target_list, beta)

        # Leftovers
        unmatched_base = base_list
        unmatched_target = target_list

        # Calculate Module Score
        if report_type == "symmetric":
            union_size = mod_base_count + mod_target_count - mod_solid_count
            mod_score = (
                mod_solid_count / union_size if union_size > 0 else 1.0
            )
        else:  # directional
            precision = stats.calculate_precision(
                mod_solid_count, mod_target_count
            )
            recall = stats.calculate_recall(mod_solid_count, mod_base_count)
            mod_score = stats.calculate_f1(precision, recall)
        
        status_icon = "❌"
        if mod_score == 1.0:
            status_icon = "✅"
        elif mod_score >= 0.8:
            status_icon = "⚠️"

        # Safe filename
        module_safe_name = module.replace(".", "_")
        module_filename = f"{module_safe_name}.md"

        # Determine ADK Value (Symmetric Only)
        row_content = ""
        if report_type == "symmetric":
             adk_parts = []
             if mod_base_count > 0:
                 adk_parts.append(base_code)
             if mod_target_count > 0:
                 adk_parts.append(target_code)
             adk_value = ", ".join(adk_parts)
             
             row_content = (
                f"| {adk_value} | `{module}` | {mod_base_count} | {mod_score:.2%} | "
                f"{status_icon} | "
                f"[View Details]({{modules_dir}}/{module_filename}) |"
            )
        else:
            row_content = (
                f"| `{module}` | {mod_base_count} | {mod_score:.2%} | "
                f"{status_icon} | "
                f"[View Details]({{modules_dir}}/{module_filename}) |"
            )

        # Add to Master
        module_rows.append((mod_score, row_content))

        if report_type == "symmetric":
            mod_total_features = (
                mod_base_count + mod_target_count - mod_solid_count
            )
        else:
            mod_total_features = mod_base_count

        # Generate Module Content
        mod_lines = []
        mod_lines.append(f"# Module: `{module}`")
        # Back link usually works if we know the relative path structure.
        # Use placeholder {master_report} which will be replaced in main.
        # It should link to the master report file.
        mod_lines.append("[⬅️ Back to Master Report](../{master_report})")
        mod_lines.append("")
        mod_lines.append(f"**Score:** {mod_score:.2%} ({status_icon})")
        
        if report_type == "directional":
             mod_lines.append(
                "| Metric | Score |\n"
                "|---|---|\n"
                f"| **Precision** | {precision:.2%} |\n"
                f"| **Recall** | {recall:.2%} |"
            )
        else:
             # For symmetric, we usually just have the score (Jaccard).
             # We can make it a table too for consistency if desired.
             pass

        mod_lines.append("")
        mod_lines.append(f"**Features:** {mod_total_features}")
        mod_lines.append("")

        # Sort matches by type
        solid_matches.sort(
            key=lambda x: (get_type_priority(x[0]), x[0].normalized_name)
        )
        potential_matches.sort(
            key=lambda x: (get_type_priority(x[0]), x[0].normalized_name)
        )

        if report_type == "symmetric":
            if solid_matches:
                mod_lines.append("### ✅ Solid Matches")
                mod_lines.append(
                    "| Type | Base Feature | Target Feature | "
                    "Similarity Score |"
                )
                mod_lines.append("|---|---|---|---|")
                for f_base, f_target, score in solid_matches:
                    f_type = get_type_display_name(f_base)
                    mod_lines.append(
                        f"| {f_type} | `{format_feature(f_base)}` | "
                        f"`{format_feature(f_target)}` | {score:.2f} |"
                    )
                mod_lines.append("")

            if potential_matches:
                mod_lines.append("### ⚠️ Potential Matches")
                mod_lines.append(
                    "| Type | Base Feature | Closest Target Candidate | "
                    "Similarity |"
                )
                mod_lines.append("|---|---|---|---|")
                for f_base, f_target, score in potential_matches:
                    f_type = get_type_display_name(f_base)
                    mod_lines.append(
                        f"| {f_type} | `{format_feature(f_base)}` | "
                        f"`{format_feature(f_target)}` | {score:.2f} |"
                    )
                mod_lines.append("")

            if unmatched_base or unmatched_target:
                mod_lines.append("### ❌ Unmatched Features")
                mod_lines.append("| Missing Feature | Missing In |")
                mod_lines.append("|---|---|")
                for f_base in unmatched_base:
                    mod_lines.append(f"| `{format_feature(f_base)}` | Target |")
                for f_target in unmatched_target:
                    mod_lines.append(f"| `{format_feature(f_target)}` | Base |")
                mod_lines.append("")
        else:  # directional
            if solid_matches:
                mod_lines.append("### ✅ Matched Features")
                mod_lines.append(
                    "| Type | Base Feature | Target Feature | "
                    "Similarity Score |"
                )
                mod_lines.append("|---|---|---|---|")
                for f_base, f_target, score in solid_matches:
                    f_type = get_type_display_name(f_base)
                    mod_lines.append(
                        f"| {f_type} | `{format_feature(f_base)}` | "
                        f"`{format_feature(f_target)}` | {score:.2f} |"
                    )
                mod_lines.append("")

            if potential_matches:
                mod_lines.append("### ⚠️ Potential Matches")
                mod_lines.append(
                    "| Type | Base Feature | Closest Target Candidate | "
                    "Similarity |"
                )
                mod_lines.append("|---|---|---|---|")
                for f_base, f_target, score in potential_matches:
                    f_type = get_type_display_name(f_base)
                    mod_lines.append(
                        f"| {f_type} | `{format_feature(f_base)}` | "
                        f"`{format_feature(f_target)}` | {score:.2f} |"
                    )
                mod_lines.append("")

            if unmatched_base:
                mod_lines.append("### ❌ Missing in Target")
                mod_lines.append("| Missing Feature |")
                mod_lines.append("|---|")
                for f_base in unmatched_base:
                    mod_lines.append(f"| `{format_feature(f_base)}` |")
                mod_lines.append("")
                
            # Directional reports usually ignore target exclusives.
            # We flag missing-in-target features only.
        module_files[module_filename] = "\n".join(mod_lines).strip()

    # Sort modules by score descending
    module_rows.sort(key=lambda x: x[0], reverse=True)
    for _, row in module_rows:
        master_lines.append(row)

    # Calculate Global Score
    if report_type == "symmetric":
        union_size = (
            total_base_features + total_target_features - total_solid_matches
        )
        parity_score = (
            total_solid_matches / union_size if union_size > 0 else 1.0
        )
        global_stats = f"**Global Jaccard Index:** {parity_score:.2%}"
    else:
        precision = stats.calculate_precision(
            total_solid_matches, total_target_features
        )
        recall = stats.calculate_recall(
            total_solid_matches, total_base_features
        )
        parity_score = stats.calculate_f1(precision, recall)
        
        global_stats = (
            "| Metric | Score |\n"
            "|---|---|\n"
            f"| **Precision** | {precision:.2%} |\n"
            f"| **Recall** | {recall:.2%} |\n"
            f"| **Global F1 Score** | {parity_score:.2%} |"
        )

    master_lines[
        global_score_idx
    ] = global_stats

    return MatchResult(
        master_content="\n".join(master_lines).strip(),
        module_files=module_files
    )


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
        default=0.8,
        help="Similarity threshold (0.0 to 1.0) defaults to 0.8.",
    )
    parser.add_argument(
        "--report-type",
        choices=["symmetric", "directional", "raw"],
        default="symmetric",
        help="Type of gap report to generate (symmetric, directional, or raw).",
    )
    args = parser.parse_args()

    try:
        base_registry = read_feature_registry(args.base)
        target_registry = read_feature_registry(args.target)
    except Exception as e:
        print(f"Error reading feature registries: {e}", file=sys.stderr)
        sys.exit(1)

    result = match_registries(
        base_registry, target_registry, args.alpha, args.report_type
    )

    output_path = Path(args.output)
    
    if args.report_type == "raw":
        # Raw report is a single file, no modules directory needed
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.master_content)
            print(f"Successfully wrote raw match report to {output_path}")
        except Exception as e:
            print(
                f"Error writing raw report to {output_path}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        return

    # Create module directory
    if result.module_files:
        modules_dir_name = f"{output_path.stem}_modules"
        modules_dir = output_path.parent / modules_dir_name
        modules_dir.mkdir(parents=True, exist_ok=True)
        
        # Write module files
        for filename, content in result.module_files.items():
            # Replace placeholder for master report link
            # The link is relative from module dir to master report
            # So name is enough.
            final_content = content.replace("{master_report}", output_path.name)
            (modules_dir / filename).write_text(final_content)
            
        # Replace placeholder in Master Report
        # We assume master report is in parent of modules_dir
        # modules_dir relative to master report is just the dir name
        master_report = result.master_content.replace(
            "{modules_dir}", modules_dir_name
        )
    else:
        master_report = result.master_content.replace("{modules_dir}", ".")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(master_report)
        print(f"Successfully wrote match report to {output_path}")
    except Exception as e:
        print(f"Error writing report to {output_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
