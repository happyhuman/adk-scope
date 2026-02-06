import argparse
import dataclasses
import logging
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from jellyfish import jaro_winkler_similarity

import numpy as np
from google.protobuf import text_format
from scipy.optimize import linear_sum_assignment

from google.adk.scope import features_pb2
from google.adk.scope.utils import args as adk_args
from google.adk.scope.utils import stats
from google.adk.scope.utils.similarity import SimilarityScorer

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
    if name == {"PYTHON", "PY"}:
        return "py"
    elif name in {"TYPESCRIPT", "TS"}:
        return "ts"
    elif name == "JAVA":
        return "java"
    elif name in {"GOLANG", "GO"}:
        return "go"
    else:
        return name.lower()


def _group_features_by_module(
    registry: features_pb2.FeatureRegistry,
) -> Dict[str, List[features_pb2.Feature]]:
    """Groups features by their module."""
    features = defaultdict(list)
    for f in registry.features:
        key = f.normalized_namespace or f.namespace or "Unknown Module"
        features[key].append(f)
    return features


def _fuzzy_match_namespaces(
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


class ReportGenerator:
    def __init__(
        self,
        base_registry: features_pb2.FeatureRegistry,
        target_registry: features_pb2.FeatureRegistry,
        alpha: float,
    ):
        self.base_registry = base_registry
        self.target_registry = target_registry

        self.features_base = _group_features_by_module(base_registry)
        self.features_target = _group_features_by_module(target_registry)
        _fuzzy_match_namespaces(self.features_base, self.features_target)
        self.alpha = alpha
    
    def generate_report(self, report_type) -> MatchResult:
        """Generates report."""
        if report_type == "raw":
            return self.generate_raw_report()
        elif report_type == "directional":  
            return self.generate_directional_report()
        elif report_type == "symmetric":
            return self.generate_symmetric_report()
        else:
            raise ValueError(f"Unknown report type: {report_type}")
    
    def generate_raw_report(self) -> MatchResult:
        """Generates a raw CSV report."""
        base_code = get_language_code(self.base_registry.language)
        target_code = get_language_code(self.target_registry.language)
        all_modules = sorted(
            set(self.features_base.keys()) | set(self.features_target.keys())
        )        
        csv_header = (
            f"{base_code}_namespace,{base_code}_member_of,{base_code}_name,"
            f"{target_code}_namespace,{target_code}_member_of,{target_code}_name,"
            "type,score"
        )
        csv_lines = [csv_header]

        def get_feature_cols(f: features_pb2.Feature) -> tuple[str, str, str]:
            ns = f.namespace or ""
            if not ns and f.normalized_namespace:
                ns = f.normalized_namespace

            mem = f.member_of or ""
            if not mem and f.normalized_member_of:
                mem = f.normalized_member_of
            if mem.lower() == "null":
                mem = ""

            name = f.original_name or f.normalized_name or ""
            return ns, mem, name

        def escape_csv(s):
            if s is None:
                return ""
            if "," in s or '"' in s or "\n" in s:
                return '"{}"'.format(s.replace('"', '""'))
            return s

        for module in all_modules:
            base_list = self.features_base.get(module, [])
            target_list = self.features_target.get(module, [])

            solid_matches = match_features(base_list, target_list, self.alpha)
            beta = max(0.0, self.alpha - _NEAR_MISS_THRESHOLD)
            potential_matches = match_features(base_list, target_list, beta)

            unmatched_base = base_list
            unmatched_target = target_list

            for f_base, f_target, score in solid_matches:
                b_ns, b_mem, b_name = get_feature_cols(f_base)
                t_ns, t_mem, t_name = get_feature_cols(f_target)
                f_type = get_type_display_name(f_base)
                csv_lines.append(
                    f"{escape_csv(b_ns)},{escape_csv(b_mem)},{escape_csv(b_name)},"
                    f"{escape_csv(t_ns)},{escape_csv(t_mem)},{escape_csv(t_name)},"
                    f"{escape_csv(f_type)},{score:.4f}"
                )

            for f_base, f_target, score in potential_matches:
                b_ns, b_mem, b_name = get_feature_cols(f_base)
                t_ns, t_mem, t_name = get_feature_cols(f_target)
                f_type = get_type_display_name(f_base)
                csv_lines.append(
                    f"{escape_csv(b_ns)},{escape_csv(b_mem)},{escape_csv(b_name)},"
                    f"{escape_csv(t_ns)},{escape_csv(t_mem)},{escape_csv(t_name)},"
                    f"{escape_csv(f_type)},{score:.4f}"
                )

            for f_base in unmatched_base:
                b_ns, b_mem, b_name = get_feature_cols(f_base)
                f_type = get_type_display_name(f_base)
                csv_lines.append(
                    f"{escape_csv(b_ns)},{escape_csv(b_mem)},{escape_csv(b_name)},"
                    f",,,{escape_csv(f_type)},0.0000"
                )

            for f_target in unmatched_target:
                t_ns, t_mem, t_name = get_feature_cols(f_target)
                f_type = get_type_display_name(f_target)
                csv_lines.append(
                    f",,,{escape_csv(t_ns)},{escape_csv(t_mem)},"
                    f"{escape_csv(t_name)},{escape_csv(f_type)},0.0000"
                )

        return MatchResult(master_content="\n".join(csv_lines), module_files={})

    def generate_directional_report(self) -> MatchResult:
        """Generates a directional report."""
        all_modules = sorted(self.features_base.keys())
        master_lines = []
        title_suffix = "Directional"
        master_lines.extend(
            [
                f"# Feature Matching Report: {title_suffix}",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"**Base:** {self.base_registry.language} ({self.base_registry.version})",
                f"**Target:** {self.target_registry.language}"
                f" ({self.target_registry.version})",
            ]
        )

        global_score_idx = len(master_lines)
        master_lines.append("GLOBAL_SCORE_PLACEHOLDER")
        master_lines.append("")

        header = "| Module | Features (Base) | Score | Status | Details |"
        divider = "|---|---|---|---|---|"
        master_lines.extend(["## Module Summary", header, divider])

        module_files = {}
        module_rows = []
        total_solid_matches = 0

        base_code = get_language_code(self.base_registry.language)
        target_code = get_language_code(self.target_registry.language)

        for module in all_modules:
            mod_base_list = self.features_base.get(module, [])
            mod_target_list = self.features_target.get(module, [])

            results = _process_module(
                module,
                mod_base_list,
                mod_target_list,
                self.alpha,
                "directional",
                base_code,
                target_code,
            )
            total_solid_matches += results["solid_matches_count"]
            module_rows.append((results["score"], results["row_content"]))
            if results.get("module_filename"):
                module_files[results["module_filename"]] = results["module_content"]

        module_rows.sort(key=lambda x: x[0], reverse=True)
        master_lines.extend([row for _, row in module_rows])

        total_base_features = len(self.base_registry.features)
        total_target_features = len(self.target_registry.features)

        precision = stats.calculate_precision(
            total_solid_matches, total_target_features
        )
        recall = stats.calculate_recall(
            total_solid_matches, total_base_features
        )
        parity_score = stats.calculate_f1(precision, recall)

        global_stats = (
            "\n| Metric | Score |\n"
            "|---|---|\n"
            f"| **Precision** | {precision:.2%} |\n"
            f"| **Recall** | {recall:.2%} |\n"
            f"| **F1 Score** | {parity_score:.2%} |\n\n"
            "> **Precision**: Of all features in the target, how many are "
            "correct matches to the base? (High score = low number of extra "
            "features in target)\n\n"
            "> **Recall**: Of all features in the base, how many were found in "
            "the target? (High score = low number of missing features in "
            "target)\n\n"
            "> **F1 Score**: A weighted average of Precision and Recall, "
            "providing a single measure of how well the target feature set "
            "matches the base."
        )

        master_lines[global_score_idx] = global_stats

        return MatchResult(
            master_content="\n".join(master_lines).strip(),
            module_files=module_files,
        )

    def generate_symmetric_report(self) -> MatchResult:
        """Generates a symmetric report."""
        all_modules = sorted(
            set(self.features_base.keys()) | set(self.features_target.keys())
        )
        master_lines = []
        title_suffix = "Symmetric"
        master_lines.extend(
            [
                f"# Feature Matching Report: {title_suffix}",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"**Base:** {self.base_registry.language} ({self.base_registry.version})",
                f"**Target:** {self.target_registry.language}"
                f" ({self.target_registry.version})",
            ]
        )

        global_score_idx = len(master_lines)
        master_lines.append("GLOBAL_SCORE_PLACEHOLDER")
        master_lines.append("")

        header = "| ADK | Module | Features (Base) | Score | Status | Details |"
        divider = "|---|---|---|---|---|---|"

        master_lines.extend(["## Module Summary", header, divider])

        module_files = {}
        module_rows = []
        total_solid_matches = 0

        base_code = get_language_code(self.base_registry.language)
        target_code = get_language_code(self.target_registry.language)

        for module in all_modules:
            mod_base_list = self.features_base.get(module, [])
            mod_target_list = self.features_target.get(module, [])

            results = _process_module(
                module,
                mod_base_list,
                mod_target_list,
                self.alpha,
                "symmetric",
                base_code,
                target_code,
            )
            total_solid_matches += results["solid_matches_count"]
            module_rows.append((results["score"], results["row_content"]))
            if results.get("module_filename"):
                module_files[results["module_filename"]] = results["module_content"]

        module_rows.sort(key=lambda x: x[0], reverse=True)
        master_lines.extend([row for _, row in module_rows])

        total_base_features = len(self.base_registry.features)
        total_target_features = len(self.target_registry.features)

        union_size = (
            total_base_features + total_target_features - total_solid_matches
        )
        parity_score = (
            total_solid_matches / union_size if union_size > 0 else 1.0
        )
        global_stats = (
            f"**Jaccard Index:** {parity_score:.2%}\n\n"
            "> The Jaccard Index measures the similarity between the "
            "two feature sets. A score of 100% indicates that both languages "
            "have identical features."
        )

        master_lines[global_score_idx] = global_stats

        return MatchResult(
            master_content="\n".join(master_lines).strip(),
            module_files=module_files,
        )


def match_registries(
    base_registry: features_pb2.FeatureRegistry,
    target_registry: features_pb2.FeatureRegistry,
    alpha: float,
    report_type: str = "symmetric",
) -> MatchResult:
    """Matches features and generates a master report + module sub-reports."""
    reporter = ReportGenerator(
        base_registry,
        target_registry,
        alpha,
    )

    return reporter.generate_report(report_type)


def _process_module(
    module: str,
    base_list: List[features_pb2.Feature],
    target_list: List[features_pb2.Feature],
    alpha: float,
    report_type: str,
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

    if report_type == "symmetric":
        union_size = mod_base_count + mod_target_count - mod_solid_count
        mod_score = mod_solid_count / union_size if union_size > 0 else 1.0
    else:  # directional
        precision = stats.calculate_precision(mod_solid_count, mod_target_count)
        recall = stats.calculate_recall(mod_solid_count, mod_base_count)
        mod_score = stats.calculate_f1(precision, recall)

    status_icon = (
        "✅" if mod_score == 1.0 else "⚠️" if mod_score >= 0.8 else "❌"
    )
    module_safe_name = module.replace(".", "_")
    module_filename = f"{module_safe_name}.md"

    details_link = f"[View Details]({{modules_dir}}/{module_filename})"
    if report_type == "symmetric":
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
    else:
        row_content = (
            f"| `{module}` | {mod_base_count} | {mod_score:.2%} | {status_icon}"
            f" | {details_link} |"
        )

    # Module Content
    mod_lines = [
        f"# Module: `{module}`",
        "[⬅️ Back to Master Report](../{master_report})",
        "",
        f"**Score:** {mod_score:.2%} ({status_icon})",
    ]
    if report_type == "directional":
        mod_lines.extend(
            [
                "\n| Metric | Score |",
                "|---|---|",
                f"| **Precision** | {precision:.2%} |",
                f"| **Recall** | {recall:.2%} |",
            ]
        )

    mod_total_features = (
        (mod_base_count + mod_target_count - mod_solid_count)
        if report_type == "symmetric"
        else mod_base_count
    )
    mod_lines.extend(["", f"**Features:** {mod_total_features}", ""])

    solid_matches.sort(
        key=lambda x: (get_type_priority(x[0]), x[0].normalized_name)
    )
    potential_matches.sort(
        key=lambda x: (get_type_priority(x[0]), x[0].normalized_name)
    )

    if solid_matches:
        mod_lines.append(
            f"### ✅ {'Solid' if report_type == 'symmetric' else 'Matched'}"
            " Features"
        )
        mod_lines.extend(
            [
                "| Type | Base Feature | Target Feature | Similarity Score |",
                "|---|---|---|---|",
            ]
        )
        mod_lines.extend(
            [
                f"| {get_type_display_name(f_base)} |"
                f" `{format_feature(f_base)}`"
                f" | `{format_feature(f_target)}` | {score:.2f} |"
                for f_base, f_target, score in solid_matches
            ]
        )
        mod_lines.append("")

    if potential_matches:
        mod_lines.extend(
            [
                "### ⚠️ Potential Matches",
                "| Type | Base Feature | Closest Target Candidate"
                " | Similarity |",
                "|---|---|---|---|",
            ]
        )
        mod_lines.extend(
            [
                f"| {get_type_display_name(f_base)} |"
                f" `{format_feature(f_base)}`"
                f" | `{format_feature(f_target)}` | {score:.2f} |"
                for f_base, f_target, score in potential_matches
            ]
        )
        mod_lines.append("")

    if report_type == "symmetric" and (unmatched_base or unmatched_target):
        mod_lines.extend(
            [
                "### ❌ Unmatched Features",
                "\n| Missing Feature | Missing In |",
                "|---|---|",
            ]
        )
        mod_lines.extend(
            [f"| `{format_feature(f)}` | Target |" for f in unmatched_base]
        )
        mod_lines.extend(
            [f"| `{format_feature(f)}` | Base |" for f in unmatched_target]
        )
        mod_lines.append("")
    elif report_type == "directional" and unmatched_base:
        mod_lines.extend(
            ["### ❌ Missing in Target", "| Missing Feature |", "|---|"]
        )
        mod_lines.extend([f"| `{format_feature(f)}` |" for f in unmatched_base])
        mod_lines.append("")

    return {
        "solid_matches_count": mod_solid_count,
        "score": mod_score,
        "row_content": row_content,
        "module_filename": module_filename,
        "module_content": "\n".join(mod_lines).strip(),
    }


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
    adk_args.add_verbose_argument(parser)
    args = parser.parse_args()
    adk_args.configure_logging(args)

    try:
        base_registry = read_feature_registry(args.base)
        target_registry = read_feature_registry(args.target)
    except Exception as e:
        logging.error(f"Error reading feature registries: {e}")
        sys.exit(1)

    result = match_registries(
        base_registry, target_registry, args.alpha, args.report_type
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.report_type == "raw":
        # Raw report is a single file, no modules directory needed
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.master_content)
            logging.info(
                f"Successfully wrote raw match report to {output_path}"
            )
        except Exception as e:
            logging.error(f"Error writing raw report to {output_path}: {e}")
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
        logging.info(f"Successfully wrote match report to {output_path}")
    except Exception as e:
        logging.error(f"Error writing report to {output_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
