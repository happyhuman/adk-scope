import argparse
import dataclasses
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from google.protobuf import text_format
from jellyfish import jaro_winkler_similarity
from scipy.optimize import linear_sum_assignment

from google.adk.scope import features_pb2
from google.adk.scope import matcher
from google.adk.scope.utils import args as adk_args
from google.adk.scope.utils import stats
from google.adk.scope.utils.similarity import SimilarityScorer

_NEAR_MISS_THRESHOLD = 0.15


def _group_features_by_module(
    registry: features_pb2.FeatureRegistry,
) -> Dict[str, List[features_pb2.Feature]]:
    """Groups features by their module."""
    features = defaultdict(list)
    for f in registry.features:
        key = f.normalized_namespace or f.namespace or "Unknown Module"
        features[key].append(f)
    return features


def _get_language_code(language_name: str) -> str:
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


def _read_feature_registry(file_path: str) -> features_pb2.FeatureRegistry:
    """Reads a FeatureRegistry from a text proto file."""
    registry = features_pb2.FeatureRegistry()
    with open(file_path, "rb") as f:
        text_format.Parse(f.read(), registry)
    return registry


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
        matcher.fuzzy_match_namespaces(self.features_base, self.features_target)
        self.alpha = alpha

    def generate_report(self, report_type) -> matcher.MatchResult:
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
        base_code = _get_language_code(self.base_registry.language)
        target_code = _get_language_code(self.target_registry.language)
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

        def esc_csv(s):
            if s is None:
                return ""
            if "," in s or '"' in s or "\n" in s:
                return '"{}"'.format(s.replace('"', '""'))
            return s

        for module in all_modules:
            base_list = self.features_base.get(module, [])
            target_list = self.features_target.get(module, [])

            solid_matches = matcher.match_features(
                base_list, target_list, self.alpha
            )
            beta = max(0.0, self.alpha - _NEAR_MISS_THRESHOLD)
            potential_matches = matcher.match_features(
                base_list, target_list, beta
            )

            unmatched_base = base_list
            unmatched_target = target_list

            for f_base, f_target, score in solid_matches:
                b_ns, b_mem, b_name = get_feature_cols(f_base)
                t_ns, t_mem, t_name = get_feature_cols(f_target)
                f_type = matcher.get_type_display_name(f_base)
                csv_lines.append(
                    f"{esc_csv(b_ns)},{esc_csv(b_mem)},{esc_csv(b_name)},"
                    f"{esc_csv(t_ns)},{esc_csv(t_mem)},{esc_csv(t_name)},"
                    f"{esc_csv(f_type)},{score:.4f}"
                )

            for f_base, f_target, score in potential_matches:
                b_ns, b_mem, b_name = get_feature_cols(f_base)
                t_ns, t_mem, t_name = get_feature_cols(f_target)
                f_type = matcher.get_type_display_name(f_base)
                csv_lines.append(
                    f"{esc_csv(b_ns)},{esc_csv(b_mem)},{esc_csv(b_name)},"
                    f"{esc_csv(t_ns)},{esc_csv(t_mem)},{esc_csv(t_name)},"
                    f"{esc_csv(f_type)},{score:.4f}"
                )

            for f_base in unmatched_base:
                b_ns, b_mem, b_name = get_feature_cols(f_base)
                f_type = matcher.get_type_display_name(f_base)
                csv_lines.append(
                    f"{esc_csv(b_ns)},{esc_csv(b_mem)},{esc_csv(b_name)},"
                    f",,,{esc_csv(f_type)},0.0000"
                )

            for f_target in unmatched_target:
                t_ns, t_mem, t_name = get_feature_cols(f_target)
                f_type = get_type_display_name(f_target)
                csv_lines.append(
                    f",,,{esc_csv(t_ns)},{esc_csv(t_mem)},"
                    f"{esc_csv(t_name)},{esc_csv(f_type)},0.0000"
                )

        return matcher.MatchResult(
            master_content="\n".join(csv_lines),
            module_files={},
        )

    def generate_directional_report(self) -> matcher.MatchResult:
        """Generates a directional report."""
        all_modules = sorted(self.features_base.keys())
        master_lines = []
        title_suffix = "Directional"
        master_lines.extend(
            [
                f"# Feature Matching Report: {title_suffix}",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                (
                    f"**Base:** {self.base_registry.language}"
                    f" ({self.base_registry.version})"
                ),
                (
                    f"**Target:** {self.target_registry.language}"
                    f" ({self.target_registry.version})"
                ),
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

        base_code = _get_language_code(self.base_registry.language)
        target_code = _get_language_code(self.target_registry.language)

        for module in all_modules:
            mod_base_list = self.features_base.get(module, [])
            mod_target_list = self.features_target.get(module, [])

            results = matcher.process_module(
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
                module_files[results["module_filename"]] = results[
                    "module_content"
                ]

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

        return matcher.MatchResult(
            master_content="\n".join(master_lines).strip(),
            module_files=module_files,
        )

    def generate_symmetric_report(self) -> matcher.MatchResult:
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
                "| Registry | Language | Version |",
                "| :--- | :--- | :--- |",
                (
                    f"| **Base** | {self.base_registry.language} |"
                    f" {self.base_registry.version} |"
                ),
                (
                    f"| **Target** | {self.target_registry.language} |"
                    f" {self.target_registry.version} |"
                ),
                "",
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

        base_code = _get_language_code(self.base_registry.language)
        target_code = _get_language_code(self.target_registry.language)

        for module in all_modules:
            mod_base_list = self.features_base.get(module, [])
            mod_target_list = self.features_target.get(module, [])

            results = matcher.process_module(
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
                module_files[results["module_filename"]] = results[
                    "module_content"
                ]

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

        return matcher.MatchResult(
            master_content="\n".join(master_lines).strip(),
            module_files=module_files,
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
    adk_args.add_verbose_argument(parser)
    args = parser.parse_args()
    adk_args.configure_logging(args)

    try:
        base_registry = _read_feature_registry(args.base)
        target_registry = _read_feature_registry(args.target)
    except Exception as e:
        logging.error(f"Error reading feature registries: {e}")
        sys.exit(1)

    result = matcher.match_registries(
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
