
import argparse
import dataclasses
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from google.protobuf import text_format
import pandas as pd

from google.adk.scope import features_pb2
from google.adk.scope.matcher import matcher
from google.adk.scope.reporter import raw
from google.adk.scope.utils import args as adk_args
from google.adk.scope.utils import stats
from google.adk.scope.utils.similarity import SimilarityScorer

_NEAR_MISS_THRESHOLD = 0.15


@dataclasses.dataclass
class MatchResult:
    master_content: str
    module_files: Dict[str, str]  # filename -> content


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
    if name in {"PYTHON", "PY"}:
        return "py"
    elif name in {"TYPESCRIPT", "TS"}:
        return "ts"
    elif name == "JAVA":
        return "java"
    elif name in {"GOLANG", "GO"}:
        return "go"
    else:
        return name.lower()


def _get_language_name(language_name: str) -> str:
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


def _read_feature_registry(file_path: str) -> features_pb2.FeatureRegistry:
    """Reads a FeatureRegistry from a text proto file."""
    registry = features_pb2.FeatureRegistry()
    with open(file_path, "rb") as f:
        text_format.Parse(f.read(), registry)
    return registry


def match_registries(
    registries: List[features_pb2.FeatureRegistry],
    alpha: float,
    report_type: str = "md",  # Kept for backward compatibility/matrix logic
    common: bool = False,
    output_path: Optional[Path] = None,
) -> MatchResult:
    """Matches features and generates reports."""
    if report_type == "matrix":
        reporter = MatrixReportGenerator(registries, alpha, common)
        return reporter.generate_report("matrix")
    else:
        if len(registries) != 2:
            raise ValueError(
                f"Report type '{report_type}' requires exactly 2 registries."
            )
        
        # New unified flow for standard reports
        generator = raw.RawReportGenerator(registries[0], registries[1])
        
        # Generate DataFrame (and CSV if path provided)
        csv_path = None
        if output_path:
            # If output is "report.md", csv will be "report.csv"
            # If output is "report.csv", md will be "report.md"
            stem = output_path.stem
            parent = output_path.parent
            csv_path = str(parent / f"{stem}.csv")
            
        df = generator.generate(output_path=csv_path)
        
        # Generate Markdown Report from DataFrame
        reporter = ReportGenerator(registries[0], registries[1], df)
        return reporter.generate_md_report()


class MatrixReportGenerator:
    def __init__(
        self,
        registries: List[features_pb2.FeatureRegistry],
        alpha: float,
        common: bool = False,
    ):
        self.registries = registries
        self.alpha = alpha
        self.common = common

        self.langs = [_get_language_name(r.language) for r in self.registries]

    def _compute_jaccard_matrix(self) -> List[str]:
        n = len(self.registries)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                if i > j:
                    matrix[i][j] = matrix[j][i]
                    continue
                
                # compute intersection
                r_base = self.registries[i]
                r_target = self.registries[j]
                
                features_base = _group_features_by_module(r_base)
                features_target = _group_features_by_module(r_target)
                matcher.fuzzy_match_namespaces(features_base, features_target)
                
                all_modules = set(features_base.keys()) | set(features_target.keys())
                total_solid = 0
                for mod in all_modules:
                    b_list = list(features_base.get(mod, []))
                    t_list = list(features_target.get(mod, []))
                    solid_matches = matcher.match_features(b_list, t_list, self.alpha)
                    total_solid += len(solid_matches)
                
                total_base = len(r_base.features)
                total_target = len(r_target.features)
                union_size = total_base + total_target - total_solid
                
                score = total_solid / union_size if union_size > 0 else 1.0
                matrix[i][j] = score

        lines = [
            "## Global Parity Matrix",
            "",
            "| Language | " + " | ".join(self.langs) + " |",
            "| :--- |" + " :--- |" * n
        ]

        for i in range(n):
            row = [f"**{self.langs[i]}**"]
            for j in range(n):
                if i == j:
                    row.append("-")
                else:
                    row.append(f"{matrix[i][j]:.2%}")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")
        return lines

    def _build_global_feature_matrix(self) -> List[str]:
        # CrossLanguageFeature: dict mapping lang_idx -> Feature
        global_features: List[Dict[int, features_pb2.Feature]] = []

        # 1. Initialize with Anchor (index 0)
        anchor_registry = self.registries[0]
        for f in anchor_registry.features:
            global_features.append({0: f})

        # 2. Iteratively align remaining registries
        for i in range(1, len(self.registries)):
            target_registry = self.registries[i]
            
            # Group current global features by module and target features by module
            global_by_mod = defaultdict(list)
            for row in global_features:
                # Use the feature representation from the earliest language that has it
                rep_idx = min(row.keys())
                rep_f = row[rep_idx]
                mod = rep_f.normalized_namespace or rep_f.namespace or "Unknown Module"
                global_by_mod[mod].append((row, rep_f))

            target_by_mod = _group_features_by_module(target_registry)
            
            # We must remap namespaces just for matching purposes in this step
            # We'll build temporary Dict[str, List[Feature]] for namespaces
            g_ns_dict = {mod: [f for _, f in lst] for mod, lst in global_by_mod.items()}
            matcher.fuzzy_match_namespaces(g_ns_dict, target_by_mod)

            all_modules = set(g_ns_dict.keys()) | set(target_by_mod.keys())

            for mod in all_modules:
                base_tuples = global_by_mod.get(mod, [])  # list of (row_dict, Feature)
                b_list = [f for _, f in base_tuples]
                t_list = target_by_mod.get(mod, [])

                # Match
                solid_matches = matcher.match_features(b_list, t_list, self.alpha)

                # Record matches
                for b_f, t_f, _ in solid_matches:
                    # Find the original row dict that owns b_f
                    for row_dict, feat in base_tuples:
                        if feat is b_f:
                            row_dict[i] = t_f
                            break
                
                # Record unmatched targets as new rows
                # t_list was mutated by match_features (items removed)
                for t_f in t_list:
                    global_features.append({i: t_f})

        # 3. Render table grouped by Module
        # Regroup final global features by module for rendering
        final_by_mod = defaultdict(list)
        for row in global_features:
            rep_idx = min(row.keys())
            rep_f = row[rep_idx]
            mod = rep_f.normalized_namespace or rep_f.namespace or "Unknown Module"
            final_by_mod[mod].append(row)

        lines = ["## Global Feature Support", ""]
        
        for mod in sorted(final_by_mod.keys()):
            mod_rows = final_by_mod[mod]
            
            if self.common:
                python_idx = self.langs.index("Python") if "Python" in self.langs else -1
                mod_rows = [row for row in mod_rows if python_idx in row or len(row) >= 2]
                
            if not mod_rows:
                continue

            lines.append(f"### Module: `{mod}`")
            header = "| Feature | Type | " + " | ".join(self.langs) + " |"
            divider = "| :--- | :--- |" + " :---: |" * len(self.langs)
            lines.extend([header, divider])

            # sort features in module
            def get_sort_key(row):
                rep_idx = min(row.keys())
                rep_f = row[rep_idx]
                return (matcher._get_type_priority(rep_f), rep_f.normalized_name or "")
                
            mod_rows.sort(key=get_sort_key)

            for row in mod_rows:
                rep_idx = min(row.keys())
                rep_f = row[rep_idx]
                f_name = matcher._format_feature(rep_f)
                f_type = matcher.get_type_display_name(rep_f)

                row_cells = [f"`{f_name}`", f_type]
                for i in range(len(self.registries)):
                    if i in row:
                        row_cells.append("✅")
                    else:
                        row_cells.append("❌")

                lines.append("| " + " | ".join(row_cells) + " |")

            lines.append("")

        return lines

    def generate_report(self, report_type: str = "matrix") -> MatchResult:
        master_lines = [
            "# Multi-SDK Feature Matrix Report",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Registries",
            "| Role | Language | Version |",
            "| :--- | :--- | :--- |",
            "| :--- | :--- | :--- |"
        ]
        
        for idx, r in enumerate(self.registries):
            role_marker = "Anchor" if idx == 0 else f"Comparison {idx}"
            master_lines.append(
                f"| **{role_marker}** | {self.langs[idx]} | {r.version} |"
            )
        
        master_lines.append("")
        master_lines.extend(self._compute_jaccard_matrix())
        master_lines.extend(self._build_global_feature_matrix())

        return MatchResult(
            master_content="\n".join(master_lines).strip(),
            module_files={},
        )


class ReportGenerator:
    def __init__(
        self,
        base_registry: features_pb2.FeatureRegistry,
        target_registry: features_pb2.FeatureRegistry,
        df: pd.DataFrame,
    ):
        self.base_registry = base_registry
        self.target_registry = target_registry
        self.df = df
        
        self.base_code = _get_language_code(base_registry.language)
        self.target_code = _get_language_code(target_registry.language)
        self.base_name = _get_language_name(base_registry.language)
        self.target_name = _get_language_name(target_registry.language)

    def generate_md_report(self) -> MatchResult:
        """Generates a Markdown parity report from the DataFrame."""
        master_lines = []
        master_lines.extend(
            [
                "# Feature Matching Parity Report",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "| Role | Language | Version |",
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

        header = f"| Module | Features ({self.base_name}) | Score | Status | Details |"
        divider = "|---|---|---|---|---|"

        master_lines.extend(["## Module Summary", header, divider])

        module_files = {}
        module_rows = []
        
        # Determine cols based on language codes
        col_ns = f"{self.base_code}_namespace"
        
        # Group by base namespace
        # If namespace is empty, group under "Unknown Module"
        self.df["_module_group"] = self.df[col_ns].replace("", "Unknown Module")
        
        grouped = self.df.groupby("_module_group")
        
        total_high = 0
        total_low = 0
        total_mismatch = 0
        total_base_features = len(self.df)

        for module, group in grouped:
            # Calculate module stats
            high = len(group[group["confidence"] == "high"])
            low = len(group[group["confidence"] == "low"])
            mismatches = len(group[group["match"] == "false"])
            
            # Actually, `high` and `low` confidence applies to matches usually
            # But let's verify what `match` column says.
            matches_high = len(group[(group["match"] == "true") & (group["confidence"] == "high")])
            matches_low = len(group[(group["match"] == "true") & (group["confidence"] == "low")])
            # Everything else is a mismatch or low confidence match?
            # Let's trust `match` column for parity score
            solid_matches_count = len(group[group["match"] == "true"])
            
            total_high += matches_high
            total_low += matches_low
            total_mismatch += mismatches
            
            module_total = len(group)
            score = solid_matches_count / module_total if module_total > 0 else 0.0
            
            # Generate Module File Content
            module_filename = f"{module}.md"
            module_content = self._generate_module_content(module, group, module_total, matches_high, matches_low, mismatches)
            module_files[module_filename] = module_content
            
            # Add summary row
            status_icon = "✅" if score == 1.0 else "⚠️" if score > 0.5 else "❌"
            row_str = (
                f"| `{module}` | {module_total} | "
                f"{score:.2%} | {status_icon} | [View Details]({{modules_dir}}/{module_filename}) |"
            )
            module_rows.append((score, row_str))

        module_rows.sort(key=lambda x: x[0], reverse=True)
        master_lines.extend([row for _, row in module_rows])

        # Summary Stats
        total_matches = total_high + total_low
        parity_score = total_matches / total_base_features if total_base_features > 0 else 1.0
        
        base_exclusive = total_base_features - total_matches
        
        global_stats = (
            "## Summary\n\n"
            "| Feature Category | Count | Details |\n"
            "| :--- | :--- | :--- |\n"
            f"| **✅ High Confidence Matches** | **{total_high}** | "
            f"Strong matches found in `{self.target_name}` |\n"
            f"| **⚠️ Low Confidence Matches** | **{total_low}** | "
            f"Likely matches needing verification |\n"
            f"| **❌ Mismatches** | **{base_exclusive}** | "
            f"No suitable match found in `{self.target_name}` |\n"
            f"| **📊 Coverage Score** | **{parity_score:.2%}** | "
            f"Matches / Total Base Features ({total_matches} / {total_base_features}) |"
        )
        
        master_lines[global_score_idx] = global_stats

        return MatchResult(
            master_content="\n".join(master_lines).strip(),
            module_files=module_files,
        )

    def _generate_module_content(
        self, 
        module: str, 
        group: pd.DataFrame,
        total_features: int,
        high_conf: int,
        low_conf: int,
        mismatches: int
    ) -> str:
        
        # Calculate scores for summary
        total_matches = high_conf + low_conf
        coverage = total_matches / total_features if total_features > 0 else 0.0
        
        summary_table = (
            "## Summary\n\n"
            "| Feature Category | Count | Details |\n"
            "| :--- | :--- | :--- |\n"
            f"| **✅ High Confidence Matches** | **{high_conf}** | "
            f"Strong matches found in `{self.target_name}` |\n"
            f"| **⚠️ Low Confidence Matches** | **{low_conf}** | "
            f"Likely matches needing verification |\n"
            f"| **❌ Mismatches** | **{mismatches}** | "
            f"No suitable match found in `{self.target_name}` |\n"
            f"| **📊 Coverage Score** | **{coverage:.2%}** | "
            f"Matches / Total Base Features ({total_matches} / {total_features}) |\n"
        )

        lines = [
            f"# Module: `{module}`",
            "",
            f"[← Back to Master Report]({{master_report}})",
            "",
            summary_table,
            "## Feature Details",
            "",
            f"| Module ({self.base_name}) | Container ({self.base_name}) | Name ({self.base_name}) | Module ({self.target_name}) | Container ({self.target_name}) | Name ({self.target_name}) | Score | Match | Confidence |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |",
        ]
        
        # Sort by score desc, then name
        group_sorted = group.sort_values(by=["score", f"{self.base_code}_name"], ascending=[False, True])
        
        for _, row in group_sorted.iterrows():
            # Base logic
            b_ns = row[f'{self.base_code}_namespace']
            b_mem = row[f'{self.base_code}_member_of']
            b_name = row[f'{self.base_code}_name']
            
            # Target logic
            t_ns = row[f'{self.target_code}_namespace']
            t_mem = row[f'{self.target_code}_member_of']
            t_name = row[f'{self.target_code}_name']
            
            if t_name == "" and t_mem == "" and t_ns == "":
                t_name = "*(None)*"
                
            score = row['score']
            match_val = row['match']
            conf_val = row['confidence']
            
            if match_val == "true":
                if conf_val == "high":
                    match_icon = "✅"
                else:
                    match_icon = "⚠️"
            else:
                match_icon = "❌"

            conf_display = conf_val.title()
            if conf_display == "High":
                conf_display = "**High**"
                
            lines.append(
                f"| `{b_ns}` | `{b_mem}` | `{b_name}` | "
                f"`{t_ns}` | `{t_mem}` | `{t_name}` | "
                f"{score:.4f} | {match_icon} | {conf_display} |"
            )
            
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Match ADK features between two languages."
    )
    parser.add_argument(
        "--base",
        required=False,
        help="Path to the base FeatureRegistry .txtpb file.",
    )
    parser.add_argument(
        "--target",
        required=False,
        help="Path to the target FeatureRegistry .txtpb file.",
    )
    parser.add_argument(
        "--registries",
        nargs="+",
        required=False,
        help="Paths to multiple FeatureRegistry .txtpb files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the Markdown report. Corresponding CSV will be saved with same stem.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Similarity threshold (0.0 to 1.0) defaults to 0.8.",
    )
    parser.add_argument(
        "--report-type",
        choices=["md", "raw", "matrix"],
        default="md",
        help="Type of gap report. 'md' or 'raw' now produce both. 'matrix' is separate.",
    )
    parser.add_argument(
        "--common",
        action="store_true",
        help="Only list features present in Python or at least 2 languages (matrix report only).",
    )
    adk_args.add_verbose_argument(parser)
    args = parser.parse_args()
    adk_args.configure_logging(args)

    try:
        registry_paths = []
        if args.registries:
            registry_paths.extend(args.registries)
        elif args.base and args.target:
            registry_paths.extend([args.base, args.target])
        else:
            logging.error("Must provide either --registries or both --base and --target")
            sys.exit(1)
            
        if len(registry_paths) < 2:
            logging.error("Must provide at least 2 registries to compare.")
            sys.exit(1)

        registries = [_read_feature_registry(p) for p in registry_paths]
    except Exception as e:
        logging.error(f"Error reading feature registries: {e}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = match_registries(
        registries, 
        args.alpha, 
        args.report_type, 
        args.common, 
        output_path=output_path
    )

    if args.report_type == "matrix":
        # Matrix only writes one file
        try:
            output_path.write_text(result.master_content)
            logging.info(f"Successfully wrote matrix report to {output_path}")
        except Exception as e:
            logging.error(f"Error writing matrix report: {e}")
            sys.exit(1)
        return

    # For standard report, we already generated CSV inside match_registries.
    # Now write the Markdown and Modules.
    
    # Create module directory
    if result.module_files:
        modules_dir_name = f"{output_path.stem}_modules"
        modules_dir = output_path.parent / modules_dir_name
        modules_dir.mkdir(parents=True, exist_ok=True)

        # Write module files
        for filename, content in result.module_files.items():
            # Replace placeholder for master report link
            # The link is relative from module dir to master report
            # We are in {stem}_modules/, so we need to go up one level.
            final_content = content.replace("{master_report}", f"../{output_path.name}")
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
        output_path.write_text(master_report)
        logging.info(f"Successfully wrote match report to {output_path}")
        # Note: CSV writing is logged inside RawReportGenerator or we should log it here
        # Actually RawReportGenerator doesn't log, so we might want to Add a log here if we knew it matched
        stem = output_path.stem
        csv_path = output_path.parent / f"{stem}.csv"
        if csv_path.exists():
             logging.info(f"Successfully wrote raw match report to {csv_path}")

    except Exception as e:
        logging.error(f"Error writing report to {output_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
