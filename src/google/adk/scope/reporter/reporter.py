import argparse
import dataclasses
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from google.protobuf import text_format

from google.adk.scope import features_pb2
from google.adk.scope.matcher import matcher
from google.adk.scope.utils import args as adk_args
from google.adk.scope.utils import stats
from google.adk.scope.utils.similarity import SimilarityScorer

_NEAR_MISS_THRESHOLD = 0.15

# Global thresholds for match confidence
# Keys are frozenset of language codes (e.g., frozenset(['py', 'go']))
SIMILARITY_THRESHOLDS = {
    frozenset(["py", "go"]): {
        "high": 0.6,
        "avg": 0.5,
    },
    frozenset(["py", "java"]): {
        "high": 0.6,
        "avg": 0.58,
    },
    frozenset(["py", "ts"]): {
        "high": 0.7,
        "avg": 0.55,
    },
}


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
    report_type: str = "md",
    common: bool = False,
) -> MatchResult:
    """Matches features and generates reports."""
    if report_type == "matrix":
        reporter = MatrixReportGenerator(registries, alpha, common)
    else:
        if len(registries) != 2:
            raise ValueError(f"Report type '{report_type}' requires exactly 2 registries.")
        reporter = ReportGenerator(
            registries[0],
            registries[1],
            alpha,
        )

    return reporter.generate_report(report_type)


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
        alpha: float,
    ):
        self.base_registry = base_registry
        self.target_registry = target_registry

        self.features_base = _group_features_by_module(base_registry)
        self.features_target = _group_features_by_module(target_registry)
        matcher.fuzzy_match_namespaces(self.features_base, self.features_target)
        self.alpha = alpha

    def generate_report(self, report_type) -> MatchResult:
        """Generates report."""
        if report_type == "raw":
            return self.generate_raw_report()
        elif report_type == "md":
            return self.generate_md_report()
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def generate_raw_report(self) -> MatchResult:
        """Generates a raw CSV report using global best-match logic.

        For every feature in the base registry, finds the best matching feature
        in the target registry with the same TYPE, regardless of module/namespace.
        """
        base_code = _get_language_code(self.base_registry.language)
        target_code = _get_language_code(self.target_registry.language)

        csv_header = (
            f"py_namespace,py_member_of,py_name,"
            f"java_namespace,java_member_of,java_name,"
            "type,score,match,confidence"
        )
        # Use user-requested headers if languages match expectation, otherwise dynamic
        if base_code == "py" and target_code == "java":
             pass # Header is already correct for the user's specific request example
        else:
             # Fallback to dynamic headers if not exactly py/java as requested
             csv_header = (
                f"{base_code}_namespace,{base_code}_member_of,{base_code}_name,"
                f"{target_code}_namespace,{target_code}_member_of,{target_code}_name,"
                "type,score,match,confidence"
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
            s = str(s)
            if "," in s or '"' in s or "\n" in s:
                return '"{}"'.format(s.replace('"', '""'))
            return s

        # 1. Index target features by Type for faster lookup
        target_by_type = defaultdict(list)
        for f in self.target_registry.features:
            target_by_type[f.type].append(f)

        scorer = SimilarityScorer(alpha=self.alpha)

        # 2. Iterate over all base features
        for f_base in self.base_registry.features:
            candidates = target_by_type.get(f_base.type, [])
            
            best_match = None
            best_score = -1.0

            if candidates:
                # Find best match among candidates of same type
                for f_target in candidates:
                    score = scorer.get_similarity_score(f_base, f_target)
                    if score > best_score:
                        best_score = score
                        best_match = f_target
            
            # 3. Write row if we have a match (even if score is 0, user might want to see it? 
            # Actually user said "pair with maximum similarity score should be included")
            # We will include it if it matches the best score logic. 
            # If no candidates exist, we print empties for target.
            
            b_ns, b_mem, b_name = get_feature_cols(f_base)
            f_type = matcher.get_type_display_name(f_base)

            if best_match:
                t_ns, t_mem, t_name = get_feature_cols(best_match)
                final_score = best_score
            else:
                t_ns, t_mem, t_name = "", "", ""
                final_score = 0.0

            # Determine match and confidence
            thresholds = SIMILARITY_THRESHOLDS.get(
                frozenset([base_code, target_code])
            )

            match_str = "false"
            confidence_str = "low"

            if thresholds:
                if final_score > thresholds["high"]:
                    match_str = "true"
                    confidence_str = "high"
                elif final_score >= thresholds["avg"]:
                    match_str = "true"
                    confidence_str = "low"
                else:
                    match_str = "false"
                    confidence_str = "high"
            else:
                # Default behavior if no thresholds defined for this pair
                # Fallback to general alpha or just say low confidence?
                # User only provided specific pairs.
                match_str = "true" if final_score >= self.alpha else "false"
                confidence_str = "low"

            csv_lines.append(
                f"{esc_csv(b_ns)},{esc_csv(b_mem)},{esc_csv(b_name)},"
                f"{esc_csv(t_ns)},{esc_csv(t_mem)},{esc_csv(t_name)},"
                f"{esc_csv(f_type)},{final_score:.4f},"
                f"{match_str},{confidence_str}"
            )

        return MatchResult(
            master_content="\n".join(csv_lines),
            module_files={},
        )

    def generate_md_report(self) -> MatchResult:
        """Generates a Markdown parity report."""
        all_modules = sorted(
            set(self.features_base.keys()) | set(self.features_target.keys())
        )
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

        b_lang = _get_language_name(self.base_registry.language)
        t_lang = _get_language_name(self.target_registry.language)

        header = f"| ADK | Module | Features ({b_lang}) | Score | Status | Details |"
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
                b_lang,
                t_lang,
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

        # Calculate metrics for the summary table
        base_exclusive = total_base_features - total_solid_matches
        target_exclusive = total_target_features - total_solid_matches

        union_size = total_base_features + total_target_features - total_solid_matches
        parity_score = total_solid_matches / union_size if union_size > 0 else 1.0

        b_lang = _get_language_name(self.base_registry.language)
        t_lang = _get_language_name(self.target_registry.language)

        global_stats = (
            "## Summary\n\n"
            "| Feature Category | Count | Details |\n"
            "| :--- | :--- | :--- |\n"
            f"| **✅ Common Shared** | **{total_solid_matches}** | "
            f"Implemented in both SDKs |\n"
            f"| **📦 Exclusive to `{b_lang}`** | **{base_exclusive}** | "
            f"Requires implementation in `{t_lang}` |\n"
            f"| **📦 Exclusive to `{t_lang}`** | **{target_exclusive}** | "
            f"Requires implementation in `{b_lang}` |\n"
            f"| **📊 Jaccard Score** | **{parity_score:.2%}** | "
            f"Overall Parity ({total_solid_matches} / {union_size}) |"
        )

        master_lines[global_score_idx] = global_stats

        return MatchResult(
            master_content="\n".join(master_lines).strip(),
            module_files=module_files,
        )


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
        choices=["md", "raw", "matrix"],
        default="md",
        help="Type of gap report to generate (md, raw, matrix).",
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

    result = match_registries(registries, args.alpha, args.report_type, args.common)

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
