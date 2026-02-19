import dataclasses
from datetime import datetime
from typing import Dict

import pandas as pd

from google.adk.scope import features_pb2


@dataclasses.dataclass
class MarkdownReport:
    main_report_content: str
    module_reports: Dict[str, str]  # filename -> content


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


class MarkdownReportGenerator:
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

    def generate(self) -> MarkdownReport:
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

        header = (
            f"| Module | Features ({self.base_name}) | Score | Status | "
            f"Details |"
        )
        divider = "|---|---|---|---|---|"

        master_lines.extend(["## Module Summary", header, divider])

        module_reports = {}
        module_rows = []

        # Determine cols based on language codes
        col_ns = f"{self.base_code}_namespace"

        # Group by base namespace
        # If namespace is empty, group under "Unknown Module"
        self.df["_module_group"] = self.df[col_ns].replace("", "Unknown Module")

        grouped = self.df.groupby("_module_group")

        total_high = 0
        total_low = 0
        total_base_features = len(self.df)

        for module, group in grouped:
            # Calculate module stats
            mismatches = len(group[group["match"] == "false"])

            # Actually, `high` and `low` confidence applies to matches usually
            # But let's verify what `match` column says.
            matches_high = len(
                group[
                    (group["match"] == "true") & (group["confidence"] == "high")
                ]
            )
            matches_low = len(
                group[
                    (group["match"] == "true") & (group["confidence"] == "low")
                ]
            )
            # Everything else is a mismatch or low confidence match?
            # Let's trust `match` column for parity score
            solid_matches_count = len(group[group["match"] == "true"])

            total_high += matches_high
            total_low += matches_low

            module_total = len(group)
            score = (
                solid_matches_count / module_total if module_total > 0 else 0.0
            )

            # Generate Module File Content
            module_filename = f"{module}.md"
            module_content = self._generate_module_content(
                module,
                group,
                module_total,
                matches_high,
                matches_low,
                mismatches,
            )
            module_reports[module_filename] = module_content

            # Add summary row
            status_icon = "✅" if score == 1.0 else "⚠️" if score > 0.5 else "❌"
            row_str = (
                f"| `{module}` | {module_total} | "
                f"{score:.2%} | {status_icon} | "
                f"[View Details]({{modules_dir}}/{module_filename}) |"
            )
            module_rows.append((score, row_str))

        module_rows.sort(key=lambda x: x[0], reverse=True)
        master_lines.extend([row for _, row in module_rows])

        # Summary Stats
        total_matches = total_high + total_low
        parity_score = (
            total_matches / total_base_features
            if total_base_features > 0
            else 1.0
        )

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
            f"Matches / Total Base Features ({total_matches} / "
            f"{total_base_features}) |"
        )

        master_lines[global_score_idx] = global_stats

        return MarkdownReport(
            main_report_content="\n".join(master_lines).strip(),
            module_reports=module_reports,
        )

    def _generate_module_content(
        self,
        module: str,
        group: pd.DataFrame,
        total_features: int,
        high_conf: int,
        low_conf: int,
        mismatches: int,
    ) -> str:

        # Calculate scores for summary
        total_matches = high_conf + low_conf
        coverage = total_matches / total_features if total_features > 0 else 0.0

        # Replace empty values for display
        group = group.fillna("___")
        group = group.replace("", "___")

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
            f"Matches / Total Base Features ({total_matches} / "
            f"{total_features}) |\n"
        )

        lines = [
            f"# Module: `{module}`",
            "",
            "[← Back to Master Report]({master_report})",
            "",
            summary_table,
            "## Feature Details",
            "",
            f"| Module ({self.base_name}) | Container ({self.base_name}) | "
            f"Name ({self.base_name}) | Module ({self.target_name}) | "
            f"Container ({self.target_name}) | Name ({self.target_name}) | "
            "Score | Match | Confidence |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | "
            ":---: |",
        ]

        # Sort by score desc, then name
        group_sorted = group.sort_values(
            by=["score", f"{self.base_code}_name"], ascending=[False, True]
        )

        for _, row in group_sorted.iterrows():
            # Base logic
            b_ns = row[f"{self.base_code}_namespace"]
            b_mem = row[f"{self.base_code}_member_of"]
            b_name = row[f"{self.base_code}_name"]

            # Target logic
            t_ns = row[f"{self.target_code}_namespace"]
            t_mem = row[f"{self.target_code}_member_of"]
            t_name = row[f"{self.target_code}_name"]

            if t_name == "___" and t_mem == "___" and t_ns == "___":
                t_name = "*(None)*"

            score = row["score"]
            match_val = row["match"]
            conf_val = row["confidence"]

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
