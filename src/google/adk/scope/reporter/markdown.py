import dataclasses
from datetime import datetime
from typing import Dict

import pandas as pd

from google.adk.scope import features_pb2
from google.adk.scope.utils import reporting, string


@dataclasses.dataclass
class MarkdownReport:
    main_report_content: str
    module_reports: Dict[str, str]  # filename -> content


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

        self.base_name = string.get_language_name(base_registry.language)
        self.target_name = string.get_language_name(target_registry.language)
        self.base_code = string.get_language_code(base_registry.language)
        self.target_code = string.get_language_code(target_registry.language)

    def generate(self) -> MarkdownReport:
        """Generates a Markdown parity report from the DataFrame."""
        master_lines = []
        master_lines.extend(
            [
                "# Feature Matching Parity Report",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "| Role | Language | Version | Last Commit |",
                "| :--- | :--- | :--- | :--- |",
                (
                    f"| **Base** | {self.base_registry.language} |"
                    f" {self.base_registry.version} |"
                    f" {self.base_registry.commit_id or 'N/A'} |"
                ),
                (
                    f"| **Target** | {self.target_registry.language} |"
                    f" {self.target_registry.version} |"
                    f" {self.target_registry.commit_id or 'N/A'} |"
                ),
                "",
            ]
        )

        global_score_idx = len(master_lines)
        master_lines.append("GLOBAL_SCORE_PLACEHOLDER")
        master_lines.append("")

        header = (
            f"| Module | Features ({self.base_name}) | Overlap | " f"Details |"
        )
        divider = "|---|---|---|---|"

        master_lines.extend(["## Module Summary", header, divider])

        module_reports = {}
        module_rows = []

        # Determine cols based on language codes
        col_ns = f"{self.base_code}_namespace"

        # Split DataFrame: Base Present vs Target Only
        # Target Only rows have empty base_name (and score 0.0)
        # Note: We check if base_name is empty/NaN.
        # In clean_dataframe terms it might be "___", but here it is "" from raw.py
        df_target_only = self.df[self.df[f"{self.base_code}_name"] == ""]
        df_base_present = self.df[self.df[f"{self.base_code}_name"] != ""]

        # Group by base namespace (for Base Present)
        # If namespace is empty, group under "Unknown Module"
        # We need to act on a copy to avoid SettingWithCopyWarning
        df_base_present = df_base_present.copy()
        df_base_present["_module_group"] = df_base_present[col_ns].replace(
            "", "Unknown Module"
        )

        grouped = df_base_present.groupby("_module_group")

        total_high = 0
        total_low = 0
        total_base_features = len(df_base_present)

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
            row_str = (
                f"| `{module}` | {module_total} | "
                f"{score:.2%} | "
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
            f"| **📊 Coverage Overlap** | **{parity_score:.2%}** | "
            f"Matches / Total Base Features ({total_matches} / "
            f"{total_base_features}) |"
        )

        master_lines[global_score_idx] = global_stats

        # -- Target Exclusive Section --
        if not df_target_only.empty:
            target_section = self._generate_target_exclusive_section(
                df_target_only
            )
            master_lines.append("")
            master_lines.append(target_section)

        return MarkdownReport(
            main_report_content="\n".join(master_lines).strip(),
            module_reports=module_reports,
        )

    def _generate_target_exclusive_section(self, df: pd.DataFrame) -> str:
        """Generates a section in the request for Target-Only features."""
        lines = [
            "## Target-Exclusive Modules",
            "",
            f"Features found in **{self.target_name}** but NOT in **{self.base_name}**.",
            "",
        ]

        col_ns = f"{self.target_code}_namespace"

        # Determine modules
        # Avoid SettingWithCopyWarning
        df_copy = df.copy()
        df_copy["_target_module"] = df_copy[col_ns].replace(
            "", "Unknown Module"
        )

        # Group
        grouped = df_copy.groupby("_target_module")

        # Table
        lines.append(f"| Target Module | Exclusive Features | Details |")
        lines.append("| :--- | :--- | :--- |")

        rows = []
        for module, group in grouped:
            count = len(group)

            # List top 3 examples
            examples = group[f"{self.target_code}_name"].head(3).tolist()
            example_str = ", ".join([f"`{e}`" for e in examples])
            if count > 3:
                example_str += ", ..."

            rows.append(f"| `{module}` | {count} | {example_str} |")

        # Sort rows by count desc or alpha? Let's sort by module name (default)
        # Actually keys are already sorted by groupby default

        lines.extend(rows)
        return "\n".join(lines)

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
        # Replace empty values for display
        group = reporting.clean_dataframe(group)

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
            f"| **📊 Coverage Overlap** | **{coverage:.2%}** | "
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
            f"| Type | Feature ({self.base_name}) | Feature ({self.target_name}) | "
            "Score | Match | Confidence |",
            "| :--- | :--- | :--- | :--- | :---: | :---: |",
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

            match_icon = reporting.get_match_icon(match_val, conf_val)

            conf_display = conf_val.title()
            if conf_display == "High":
                conf_display = "**High**"

            feat_type = row.get("type", "unknown")

            # Helper to format path
            def fmt_path(ns, mem, name):
                parts = [ns, mem, name]
                return "/".join([p for p in parts if p and p != "___"])

            b_path = fmt_path(b_ns, b_mem, b_name)
            t_path = fmt_path(t_ns, t_mem, t_name)

            if not t_path:
                t_path = "*(None)*"

            lines.append(
                f"| `{feat_type}` | `{b_path}` | `{t_path}` | "
                f"{score:.4f} | {match_icon} | {conf_display} |"
            )

        return "\n".join(lines)
