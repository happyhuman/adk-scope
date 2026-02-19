import dataclasses
from datetime import datetime
from typing import Dict, List

import pandas as pd

from google.adk.scope import features_pb2
from google.adk.scope.utils import reporting, string


@dataclasses.dataclass
class MatrixReport:
    content: str


class MatrixReportGenerator:
    def __init__(
        self,
        match_dataframes: Dict[str, pd.DataFrame],
        base_registry: features_pb2.FeatureRegistry = None,
        target_registries: List[features_pb2.FeatureRegistry] = None,
        base_language: str = None,
        base_version: str = None,
    ):
        # Clean dataframes: replace NaN and empty strings with "___"
        # We process copies to avoid side effects
        cleaned_dfs = {}
        for k, df in match_dataframes.items():
            # Clean dataframe
            df_clean = reporting.clean_dataframe(df)
            cleaned_dfs[k] = df_clean

        self.match_dataframes = cleaned_dfs
        self.base_registry = base_registry
        self.target_registries = target_registries or []

        if base_registry:
            self.base_name = string.get_language_name(base_registry.language)
            self.base_code = self.base_name.lower()
            self.base_version = base_registry.version
        else:
            self.base_name = string.get_language_name(
                base_language or "Unknown"
            )
            self.base_code = self.base_name.lower()
            self.base_version = base_version or "Unknown"

    def generate(self) -> MatrixReport:
        """Generates a Markdown matrix report."""
        lines = []
        lines.extend(
            [
                "# Feature Matrix Report",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"**Base Language**: {self.base_name} "
                f"({self.base_version})",
                "",
            ]
        )

        if not self.match_dataframes:
            return MatrixReport("No data available.")

        # Use the first dataframe to get the base feature list
        first_target_code = list(self.match_dataframes.keys())[0]
        base_df = self.match_dataframes[first_target_code].copy()

        # Filter base dataframe to exclude empty base keys (target-only)
        # This is critical for matrix report to avoid explosion
        # Note: `clean_dataframe` replaces NaN and "" with "___"
        col_name = f"{self.base_code}_name"
        base_df = base_df[
            (base_df[col_name] != "") & (base_df[col_name] != "___")
        ]

        # We only really need the base columns from this one
        base_cols = [
            f"{self.base_code}_namespace",
            f"{self.base_code}_member_of",
            f"{self.base_code}_name",
        ]

        # Initialize the consolidated dataframe with base columns
        matrix_df = base_df[base_cols].copy()

        target_cols_info = []

        # We iterate over match_dataframes keys if registries are not provided
        # or iterate based on preferred order if registries ARE provided.
        if self.target_registries:
            target_codes = [
                string.get_language_name(r.language).lower()
                for r in self.target_registries
            ]
            # Ensure we only use those that are in dataframes
            target_iterator = [
                (code, string.get_language_name(code))
                for code in target_codes
                if code in self.match_dataframes
            ]
        else:
            # Sort keys for consistent output
            target_codes = sorted(list(self.match_dataframes.keys()))
            target_iterator = [
                (code, string.get_language_name(code)) for code in target_codes
            ]

        for target_code, target_name in target_iterator:

            df = self.match_dataframes[target_code]

            def get_icon(row):
                match = row.get("match", "false")
                conf = row.get("confidence", "low")
                return reporting.get_match_icon(match, conf)

            # Better approach: Just build the `icon` column in the source DF
            # and rename it.
            # Filter out target-only rows (where base_name is empty or placeholder) before processing
            # to avoid merging on empty keys which causes explosion
            # Note: `clean_dataframe` replaces NaN and "" with "___"
            col_name = f"{self.base_code}_name"
            # Explicitly verify column exists to avoid KeyError if base language was guessed wrong
            if col_name not in df.columns:
                # Try to fallback or skip?
                # If base code is wrong, everything is broken.
                pass

            df = df[(df[col_name] != "") & (df[col_name] != "___")].copy()

            df[f"status_{target_code}"] = df.apply(get_icon, axis=1)

            # We only need the status column to merge
            matrix_df = pd.merge(
                matrix_df,
                df[[*base_cols, f"status_{target_code}"]],
                on=base_cols,
                how="left",
            )

            target_cols_info.append(
                {
                    "code": target_code,
                    "name": target_name,
                    "col": f"status_{target_code}",
                }
            )

        # Now we have `matrix_df` with base cols + status_{target} cols.
        # Let's group by namespace (Module).

        col_ns = f"{self.base_code}_namespace"
        matrix_df["_module_group"] = matrix_df[col_ns].replace(
            ["", "___"], "Unknown Module"
        )
        grouped = matrix_df.groupby("_module_group")

        # Table Header
        # | Module | Container | Name | Java | Go | TS |
        target_headers = " | ".join([f"{t['name']}" for t in target_cols_info])
        header = (
            f"| Module ({self.base_name}) | Container | Name | "
            f"{target_headers} |"
        )
        # Dynamic divider
        target_dividers = " | ".join(["---"] * len(target_cols_info))
        divider = f"| :--- | :--- | :--- | {target_dividers} |"

        lines.extend([header, divider])

        for module, group in grouped:
            # Sort by name
            group_sorted = group.sort_values(
                by=[f"{self.base_code}_name"], ascending=[True]
            )

            for _, row in group_sorted.iterrows():
                ns = row[f"{self.base_code}_namespace"]
                mem = row[f"{self.base_code}_member_of"]
                name = row[f"{self.base_code}_name"]

                # Build row string
                row_parts = [f"`{ns}`", f"`{mem}`", f"`{name}`"]

                for t in target_cols_info:
                    status = row[t["col"]]
                    # Handle NaN if merge failed (shouldn't happen)
                    if pd.isna(status):
                        status = "❓"
                    row_parts.append(status)

                lines.append(f"| {' | '.join(row_parts)} |")

        return MatrixReport(content="\n".join(lines))
