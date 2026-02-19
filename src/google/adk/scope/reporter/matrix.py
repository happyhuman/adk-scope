import dataclasses
from datetime import datetime
from typing import Dict, List

import pandas as pd

from google.adk.scope import features_pb2
from google.adk.scope.utils import string, reporting




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
            self.base_name = string.get_language_name(base_language or "Unknown")
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

        # Prepare columns
        # We need to join all dataframes on the base feature unique identifier.
        # Unique ID = (namespace, member_of, name)
        # Note: In raw report, these are `base_namespace`, ranges...
        # actually `base_name` etc.
        # (Actually `py_namespace` etc. if base is Python)

        # Let's collect all base features first.
        # We can just use one of the dataframes as the driver, or the registry
        # itself. But using the dataframe is easier as it already has the rows.
        # However, multiple targets might have different subsets if there were
        # bugs, but theoretically all dataframes should have ALL base features
        # as rows (outer join or left join from base).
        # `RawReportGenerator` iterates over ALL base features, so every DF
        # has all base features.

        if not self.match_dataframes:
            return MatrixReport("No data available.")

        # Use the first dataframe to get the base feature list
        first_target_code = list(self.match_dataframes.keys())[0]
        base_df = self.match_dataframes[first_target_code].copy()

        # We only really need the base columns from this one
        base_cols = [
            f"{self.base_code}_namespace",
            f"{self.base_code}_member_of",
            f"{self.base_code}_name",
        ]
        
        # Initialize the consolidated dataframe with base columns
        matrix_df = base_df[base_cols].copy()
        
        # Create a display name for the base feature
        # E.g. "google.adk.scope.Feature"
        # Or just use the columns as is.
        # Let's create a specific 'Feature' column for readability if we want,
        # but the task request says "similar to markdown report".
        # The markdown report groups by module.

        target_cols_info = []

        # We iterate over match_dataframes keys if registries are not provided
        # or iterate based on preferred order if registries ARE provided.
        if self.target_registries:
            target_codes = [
                string.get_language_name(r.language).lower() for r in self.target_registries
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
            
            # We assume df is aligned with matrix_df because
            # `RawReportGenerator` iterates `base_registry.features` in order.
            # TO BE SAFE: we should merge on the base columns.
            
            # Rename `match` and `confidence` to include target code
            # We want to show a symbol based on match/confidence.
            
            def get_icon(row):
                match = row.get("match", "false")
                conf = row.get("confidence", "low")
                return reporting.get_match_icon(match, conf)

            # We can't apply this directly to `df` efficiently if we are going
            # to merge, unless we create a temp column.
            
            # Let's just merge 'match' and 'confidence' first.
            # suffix = f"_{target_code}"
            # Merging logic removed as we do it simpler below.
            
            # Now calculate the icon for this target
            # Note: after merge, columns might be named `match` (if first) or
            # `match_go` etc.
            # Wait, `pd.merge` might create duplicates if we are not careful
            # with suffixes.
            # Actually, `matrix_df` starts with NO match/confidence columns.
            # So the first merge adds `match`, `confidence`.
            # Subsequent merges need suffixes.

            # Better approach: Just build the `icon` column in the source DF
            # and rename it.
            df = df.copy()
            df[f"status_{target_code}"] = df.apply(get_icon, axis=1)
            
            # We only need the status column to merge
            matrix_df = pd.merge(
                matrix_df,
                df[[*base_cols, f"status_{target_code}"]],
                on=base_cols,
                how="left"
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

        # Same logic as Markdown report for grouping
        col_ns = f"{self.base_code}_namespace"
        matrix_df["_module_group"] = matrix_df[col_ns].replace(
            "", "Unknown Module"
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
