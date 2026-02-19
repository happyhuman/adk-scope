import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from google.protobuf import text_format

from google.adk.scope import features_pb2
from google.adk.scope.reporter import markdown, matrix, raw
from google.adk.scope.utils import args as adk_args


def _read_feature_registry(file_path: str) -> features_pb2.FeatureRegistry:
    """Reads a FeatureRegistry from a text proto file."""
    registry = features_pb2.FeatureRegistry()
    with open(file_path, "rb") as f:
        text_format.Parse(f.read(), registry)
    return registry


def generate_matrix_report_from_csvs(
    csv_paths: List[str],
    output_path: Path,
) -> None:
    """Generates a matrix report from a list of CSV files."""
    match_dataframes = {}

    # We need to infer base language and version from the first CSV
    # or just assume something common.
    # The columns in CSV are: {base}_namespace, {base}_member_of, {base}_name, ...

    base_lang = None

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            # Infer language codes from columns
            # Expected columns: {base}_namespace, ..., {target}_namespace, ...

            # Find the base columns (first 3 usually)
            # Actually we can regex the columns ending in _namespace

            cols = df.columns
            namespaces = [c for c in cols if c.endswith("_namespace")]

            if len(namespaces) < 2:
                logging.warning(
                    f"Skipping {csv_path}: Could not detect base/target languages."
                )
                continue

            # Assume first namespace col is base (or find specific one?)
            # Usually base is first in raw report.
            base_code = namespaces[0].split("_")[0]
            target_code = namespaces[1].split("_")[0]

            if base_lang is None:
                base_lang = base_code
            elif base_lang != base_code:
                logging.warning(
                    f"Skipping {csv_path}: Base language mismatch ({base_code} != {base_lang})"
                )
                continue

            # Add to dataframes
            match_dataframes[target_code] = df

        except Exception as e:
            logging.error(f"Error reading CSV {csv_path}: {e}")
            sys.exit(1)

    if not match_dataframes:
        logging.error("No valid CSV data found for matrix report.")
        sys.exit(1)

    # Generate matrix report
    # We pass base_language code. Version is unknown.
    report_gen = matrix.MatrixReportGenerator(
        match_dataframes, base_language=base_lang, base_version="Unknown"
    )
    report = report_gen.generate()

    try:
        output_path.write_text(report.content)
        logging.info(
            f"Successfully wrote matrix report from CSVs to {output_path}"
        )
    except Exception as e:
        logging.error(f"Error writing matrix report to {output_path}: {e}")
        sys.exit(1)


def generate_markdown_raw_reports(
    registries: List[features_pb2.FeatureRegistry],
    output_path: Optional[Path] = None,
):
    """Matches features and generates reports."""

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
    reporter = markdown.MarkdownReportGenerator(
        registries[0], registries[1], df
    )
    result = reporter.generate()
    if result.module_reports:
        modules_dir_name = f"{output_path.stem}_modules"
        modules_dir = output_path.parent / modules_dir_name
        modules_dir.mkdir(parents=True, exist_ok=True)

        # Write module files
        for filename, content in result.module_reports.items():
            # Replace placeholder for master report link
            # The link is relative from module dir to master report
            # We are in {stem}_modules/, so we need to go up one level.
            final_content = content.replace(
                "{master_report}", f"../{output_path.name}"
            )
            (modules_dir / filename).write_text(final_content)

        # Replace placeholder in Master Report
        # We assume master report is in parent of modules_dir
        # modules_dir relative to master report is just the dir name
        master_report = result.main_report_content.replace(
            "{modules_dir}", modules_dir_name
        )
    else:
        master_report = result.main_report_content.replace("{modules_dir}", ".")

    try:
        output_path.write_text(master_report)
        logging.info(f"Successfully wrote match report to {output_path}")
        # Note: CSV writing is logged inside RawReportGenerator or we should
        # log it here. Actually RawReportGenerator doesn't log, so we might
        # want to Add a log here if we knew it matched
        stem = output_path.stem
        csv_path = output_path.parent / f"{stem}.csv"
        if csv_path.exists():
            logging.info(f"Successfully wrote raw match report to {csv_path}")

    except Exception as e:
        logging.error(f"Error writing report to {output_path}: {e}")
        sys.exit(1)

    return result


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
        help=(
            "Path to save the Markdown report. Corresponding CSV will be "
            "saved with same stem."
        ),
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
            logging.error(
                "Must provide either --registries or both --base and --target"
            )
            sys.exit(1)

        if len(registry_paths) < 2:
            logging.error("Must provide registries or CSV files to compare.")
            sys.exit(1)

        # Infer output path if directory
        output_path = Path(args.output)
        is_matrix_input = registry_paths[0].endswith(".csv")

        # If output looks like a directory (no extension) or is an existing dir
        if output_path.suffix == "" or output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            filename = "matrix_report.md" if is_matrix_input else "report.md"
            output_path = output_path / filename
        else:
            # Check if we are in matrix mode and the filename is _.md (generated by script when langs unknown)
            if is_matrix_input and output_path.name == "_.md":
                output_path = output_path.with_name("matrix_report.md")

            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if inputs are CSVs
        if is_matrix_input:
            # Verify all are CSVs
            if not all(p.endswith(".csv") for p in registry_paths):
                logging.error(
                    "All inputs must be CSV files for CSV matrix report."
                )
                sys.exit(1)

            generate_matrix_report_from_csvs(registry_paths, output_path)
            return

        if len(registry_paths) < 2:
            logging.error("Must provide at least 2 registries to compare.")
            sys.exit(1)

        registries = [_read_feature_registry(p) for p in registry_paths]
    except Exception as e:
        logging.error(f"Error reading feature registries: {e}")
        sys.exit(1)

    generate_markdown_raw_reports(registries, output_path=output_path)


if __name__ == "__main__":
    main()
