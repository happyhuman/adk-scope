import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from google.protobuf import text_format

from google.adk.scope import features_pb2
from google.adk.scope.reporter import markdown, raw
from google.adk.scope.utils import args as adk_args


def _read_feature_registry(file_path: str) -> features_pb2.FeatureRegistry:
    """Reads a FeatureRegistry from a text proto file."""
    registry = features_pb2.FeatureRegistry()
    with open(file_path, "rb") as f:
        text_format.Parse(f.read(), registry)
    return registry


def generate_markdown_raw_reports(
    registries: List[features_pb2.FeatureRegistry],
    report_type: str = "md",  # Kept for backward compatibility
    output_path: Optional[Path] = None,
) -> markdown.MarkdownReport:
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

        return result

    except Exception as e:
        logging.error(f"Error writing report to {output_path}: {e}")
        sys.exit(1)


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
    parser.add_argument(
        "--report-type",
        choices=["md", "matrix"],
        default="md",
        help="Type of gap report. 'md' or 'matrix' now produce both.",
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
            logging.error("Must provide at least 2 registries to compare.")
            sys.exit(1)

        registries = [_read_feature_registry(p) for p in registry_paths]
    except Exception as e:
        logging.error(f"Error reading feature registries: {e}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_markdown_raw_reports(
        registries, args.report_type, output_path=output_path
    )


if __name__ == "__main__":
    main()
