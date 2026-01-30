import logging
import sys
from pathlib import Path

from google.protobuf.json_format import MessageToJson
from google.adk.scope.features_pb2 import FeatureRegistry
from google.adk.scope.utils.args import parse_args
from google.adk.scope.extractors import extractor_py, extractor_ts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EXTRACTORS = {
    "python": extractor_py,
    "typescript": extractor_ts,
}

REPO_ROOT_MARKERS = {
    "python": ["src"],
    "typescript": ["package.json", "tsconfig.json"],
}

REPO_SRC_SUBDIRS = {
    "python": ["src"],
    "typescript": ["core/src", "src"],
}


def get_repo_root(input_path: Path, language: str) -> Path | None:
    markers = REPO_ROOT_MARKERS.get(language, [])
    for parent in input_path.parents:
        if any((parent / marker).exists() for marker in markers):
            return parent
    return None


def get_search_dir(input_path: Path, language: str) -> Path:
    subdirs = REPO_SRC_SUBDIRS.get(language, [])
    for subdir in subdirs:
        potential_dir = input_path / subdir
        if potential_dir.exists():
            return potential_dir

    logger.warning(
        "Could not find standard source directories (%s) in %s. Scanning root.",
        subdirs,
        input_path,
    )
    return input_path


def main():
    args = (
        parse_args()
    )  # language is already normalized to "python" or "typescript"

    logger.info("Extractor - Language: %s", args.language)
    logger.info("Output will be saved to: %s", args.output)

    extractor_module = EXTRACTORS.get(args.language)
    if not extractor_module:
        logger.error("Unsupported language: %s", args.language)
        sys.exit(1)

    all_features = []
    repo_root = None

    if args.input_file:
        input_path = args.input_file
        if not input_path.exists():
            logger.error("Input file '%s' does not exist.", input_path)
            sys.exit(1)

        logger.info("Mode: Single file extraction: %s", input_path)

        # Determine repo root
        repo_root = input_path.parent
        if root := get_repo_root(input_path, args.language):
            repo_root = root

        features = extractor_module.extract_features(input_path, repo_root)
        all_features.extend(features)

        try:
            rel_path = input_path.relative_to(repo_root)
            logger.info("File: %s - Found %d features", rel_path, len(features))
        except ValueError:
            logger.info(
                "File: %s - Found %d features", input_path.name, len(features)
            )

    elif args.input_dir:
        input_path = args.input_dir
        if not input_path.exists():
            logger.error("Input directory '%s' does not exist.", input_path)
            sys.exit(1)

        logger.info(
            "Mode: Directory extraction (non-recursive): %s", input_path
        )
        repo_root = input_path

        files = list(extractor_module.find_files(input_path, recursive=False))
        logger.info("Found %d %s files.", len(files), args.language)

        for p in files:
            features = extractor_module.extract_features(p, repo_root)
            all_features.extend(features)
            # Log only if features found? Or keep unified summary at end.
            if features:
                try:
                    display_path = p.relative_to(input_path)
                except ValueError:
                    display_path = p.name
                logger.info(
                    "File: %s - Found %d features", display_path, len(features)
                )

    elif args.input_repo:
        input_path = args.input_repo
        if not input_path.exists():
            logger.error("Input repo '%s' does not exist.", input_path)
            sys.exit(1)

        logger.info("Mode: Repo extraction: %s", input_path)

        search_dir = get_search_dir(input_path, args.language)
        repo_root = input_path

        # Type checkers might complain about assignment if not specific list
        files = list(extractor_module.find_files(search_dir, recursive=True))

        logger.info(
            "Found %d %s files in %s.", len(files), args.language, search_dir
        )

        for p in files:
            features = extractor_module.extract_features(p, repo_root)
            all_features.extend(features)

    else:
        logger.error("No input mode specified.")
        sys.exit(1)

    logger.info("Total features found: %d", len(all_features))

    # Version extraction
    version = extractor_module.get_version(
        repo_root if repo_root else Path(".")
    )

    registry = FeatureRegistry(
        language=args.language.upper(),
        version=version,
        features=all_features,
    )

    try:
        with open(args.output, "w") as f:
            f.write(
                MessageToJson(
                    registry,
                    indent=2,
                    preserving_proto_field_name=True,
                    always_print_fields_with_no_presence=True,
                )
            )
        logger.info("Successfully wrote output to %s", args.output)
    except IOError as e:
        logger.error("Failed to write output: %s", e)


if __name__ == "__main__":
    main()
