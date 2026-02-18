import argparse
import logging
import sys
from pathlib import Path

from google.protobuf import text_format

from google.adk.scope import features_pb2
from google.adk.scope.utils.similarity import SimilarityScorer


def main():
    parser = argparse.ArgumentParser(
        description="Calculate similarity score between two features."
    )
    parser.add_argument(
        "feature1", type=Path, help="Path to first feature file (text proto)."
    )
    parser.add_argument(
        "feature2", type=Path, help="Path to second feature file (text proto)."
    )

    args = parser.parse_args()

    # Configure logging to DEBUG
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        f1_content = args.feature1.read_text()
        f2_content = args.feature2.read_text()

        f1 = features_pb2.Feature()
        text_format.Parse(f1_content, f1)

        f2 = features_pb2.Feature()
        text_format.Parse(f2_content, f2)

        scorer = SimilarityScorer()
        score = scorer.get_similarity_score(f1, f2)

        print("-" * 40)
        print(f"Similarity Score: {score:.4f}")
        print("-" * 40)

    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
