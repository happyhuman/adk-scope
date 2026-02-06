# ADK Scope

`adk-scope` is a tool designed to extract semantic metadata (features) from Agent Development Kit (ADK) repositories. It utilizes [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) to parse source code across multiple languages (Python, Java, Go, TypeScript) and generates structured metadata in Protocol Buffers format.

## Features

-   **Multi-Language Support**: Extracts features from Python, Java, Go, and TypeScript.
-   **Semantic Extraction**: Identifying Agents, Tools, and other high-level constructs.
-   **Flexible Input**: Supports extraction from single files, directories, or full repositories.
-   **Structured Output**: Generates `FeatureRegistry` objects serialized to JSON.

## Installation

This project uses `pyproject.toml` for dependency management.

1.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies**:
    ```bash
    pip install .
    # OR for development
    pip install -e ".[dev]"
    ```


## Usage

You can run the extractor using the provided shell script or directly via Python.

### Wrapper Script

The `extract.sh` wrapper helps set up the `PYTHONPATH` correctly.
The script requires a `--language` argument to specify the target language (`py` or `ts`).

```bash
# For Python
./extract.sh --language py --input-repo /path/to/adk-python output_dir

# For TypeScript
./extract.sh --language ts --input-repo /path/to/adk-js output_dir
```

### CLI Arguments

| Argument | Description |
| :--- | :--- |
| `--language <lang>` | **Required.** Language to extract (`python` or `typescript`). |
| `--input-file <path>` | Path to a single file to process. |
| `--input-dir <path>` | Path to a directory containing files. |
| `--input-repo <path>` | Path to the root of an ADK repository. Recursive search in `src` (Python) or `core/src` (TS). |
| `output` | **Required.** Path to the output directory. |

**Examples:**

```bash
# Process a single file
./extract.sh --language python --input-file src/my_agent.py output_dir

# Process a directory
python3 -m google.adk.scope.extractors.python.extractor \
  --input-dir src/google/adk \
  output_dir
```

```

### Feature Matching

Once you have extracted features from two languages (e.g., Python and TypeScript), you can compare them using the `match.sh` script.

```bash
./match.sh \
  --base output/py.txtpb \
  --target output/ts.txtpb \
  --output output/ \
  --report-type directional
```

| Argument | Description |
| :--- | :--- |
| `--base <path>` | **Required.** Path to the "source of truth" feature registry (e.g., Python). |
| `--target <path>` | **Required.** Path to the comparison registry (e.g., TypeScript). |
| `--output <dir>` | **Required.** Path for the output directory. The report filename is auto-generated. |
| `--report-type <type>` | `symmetric` (default) for Jaccard Index, `directional` for F1/Precision/Recall, or `raw` for CSV. |
| `--alpha <float>` | Similarity threshold (0.0 - 1.0). Default is `0.8`. |

#### How Matching Works

The matcher uses the **Hungarian Algorithm** to find the optimal assignment between features in the Base and Target registries.
-   **Cost Function**: Based on a similarity score derived from:
    -   Feature Name (normalized)
    -   Namespace / Module
    -   Feature Type (Function, Method, Class, etc.)
-   **Thresholding**: Pairs with a similarity score below `--alpha` are discarded.

#### Understanding the Reports

`adk-scope` can generate three types of reports to help you understand the feature overlap between two languages.

##### Symmetric Report (`--report-type symmetric`)

This report is best for measuring the general similarity between two feature sets, where neither is considered the "source of truth". It uses the **Jaccard Index** to calculate a global similarity score.

-   **What it measures**: The Jaccard Index measures the similarity between two sets by dividing the size of their intersection by the size of their union. The score ranges from 0% (no similarity) to 100% (identical sets).
-   **What it means**: A high Jaccard Index indicates that both languages have a very similar set of features, with few features unique to either one. It penalizes both missing and extra features equally.

##### Directional Report (`--report-type directional`)

This report is ideal when you have a "base" or "source of truth" language and you want to measure how well a "target" language conforms to it. It uses **Precision**, **Recall**, and **F1-Score**.

-   **Precision**: Answers the question: *"Of all the features implemented in the target language, how many of them are correct matches to features in the base language?"* A low score indicates the target has many extra features not present in the base.
-   **Recall**: Answers the question: *"Of all the features that should be in the target language (i.e., all features in the base), how many were actually found?"* A low score indicates the target is missing many features from the base.
-   **F1-Score**: The harmonic mean of Precision and Recall, providing a single score that balances both. A high F1-Score indicates the target is a close match to the base, having most of the required features and not too many extra ones.

##### Raw Report (`--report-type raw`)

This report provides a simple CSV output of all features (matched and unmatched) from both the base and target registries. It is useful for programmatic analysis or for importing the data into other tools.$

## Development

### Running Tests

We use `pytest` for testing.

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=google.adk.scope --cov-report=term-missing
```

### Linting

We use `ruff` for linting.

```bash
ruff check .
```

### Protobuf Generation

If you modify `proto/features.proto`, you need to regenerate the Python code:

```bash
./proto2py.sh
```

## Project Structure

-   `src/google/adk/scope/`: Main source code.
    -   `extractors/`: Language-specific extractors (currently Python).
    -   `utils/`: Utility modules (strings, args).
-   `proto/`: Protocol Buffer definitions (`features.proto`).
-   `test/`: Unit tests.
