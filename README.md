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

#### Scoring Metrics

**Symmetric Report (Jaccard Index)**
-   Best for measuring general parity between two equal implementations.
-   **Score**: $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$
-    Penalizes both missing features and extra features.

**Directional Report (F1 Score)**
-   Best when checking if a Target implementation covers the Base implementation (e.g., "Is the TS SDK feature-complete vs Python?").
-   **Precision**: $\frac{\text{Matches}}{\text{Total Target Features}}$ (How accurate is the target?)
-   **Recall**: $\frac{\text{Matches}}{\text{Total Base Features}}$ (How complete is the target?)
-   **F1 Score**: Harmonic mean of Precision and Recall. $F1 = 2 \cdot \frac{P \cdot R}{P + R}$

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
