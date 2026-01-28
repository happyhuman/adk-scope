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

### wrapper Script

The `extract_python.sh` wrapper helps set up the `PYTHONPATH` correctly.

```bash
./extract_python.sh --input-repo /path/to/adk-python output.json
```

### CLI Arguments

| Argument | Description |
| :--- | :--- |
| `--input-file <path>` | Path to a single Python file to process. |
| `--input-dir <path>` | Path to a directory containing Python files. |
| `--input-repo <path>` | Path to the root of an ADK repository. Extracts ADK version from `src/google/adk/version.py`. |
| `output` | **Required.** Path to the output JSON file. |

**Examples:**

```bash
# Process a single file
python3 -m google.adk.scope.extractors.python.extractor \
  --input-file src/my_agent.py \
  output.json

# Process a directory
python3 -m google.adk.scope.extractors.python.extractor \
  --input-dir src/google/adk \
  output.json
```

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
./generate_protos.sh
```

## Project Structure

-   `src/google/adk/scope/`: Main source code.
    -   `extractors/`: Language-specific extractors (currently Python).
    -   `utils/`: Utility modules (strings, args).
-   `proto/`: Protocol Buffer definitions (`features.proto`).
-   `test/`: Unit tests.
