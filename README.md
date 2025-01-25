# Autobyteus LLM Client

Async Python client for Autobyteus LLM API.

## Installation

```bash
pip install autobyteus_llm_client
```

## Building and Publishing the Package

### Prerequisites

1. Create a PyPI account at https://pypi.org/account/register/
2. Install build and twine:
```bash
pip install build twine
```

### Building the Package

To build the package, run:
```bash
python -m build
```

This will create two files in the `dist` directory:
- A source archive (.tar.gz)
- A wheel (.whl)

### Publishing to PyPI

#### Test PyPI (Recommended for Testing)

1. Register an account at https://test.pypi.org/
2. Upload to Test PyPI:
```bash
python -m twine upload --repository testpypi dist/*
```
3. Install from Test PyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ autobyteus_llm_client
```

#### Production PyPI

When ready to publish to production:
```bash
python -m twine upload dist/*
```

Note: You'll need to provide your PyPI username and password when uploading.

## Development

### Requirements
- Python 3.8 or higher
- httpx

### Installing Development Dependencies
```bash
pip install -e ".[test]"
```

## License

This project is licensed under the MIT License.
