# mlx-audiocraft

MLX-native inference runtime for AudioCraft-family pretrained models on Apple Silicon.

Analogous to [`mlx-lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms) but for Meta AudioCraft model inference — starting with MusicGen.

## Status

**Pre-alpha / bootstrap phase.** No inference code exists yet.

## Scope

- Loading and converting pretrained AudioCraft weights (PyTorch → MLX).
- Running inference on Apple Silicon with MLX.

## Project Structure

```
mlx-audiocraft/
├── src/mlx_audiocraft/       # Package source
│   ├── __init__.py
│   └── _musicgen.py          # MusicGen inference (placeholder)
├── tests/                    # Test suite
├── pyproject.toml            # Build, deps, tool config
├── openspec/                 # SDD planning artifacts
└── .atl/                     # Agent skill registry
```

## Development

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint and type-check
ruff check src/
mypy src/
```

## License

MIT
