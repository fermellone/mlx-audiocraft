"""
mlx-audiocraft: MLX-native inference runtime for AudioCraft-family pretrained models.

This package provides MLX-optimized inference for Meta AudioCraft models
(starting with MusicGen). Analogous to mlx-lm but for audio generation.

No model training or training-from-scratch is included. Scope is limited to:
- Loading/converting pretrained weights (PyTorch → MLX)
- Running inference on Apple Silicon with MLX
"""

__version__ = "0.1.0.dev0"

from ._musicgen import MusicGen  # noqa: F401
