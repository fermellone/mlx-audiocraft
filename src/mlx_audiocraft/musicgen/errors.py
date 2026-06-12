"""Typed, actionable errors for MusicGen loading and inference.

These errors enforce strict failure boundaries: unsupported checkpoints, configs,
model features, cache incompatibilities, and not-yet-implemented capabilities are
reported clearly rather than falling back to PyTorch or partial outputs.
"""


class MusicGenError(Exception):
    """Base error for all MusicGen-specific failures."""


class UnsupportedCheckpointError(MusicGenError):
    """Raised when a model_id is not a supported official checkpoint.

    Only ``facebook/musicgen-*`` checkpoints are supported in this slice.
    """


class UnsupportedConfigError(MusicGenError):
    """Raised when the checkpoint config contains unsupported sections or values."""


class UnsupportedModelFeatureError(MusicGenError):
    """Raised when a supported checkpoint uses a feature not yet implemented in MLX."""


class IncompatibleCacheError(MusicGenError):
    """Raised when the converted-cache metadata is missing or incompatible."""


class NotYetSupportedError(MusicGenError):
    """Raised when a requested feature is not yet implemented.

    This is an explicit capability boundary — not a bug. Audio generation
    itself falls here until end-to-end inference lands.
    """
