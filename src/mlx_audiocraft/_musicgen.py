"""MusicGen MLX-native inference runtime.

Provides the public MusicGen.from_pretrained API for loading official
``facebook/musicgen-*`` checkpoints and assembling an MLX-native runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from mlx_audiocraft.musicgen.errors import NotYetSupportedError, UnsupportedCheckpointError


class MusicGen:
    """MLX-native MusicGen inference runtime.

    Loaded via :meth:`from_pretrained`.  This scaffold validates the requested
    checkpoint identity and returns an MLX-native runtime object.  Audio
    generation is not yet wired — calling :meth:`generate` raises
    :exc:`NotYetSupportedError`.
    """

    _OFFICIAL_PREFIX = "facebook/musicgen-"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> MusicGen:
        """Load a pretrained MusicGen model.

        Validates that *model_id* is an official ``facebook/musicgen-*``
        checkpoint and returns an MLX-native runtime scaffold.  Full
        checkpoint resolution, config parsing, weight conversion, and
        caching will be wired through internal services in later slices.

        Args:
            model_id: HuggingFace model identifier.  Must start with
                ``facebook/musicgen-``.
            cache_dir: Optional directory for converted MLX weight cache.
                When ``None``, the default HuggingFace cache is used.

        Returns:
            An MLX-native :class:`MusicGen` runtime object.

        Raises:
            UnsupportedCheckpointError: If *model_id* is not an official
                ``facebook/musicgen-*`` checkpoint.
        """
        # cache_dir is consumed by later phases (checkpoint resolution / cache lookup).
        _ = cache_dir
        if not model_id.startswith(cls._OFFICIAL_PREFIX):
            raise UnsupportedCheckpointError(
                f"Unsupported checkpoint: {model_id!r}. "
                f"Only official {cls._OFFICIAL_PREFIX}* checkpoints are supported."
            )
        return cls(model_id=model_id)

    def generate(self, text: str, **kwargs: object) -> None:
        """Generate audio from a text description (not yet implemented).

        Args:
            text: Text prompt describing the audio to generate.  (Accepted
                for forward compatibility; currently unused as generation
                is not yet implemented.)
            **kwargs: Forward-compatible keyword arguments (unused).

        Raises:
            NotYetSupportedError: Audio generation is not yet supported.
        """
        # text and kwargs accepted for forward compatibility.
        _ = (text, kwargs)
        raise NotYetSupportedError(
            "Audio generation is not yet supported in this architecture slice. "
            "Text-to-audio inference will be implemented in a future change."
        )
