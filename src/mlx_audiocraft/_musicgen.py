"""MusicGen placeholder interface."""

from __future__ import annotations


class MusicGen:
    """Placeholder for the MLX-native MusicGen runtime."""

    @classmethod
    def load(cls, name: str) -> "MusicGen":
        """Load a pretrained MusicGen model by name."""
        raise NotImplementedError("MusicGen loading is not implemented yet.")

    def generate(self, prompt: str) -> bytes:
        """Generate audio from a text prompt."""
        raise NotImplementedError("MusicGen generation is not implemented yet.")
