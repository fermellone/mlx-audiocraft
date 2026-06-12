"""Basic import smoke test for mlx-audiocraft."""


class TestImport:
    """Verify the package structure imports correctly."""

    def test_package_import(self) -> None:
        """Package can be imported and exposes expected version."""
        import mlx_audiocraft  # noqa: F401

        assert hasattr(mlx_audiocraft, "__version__")

    def test_musicgen_class_exists(self) -> None:
        """MusicGen class is exposed at package level."""
        from mlx_audiocraft import MusicGen  # noqa: F401

        assert MusicGen is not None

    def test_musicgen_from_pretrained_callable(self) -> None:
        """MusicGen.from_pretrained is exposed and callable."""
        from mlx_audiocraft import MusicGen

        assert callable(MusicGen.from_pretrained)
