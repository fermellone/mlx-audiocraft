"""Tests for MusicGen loading API (from_pretrained and generation boundary)."""

import pytest

from mlx_audiocraft import MusicGen
from mlx_audiocraft.musicgen.errors import NotYetSupportedError, UnsupportedCheckpointError


class TestMusicGenFromPretrained:
    """``MusicGen.from_pretrained`` checkpoint validation."""

    def test_official_id_accepted(self) -> None:
        """Loading an official checkpoint ID returns a MusicGen instance."""
        model = MusicGen.from_pretrained("facebook/musicgen-small")
        assert isinstance(model, MusicGen)
        assert model.model_id == "facebook/musicgen-small"

    def test_official_id_medium_accepted(self) -> None:
        """Loading facebook/musicgen-medium also succeeds."""
        model = MusicGen.from_pretrained("facebook/musicgen-medium")
        assert isinstance(model, MusicGen)
        assert model.model_id == "facebook/musicgen-medium"

    def test_official_id_large_accepted(self) -> None:
        """Loading facebook/musicgen-large also succeeds."""
        model = MusicGen.from_pretrained("facebook/musicgen-large")
        assert isinstance(model, MusicGen)
        assert model.model_id == "facebook/musicgen-large"

    def test_official_id_melody_accepted(self) -> None:
        """Loading facebook/musicgen-melody also succeeds."""
        model = MusicGen.from_pretrained("facebook/musicgen-melody")
        assert isinstance(model, MusicGen)
        assert model.model_id == "facebook/musicgen-melody"

    def test_unofficial_id_raises_unsupported_checkpoint_error(self) -> None:
        """A non-official checkpoint ID raises UnsupportedCheckpointError."""
        with pytest.raises(UnsupportedCheckpointError):
            MusicGen.from_pretrained("some-other/model")

    def test_unofficial_id_error_message_contains_model_id(self) -> None:
        """The error message includes the offending model_id."""
        with pytest.raises(UnsupportedCheckpointError, match="some-other/model"):
            MusicGen.from_pretrained("some-other/model")

    def test_empty_model_id_raises(self) -> None:
        """An empty string is not a valid official checkpoint."""
        with pytest.raises(UnsupportedCheckpointError):
            MusicGen.from_pretrained("")

    def test_bare_facebook_prefix_rejected(self) -> None:
        """The bare 'facebook/' prefix without 'musicgen-' is rejected."""
        with pytest.raises(UnsupportedCheckpointError):
            MusicGen.from_pretrained("facebook/some-model")


class TestMusicGenGenerate:
    """``MusicGen.generate`` not-yet-supported boundary."""

    def test_generate_raises_not_yet_supported_error(self) -> None:
        """Calling generate() raises NotYetSupportedError."""
        model = MusicGen.from_pretrained("facebook/musicgen-small")
        with pytest.raises(NotYetSupportedError):
            model.generate("a test prompt")

    def test_generate_with_kwargs_also_raises(self) -> None:
        """Calling generate() with extra kwargs also raises NotYetSupportedError."""
        model = MusicGen.from_pretrained("facebook/musicgen-small")
        with pytest.raises(NotYetSupportedError):
            model.generate("prompt", duration=10, top_k=250)
