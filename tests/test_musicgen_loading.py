"""Tests for MusicGen loading API (from_pretrained, generation boundary, checkpoint
resolution, and config validation)."""

from pathlib import Path
from typing import Any

import pytest

from mlx_audiocraft import MusicGen
from mlx_audiocraft.musicgen.checkpoint import (
    ResolvedCheckpoint,
    is_official_musicgen_id,
    resolve_artifacts,
)
from mlx_audiocraft.musicgen.config import (
    SUPPORTED_SECTIONS,
    parse_xp_cfg,
    validate_supported,
)
from mlx_audiocraft.musicgen.errors import (
    NotYetSupportedError,
    UnsupportedCheckpointError,
    UnsupportedConfigError,
)


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


# ---------------------------------------------------------------------------
# Phase 2: Checkpoint validation
# ---------------------------------------------------------------------------


class TestIsOfficialMusicgenId:
    """``is_official_musicgen_id`` checkpoint identity checks."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "facebook/musicgen-small",
            "facebook/musicgen-medium",
            "facebook/musicgen-large",
            "facebook/musicgen-melody",
            "facebook/musicgen-stereo-small",
            "facebook/musicgen-stereo-medium",
            "facebook/musicgen-stereo-large",
            "facebook/musicgen-stereo-melody",
        ],
    )
    def test_official_ids_return_true(self, model_id: str) -> None:
        """Every known official checkpoint variant is recognized."""
        assert is_official_musicgen_id(model_id) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "",
            "some-other/model",
            "facebook/musicgen",
            "facebook/",
            "musicgen-small",
            "facebook/audiogen-medium",
        ],
    )
    def test_unofficial_ids_return_false(self, model_id: str) -> None:
        """Non-official or malformed IDs are not recognized."""
        assert is_official_musicgen_id(model_id) is False


# ---------------------------------------------------------------------------
# Phase 2: Artifact resolution with stubbed downloader
# ---------------------------------------------------------------------------


class TestResolveArtifacts:
    """``resolve_artifacts`` with an injectable downloader stub."""

    @staticmethod
    def _fake_downloader(repo_id: str, filename: str, **kwargs: Any) -> str:  # noqa: ARG004
        """Stub that returns a path under *cache_dir* (or /tmp)."""
        cache_dir = kwargs.get("cache_dir")
        base = Path(cache_dir) if cache_dir else Path("/tmp")
        return str(base / filename)

    def test_returns_resolved_checkpoint_dataclass(self, tmp_path: Path) -> None:
        """A successful resolution returns a ResolvedCheckpoint with expected fields."""
        resolved = resolve_artifacts(
            "facebook/musicgen-small",
            cache_dir=tmp_path,
            downloader=self._fake_downloader,
        )
        assert isinstance(resolved, ResolvedCheckpoint)
        assert resolved.model_id == "facebook/musicgen-small"
        assert resolved.config_path == tmp_path / "xp.cfg"
        assert resolved.state_dict_path == tmp_path / "state_dict.bin"
        assert resolved.compression_state_dict_path == tmp_path / "compression_state_dict.bin"
        assert resolved.cache_dir == tmp_path

    def test_downloader_receives_correct_arguments(self, tmp_path: Path) -> None:
        """The injected downloader is called with the expected repo/filename."""
        calls: list[dict] = []

        def recording_downloader(repo_id: str, filename: str, **kwargs: Any) -> str:
            calls.append({"repo_id": repo_id, "filename": filename, "kwargs": kwargs})
            return str(tmp_path / filename)

        resolve_artifacts(
            "facebook/musicgen-medium",
            cache_dir=tmp_path,
            downloader=recording_downloader,
        )

        assert len(calls) == 3
        filenames = [c["filename"] for c in calls]
        assert "xp.cfg" in filenames
        assert "state_dict.bin" in filenames
        assert "compression_state_dict.bin" in filenames
        for c in calls:
            assert c["repo_id"] == "facebook/musicgen-medium"

    def test_rejects_unofficial_checkpoint(self, tmp_path: Path) -> None:
        """Non-official model_id raises UnsupportedCheckpointError immediately."""
        with pytest.raises(UnsupportedCheckpointError, match="some-other/model"):
            resolve_artifacts(
                "some-other/model",
                cache_dir=tmp_path,
                downloader=self._fake_downloader,
            )

    def test_rejects_empty_model_id(self, tmp_path: Path) -> None:
        """Empty string is not a valid official checkpoint."""
        with pytest.raises(UnsupportedCheckpointError):
            resolve_artifacts(
                "",
                cache_dir=tmp_path,
                downloader=self._fake_downloader,
            )

    def test_cache_dir_remains_none_when_not_provided(self) -> None:
        """When cache_dir is None, the dataclass does not invent a cache path."""
        resolved = resolve_artifacts(
            "facebook/musicgen-small",
            cache_dir=None,
            downloader=self._fake_downloader,
        )
        # The downloader receives cache_dir=None and HuggingFace Hub decides
        # its actual cache location; ResolvedCheckpoint only records caller input.
        assert resolved.cache_dir is None


# ---------------------------------------------------------------------------
# Phase 2: Config parsing
# ---------------------------------------------------------------------------


class TestParseXpCfg:
    """``parse_xp_cfg`` checkpoint config parsing."""

    def test_parses_yaml_style_config(self, tmp_path: Path) -> None:
        """A YAML-style xp.cfg is parsed into a dictionary."""
        cfg_path = tmp_path / "xp.cfg"
        cfg_path.write_text(
            """\
transformer_lm:
  dim: 1024
  num_layers: 24
compression_model:
  sample_rate: 32000
"""
        )
        result = parse_xp_cfg(cfg_path)
        assert isinstance(result, dict)
        assert "transformer_lm" in result
        assert "compression_model" in result
        assert result["transformer_lm"]["dim"] == 1024

    def test_parses_key_value_style_config(self, tmp_path: Path) -> None:
        """A flat key=value config is parsed into a dictionary."""
        cfg_path = tmp_path / "xp.cfg"
        cfg_path.write_text(
            """\
[transformer_lm]
dim = 1024
num_layers = 24

[pattern]
delay = 4
"""
        )
        result = parse_xp_cfg(cfg_path)
        assert isinstance(result, dict)
        assert "transformer_lm" in result
        assert result["transformer_lm"]["dim"] == 1024
        assert "pattern" in result
        assert result["pattern"]["delay"] == 4

    def test_parses_value_types(self, tmp_path: Path) -> None:
        """String values are coerced to int/float/bool where possible."""
        cfg_path = tmp_path / "xp.cfg"
        cfg_path.write_text(
            """\
[model]
layers = 24
lr = 1e-4
use_custom = True
name = musicgen
"""
        )
        result = parse_xp_cfg(cfg_path)
        assert result["model"]["layers"] == 24
        assert result["model"]["lr"] == 0.0001
        assert result["model"]["use_custom"] is True
        assert result["model"]["name"] == "musicgen"

    def test_parses_dora_omegaconf_style(self, tmp_path: Path) -> None:
        """Dora/OmegaConf-style config blocks are parsed."""
        cfg_path = tmp_path / "xp.cfg"
        cfg_path.write_text(
            """\
dset:
  train: /path/to/data
  batch_size: 64

solver:
  optimizer: adam
  lr: 0.001
"""
        )
        result = parse_xp_cfg(cfg_path)
        assert "dset" in result
        assert "solver" in result
        assert result["dset"]["batch_size"] == 64

    def test_raises_file_not_found_for_missing_path(self, tmp_path: Path) -> None:
        """A nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_xp_cfg(tmp_path / "nonexistent.cfg")


class TestValidateSupported:
    """``validate_supported`` config section validation."""

    def test_accepts_known_sections(self) -> None:
        """A config with only known sections passes validation."""
        cfg: dict[str, Any] = {
            "transformer_lm": {"dim": 1024},
            "pattern": {"delay": 4},
        }
        # Should not raise.
        validate_supported(cfg)

    def test_accepts_empty_config(self) -> None:
        """An empty config dictionary passes validation."""
        validate_supported({})

    def test_rejects_unknown_section(self) -> None:
        """A config with an unknown section raises UnsupportedConfigError."""
        cfg = {"unknown_section": {"foo": 1}}
        with pytest.raises(UnsupportedConfigError, match="unknown_section"):
            validate_supported(cfg)

    def test_rejects_multiple_unknown_sections(self) -> None:
        """Multiple unknown sections are listed in the error message."""
        cfg = {"foo": {}, "bar": {}}
        with pytest.raises(UnsupportedConfigError, match="foo"):
            validate_supported(cfg)

    def test_error_message_includes_supported_list(self) -> None:
        """The error message mentions supported sections for guidance."""
        cfg = {"bogus": {}}
        with pytest.raises(UnsupportedConfigError, match="Supported sections"):
            validate_supported(cfg)

    def test_standard_musicgen_sections_are_supported(self) -> None:
        """Core MusicGen config sections are in the supported set."""
        assert "transformer_lm" in SUPPORTED_SECTIONS
        assert "compression_model" in SUPPORTED_SECTIONS
        assert "pattern" in SUPPORTED_SECTIONS
        assert "condition_provider" in SUPPORTED_SECTIONS
        assert "dset" in SUPPORTED_SECTIONS


# ---------------------------------------------------------------------------
# Phase 2: Parse + Validate integration
# ---------------------------------------------------------------------------


class TestConfigParseAndValidate:
    """Integration: parse xp.cfg then validate sections."""

    def test_supported_config_passes(self, tmp_path: Path) -> None:
        """Parse a supported config file and validate it passes."""
        cfg_path = tmp_path / "xp.cfg"
        cfg_path.write_text(
            """\
transformer_lm:
  dim: 1024
  num_layers: 24
pattern:
  delay: 4
"""
        )
        cfg = parse_xp_cfg(cfg_path)
        validate_supported(cfg)  # Should not raise.

    def test_unsupported_config_raises_after_parse(self, tmp_path: Path) -> None:
        """Parse a config with unknown sections and validate raises."""
        cfg_path = tmp_path / "xp.cfg"
        cfg_path.write_text(
            """\
unsupported_block:
  param: value
"""
        )
        cfg = parse_xp_cfg(cfg_path)
        with pytest.raises(UnsupportedConfigError):
            validate_supported(cfg)
