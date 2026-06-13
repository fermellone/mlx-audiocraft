"""Official MusicGen checkpoint validation and artifact resolution.

Provides :func:`is_official_musicgen_id` for checkpoint identity checks and
:func:`resolve_artifacts` for downloading required HuggingFace artifacts with
an injectable downloader hook for test isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from mlx_audiocraft.musicgen.errors import UnsupportedCheckpointError

#: Injectable downloader callable signature: ``(repo_id, filename, *, cache_dir=None) -> str``.
Downloader = Callable[..., str]


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Paths to resolved HuggingFace checkpoint artifacts.

    Attributes:
        model_id: The HuggingFace model identifier (e.g. ``facebook/musicgen-small``).
        config_path: Local path to the ``xp.cfg`` checkpoint config file.
        state_dict_path: Local path to the ``state_dict.bin`` PyTorch weights file.
        compression_state_dict_path: Local path to the ``compression_state_dict.bin``
            codec weights file.
        cache_dir: Caller-provided cache root used for artifact resolution, if any.
    """

    model_id: str
    config_path: Path
    state_dict_path: Path
    compression_state_dict_path: Path
    cache_dir: Optional[Path]


def is_official_musicgen_id(model_id: str) -> bool:
    """Return ``True`` if *model_id* names an official ``facebook/musicgen-*`` checkpoint."""
    return model_id.startswith("facebook/musicgen-")


def resolve_artifacts(
    model_id: str,
    cache_dir: Optional[Union[str, Path]] = None,
    *,
    downloader: Optional[Downloader] = None,
) -> ResolvedCheckpoint:
    """Download and resolve all required checkpoint artifacts.

    Args:
        model_id: Official ``facebook/musicgen-*`` checkpoint identifier.
        cache_dir: Optional directory for caching downloaded artifacts.
            When ``None``, the default HuggingFace cache is used.
        downloader: Injectable download function with the same call signature
            as :func:`huggingface_hub.hf_hub_download`.  When ``None``, the
            real ``hf_hub_download`` is lazy-imported.  Tests inject a stub
            to avoid network access.

    Returns:
        A :class:`ResolvedCheckpoint` with local paths to all required artifacts.

    Raises:
        UnsupportedCheckpointError: If *model_id* is not an official checkpoint.
    """
    if not is_official_musicgen_id(model_id):
        raise UnsupportedCheckpointError(
            f"Unsupported checkpoint: {model_id!r}. "
            f"Only official facebook/musicgen-* checkpoints are supported."
        )

    _cache_dir: Optional[str] = str(cache_dir) if cache_dir is not None else None
    _download: Callable[..., str]

    if downloader is not None:
        _download = downloader
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as err:
            raise RuntimeError(
                "huggingface_hub is required for artifact resolution. "
                "Install it with `pip install huggingface_hub` or inject a custom downloader."
            ) from err
        _download = hf_hub_download

    config_path = Path(
        _download(repo_id=model_id, filename="xp.cfg", cache_dir=_cache_dir)
    )
    state_dict_path = Path(
        _download(repo_id=model_id, filename="state_dict.bin", cache_dir=_cache_dir)
    )
    compression_path = Path(
        _download(
            repo_id=model_id,
            filename="compression_state_dict.bin",
            cache_dir=_cache_dir,
        )
    )

    return ResolvedCheckpoint(
        model_id=model_id,
        config_path=config_path,
        state_dict_path=state_dict_path,
        compression_state_dict_path=compression_path,
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
    )
