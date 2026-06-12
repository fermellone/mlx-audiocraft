"""MusicGen checkpoint config parsing and validation.

Parses the ``xp.cfg`` checkpoint file and validates that the config only
contains supported sections for the current MLX-native architecture slice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from mlx_audiocraft.musicgen.errors import UnsupportedConfigError

#: Config section names that the current MLX-native architecture slice can
#: safely consume.  Unknown sections trigger :exc:`UnsupportedConfigError`
#: to avoid silent misconfiguration.
SUPPORTED_SECTIONS: frozenset[str] = frozenset(
    {
        "transformer_lm",
        "lm",
        "pattern",
        "condition_provider",
        "dset",
        "codec",
        "compression_model",
        "dataset",
        "datasource",
        "diffusion",
        "model",
        "solvers",
        "audio",
        "channels",
        "continue_from",
        "deadlock",
        "evaluate",
        "execute_only",
        "generate",
        "grid",
        "log",
        "optim",
        "solver",
        "device",  # some older configs include this
    }
)


def parse_xp_cfg(path: Path) -> dict[str, Any]:
    """Parse a MusicGen ``xp.cfg`` checkpoint config file.

    Reads the file at *path* and returns the parsed configuration as a
    dictionary.  The file is expected to be a YAML-style config. If the
    content cannot be loaded as a YAML mapping, a minimal flat-config parser
    handles simple ``[section]``/``key = value`` files used by tests.

    Args:
        path: Filesystem path to the ``xp.cfg`` file.

    Returns:
        Parsed configuration dictionary whose keys are top-level section names.

    Raises:
        FileNotFoundError: If *path* does not exist.
        UnsupportedConfigError: If the config contains unsupported sections
            (enforced by :func:`validate_supported`; callers may choose to
            separate parse-time and validation-time failures).
    """
    raw_text = path.read_text()

    # PyYAML is a runtime dependency for checkpoint config parsing. If the
    # file is not a YAML mapping, use the minimal flat-config parser.
    try:
        import yaml  # type: ignore[import-untyped]

        parsed = yaml.safe_load(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return _parse_flat_config(raw_text)


def validate_supported(cfg: dict[str, Any]) -> None:
    """Validate that *cfg* only contains supported MusicGen config sections.

    Any key in *cfg* that is not in :data:`SUPPORTED_SECTIONS` causes an
    :exc:`UnsupportedConfigError` listing the offending section name(s).

    Args:
        cfg: Parsed checkpoint configuration dictionary.

    Raises:
        UnsupportedConfigError: If one or more unsupported sections are found.
    """
    unknown = [k for k in cfg if k not in SUPPORTED_SECTIONS]
    if unknown:
        raise UnsupportedConfigError(
            f"Unsupported config section(s): {', '.join(sorted(unknown))!r}. "
            f"Supported sections: {sorted(SUPPORTED_SECTIONS)}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_flat_config(text: str) -> dict[str, Any]:
    """Minimal fallback parser for flat key-value config text.

    Handles both ``section.key = value`` and ``key: value`` styles.
    Nested sections are flattened with dot-separated keys.
    """
    result: dict[str, Any] = {}
    current_section: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Section header: [section_name]  or  section_name:
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip()
            continue
        if line.endswith(":") and "=" not in line and not line.startswith("-"):
            candidate = line[:-1].strip()
            # Heuristic: short alphanumeric name without spaces → section header.
            if " " not in candidate and candidate:
                current_section = candidate
                continue

        # Key-value pair: key = value, key: value, section.key = value.
        for sep in (" = ", "=", ": "):
            if sep in line:
                k, v = line.split(sep, 1)
                key = k.strip()
                val: Any = _coerce_value(v.strip().rstrip(","))
                if current_section and "." not in key:
                    result.setdefault(current_section, {})[key] = val
                else:
                    result[key] = val  # top-level or already dotted
                break

    return result


def _coerce_value(raw: str) -> Any:
    """Coerce a raw config value string to int, float, bool, or str."""
    raw = raw.strip().strip("\"'")
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw
