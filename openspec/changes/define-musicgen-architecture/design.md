# Design: Define MusicGen Architecture

## Technical Approach

Build a thin public `MusicGen.from_pretrained(...)` API over strict internal services: checkpoint validation/resolution, `xp.cfg` parsing, versioned MLX cache lookup, automatic PyTorch-to-MLX conversion on cache miss, and MLX runtime assembly. This mirrors upstream AudioCraft loading shape (`get_pretrained` downloads HF artifacts, reads `xp.cfg`, builds modules, loads PyTorch state dicts) but removes short aliases, PyTorch fallback, training paths, and end-to-end generation claims for this first slice.

## Architecture Decisions

| Decision | Options considered | Choice and rationale |
|---|---|---|
| Public API | Keep constructor+`load()`, AudioCraft-style `get_pretrained`, or HF/MLX-style `from_pretrained` | Use `MusicGen.from_pretrained(model_id, cache_dir=None)`. It matches the approved contract and avoids exposing half-loaded instances. |
| Runtime boundaries | Keep everything in `_musicgen.py` or split internals | Keep `MusicGen` exported from `src/mlx_audiocraft/_musicgen.py`, move services under `src/mlx_audiocraft/musicgen/`. This preserves today’s import path while making conversion and runtime logic reviewable. |
| Conversion UX | CLI/manual conversion or automatic conversion | Convert automatically on cache miss inside `from_pretrained`. The spec requires first-use loading to resolve source artifacts, convert, cache, then load MLX weights. |
| Failure policy | Best-effort partial loading or strict errors | Use typed, actionable exceptions and no PyTorch fallback. Pre-alpha users need truthful capability boundaries more than silent experimental paths. |
| Cache format | Raw arrays only or metadata+weights | Store metadata beside MLX weights with source model id, converter version, and cache format. This enables safe reuse/reconversion as mappings evolve. |

## Data Flow

```text
caller
  └─ MusicGen.from_pretrained(model_id)
      ├─ validate official facebook/musicgen-* id
      ├─ resolve HF artifacts: xp.cfg, state_dict.bin, compression_state_dict.bin
      ├─ parse checkpoint config
      ├─ check versioned MLX cache
      ├─ convert PyTorch weights on cache miss
      └─ build MLX runtime object

MusicGen.generate(...) -> NotYetSupportedError until inference lands.
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `src/mlx_audiocraft/_musicgen.py` | Modify | Define public `MusicGen.from_pretrained`, loaded runtime state, and generation-not-supported behavior. |
| `src/mlx_audiocraft/musicgen/__init__.py` | Create | Internal MusicGen service package marker. |
| `src/mlx_audiocraft/musicgen/errors.py` | Create | Typed actionable errors for unsupported checkpoints, config gaps, conversion gaps, and generation boundary. |
| `src/mlx_audiocraft/musicgen/checkpoint.py` | Create | Official checkpoint validation plus HF artifact resolution contract. |
| `src/mlx_audiocraft/musicgen/config.py` | Create | `xp.cfg` parsing and supported-config validation boundary. |
| `src/mlx_audiocraft/musicgen/cache.py` | Create | Versioned converted-cache metadata lookup and compatibility checks. |
| `src/mlx_audiocraft/musicgen/conversion.py` | Create | PyTorch state-dict to MLX weight mapping scaffold and unsupported tensor reporting. |
| `src/mlx_audiocraft/musicgen/runtime.py` | Create | MLX module factory scaffold for LM and compression/codec boundaries. |
| `tests/test_musicgen_loading.py` | Create | Public API, unsupported id, cache-hit/cache-miss orchestration tests. |
| `tests/test_musicgen_conversion.py` | Create | Conversion metadata and strict unsupported-input tests. |

## Interfaces / Contracts

```python
MusicGen.from_pretrained(model_id: str, cache_dir: str | Path | None = None) -> MusicGen
```

`model_id` MUST start with `facebook/musicgen-`. The returned object is MLX-native scaffolding only; audio generation MUST raise `NotYetSupportedError`. Cache metadata MUST include `source_model_id`, `source_revision`, `cache_format_version`, and `converter_version`.

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | Identifier validation, typed errors, config/cache metadata compatibility | pytest with temp dirs and small fixtures/mocks; no network required by default. |
| Integration | `from_pretrained` orchestration on cache hit and cache miss | Mock HF resolver and converter to prove call order and no PyTorch fallback. |
| E2E | Real audio generation | Out of scope; assert `generate` fails clearly. |

## Migration / Rollout

No migration required. This replaces placeholder behavior with a stricter loading scaffold while preserving `from mlx_audiocraft import MusicGen`.

## Open Questions

- [ ] None blocking. Exact tensor-name mappings and MLX module internals should be finalized during implementation against real checkpoint samples.
