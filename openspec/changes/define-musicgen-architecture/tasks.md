# Tasks: Define MusicGen Architecture

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~510 (add ~535, del ~25) |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 (API+errors) -> PR 2 (checkpoint/config) -> PR 3 (cache/conversion) -> PR 4 (runtime wiring) |
| Delivery strategy | ask-on-risk |
| Chain strategy | stacked-to-main |

Decision needed before apply: No
Chained PRs recommended: Yes
Chain strategy: stacked-to-main
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | `from_pretrained` scaffold + typed errors | PR 1 | Base: main; preserves `from mlx_audiocraft import MusicGen` |
| 2 | Official id validation + `xp.cfg` parsing | PR 2 | Base: main; depends on errors from PR 1 |
| 3 | Versioned cache + PyTorch->MLX scaffold | PR 3 | Base: main; depends on checkpoint/config from PR 2 |
| 4 | MLX runtime assembly + orchestration wiring | PR 4 | Base: main; depends on PR 1+2+3 |

## Phase 1: Foundation - API & Errors (Work Unit 1)

- [ ] 1.1 Create `src/mlx_audiocraft/musicgen/__init__.py` exporting package marker.
- [ ] 1.2 Create `src/mlx_audiocraft/musicgen/errors.py` with `UnsupportedCheckpointError`, `UnsupportedConfigError`, `UnsupportedModelFeatureError`, `IncompatibleCacheError`, `NotYetSupportedError`.
- [ ] 1.3 Modify `src/mlx_audiocraft/_musicgen.py`: replace placeholder `__init__`/`load` with classmethod `MusicGen.from_pretrained(model_id, cache_dir=None)` returning MLX scaffold; `generate()` raises `NotYetSupportedError`.
- [ ] 1.4 Create `tests/test_musicgen_loading.py` covering: official id accepted, non-official id raises `UnsupportedCheckpointError`, `generate()` raises `NotYetSupportedError`.
- [ ] 1.5 Extend `tests/test_import.py` asserting `MusicGen.from_pretrained` is callable.

## Phase 2: Checkpoint & Config Resolution (Work Unit 2)

- [ ] 2.1 Create `src/mlx_audiocraft/musicgen/checkpoint.py` with `is_official_musicgen_id()` and `resolve_artifacts(model_id, cache_dir)` returning a `ResolvedCheckpoint` dataclass.
- [ ] 2.2 Add injectable downloader hook to `resolve_artifacts` so tests stub `hf_hub_download` without network.
- [ ] 2.3 Create `src/mlx_audiocraft/musicgen/config.py` with `parse_xp_cfg(path)` and `validate_supported(cfg)` raising `UnsupportedConfigError` for unsupported sections.
- [ ] 2.4 Extend `tests/test_musicgen_loading.py` with resolver + xp.cfg tests using `tmp_path` and the stubbed downloader.

## Phase 3: Cache & Conversion Scaffold (Work Unit 3)

- [ ] 3.1 Create `src/mlx_audiocraft/musicgen/cache.py` with `CacheMetadata` (`source_model_id`, `source_revision`, `cache_format_version`, `converter_version`) plus `lookup()` and `is_compatible()`.
- [ ] 3.2 Add `write_metadata()` and `reconvert_or_fail()` that raises `IncompatibleCacheError` on missing/incompatible metadata.
- [ ] 3.3 Create `src/mlx_audiocraft/musicgen/conversion.py` with `convert_state_dict()` scaffold returning MLX arrays; raise `UnsupportedModelFeatureError` listing offending tensor names. No PyTorch fallback.
- [ ] 3.4 Create `tests/test_musicgen_conversion.py` covering cache hit reuse, strict fail on incompatible metadata, and unsupported tensor reporting.

## Phase 4: Runtime Assembly & Wiring (Work Unit 4)

- [ ] 4.1 Create `src/mlx_audiocraft/musicgen/runtime.py` with `build_runtime(cfg, mlx_weights)` returning an MLX-native runtime scaffold (no inference body).
- [ ] 4.2 Wire `MusicGen.from_pretrained` to call checkpoint -> config -> cache -> conversion -> runtime in order; accept injectable hooks.
- [ ] 4.3 Add integration test in `tests/test_musicgen_loading.py` asserting call order on cache miss vs cache hit (mocked resolver/converter) and no PyTorch fallback path.

## Phase 5: Verification & Cleanup

- [ ] 5.1 Run `pytest`, `ruff check src/`, `mypy src/`; fix any new violations.
- [ ] 5.2 Update `README.md` to document the `from_pretrained` contract and the generation-not-supported boundary.
