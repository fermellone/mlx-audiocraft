# Proposal: Define MusicGen Architecture

## Intent

Define the first MLX-native MusicGen architecture slice: public API, model-loading flow, and PyTorch-to-MLX weight-conversion plan/scaffold. This turns the current placeholder into a reviewable contract for official HuggingFace `facebook/musicgen-*` checkpoints without promising end-to-end generation yet.

Reference behavior: AudioCraft `MusicGen.get_pretrained(...)` downloads HF artifacts, reads `xp.cfg`, rebuilds modules, and loads PyTorch state dicts.

## Scope

### In Scope
- Specify `MusicGen.from_pretrained("facebook/musicgen-small")` as the public loading API.
- Define first-use flow: download checkpoint, read `xp.cfg`, convert missing MLX cache, save converted weights, load MLX modules.
- Define architecture boundaries for MusicGen LM, compression/codec loading, conversion/cache services, and actionable unsupported-feature errors.

### Out of Scope
- Full text-to-audio inference quality or parity.
- Training, fine-tuning, or training-from-scratch.
- Non-official checkpoints, short-name aliases, CLI conversion, or fallback to PyTorch.

## Capabilities

### New Capabilities
- `musicgen-runtime`: MLX-native MusicGen loading API, supported checkpoint policy, runtime module boundaries, and failure behavior.
- `musicgen-weight-conversion`: automatic PyTorch-to-MLX conversion/cache behavior for official `facebook/musicgen-*` checkpoints.

### Modified Capabilities
- None; `openspec/specs/` is empty.

## Approach

Document and scaffold a layered runtime: `MusicGen.from_pretrained` orchestrates HF artifact resolution, `xp.cfg` parsing, cache lookup, conversion when needed, and MLX module construction. Conversion remains explicit and strict: unsupported config/model gaps fail with actionable errors rather than partial experimental fallback.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/mlx_audiocraft/_musicgen.py` | Modified | Replace placeholder loading contract with `from_pretrained` architecture surface. |
| `src/mlx_audiocraft/` | New | Add runtime/conversion module boundaries for MusicGen architecture. |
| `tests/` | Modified | Add API/conversion-cache behavior tests around the scaffold. |
| `openspec/specs/` | New | Add specs for runtime and conversion capabilities. |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Upstream AudioCraft config/weights differ across checkpoints | Med | Limit first slice to official `facebook/musicgen-*`; validate `xp.cfg` before conversion. |
| Architecture over-promises inference readiness | Med | Keep success criteria focused on contract and scaffold, not audio generation parity. |
| Cache format churn | Med | Version converted cache metadata from the first scaffold. |

## Rollback Plan

Revert the OpenSpec change and any scaffold files/tests introduced by the implementation. Existing placeholder import behavior remains the baseline.

## Dependencies

- HuggingFace Hub artifacts for official `facebook/musicgen-*` checkpoints.
- MLX array/module APIs on Apple Silicon.
- Upstream AudioCraft loading behavior as architectural reference.

## Success Criteria

- [ ] Proposal/spec/design define MusicGen runtime and weight-conversion contracts.
- [ ] Public API contract uses `MusicGen.from_pretrained(...)`.
- [ ] First-use conversion/cache flow is specified with clear unsupported-gap errors.
