# musicgen-runtime Specification

## Purpose

Define the supported MusicGen loading contract, MLX runtime boundaries, and failure behavior for the first architecture slice.

## Requirements

### Requirement: Public MusicGen Loading API

The system MUST expose `MusicGen.from_pretrained(model_id)` as the supported loading API for official `facebook/musicgen-*` checkpoints and MUST NOT support short-name aliases in this slice.

#### Scenario: Load official checkpoint

- GIVEN `model_id` is `facebook/musicgen-small`
- WHEN the caller loads it with `MusicGen.from_pretrained(model_id)`
- THEN the system MUST attempt the MLX-native loading flow for that checkpoint

#### Scenario: Reject unsupported identifier

- GIVEN `model_id` is not an official `facebook/musicgen-*` checkpoint
- WHEN the caller loads it
- THEN the system MUST fail with an actionable unsupported-checkpoint error

### Requirement: MLX Runtime Boundary

The system SHALL keep loading orchestration, checkpoint configuration, conversion/cache access, and MLX module construction as separate runtime responsibilities.

#### Scenario: Build runtime from resolved artifacts

- GIVEN supported checkpoint artifacts and converted MLX weights are available
- WHEN loading completes
- THEN the system MUST return an MLX-native `MusicGen` runtime object

### Requirement: Strict Failure Behavior

The system MUST fail clearly for unsupported checkpoint config, MLX model gaps, or unavailable runtime features, and MUST NOT fall back to PyTorch or partial experimental paths.

#### Scenario: Unsupported model gap

- GIVEN a supported checkpoint exposes a feature not implemented in MLX
- WHEN loading reaches that feature
- THEN the system MUST fail with the missing capability and suggested next action

### Requirement: Audio Output Boundary

The system MUST NOT claim successful text-to-audio generation until end-to-end inference is implemented.

#### Scenario: Audio generation requested before support

- GIVEN a `MusicGen` object was loaded by this architecture slice
- WHEN audio generation is requested
- THEN the system MUST fail with a clear not-yet-supported generation error
