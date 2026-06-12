# musicgen-weight-conversion Specification

## Purpose

Define automatic PyTorch-to-MLX conversion and cache behavior for official `facebook/musicgen-*` checkpoints.

## Requirements

### Requirement: Official Checkpoint Artifacts

The system MUST resolve official HuggingFace `facebook/musicgen-*` checkpoint artifacts, including checkpoint configuration and PyTorch weights, before conversion.

#### Scenario: Resolve official artifacts

- GIVEN an official `facebook/musicgen-*` checkpoint has no converted cache
- WHEN `MusicGen.from_pretrained(...)` is called
- THEN the system MUST resolve the required source artifacts for conversion

### Requirement: Automatic Conversion on Cache Miss

The system MUST run PyTorch-to-MLX conversion automatically inside `MusicGen.from_pretrained(...)` when no compatible converted MLX cache exists.

#### Scenario: Convert missing cache

- GIVEN source artifacts are valid and the MLX cache is missing
- WHEN loading begins
- THEN the system MUST convert supported weights and continue loading from MLX weights

### Requirement: Versioned Conversion Cache

The system SHALL persist converted MLX weights with metadata that identifies source checkpoint, cache format, and converter version.

#### Scenario: Reuse compatible cache

- GIVEN a compatible converted cache exists for the requested checkpoint
- WHEN `MusicGen.from_pretrained(...)` is called
- THEN the system MUST reuse the cache without reconverting weights

#### Scenario: Reject incompatible cache

- GIVEN cache metadata is missing or incompatible
- WHEN loading checks the cache
- THEN the system MUST reject it and either safely reconvert or fail actionably

### Requirement: Strict Conversion Failure

The system MUST fail before exposing a usable runtime when conversion encounters unsupported tensors, config, or model structure, and MUST NOT fall back to PyTorch.

#### Scenario: Unsupported conversion input

- GIVEN conversion reaches an unsupported weight or config field
- WHEN conversion runs
- THEN the system MUST report the unsupported item and avoid presenting partial audio output support
