# mlx-audiocraft handoff

## Current status

This repository was initialized as an independent open-source Python library for
MLX-native inference of AudioCraft-family pretrained models.

The current repo is a skeleton only. It does not implement MusicGen inference

## SDD preflight decisions

- Execution mode: `interactive`
- Artifact store: `openspec`
- PR strategy: `ask-always`
- Review budget: `400` changed lines

## Product decisions

- Repository/package name: `mlx-audiocraft`
- Scope: AudioCraft-family runtime, not generic audio ML
- First target: MusicGen inference
- No training and no training-from-scratch in scope
- Preferred API style: MLX/HuggingFace-style `from_pretrained`, not AudioCraft's
  `get_pretrained`
- Initial positioning: experimental/research; practical developer runtime later

## Verified AudioCraft loading behavior

AudioCraft's `MusicGen.get_pretrained(...)`:

1. Maps short names like `small`, `medium`, and `melody` to HuggingFace ids.
2. Downloads checkpoints via `hf_hub_download`.
3. Loads `state_dict.bin` for the language model.
4. Loads `compression_state_dict.bin` for the compression/audio codec model.
5. Reads `xp.cfg` from each checkpoint.
6. Rebuilds model objects from config.
7. Loads PyTorch state dicts.

For `mlx-audiocraft`, the desired behavior is:

```python
from mlx_audiocraft import MusicGen

model = MusicGen.from_pretrained("facebook/musicgen-small")
audio = model.generate("lofi hip hop, warm pads")
```

Internally, first-use behavior should likely be:

1. Download the HuggingFace checkpoint.
2. Read `xp.cfg`.
3. Convert PyTorch weights to MLX format if no converted cache exists.
4. Save converted MLX weights locally.
5. Load MLX-native modules.
6. Run inference.

An explicit CLI may also be useful later:

```bash
mlx-audiocraft convert facebook/musicgen-small
```

## Recommended next step

Start the next session from this directory:

```bash
cd /Users/fermellone/Desktop/projects/mlx-audiocraft
```

Then continue SDD planning with exploration/proposal for the first change:

> Define the MLX-native MusicGen inference and weight-conversion architecture.

Because execution mode is interactive, pause after each SDD phase and ask before
continuing.
