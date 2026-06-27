# kokoro-voiceopt

This package implements black-box Kokoro voice optimization.

It does **not** train Kokoro, update model weights, or require a differentiable audio generation path. It searches for a better Kokoro `voice.pt` tensor by synthesizing candidate voices, scoring generated audio with a speaker-verification model, and optimizing over a constrained voice manifold built from existing Kokoro voices.

## Install dependencies

```bash
uv sync
```

The default speaker encoder uses:

```text
microsoft/wavlm-base-plus-sv
```

via `transformers`.

Silero VAD is used for target speech segmentation.

## CLI usage

```bash
uv run python scripts/optimize_voice.py \
  --target-audio path/to/target.wav \
  --target-transcript "exact words spoken in the target clip" \
  --output-dir runs/my_voice \
  --lang-code a \
  --device cuda
```

Important options:

```bash
uv run python scripts/optimize_voice.py \
  --target-audio target.wav \
  --target-transcript "The transcript of the target audio goes here." \
  --output-dir runs/example \
  --voices-dir voices \
  --speaker-model microsoft/wavlm-base-plus-sv \
  --speaker-batch-size 16 \
  --top-k 8 \
  --latent-dim 32 \
  --blend-iterations 60 \
  --latent-iterations 180 \
  --population-pairs 12 \
  --seed 1234
```

## Target transcript requirement

A non-empty target transcript is required.

Speaker encoders are not perfectly content-invariant. If generated text and target speech contain unrelated words, the score can be contaminated by phonetic content, rhythm, duration, and prosody. The optimizer therefore uses text derived from the target transcript for the optimization stage.

Validation prompts are separate and are used to detect overfitting to the optimization text.

Validation texts can be supplied with:

```bash
--validation-texts validation_prompts.txt
```

The file should contain one prompt per line. Blank lines and lines starting with `#` are ignored.

## What the optimizer does

The pipeline performs these stages:

1. Load and preprocess target audio.
2. Use VAD to split target speech into clean segments.
3. Build a normalized target speaker profile from segment embeddings.
4. Load Kokoro voice tensors.
5. Evaluate all stock voices as a baseline.
6. Optimize a convex blend of the best stock voices.
7. Fit a full-sequence low-rank manifold over the Kokoro voice corpus.
8. Optimize bounded latent coordinates in that manifold.
9. Validate baseline, blend, latent, final, and checkpoint candidates.
10. Save the best validation-selected voice.

## Output artifacts

A run directory contains:

```text
runs/my_voice/
  config.json
  run_info.json
  text_plan.json

  target_profile.pt
  target_profile.json

  corpus/
    corpus_manifest.json

  manifold/
    manifold.pt
    manifold_report.json

  stages/
    baseline.json
    blend_history.json
    latent_history.json
    validation.json

  voices/
    voice_best.pt
    voice_final.pt
    voice_best_optimization.pt
    voice_best_blend.pt
    voice_best_baseline.pt
    *_meta.json

  samples/
    validation/
    optimization_bests/

  logs/
```

The main artifact is:

```text
voices/voice_best.pt
```

It is saved as a CPU `float32` Kokoro-compatible tensor shaped:

```text
[T, 1, 256]
```

## Loading the optimized voice

```py
from kokoro import KPipeline
import torch

pipeline = KPipeline(lang_code="a")
voice = torch.load("runs/my_voice/voices/voice_best.pt", map_location="cpu")

for result in pipeline("Your text here.", voice=voice):
    audio = result.audio
```

## Notes

- The selected `voice_best.pt` is chosen by validation loss.
- `voice_best_optimization.pt` is the best candidate on the optimization prompts.
- `voice_final.pt` is the last latent candidate.
- Generated WAV samples are saved when sample saving is enabled.
