# kokoro-voiceopt

This package implements black-box Kokoro voice optimization.

It optimizes a Kokoro voicepack against prepared, validated target speaker clips.

## Command flow

```bash
kokoro-voiceopt assets  --config config.yaml --project-root . --force

kokoro-voiceopt prepare \
  --config config.yaml \
  --project-root . \
  --audio-dir raw_audio \
  --transcripts target.jsonl \
  --force

kokoro-voiceopt check   --config config.yaml --project-root .
kokoro-voiceopt profile --config config.yaml --project-root . --force
kokoro-voiceopt doctor  --config config.yaml --project-root .
kokoro-voiceopt optimize --config config.yaml --project-root .
kokoro-voiceopt export  --config config.yaml --project-root . --candidate best
kokoro-voiceopt preview --config config.yaml --project-root .
```

All relative CLI paths are resolved against `--project-root`.

## Transcript formats

Preferred JSONL format:

```json
{"audio":"speaker/file001.wav","text":"This is the spoken text.","start_s":12.5,"end_s":18.2}
{"audio":"speaker/file002.wav","text":"This whole file is one target clip."}
```

Fields:

- `audio` or `path`: required.
- `text`: required.
- `start_s`: optional.
- `end_s`: optional.
- `id`: optional.

If `start_s` and `end_s` are absent, the whole file is used. If timestamps are present, the exact slice is used. There is no automatic splitting.

Pipe format is also supported for simple whole-file clips:

```text
relative/audio.wav|This is the spoken text.
```

## Text normalization and spoken-form checks

Preparation applies deterministic normalization:

- Unicode NFKC normalization,
- curly quote normalization,
- em/en dash normalization,
- bracketed noise-marker removal,
- whitespace collapse.

When `data.require_spoken_form` is true, rows are rejected if the normalized text contains URLs, email addresses, digits, currency symbols, or raw `@`.

Use spoken-form transcript text.

## Run layout

A run is self-describing on disk. Artifact metadata and cache fingerprints are stored beside artifact data.

```text
runs/<id>/
  config.resolved.yaml

  data/
    audio/
      000000.wav
      000001.wav
    manifests/
      target.jsonl
      rejected.jsonl
    report.json
    text_plan.json
    text_plan.meta.json

  profile/
    target_profile.pt
    target_profile.json

  corpus/
    corpus.pt
    corpus_manifest.json

  manifold/
    manifold.pt
    manifold_report.json

  optimize/
    stages/
      baseline.json
      blend_history.jsonl
      latent_history.jsonl
      validation.json
    checkpoints/
      blend_000010_<candidate-id>.pt
      latent_000020_<candidate-id>.pt
    voices/
      <voice_hash>.pt
    evaluations/
      <candidate_id>.json
    candidates/
      <candidate_id>.json
      <candidate_id>.pt
    run_info.json

  export/
    voice.pt
    voice_meta.json
    voice_best.pt
    voice_final.pt
    voice_best_optimization.pt
    voice_best_meta.json
    voice_final_meta.json
    voice_best_optimization_meta.json

  preview/
    preview_00.wav
    preview_00.json
```

## Artifact metadata

Artifacts use a common metadata schema:

```json
{
  "schema_version": 1,
  "artifact": "target_profile",
  "fingerprint": {
    "target_manifest_sha256": "...",
    "audio_config": "...",
    "speaker_encoder_config": "..."
  },
  "data_path": "target_profile.pt",
  "data_sha256": "...",
  "created_at": "2026-06-27T00:00:00+00:00",
  "...": "artifact-specific summary fields"
}
```

If an artifact exists but its fingerprint no longer matches the current config/input files, the command raises a stale-artifact error or rebuilds when the stage owns the artifact.

## Prepared target manifest

`runs/<id>/data/manifests/target.jsonl` rows look like:

```json
{
  "id": "000000",
  "audio": "audio/000000.wav",
  "source_audio": "speaker/file001.wav",
  "source_start_s": 12.5,
  "source_end_s": 18.2,
  "text_original": "This is the spoken text.",
  "text_normalized": "This is the spoken text.",
  "speaker": "my_speaker",
  "lang_code": "a",
  "duration_s": 5.7,
  "n_chars": 24,
  "peak": 0.98,
  "rms": 0.0643,
  "clip_ratio": 0.0,
  "audio_sha256": "..."
}
```

Rejected rows are written to:

```text
runs/<id>/data/manifests/rejected.jsonl
```

Stable reject reasons include:

- `missing_audio`
- `bad_timestamp`
- `empty_text`
- `text_not_spoken_form`
- `audio_decode_failed`
- `empty_audio`
- `all_silence`
- `clipping`
- `too_short`
- `too_long`
- `hard_end`
- `save_failed`

Hard-end detection runs before fade application, so fades cannot hide abrupt endings. Clips beyond `data.max_target_clips` are selected before WAV writing, so discarded accepted clips do not leave orphan files.

## Target profile

The profile command reads only prepared manifest rows and canonical WAVs. It does not load raw target audio and does not segment anything.

The saved profile contains:

- duration-weighted target speaker embedding,
- per-clip embeddings,
- segment durations,
- segment IDs,
- prepared audio hashes,
- target manifest hash,
- audio config hash,
- speaker encoder config hash,
- speaker model name,
- speech-rate estimate in normalized text chars per second.

Duration loss during optimization uses this speech rate to compute text-length-aware expected durations.

## Prepared corpus

The `assets` command resolves/downloads Kokoro voicepacks, canonicalizes them to `[T, D]`, validates that all selected voices have the same shape, hashes sources and tensors, and writes:

```text
corpus.pt
corpus_manifest.json
```

Optimization never downloads voicepacks. It loads only the prepared corpus artifact.

## Optimization artifact model

Voice tensor persistence is separated from stage/evaluation records:

- `voice_hash` identifies voice tensor content.
- `candidate_id` identifies a stage/evaluation record for a voice.
- The same `voice_hash` can have multiple candidate/evaluation records, for example baseline and validation.

Candidate metadata includes:

- text-plan hash,
- optimization text hash,
- validation text hash,
- objective config hash,
- search config hash,
- corpus hash,
- target profile hash.

Validation explicitly records whether search priors/bounds were included. By default final validation excludes latent prior/bound penalties and evaluates only the validation objective.

## Export and preview

Export selectors:

```bash
kokoro-voiceopt export --config config.yaml --candidate best
kokoro-voiceopt export --config config.yaml --candidate final
kokoro-voiceopt export --config config.yaml --candidate best_optimization
kokoro-voiceopt export --config config.yaml --candidate <voice_hash>
```

Preview synthesizes validation texts from `data/text_plan.json` using an exported voice.
