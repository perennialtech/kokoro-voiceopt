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

A run is self-describing on disk. The expensive/stable artifacts are the prepared target data, target profile, and prepared voice corpus. Optimization uses a simplified single-writer layout.

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

  profile/
    target_profile.pt
    target_profile.json

  corpus/
    corpus.pt
    corpus_manifest.json

  optimize/
    run.json
    history.jsonl
    candidates.jsonl
    voices/
      <voice_hash>.pt

  export/
    voice.pt
    voice_meta.json

  preview/
    preview_00.wav
    preview_00.json
```

`optimize/voices/<voice_hash>.pt` stores each retained unique optimized voice tensor once. `optimize/candidates.jsonl` stores retained baseline, blend, latent, checkpoint, final, and validation candidate records. `optimize/history.jsonl` stores per-iteration convergence summaries. `optimize/run.json` stores run fingerprints, exact optimization and validation texts, selected candidate hashes, selected voice hashes, and summary metrics.

## Artifact metadata

Only the prepared corpus and target speaker profile use stale-artifact metadata. They use a common metadata schema:

```json
{
  "schema_version": 1,
  "artifact": "target_profile",
  "fingerprint": {
    "target_manifest_sha256": "...",
    "audio_config_sha256": "...",
    "speaker_encoder_config_sha256": "..."
  },
  "data_path": "target_profile.pt",
  "data_sha256": "...",
  "created_at": "2026-06-27T00:00:00+00:00",
  "...": "artifact-specific summary fields"
}
```

If one of these artifacts exists but its fingerprint no longer matches the current config/input files, the command raises a stale-artifact error or rebuilds when the stage owns the artifact.

`data/text_plan.json` is a cheap deterministic convenience file and is regenerated from the current prepared manifest/config at command time.

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

Optimization writes only:

```text
optimize/
  run.json
  history.jsonl
  candidates.jsonl
  voices/
    <voice_hash>.pt
```

Candidate records include:

- role, such as `baseline_best`, `blend_best`, `latent_checkpoint`, `latent_final`, or `validation_best`;
- stage;
- iteration;
- voice hash;
- inline blend/latent params when present;
- evaluation set, either `optimization` or `validation`;
- evaluation terms;
- text-plan hash;
- optimization text hash;
- validation text hash;
- objective config hash;
- search config hash;
- corpus hash;
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

Every export selector writes:

```text
export/
  voice.pt
  voice_meta.json
```

`voice_meta.json` includes the exact optimization and validation texts used for the selected optimization run.

Preview reads validation texts from `export/voice_meta.json` first. If no exported metadata is available, it rebuilds the deterministic text plan from the current prepared manifest/config.
