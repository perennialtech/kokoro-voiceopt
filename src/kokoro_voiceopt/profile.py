from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import hash_file, read_jsonl, stable_hash_json


@dataclass
class TargetSpeakerProfile:
    embedding: torch.Tensor
    segment_embeddings: torch.Tensor
    segment_durations: list[float]
    segment_ids: list[str]
    segment_audio_sha256: list[str]
    source_sample_rate: int
    total_speech_seconds: float
    total_audio_seconds: float
    speech_rate_chars_per_second: float
    metadata: dict

    def to_json(self) -> dict:
        return {
            "schema_version": self.metadata.get("schema_version"),
            "source_sample_rate": self.source_sample_rate,
            "num_segments": len(self.segment_ids),
            "segment_ids": self.segment_ids,
            "segment_durations": self.segment_durations,
            "total_speech_seconds": self.total_speech_seconds,
            "total_audio_seconds": self.total_audio_seconds,
            "speech_rate_chars_per_second": self.speech_rate_chars_per_second,
            "embedding_dim": int(self.embedding.numel()),
            "metadata": self.metadata,
        }


def profile_metadata(run, speaker_encoder, rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "target_manifest_sha256": hash_file(run.paths.target_manifest),
        "segment_ids": [row["id"] for row in rows],
        "segment_audio_sha256": [row["audio_sha256"] for row in rows],
        "audio_config_sha256": stable_hash_json(run.audio),
        "speaker_encoder_config_sha256": stable_hash_json(run.speaker_encoder),
        "speaker_model_name": run.speaker_encoder.model_name,
        "embedding_dim": speaker_encoder.embedding_dim,
        "target_id": run.target.id,
        "lang_code": run.target.lang_code,
    }


def _load_profile_pt(path: Path) -> TargetSpeakerProfile:
    data = torch.load(path, map_location="cpu")
    return TargetSpeakerProfile(
        embedding=data["embedding"].to(torch.float32).contiguous(),
        segment_embeddings=data["segment_embeddings"].to(torch.float32).contiguous(),
        segment_durations=list(data["segment_durations"]),
        segment_ids=list(data["segment_ids"]),
        segment_audio_sha256=list(data["segment_audio_sha256"]),
        source_sample_rate=int(data["source_sample_rate"]),
        total_speech_seconds=float(data["total_speech_seconds"]),
        total_audio_seconds=float(data["total_audio_seconds"]),
        speech_rate_chars_per_second=float(data["speech_rate_chars_per_second"]),
        metadata=dict(data["metadata"]),
    )


def load_target_speaker_profile(run) -> TargetSpeakerProfile:
    run.require_profile()
    profile = _load_profile_pt(run.paths.target_profile_pt)

    rows = read_jsonl(run.paths.target_manifest)
    expected = {
        "schema_version": 1,
        "target_manifest_sha256": hash_file(run.paths.target_manifest),
        "segment_ids": [row["id"] for row in rows],
        "segment_audio_sha256": [row["audio_sha256"] for row in rows],
        "audio_config_sha256": stable_hash_json(run.audio),
        "speaker_encoder_config_sha256": stable_hash_json(run.speaker_encoder),
        "speaker_model_name": run.speaker_encoder.model_name,
        "target_id": run.target.id,
        "lang_code": run.target.lang_code,
    }

    for key, value in expected.items():
        if profile.metadata.get(key) != value:
            raise ValueError(
                f"Target profile cache metadata mismatch for {key}; rebuild with profile --force"
            )

    return profile


def build_target_speaker_profile(
    run, speaker_encoder, force: bool = False
) -> TargetSpeakerProfile:
    run.require_manifests()
    run.write_resolved_config()

    rows = read_jsonl(run.paths.target_manifest)
    if not rows:
        raise ValueError("Target manifest is empty")

    expected_metadata = profile_metadata(run, speaker_encoder, rows)

    if run.paths.target_profile_pt.exists() and run.paths.target_profile_json.exists():
        existing = _load_profile_pt(run.paths.target_profile_pt)
        if existing.metadata == expected_metadata and not force:
            print(f"Target profile cache is current: {run.paths.target_profile_pt}")
            return existing
        if not force:
            raise SystemExit(
                "Target profile exists but metadata does not match; use --force to rebuild"
            )

    from .data import load_audio

    audios = []
    rates = []
    durations = []

    for row in rows:
        path = run.paths.data_dir / row["audio"]
        audio, sr = load_audio(path)
        audios.append(audio)
        rates.append(sr)
        durations.append(float(row["duration_s"]))

    embeddings = speaker_encoder.encode_batch(audios, rates)
    embeddings = F.normalize(embeddings, dim=-1).cpu().to(torch.float32).contiguous()

    weights = torch.tensor(durations, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-8)
    profile_embedding = F.normalize(
        (embeddings * weights.unsqueeze(1)).sum(dim=0), dim=0
    )

    total_speech = float(sum(durations))
    total_chars = sum(len(str(row["text_normalized"])) for row in rows)
    speech_rate = total_chars / max(total_speech, 1e-6)

    metadata = {
        **expected_metadata,
        "embedding_dim": int(profile_embedding.numel()),
    }

    profile = TargetSpeakerProfile(
        embedding=profile_embedding.cpu().contiguous(),
        segment_embeddings=embeddings,
        segment_durations=durations,
        segment_ids=[row["id"] for row in rows],
        segment_audio_sha256=[row["audio_sha256"] for row in rows],
        source_sample_rate=run.audio.target_sample_rate,
        total_speech_seconds=total_speech,
        total_audio_seconds=total_speech,
        speech_rate_chars_per_second=float(speech_rate),
        metadata=metadata,
    )

    run.paths.profile_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embedding": profile.embedding,
            "segment_embeddings": profile.segment_embeddings,
            "segment_durations": profile.segment_durations,
            "segment_ids": profile.segment_ids,
            "segment_audio_sha256": profile.segment_audio_sha256,
            "source_sample_rate": profile.source_sample_rate,
            "total_speech_seconds": profile.total_speech_seconds,
            "total_audio_seconds": profile.total_audio_seconds,
            "speech_rate_chars_per_second": profile.speech_rate_chars_per_second,
            "metadata": profile.metadata,
        },
        run.paths.target_profile_pt,
    )

    with open(run.paths.target_profile_json, "w", encoding="utf-8") as file:
        json.dump(profile.to_json(), file, indent=2, ensure_ascii=False)

    print(f"Wrote target profile: {run.paths.target_profile_pt}")
    return profile
