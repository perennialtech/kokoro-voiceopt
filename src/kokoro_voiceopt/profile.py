from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .audio import load_wave
from .serde import fingerprint, read_jsonl, sha256_file


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

    def summary(self) -> dict[str, Any]:
        return {
            "source_sample_rate": self.source_sample_rate,
            "num_segments": len(self.segment_ids),
            "segment_ids": self.segment_ids,
            "segment_durations": self.segment_durations,
            "total_speech_seconds": self.total_speech_seconds,
            "total_audio_seconds": self.total_audio_seconds,
            "speech_rate_chars_per_second": self.speech_rate_chars_per_second,
            "embedding_dim": int(self.embedding.numel()),
        }


def profile_fingerprint(ctx, rows: list[dict]) -> dict[str, Any]:
    return {
        "target_manifest_sha256": sha256_file(ctx.paths.data("manifests/target.jsonl")),
        "segment_ids": [row["id"] for row in rows],
        "segment_audio_sha256": [row["audio_sha256"] for row in rows],
        "audio_config_sha256": fingerprint(ctx.audio),
        "speaker_encoder_config_sha256": fingerprint(ctx.speaker_encoder),
        "speaker_model_name": ctx.speaker_encoder.model_name,
        "target_id": ctx.target.id,
        "lang_code": ctx.target.lang_code,
    }


def profile_artifact(ctx, rows: list[dict]):
    return ctx.artifacts.spec(
        "target_profile",
        data=ctx.paths.profile("target_profile.pt"),
        meta=ctx.paths.profile("target_profile.json"),
        fingerprint=profile_fingerprint(ctx, rows),
    )


def _profile_from_payload(data: dict, metadata: dict) -> TargetSpeakerProfile:
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
        metadata=metadata,
    )


def load_target_speaker_profile(ctx) -> TargetSpeakerProfile:
    ctx.require_profile()
    rows = read_jsonl(ctx.paths.data("manifests/target.jsonl"))
    artifact = profile_artifact(ctx, rows)
    metadata = artifact.require_current()
    return _profile_from_payload(artifact.load_pt(), metadata)


def build_target_speaker_profile(
    ctx,
    speaker_encoder,
    force: bool = False,
) -> TargetSpeakerProfile:
    ctx.require_manifests()
    ctx.write_resolved_config()

    rows = read_jsonl(ctx.paths.data("manifests/target.jsonl"))
    if not rows:
        raise ValueError("Target manifest is empty")

    artifact = profile_artifact(ctx, rows)

    if artifact.data_path.exists() and artifact.meta_path.exists():
        if artifact.is_current() and not force:
            print(f"Target profile cache is current: {artifact.data_path}")
            metadata = artifact.metadata()
            return _profile_from_payload(artifact.load_pt(), metadata)
        if not force:
            raise SystemExit(
                "Target profile exists but metadata does not match; use --force to rebuild"
            )

    audios = []
    rates = []
    durations = []

    for row in rows:
        path = ctx.paths.data(row["audio"])
        wave = load_wave(path)
        audios.append(wave.samples)
        rates.append(wave.sample_rate)
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

    payload = {
        "embedding": profile_embedding.cpu().contiguous(),
        "segment_embeddings": embeddings,
        "segment_durations": durations,
        "segment_ids": [row["id"] for row in rows],
        "segment_audio_sha256": [row["audio_sha256"] for row in rows],
        "source_sample_rate": ctx.audio.target_sample_rate,
        "total_speech_seconds": total_speech,
        "total_audio_seconds": total_speech,
        "speech_rate_chars_per_second": float(speech_rate),
    }

    temp_profile = _profile_from_payload(payload, {})
    metadata = artifact.save_pt(payload, extra=temp_profile.summary())
    profile = _profile_from_payload(payload, metadata)

    print(f"Wrote target profile: {artifact.data_path}")
    return profile
