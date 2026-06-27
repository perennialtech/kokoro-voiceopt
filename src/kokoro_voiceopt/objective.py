from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .audio import generated_audio_metrics, preprocess_generated_audio
from .config import AudioConfig, ObjectiveConfig
from .data import normalize_text
from .manifold import VoiceManifold
from .profile import TargetSpeakerProfile
from .speaker import SpeakerEncoder
from .synth import KokoroSynthesizer


@dataclass
class LatentInfo:
    z: torch.Tensor
    manifold: VoiceManifold


@dataclass
class CandidateEval:
    total_loss: float
    speaker_loss: float
    prior_loss: float
    bound_loss: float
    audio_quality_loss: float
    mean_similarity: float
    per_text: list[dict]

    def to_dict(self) -> dict:
        return {
            "total_loss": self.total_loss,
            "speaker_loss": self.speaker_loss,
            "prior_loss": self.prior_loss,
            "bound_loss": self.bound_loss,
            "audio_quality_loss": self.audio_quality_loss,
            "mean_similarity": self.mean_similarity,
            "per_text": self.per_text,
        }


class VoiceObjective:
    def __init__(
        self,
        synthesizer: KokoroSynthesizer,
        speaker_encoder: SpeakerEncoder,
        target_profile: TargetSpeakerProfile,
        audio_config: AudioConfig,
        objective_config: ObjectiveConfig,
    ):
        self.synthesizer = synthesizer
        self.speaker_encoder = speaker_encoder
        self.target_profile = target_profile
        self.audio_config = audio_config
        self.config = objective_config

    def expected_duration(self, normalized_text: str) -> float:
        rate = max(float(self.target_profile.speech_rate_chars_per_second), 1e-6)
        return max(len(normalized_text) / rate, 0.25)

    def evaluate_voices(
        self,
        voices: list[torch.Tensor],
        texts: list[str],
        latent_info: list[LatentInfo | None] | None = None,
    ) -> list[CandidateEval]:
        if not voices:
            return []
        if not texts:
            raise ValueError("evaluate_voices requires at least one text")

        if latent_info is None:
            latent_info = [None] * len(voices)
        if len(latent_info) != len(voices):
            raise ValueError("latent_info must be None or have one item per voice")

        normalized_texts = [normalize_text(text) for text in texts]

        generated: list[dict] = []
        encode_audios: list[torch.Tensor] = []
        encode_rates: list[int] = []

        for voice_idx, voice in enumerate(voices):
            for text_idx, text in enumerate(texts):
                normalized = normalized_texts[text_idx]
                item = {
                    "voice_idx": voice_idx,
                    "text_idx": text_idx,
                    "text": text,
                    "normalized_text": normalized,
                    "expected_duration_seconds": self.expected_duration(normalized),
                    "valid": False,
                    "embedding_index": None,
                    "error": None,
                    "metrics": None,
                }

                try:
                    audio = self.synthesizer.synthesize(text, voice)
                    metrics = generated_audio_metrics(
                        audio, self.synthesizer.sample_rate
                    )
                    processed = preprocess_generated_audio(
                        audio, self.synthesizer.sample_rate, self.audio_config
                    )

                    if processed.numel() < int(
                        0.05 * self.audio_config.target_sample_rate
                    ):
                        raise RuntimeError(
                            "Generated audio too short after preprocessing"
                        )

                    item["valid"] = True
                    item["metrics"] = metrics
                    item["embedding_index"] = len(encode_audios)
                    encode_audios.append(processed)
                    encode_rates.append(self.audio_config.target_sample_rate)
                except Exception as exc:
                    item["error"] = str(exc)

                generated.append(item)

        embeddings = None
        if encode_audios:
            embeddings = self.speaker_encoder.encode_batch(encode_audios, encode_rates)
            embeddings = F.normalize(embeddings, dim=-1)

        target = self.target_profile.embedding.cpu()

        results: list[CandidateEval] = []
        for voice_idx in range(len(voices)):
            per_text: list[dict] = []
            speaker_losses: list[float] = []
            similarities: list[float] = []
            silence_losses: list[float] = []
            clipping_losses: list[float] = []
            duration_losses: list[float] = []
            invalid_losses: list[float] = []

            for item in [g for g in generated if g["voice_idx"] == voice_idx]:
                if not item["valid"]:
                    per_text.append(
                        {
                            "text": item["text"],
                            "normalized_text": item["normalized_text"],
                            "expected_duration_seconds": item[
                                "expected_duration_seconds"
                            ],
                            "actual_duration_seconds": None,
                            "valid": False,
                            "error": item["error"],
                            "peak": None,
                            "silence_ratio": None,
                            "clip_ratio": None,
                            "similarity": -1.0,
                            "speaker_loss": self.config.invalid_audio_loss,
                            "silence_loss": 0.0,
                            "clipping_loss": 0.0,
                            "duration_loss": 0.0,
                            "quality_loss": self.config.invalid_audio_loss,
                        }
                    )
                    speaker_losses.append(self.config.invalid_audio_loss)
                    similarities.append(-1.0)
                    invalid_losses.append(self.config.invalid_audio_loss)
                    continue

                embedding = embeddings[item["embedding_index"]]
                similarity = float(torch.dot(embedding.cpu(), target))
                speaker_loss = 1.0 - similarity
                metrics = item["metrics"]

                silence_loss = (
                    max(0.0, metrics.silence_ratio - self.config.max_silence_ratio) ** 2
                )
                clipping_loss = (
                    max(0.0, metrics.clip_ratio - self.config.max_clip_ratio) ** 2
                )

                expected_duration = float(item["expected_duration_seconds"])
                duration_ratio = max(metrics.duration_seconds, 1e-6) / expected_duration
                duration_margin = math.log(1.6)
                duration_loss = (
                    max(0.0, abs(math.log(duration_ratio)) - duration_margin) ** 2
                )

                quality_loss = (
                    self.config.silence_loss_weight * silence_loss
                    + self.config.clipping_loss_weight * clipping_loss
                    + self.config.duration_loss_weight * duration_loss
                )

                per_text.append(
                    {
                        "text": item["text"],
                        "normalized_text": item["normalized_text"],
                        "expected_duration_seconds": expected_duration,
                        "actual_duration_seconds": metrics.duration_seconds,
                        "valid": True,
                        "error": None,
                        "peak": metrics.peak,
                        "silence_ratio": metrics.silence_ratio,
                        "clip_ratio": metrics.clip_ratio,
                        "similarity": similarity,
                        "speaker_loss": speaker_loss,
                        "silence_loss": silence_loss,
                        "clipping_loss": clipping_loss,
                        "duration_loss": duration_loss,
                        "quality_loss": quality_loss,
                    }
                )

                speaker_losses.append(speaker_loss)
                similarities.append(similarity)
                silence_losses.append(silence_loss)
                clipping_losses.append(clipping_loss)
                duration_losses.append(duration_loss)
                invalid_losses.append(0.0)

            speaker_loss = float(sum(speaker_losses) / max(len(speaker_losses), 1))
            mean_similarity = float(sum(similarities) / max(len(similarities), 1))
            silence_loss = (
                float(sum(silence_losses) / max(len(silence_losses), 1))
                if silence_losses
                else 0.0
            )
            clipping_loss = (
                float(sum(clipping_losses) / max(len(clipping_losses), 1))
                if clipping_losses
                else 0.0
            )
            duration_loss = (
                float(sum(duration_losses) / max(len(duration_losses), 1))
                if duration_losses
                else 0.0
            )
            invalid_loss = (
                float(sum(invalid_losses) / max(len(invalid_losses), 1))
                if invalid_losses
                else 0.0
            )

            prior_loss = 0.0
            bound_loss = 0.0
            info = latent_info[voice_idx]
            if info is not None:
                z = info.z.detach().cpu()
                prior_loss = float(info.manifold.prior_loss(z))
                bound_loss = float(info.manifold.soft_bound_loss(z))

            audio_quality_loss = (
                self.config.silence_loss_weight * silence_loss
                + self.config.clipping_loss_weight * clipping_loss
                + self.config.duration_loss_weight * duration_loss
                + invalid_loss
            )

            total_loss = (
                self.config.speaker_loss_weight * speaker_loss
                + self.config.prior_loss_weight * prior_loss
                + self.config.bound_loss_weight * bound_loss
                + audio_quality_loss
            )

            results.append(
                CandidateEval(
                    total_loss=float(total_loss),
                    speaker_loss=float(speaker_loss),
                    prior_loss=float(prior_loss),
                    bound_loss=float(bound_loss),
                    audio_quality_loss=float(audio_quality_loss),
                    mean_similarity=float(mean_similarity),
                    per_text=per_text,
                )
            )

        return results
