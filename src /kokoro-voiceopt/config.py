from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class AudioConfig:
    speaker_sample_rate: int = 16000
    kokoro_sample_rate: int = 24000
    vad_model: str = "silero"
    min_segment_seconds: float = 1.5
    max_segment_seconds: float = 8.0
    max_target_segments: int = 12
    trim_silence: bool = True
    peak_normalize: bool = True
    dc_remove: bool = True


@dataclass
class SpeakerConfig:
    backend: Literal["wavlm_xvector"] = "wavlm_xvector"
    model_name: str = "microsoft/wavlm-base-plus-sv"
    batch_size: int = 16
    normalize_embeddings: bool = True


@dataclass
class TextConfig:
    target_transcript: str
    max_optimization_texts: int = 3
    max_validation_texts: int = 6
    min_text_chars: int = 20
    max_text_chars: int = 220
    validation_texts_path: Path | None = None


@dataclass
class CorpusConfig:
    voices_dir: Path | None = None
    repo_id: str = "hexgrad/Kokoro-82M"
    lang_code: str = "a"
    include_cross_language_voices: bool = True
    require_consistent_shape: bool = True
    dtype: str = "float32"


@dataclass
class ManifoldConfig:
    center: Literal["median", "mean"] = "median"
    max_latent_dim: int = 32
    variance_coverage: float = 0.98
    z_soft_bound: float = 2.5
    z_hard_bound: float = 4.0
    save_manifold: bool = True


@dataclass
class ObjectiveConfig:
    speaker_loss_weight: float = 1.0
    prior_loss_weight: float = 0.02
    bound_loss_weight: float = 0.10
    silence_loss_weight: float = 0.05
    clipping_loss_weight: float = 0.10
    duration_loss_weight: float = 0.02
    max_silence_ratio: float = 0.25
    max_clip_ratio: float = 0.001
    invalid_audio_loss: float = 100.0


@dataclass
class SearchConfig:
    seed: int = 1234
    top_k_for_blend: int = 8

    blend_iterations: int = 60
    blend_population_pairs: int = 8
    blend_sigma_initial: float = 0.75
    blend_sigma_final: float = 0.15
    blend_learning_rate: float = 0.08

    latent_iterations: int = 180
    latent_population_pairs: int = 12
    latent_sigma_initial: float = 0.35
    latent_sigma_final: float = 0.06
    latent_learning_rate: float = 0.04

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    save_every: int = 10
    validate_every: int = 20


@dataclass
class OutputConfig:
    output_dir: Path
    save_generated_samples: bool = True
    save_candidate_history: bool = True
    save_all_stage_bests: bool = True


@dataclass
class VoiceOptConfig:
    text: TextConfig
    output: OutputConfig
    audio: AudioConfig = field(default_factory=AudioConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    device: str = "cuda"
