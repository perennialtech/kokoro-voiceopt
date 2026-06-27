from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import torchaudio

try:
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
except ImportError as exc:  # pragma: no cover
    Wav2Vec2FeatureExtractor = None
    WavLMForXVector = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

from .config import SpeakerEncoderConfig


class SpeakerEncoder(ABC):
    sample_rate: int
    embedding_dim: int | None

    @abstractmethod
    def encode_batch(
        self, audios: list[torch.Tensor], sample_rates: list[int]
    ) -> torch.Tensor:
        """Return speaker embeddings shaped [batch, speaker_dim]."""


class WavLMXVectorSpeakerEncoder(SpeakerEncoder):
    sample_rate = 16000

    def __init__(self, config: SpeakerEncoderConfig, device: str):
        if config.backend != "wavlm_xvector":
            raise ValueError(f"Unsupported speaker encoder backend: {config.backend}")

        if _TRANSFORMERS_IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError(
                "Install transformers to use WavLM speaker encoding"
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.config = config
        self.device = device
        self.batch_size = config.batch_size
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.model_name
        )
        self.model = (
            WavLMForXVector.from_pretrained(config.model_name).to(device).eval()
        )
        self.embedding_dim = getattr(self.model.config, "xvector_output_dim", None)

    def _prepare_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        elif audio.ndim != 1:
            raise ValueError(
                f"Expected audio shape [samples] or [channels, samples], got {tuple(audio.shape)}"
            )

        audio = audio.to(torch.float32).cpu()
        if sample_rate != self.sample_rate:
            audio = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(audio)
        return audio.contiguous()

    @torch.no_grad()
    def encode_batch(
        self, audios: list[torch.Tensor], sample_rates: list[int]
    ) -> torch.Tensor:
        if len(audios) != len(sample_rates):
            raise ValueError("audios and sample_rates must have the same length")
        if not audios:
            raise ValueError("encode_batch requires at least one audio tensor")

        prepared = [self._prepare_audio(a, sr) for a, sr in zip(audios, sample_rates)]
        outputs: list[torch.Tensor] = []

        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            arrays = [a.numpy() for a in batch]

            inputs = self.feature_extractor(
                arrays,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            model_outputs = self.model(**inputs)
            embeddings = model_outputs.embeddings
            embeddings = F.normalize(embeddings, dim=-1)

            outputs.append(embeddings.detach().cpu())

        return torch.cat(outputs, dim=0).contiguous()
