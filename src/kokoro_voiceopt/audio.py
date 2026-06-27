from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from .config import AudioConfig


@dataclass
class GeneratedAudioMetrics:
    duration_seconds: float
    silence_ratio: float
    clip_ratio: float
    peak: float


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    return to_mono(waveform).to(torch.float32).contiguous(), int(sample_rate)


def to_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(
            f"Expected audio shape [samples] or [channels, samples], got {tuple(audio.shape)}"
        )
    return audio.mean(dim=0)


def remove_dc_offset(audio: torch.Tensor) -> torch.Tensor:
    if audio.numel() == 0:
        return audio
    return audio - audio.mean()


def peak_normalize(audio: torch.Tensor, peak: float = 0.98) -> torch.Tensor:
    if audio.numel() == 0:
        return audio
    current = audio.abs().max()
    if current <= 1e-8:
        return audio
    return audio * (float(peak) / current)


def resample_audio(audio: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return audio.to(torch.float32).contiguous()
    return (
        torchaudio.functional.resample(
            audio.to(torch.float32).unsqueeze(0), orig_sr, new_sr
        )
        .squeeze(0)
        .contiguous()
    )


def trim_edges(
    audio: torch.Tensor,
    sample_rate: int,
    threshold: float,
    pad_ms: int,
    enabled: bool = True,
) -> torch.Tensor:
    audio = to_mono(audio).to(torch.float32)
    if not enabled or audio.numel() == 0:
        return audio.contiguous()

    mask = audio.abs() > float(threshold)
    if not mask.any():
        return audio.contiguous()

    active = torch.nonzero(mask, as_tuple=False).flatten()
    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, int(active[0]) - pad)
    end = min(int(audio.numel()), int(active[-1]) + pad + 1)
    return audio[start:end].contiguous()


def apply_fades(audio: torch.Tensor, sample_rate: int, fade_ms: int) -> torch.Tensor:
    n = min(int(sample_rate * fade_ms / 1000), audio.numel() // 2)
    if n <= 1:
        return audio.contiguous()

    audio = audio.clone()
    audio[:n] *= torch.linspace(0, 1, n, dtype=audio.dtype)
    audio[-n:] *= torch.linspace(1, 0, n, dtype=audio.dtype)
    return audio.contiguous()


def preprocess_waveform(audio: torch.Tensor, config: AudioConfig) -> torch.Tensor:
    audio = to_mono(audio).to(torch.float32)
    if config.dc_remove:
        audio = remove_dc_offset(audio)
    if config.peak_normalize:
        audio = peak_normalize(audio, peak=config.max_peak)
    return audio.contiguous()


def generated_audio_metrics(
    audio: torch.Tensor, sample_rate: int
) -> GeneratedAudioMetrics:
    audio = to_mono(audio).to(torch.float32)
    if audio.numel() == 0:
        return GeneratedAudioMetrics(0.0, 1.0, 0.0, 0.0)

    peak = float(audio.abs().max())
    threshold = max(peak * 0.01, 1e-4)
    silence_ratio = float((audio.abs() <= threshold).float().mean())
    clip_ratio = float((audio.abs() >= 0.999).float().mean())
    duration = audio.numel() / float(sample_rate)

    return GeneratedAudioMetrics(
        duration_seconds=duration,
        silence_ratio=silence_ratio,
        clip_ratio=clip_ratio,
        peak=peak,
    )


def preprocess_generated_audio(
    audio: torch.Tensor, sample_rate: int, config: AudioConfig
) -> torch.Tensor:
    audio = to_mono(audio).to(torch.float32)
    if config.dc_remove:
        audio = remove_dc_offset(audio)
    if config.peak_normalize:
        audio = peak_normalize(audio, peak=config.max_peak)
    audio = trim_edges(
        audio,
        sample_rate,
        threshold=config.trim_threshold,
        pad_ms=config.trim_pad_ms,
        enabled=config.trim_edges,
    )
    audio = resample_audio(audio, sample_rate, config.target_sample_rate)
    return audio.contiguous()
