from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio

from .config import AudioConfig, DataConfig


class AudioError(Exception):
    reason = "audio_error"


class AudioDecodeError(AudioError):
    reason = "audio_decode_failed"


class TimestampError(AudioError):
    reason = "bad_timestamp"


class EmptyAudioError(AudioError):
    reason = "empty_audio"


class SilenceError(AudioError):
    reason = "all_silence"


class ClippingError(AudioError):
    reason = "clipping"

    def __init__(self, metrics: dict[str, Any]):
        super().__init__("clipping")
        self.metrics = metrics


class DurationRejectionError(AudioError):
    def __init__(self, reason: str, metrics: dict[str, Any]):
        super().__init__(reason)
        self.reason = reason
        self.metrics = metrics


class HardEndError(AudioError):
    reason = "hard_end"

    def __init__(self, metrics: dict[str, Any]):
        super().__init__("hard_end")
        self.metrics = metrics


class AudioSaveError(AudioError):
    reason = "save_failed"


@dataclass(frozen=True)
class Wave:
    samples: torch.Tensor
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return self.samples.numel() / float(self.sample_rate)


@dataclass(frozen=True)
class AudioMetrics:
    duration_s: float
    peak: float
    rms: float
    clip_ratio: float
    silence_ratio: float
    hard_end: bool

    def to_manifest(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 4),
            "peak": round(self.peak, 6),
            "rms": round(self.rms, 6),
            "clip_ratio": round(self.clip_ratio, 8),
            "silence_ratio": round(self.silence_ratio, 8),
            "hard_end": self.hard_end,
        }


@dataclass(frozen=True)
class PreparedClip:
    wave: Wave
    source_start_s: float | None
    source_end_s: float | None
    metrics: AudioMetrics


@dataclass(frozen=True)
class GeneratedAudioMetrics:
    duration_seconds: float
    silence_ratio: float
    clip_ratio: float
    peak: float


def to_mono(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim == 1:
        return samples
    if samples.ndim != 2:
        raise ValueError(
            f"Expected audio shape [samples] or [channels, samples], got {tuple(samples.shape)}"
        )
    return samples.mean(dim=0)


def load_wave(path: str | Path) -> Wave:
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        waveform = torch.from_numpy(audio)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        elif waveform.ndim != 1:
            raise AudioDecodeError(f"Unsupported audio shape: {tuple(waveform.shape)}")
        return Wave(waveform.to(torch.float32).contiguous(), int(sr))
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError(str(exc)) from exc


def slice_wave(wave: Wave, start_s, end_s) -> tuple[Wave, float | None, float | None]:
    if start_s is None and end_s is None:
        return wave, None, None
    if start_s is None or end_s is None:
        raise TimestampError("start_s and end_s must be provided together")

    try:
        start = float(start_s)
        end = float(end_s)
    except Exception as exc:
        raise TimestampError(
            f"timestamps must be numeric: start_s={start_s}, end_s={end_s}"
        ) from exc

    if start < 0 or end <= start:
        raise TimestampError(f"bad timestamp range: start_s={start}, end_s={end}")

    start_sample = int(round(start * wave.sample_rate))
    end_sample = int(round(end * wave.sample_rate))
    if (
        start_sample < 0
        or end_sample > wave.samples.numel()
        or end_sample <= start_sample
    ):
        raise TimestampError(
            f"timestamp range outside audio: start_s={start}, end_s={end}"
        )

    return (
        Wave(wave.samples[start_sample:end_sample].contiguous(), wave.sample_rate),
        start,
        end,
    )


def resample_wave(wave: Wave, new_sample_rate: int) -> Wave:
    if wave.sample_rate == new_sample_rate:
        return Wave(wave.samples.to(torch.float32).contiguous(), wave.sample_rate)

    samples = (
        torchaudio.functional.resample(
            wave.samples.to(torch.float32).unsqueeze(0),
            wave.sample_rate,
            new_sample_rate,
        )
        .squeeze(0)
        .contiguous()
    )
    return Wave(samples, int(new_sample_rate))


def remove_dc_offset(samples: torch.Tensor) -> torch.Tensor:
    if samples.numel() == 0:
        return samples
    return (samples - samples.mean()).contiguous()


def peak_normalize(samples: torch.Tensor, peak: float = 0.98) -> torch.Tensor:
    if samples.numel() == 0:
        return samples
    current = samples.abs().max()
    if current <= 1e-8:
        return samples
    return (samples * (float(peak) / current)).contiguous()


def trim_wave(
    wave: Wave,
    *,
    threshold: float,
    pad_ms: int,
    enabled: bool = True,
    raise_on_silence: bool = False,
) -> Wave:
    samples = to_mono(wave.samples).to(torch.float32)
    if not enabled or samples.numel() == 0:
        return Wave(samples.contiguous(), wave.sample_rate)

    mask = samples.abs() > float(threshold)
    if not mask.any():
        if raise_on_silence:
            raise SilenceError("all_silence")
        return Wave(samples.contiguous(), wave.sample_rate)

    active = torch.nonzero(mask, as_tuple=False).flatten()
    pad = int(wave.sample_rate * pad_ms / 1000)
    start = max(0, int(active[0]) - pad)
    end = min(int(samples.numel()), int(active[-1]) + pad + 1)
    return Wave(samples[start:end].contiguous(), wave.sample_rate)


def apply_fades(samples: torch.Tensor, sample_rate: int, fade_ms: int) -> torch.Tensor:
    n = min(int(sample_rate * fade_ms / 1000), samples.numel() // 2)
    if n <= 1:
        return samples.contiguous()

    samples = samples.clone()
    samples[:n] *= torch.linspace(0, 1, n, dtype=samples.dtype)
    samples[-n:] *= torch.linspace(1, 0, n, dtype=samples.dtype)
    return samples.contiguous()


def measure_wave(wave: Wave, hard_end_threshold: float = 0.2) -> AudioMetrics:
    samples = to_mono(wave.samples).to(torch.float32)
    if samples.numel() == 0:
        return AudioMetrics(
            duration_s=0.0,
            peak=0.0,
            rms=0.0,
            clip_ratio=1.0,
            silence_ratio=1.0,
            hard_end=True,
        )

    peak = float(samples.abs().max().item())
    threshold = max(peak * 0.01, 1e-4)
    edge = max(1, min(int(0.02 * wave.sample_rate), samples.numel() // 10))

    return AudioMetrics(
        duration_s=samples.numel() / float(wave.sample_rate),
        peak=peak,
        rms=float(torch.sqrt(torch.mean(samples.square())).item()),
        clip_ratio=float((samples.abs() >= 0.999).float().mean().item()),
        silence_ratio=float((samples.abs() <= threshold).float().mean().item()),
        hard_end=bool(samples[-edge:].abs().max().item() > hard_end_threshold),
    )


def save_wave(path: str | Path, wave: Wave) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(
            path,
            wave.samples.detach().cpu().to(torch.float32).numpy(),
            wave.sample_rate,
            subtype="PCM_16",
        )
    except Exception as exc:
        raise AudioSaveError(str(exc)) from exc


def prepare_target_clip(
    source_path: str | Path,
    start_s,
    end_s,
    audio_config: AudioConfig,
    data_config: DataConfig,
) -> PreparedClip:
    wave = load_wave(source_path)
    wave, source_start_s, source_end_s = slice_wave(wave, start_s, end_s)

    if wave.samples.numel() == 0:
        raise EmptyAudioError("empty_audio")

    wave = resample_wave(wave, audio_config.target_sample_rate)

    samples = wave.samples.to(torch.float32).contiguous()
    if audio_config.dc_remove:
        samples = remove_dc_offset(samples)
    wave = Wave(samples, wave.sample_rate)

    if audio_config.trim_edges:
        wave = trim_wave(
            wave,
            threshold=audio_config.trim_threshold,
            pad_ms=audio_config.trim_pad_ms,
            enabled=True,
            raise_on_silence=True,
        )

    pre_fade_metrics = measure_wave(wave, data_config.hard_end_threshold)
    pre_fade_dict = pre_fade_metrics.to_manifest()

    if pre_fade_metrics.clip_ratio > data_config.max_clip_ratio:
        raise ClippingError(pre_fade_dict)

    if pre_fade_metrics.hard_end:
        raise HardEndError(pre_fade_dict)

    samples = wave.samples
    peak = samples.abs().max().item() if samples.numel() else 0.0
    if peak > audio_config.max_peak:
        samples = samples / peak * audio_config.max_peak
    elif audio_config.peak_normalize and peak > 1e-8:
        samples = samples / peak * audio_config.max_peak

    samples = apply_fades(samples, wave.sample_rate, audio_config.fade_ms)
    wave = Wave(samples.contiguous(), wave.sample_rate)

    metrics = measure_wave(wave, data_config.hard_end_threshold)
    metrics_dict = metrics.to_manifest()

    if metrics.duration_s < data_config.min_duration_s:
        raise DurationRejectionError("too_short", metrics_dict)
    if metrics.duration_s > data_config.max_duration_s:
        raise DurationRejectionError("too_long", metrics_dict)

    return PreparedClip(
        wave=wave,
        source_start_s=source_start_s,
        source_end_s=source_end_s,
        metrics=metrics,
    )


def generated_audio_metrics(
    audio: torch.Tensor, sample_rate: int
) -> GeneratedAudioMetrics:
    samples = to_mono(audio).to(torch.float32)
    if samples.numel() == 0:
        return GeneratedAudioMetrics(0.0, 1.0, 0.0, 0.0)

    peak = float(samples.abs().max())
    threshold = max(peak * 0.01, 1e-4)
    silence_ratio = float((samples.abs() <= threshold).float().mean())
    clip_ratio = float((samples.abs() >= 0.999).float().mean())
    duration = samples.numel() / float(sample_rate)

    return GeneratedAudioMetrics(
        duration_seconds=duration,
        silence_ratio=silence_ratio,
        clip_ratio=clip_ratio,
        peak=peak,
    )


def preprocess_generated_audio(
    audio: torch.Tensor,
    sample_rate: int,
    config: AudioConfig,
) -> torch.Tensor:
    wave = Wave(to_mono(audio).to(torch.float32).contiguous(), sample_rate)

    samples = wave.samples
    if config.dc_remove:
        samples = remove_dc_offset(samples)
    if config.peak_normalize:
        samples = peak_normalize(samples, peak=config.max_peak)

    wave = Wave(samples, wave.sample_rate)
    wave = trim_wave(
        wave,
        threshold=config.trim_threshold,
        pad_ms=config.trim_pad_ms,
        enabled=config.trim_edges,
        raise_on_silence=False,
    )
    wave = resample_wave(wave, config.target_sample_rate)
    return wave.samples.contiguous()
