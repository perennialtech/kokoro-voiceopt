from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torchaudio

from .config import AudioConfig


@dataclass
class AudioSegment:
    waveform: torch.Tensor
    sample_rate: int
    start_sample: int
    end_sample: int

    @property
    def duration_seconds(self) -> float:
        return (self.end_sample - self.start_sample) / float(self.sample_rate)


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


def peak_normalize(audio: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    if audio.numel() == 0:
        return audio
    current = audio.abs().max()
    if current <= 1e-8:
        return audio
    return audio * (peak / current)


def resample_audio(audio: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return audio.to(torch.float32).contiguous()
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)
    return resampler(audio.cpu()).to(torch.float32).contiguous()


def preprocess_waveform(audio: torch.Tensor, config: AudioConfig) -> torch.Tensor:
    audio = to_mono(audio).to(torch.float32)
    if config.dc_remove:
        audio = remove_dc_offset(audio)
    if config.peak_normalize:
        audio = peak_normalize(audio)
    return audio.contiguous()


def load_target_audio(
    path: str | Path, config: AudioConfig
) -> tuple[torch.Tensor, int]:
    audio, sample_rate = load_audio(path)
    audio = preprocess_waveform(audio, config)
    audio = resample_audio(audio, sample_rate, config.speaker_sample_rate)
    return audio.contiguous(), config.speaker_sample_rate


def _load_silero_get_timestamps() -> Callable | None:
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad

        model = load_silero_vad()

        def detect(audio: torch.Tensor, sample_rate: int):
            return get_speech_timestamps(
                audio, model, sampling_rate=sample_rate, return_seconds=False
            )

        return detect
    except Exception:
        pass

    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            verbose=False,
        )
        get_speech_timestamps = utils[0]

        def detect(audio: torch.Tensor, sample_rate: int):
            return get_speech_timestamps(
                audio, model, sampling_rate=sample_rate, return_seconds=False
            )

        return detect
    except Exception as exc:
        warnings.warn(f"Silero VAD unavailable; falling back to energy VAD: {exc}")
        return None


def _energy_vad(audio: torch.Tensor, sample_rate: int) -> list[dict[str, int]]:
    if audio.numel() == 0:
        return []

    frame = max(1, int(0.03 * sample_rate))
    hop = max(1, int(0.01 * sample_rate))
    padded = torch.nn.functional.pad(audio.abs(), (0, frame))
    frames = padded.unfold(0, frame, hop)
    energy = frames.mean(dim=-1)

    threshold = max(float(energy.mean() * 0.5), 1e-4)
    voiced = energy > threshold

    regions: list[dict[str, int]] = []
    start = None
    for idx, is_voiced in enumerate(voiced.tolist()):
        if is_voiced and start is None:
            start = idx * hop
        elif not is_voiced and start is not None:
            end = min(idx * hop + frame, audio.numel())
            regions.append({"start": start, "end": end})
            start = None

    if start is not None:
        regions.append({"start": start, "end": audio.numel()})

    return regions


def detect_speech_regions(
    audio: torch.Tensor, sample_rate: int, config: AudioConfig
) -> list[dict[str, int]]:
    if config.vad_model.lower() == "silero":
        detector = _load_silero_get_timestamps()
        if detector is not None:
            regions = detector(audio.cpu(), sample_rate)
            return [{"start": int(r["start"]), "end": int(r["end"])} for r in regions]

    return _energy_vad(audio.cpu(), sample_rate)


def segment_target_speech(
    audio: torch.Tensor, sample_rate: int, config: AudioConfig
) -> list[AudioSegment]:
    regions = detect_speech_regions(audio, sample_rate, config)
    if not regions:
        regions = [{"start": 0, "end": int(audio.numel())}]

    min_samples = int(config.min_segment_seconds * sample_rate)
    max_samples = int(config.max_segment_seconds * sample_rate)

    segments: list[AudioSegment] = []
    for region in regions:
        start = max(0, int(region["start"]))
        end = min(int(audio.numel()), int(region["end"]))
        if end <= start:
            continue

        cursor = start
        while cursor < end:
            chunk_end = min(cursor + max_samples, end)
            if chunk_end - cursor >= min_samples:
                segments.append(
                    AudioSegment(
                        waveform=audio[cursor:chunk_end].contiguous(),
                        sample_rate=sample_rate,
                        start_sample=cursor,
                        end_sample=chunk_end,
                    )
                )
            cursor = chunk_end

    if not segments and audio.numel() > 0:
        segments = [
            AudioSegment(
                waveform=audio.contiguous(),
                sample_rate=sample_rate,
                start_sample=0,
                end_sample=int(audio.numel()),
            )
        ]

    segments.sort(key=lambda s: s.duration_seconds, reverse=True)
    segments = segments[: config.max_target_segments]
    segments.sort(key=lambda s: s.start_sample)
    return segments


def trim_generated_audio(
    audio: torch.Tensor, sample_rate: int, enabled: bool = True
) -> torch.Tensor:
    audio = to_mono(audio).to(torch.float32)
    if not enabled or audio.numel() == 0:
        return audio.contiguous()

    peak = float(audio.abs().max())
    if peak <= 1e-8:
        return audio.contiguous()

    threshold = max(peak * 0.01, 1e-4)
    active = torch.nonzero(audio.abs() > threshold).flatten()
    if active.numel() == 0:
        return audio.contiguous()

    margin = int(0.05 * sample_rate)
    start = max(0, int(active[0]) - margin)
    end = min(int(audio.numel()), int(active[-1]) + margin + 1)
    return audio[start:end].contiguous()


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
    audio = preprocess_waveform(audio, config)
    audio = trim_generated_audio(audio, sample_rate, config.trim_silence)
    audio = resample_audio(audio, sample_rate, config.speaker_sample_rate)
    return audio.contiguous()
