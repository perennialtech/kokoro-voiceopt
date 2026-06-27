from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio

from .search import Candidate
from .synth import KokoroSynthesizer, to_kokoro_voice


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_pt(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def save_voice(path: str | Path, voice: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kokoro_voice = to_kokoro_voice(voice).cpu().to(torch.float32).contiguous()

    if (
        kokoro_voice.ndim != 3
        or kokoro_voice.shape[1] != 1
        or kokoro_voice.shape[2] != 256
    ):
        raise ValueError(
            f"Saved voice must be [T,1,256], got {tuple(kokoro_voice.shape)}"
        )

    torch.save(kokoro_voice, path)


def save_candidate_report(path: str | Path, candidate: Candidate) -> None:
    save_json(path, candidate.to_dict())


def save_candidate_samples(
    output_dir: str | Path,
    synthesizer: KokoroSynthesizer,
    candidates: list[Candidate],
    texts: list[str],
    max_candidates: int | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = candidates if max_candidates is None else candidates[:max_candidates]
    for cand_idx, candidate in enumerate(selected):
        safe_stage = candidate.stage.replace(":", "_").replace("/", "_")
        cand_dir = (
            output_dir / f"{cand_idx:03d}_{safe_stage}_{candidate.candidate_hash[:10]}"
        )
        cand_dir.mkdir(parents=True, exist_ok=True)

        save_json(cand_dir / "candidate.json", candidate.to_dict())

        for text_idx, text in enumerate(texts):
            audio = synthesizer.synthesize(text, candidate.voice)
            torchaudio.save(
                str(cand_dir / f"text_{text_idx:02d}.wav"),
                audio.cpu().to(torch.float32).unsqueeze(0),
                synthesizer.sample_rate,
            )
            save_json(cand_dir / f"text_{text_idx:02d}.json", {"text": text})
