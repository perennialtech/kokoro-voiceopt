from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from .search import Candidate
from .serde import write_json
from .synth import KokoroSynthesizer


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
            output_dir / f"{cand_idx:03d}_{safe_stage}_{candidate.candidate_id[:10]}"
        )
        cand_dir.mkdir(parents=True, exist_ok=True)

        write_json(cand_dir / "candidate.json", candidate.to_dict())

        for text_idx, text in enumerate(texts):
            audio = synthesizer.synthesize(text, candidate.voice)
            torchaudio.save(
                str(cand_dir / f"text_{text_idx:02d}.wav"),
                audio.cpu().to(torch.float32).unsqueeze(0),
                synthesizer.sample_rate,
            )
            write_json(cand_dir / f"text_{text_idx:02d}.json", {"text": text})
