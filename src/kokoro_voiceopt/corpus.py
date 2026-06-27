from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .assets import hash_tensor, resolve_voice_names
from .config import hash_file


@dataclass
class VoiceRecord:
    name: str
    tensor: torch.Tensor
    source_path: Path | None
    language_prefix: str


@dataclass
class VoiceCorpus:
    records: list[VoiceRecord]
    T: int
    D: int
    manifest: dict
    manifest_sha256: str
    corpus_sha256: str

    @classmethod
    def load(cls, run) -> "VoiceCorpus":
        run.require_corpus()

        with open(run.corpus_manifest, encoding="utf-8") as file:
            manifest = json.load(file)

        data = torch.load(run.corpus_pt, map_location="cpu")
        voices = data["voices"].cpu().contiguous()
        names = list(data["names"])
        prefixes = list(data["language_prefixes"])

        expected_names = resolve_voice_names(run.assets, run.target.lang_code)
        errors = []

        if manifest.get("schema_version") != 1:
            errors.append("unsupported corpus manifest schema_version")
        if manifest.get("repo_id") != run.assets.repo_id:
            errors.append("repo_id mismatch")
        if manifest.get("selected_voice_names") != expected_names:
            errors.append("selected voice names mismatch")
        if (
            manifest.get("include_cross_language_voices")
            != run.assets.include_cross_language_voices
        ):
            errors.append("include_cross_language_voices mismatch")
        if manifest.get("dtype") != run.assets.dtype:
            errors.append("dtype mismatch")
        if names != manifest.get("selected_voice_names"):
            errors.append("corpus.pt names do not match manifest")
        if hash_tensor(voices) != manifest.get("corpus_sha256"):
            errors.append("corpus tensor hash mismatch")

        if voices.ndim != 3 or voices.shape[-1] != 256:
            errors.append(f"invalid corpus tensor shape: {tuple(voices.shape)}")

        if run.assets.require_consistent_shape:
            shapes = {tuple(v.shape) for v in voices}
            if len(shapes) != 1:
                errors.append("corpus tensor shapes are inconsistent")

        if errors:
            raise ValueError(
                "Prepared corpus validation failed:\n  - " + "\n  - ".join(errors)
            )

        records = [
            VoiceRecord(
                name=name,
                tensor=voices[idx].to(torch.float32).contiguous(),
                source_path=Path(manifest["voices"][idx]["source_path"]),
                language_prefix=prefixes[idx],
            )
            for idx, name in enumerate(names)
        ]

        return cls(
            records=records,
            T=int(voices.shape[1]),
            D=int(voices.shape[2]),
            manifest=manifest,
            manifest_sha256=hash_file(run.corpus_manifest),
            corpus_sha256=str(manifest["corpus_sha256"]),
        )

    def tensors(self) -> torch.Tensor:
        return torch.stack(
            [record.tensor for record in self.records], dim=0
        ).contiguous()

    def names(self) -> list[str]:
        return [record.name for record in self.records]
