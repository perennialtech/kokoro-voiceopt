from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .assets import corpus_artifact, resolve_voice_names
from .serde import sha256_tensor


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
    def load(cls, ctx) -> "VoiceCorpus":
        ctx.require_corpus()
        artifact = corpus_artifact(ctx)
        manifest = artifact.require_current()
        data = artifact.load_pt()

        voices = data["voices"].cpu().contiguous()
        names = list(data["names"])
        prefixes = list(data["language_prefixes"])

        expected_names = resolve_voice_names(ctx.assets, ctx.target.lang_code)
        errors = []

        if manifest.get("selected_voice_names") != expected_names:
            errors.append("selected voice names mismatch")
        if manifest.get("repo_id") != ctx.assets.repo_id:
            errors.append("repo_id mismatch")
        if (
            manifest.get("include_cross_language_voices")
            != ctx.assets.include_cross_language_voices
        ):
            errors.append("include_cross_language_voices mismatch")
        if manifest.get("dtype") != ctx.assets.dtype:
            errors.append("dtype mismatch")
        if names != manifest.get("selected_voice_names"):
            errors.append("corpus.pt names do not match manifest")
        if sha256_tensor(voices) != manifest.get("corpus_sha256"):
            errors.append("corpus tensor hash mismatch")

        if voices.ndim != 3 or voices.shape[-1] != 256:
            errors.append(f"invalid corpus tensor shape: {tuple(voices.shape)}")

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
            manifest_sha256=str(manifest["data_sha256"]),
            corpus_sha256=str(manifest["corpus_sha256"]),
        )

    def tensors(self) -> torch.Tensor:
        return torch.stack(
            [record.tensor for record in self.records], dim=0
        ).contiguous()

    def names(self) -> list[str]:
        return [record.name for record in self.records]
