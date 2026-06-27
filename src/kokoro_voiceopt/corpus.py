from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from .config import CorpusConfig

DEFAULT_VOICE_NAMES = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
    "ef_arlet",
    "ef_dora",
    "em_danel",
    "em_ramos",
    "ff_adele",
    "fm_antoine",
    "hf_diya",
    "hm_prabhu",
    "if_chiara",
    "im_raffa",
    "jf_himari",
    "jf_nana",
    "jm_eiji",
    "jm_kaito",
    "pf_ana",
    "pm_dinis",
    "zf_meimei",
    "zm_haoran",
]


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

    @classmethod
    def load(
        cls, config: CorpusConfig, voice_names: list[str] | None = None
    ) -> "VoiceCorpus":
        names = voice_names or list(DEFAULT_VOICE_NAMES)
        if not config.include_cross_language_voices:
            names = [name for name in names if name.startswith(config.lang_code)]

        records: list[VoiceRecord] = []
        for name in names:
            path = _find_local_voice(config.voices_dir, name)
            if path is None:
                path = Path(
                    hf_hub_download(config.repo_id, filename=f"voices/{name}.pt")
                )

            tensor = _torch_load_voice(path)
            canonical = canonicalize_voice_tensor(tensor)
            records.append(
                VoiceRecord(
                    name=name,
                    tensor=canonical.cpu().to(torch.float32).contiguous(),
                    source_path=path,
                    language_prefix=name.split("_", 1)[0],
                )
            )

        if not records:
            raise ValueError("No voices were loaded")

        T, D = records[0].tensor.shape
        if config.require_consistent_shape:
            for record in records:
                if tuple(record.tensor.shape) != (T, D):
                    raise ValueError(
                        f"Inconsistent voice shape for {record.name}: "
                        f"expected {(T, D)}, got {tuple(record.tensor.shape)}"
                    )

        return cls(records=records, T=T, D=D)

    def manifest(self) -> dict:
        return {
            "num_voices": len(self.records),
            "T": self.T,
            "D": self.D,
            "voices": [
                {
                    "name": r.name,
                    "shape": list(r.tensor.shape),
                    "source_path": str(r.source_path) if r.source_path else None,
                    "language_prefix": r.language_prefix,
                }
                for r in self.records
            ],
        }

    def tensors(self) -> torch.Tensor:
        return torch.stack([r.tensor for r in self.records], dim=0).contiguous()

    def names(self) -> list[str]:
        return [r.name for r in self.records]


def canonicalize_voice_tensor(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t)!r}")

    if t.ndim == 3:
        if t.shape[1] != 1:
            raise ValueError(
                f"Expected middle dimension 1 for [T,1,256], got {tuple(t.shape)}"
            )
        t = t[:, 0, :]
    elif t.ndim == 2:
        pass
    else:
        raise ValueError(f"Expected [T,1,256] or [T,256], got {tuple(t.shape)}")

    if t.shape[-1] != 256:
        raise ValueError(f"Expected D=256, got {tuple(t.shape)}")

    return t.to(dtype=torch.float32).contiguous()


def _find_local_voice(voices_dir: Path | None, name: str) -> Path | None:
    if voices_dir is None:
        return None

    voices_dir = Path(voices_dir)
    candidates = [
        voices_dir / f"{name}.pt",
        voices_dir / "voices" / f"{name}.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _torch_load_voice(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
