from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .config import hash_file

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


def resolve_voice_names(config, lang_code: str) -> list[str]:
    names = (
        list(config.voice_names) if config.voice_names else list(DEFAULT_VOICE_NAMES)
    )
    if not config.include_cross_language_voices:
        names = [name for name in names if name.startswith(lang_code)]
    return names


def canonicalize_voice_tensor(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t)!r}")

    if t.ndim == 3:
        if t.shape[1] != 1:
            raise ValueError(
                f"Expected middle dimension 1 for [T,1,256], got {tuple(t.shape)}"
            )
        t = t[:, 0, :]
    elif t.ndim != 2:
        raise ValueError(f"Expected [T,1,256] or [T,256], got {tuple(t.shape)}")

    if t.shape[-1] != 256:
        raise ValueError(f"Expected D=256, got {tuple(t.shape)}")

    return t.contiguous()


def hash_tensor(t: torch.Tensor) -> str:
    array = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported corpus dtype: {name}")


def find_local_voice(voices_dir: Path | None, name: str) -> Path | None:
    if voices_dir is None:
        return None

    candidates = [
        voices_dir / f"{name}.pt",
        voices_dir / "voices" / f"{name}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def torch_load_voice(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_voice_corpus(run, force: bool = False) -> dict:
    corpus_dir = run.corpus_dir
    if corpus_dir.exists() and any(corpus_dir.iterdir()):
        if not force:
            raise SystemExit(f"{corpus_dir} already exists, use --force to recreate it")
        shutil.rmtree(corpus_dir)

    run.write_resolved_config()
    corpus_dir.mkdir(parents=True, exist_ok=True)

    names = resolve_voice_names(run.assets, run.target.lang_code)
    if not names:
        raise ValueError("No voice names selected for the prepared corpus")

    dtype = dtype_from_name(run.assets.dtype)

    records = []
    tensors = []

    for name in tqdm(names, desc="Preparing voice corpus"):
        path = find_local_voice(run.assets.voices_dir, name)
        if path is None:
            path = Path(
                hf_hub_download(run.assets.repo_id, filename=f"voices/{name}.pt")
            )

        raw = torch_load_voice(path)
        canonical = canonicalize_voice_tensor(raw).to(dtype=dtype).cpu().contiguous()

        records.append(
            {
                "name": name,
                "language_prefix": name.split("_", 1)[0],
                "source_path": str(path),
                "source_sha256": hash_file(path),
                "tensor_sha256": hash_tensor(canonical),
                "shape": list(canonical.shape),
            }
        )
        tensors.append(canonical)

    T, D = tensors[0].shape
    if run.assets.require_consistent_shape:
        for record, tensor in zip(records, tensors):
            if tuple(tensor.shape) != (T, D):
                raise ValueError(
                    f"Inconsistent voice shape for {record['name']}: "
                    f"expected {(T, D)}, got {tuple(tensor.shape)}"
                )

    voices = torch.stack(tensors, dim=0).contiguous()
    corpus_sha256 = hash_tensor(voices)

    metadata = {
        "schema_version": 1,
        "repo_id": run.assets.repo_id,
        "voice_names_config": (
            list(run.assets.voice_names) if run.assets.voice_names else None
        ),
        "selected_voice_names": names,
        "include_cross_language_voices": run.assets.include_cross_language_voices,
        "require_consistent_shape": run.assets.require_consistent_shape,
        "dtype": run.assets.dtype,
        "num_voices": len(names),
        "T": T,
        "D": D,
        "corpus_sha256": corpus_sha256,
    }

    torch.save(
        {
            "voices": voices,
            "names": names,
            "language_prefixes": [r["language_prefix"] for r in records],
            "metadata": metadata,
        },
        run.corpus_pt,
    )

    manifest = {
        **metadata,
        "voices": records,
    }
    with open(run.corpus_manifest, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest
