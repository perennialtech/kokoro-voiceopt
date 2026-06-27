from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .serde import fingerprint, load_pt, sha256_file, sha256_tensor
from .voice import as_voice_2d

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


def load_voice_tensor(path: Path) -> torch.Tensor:
    try:
        return load_pt(path, weights_only=True)
    except TypeError:
        return load_pt(path)


def corpus_fingerprint(ctx) -> dict[str, Any]:
    return {
        "repo_id": ctx.assets.repo_id,
        "selected_voice_names": resolve_voice_names(ctx.assets, ctx.target.lang_code),
        "include_cross_language_voices": ctx.assets.include_cross_language_voices,
        "dtype": ctx.assets.dtype,
        "asset_config_sha256": fingerprint(ctx.assets),
    }


def corpus_artifact(ctx):
    return ctx.artifacts.spec(
        "voice_corpus",
        data=ctx.paths.corpus("corpus.pt"),
        meta=ctx.paths.corpus("corpus_manifest.json"),
        fingerprint=corpus_fingerprint(ctx),
    )


def prepare_voice_corpus(ctx, force: bool = False) -> dict:
    corpus_dir = ctx.paths.corpus_dir
    if corpus_dir.exists() and any(corpus_dir.iterdir()):
        if not force:
            raise SystemExit(f"{corpus_dir} already exists, use --force to recreate it")
        shutil.rmtree(corpus_dir)

    ctx.write_resolved_config()
    corpus_dir.mkdir(parents=True, exist_ok=True)

    names = resolve_voice_names(ctx.assets, ctx.target.lang_code)
    if not names:
        raise ValueError("No voice names selected for the prepared corpus")

    dtype = dtype_from_name(ctx.assets.dtype)

    records = []
    tensors = []

    for name in tqdm(names, desc="Preparing voice corpus"):
        path = find_local_voice(ctx.assets.voices_dir, name)
        if path is None:
            path = Path(
                hf_hub_download(ctx.assets.repo_id, filename=f"voices/{name}.pt")
            )

        raw = load_voice_tensor(path)
        canonical = as_voice_2d(raw).to(dtype=dtype).cpu().contiguous()

        records.append(
            {
                "name": name,
                "language_prefix": name.split("_", 1)[0],
                "source_path": str(path),
                "source_sha256": sha256_file(path),
                "tensor_sha256": sha256_tensor(canonical),
                "shape": list(canonical.shape),
            }
        )
        tensors.append(canonical)

    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        detail = "\n".join(
            f"  - {record['name']}: {record['shape']}" for record in records
        )
        raise ValueError(f"Selected voices have inconsistent shapes:\n{detail}")

    T, D = tensors[0].shape
    voices = torch.stack(tensors, dim=0).contiguous()
    corpus_sha256 = sha256_tensor(voices)

    payload = {
        "voices": voices,
        "names": names,
        "language_prefixes": [record["language_prefix"] for record in records],
    }

    manifest_extra = {
        "repo_id": ctx.assets.repo_id,
        "voice_names_config": (
            list(ctx.assets.voice_names) if ctx.assets.voice_names else None
        ),
        "selected_voice_names": names,
        "include_cross_language_voices": ctx.assets.include_cross_language_voices,
        "dtype": ctx.assets.dtype,
        "num_voices": len(names),
        "T": T,
        "D": D,
        "corpus_sha256": corpus_sha256,
        "voices": records,
    }

    metadata = corpus_artifact(ctx).save_pt(payload, extra=manifest_extra)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata
