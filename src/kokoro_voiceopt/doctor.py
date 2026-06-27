from __future__ import annotations

import torch

from .corpus import VoiceCorpus
from .data import check_target_dataset
from .profile import load_target_speaker_profile


def check_config(ctx) -> None:
    ctx.write_resolved_config()


def check_target_data(ctx) -> None:
    check_target_dataset(ctx)


def check_corpus(ctx) -> None:
    corpus = VoiceCorpus.load(ctx)
    print(f"Corpus voices: {len(corpus.records)} shape=({corpus.T}, {corpus.D})")
    for record in corpus.records[:1]:
        if record.tensor.ndim != 2 or record.tensor.shape[-1] != 256:
            raise ValueError(
                f"voice tensor shape incompatible: {record.name} {tuple(record.tensor.shape)}"
            )


def check_profile(ctx) -> None:
    profile = load_target_speaker_profile(ctx)
    print(f"Profile embedding dim: {profile.embedding.numel()}")


def check_device(ctx) -> None:
    if ctx.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is cuda but CUDA is unavailable")


def check_imports(ctx) -> None:
    try:
        import kokoro  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"kokoro import failed: {exc}") from exc

    try:
        import transformers  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"transformers import failed: {exc}") from exc


def check_speaker_encoder(ctx) -> None:
    ctx.services.speaker_encoder()


def check_kokoro_pipeline(ctx) -> None:
    ctx.services.kokoro_pipeline()


CHECKS = [
    ("config", check_config),
    ("prepared target data", check_target_data),
    ("prepared corpus", check_corpus),
    ("target profile", check_profile),
    ("device", check_device),
    ("imports", check_imports),
    ("speaker encoder loading", check_speaker_encoder),
    ("Kokoro pipeline construction", check_kokoro_pipeline),
]


def doctor(ctx) -> None:
    errors: list[str] = []

    for name, check in CHECKS:
        print(f"Checking {name}")
        try:
            check(ctx)
        except SystemExit as exc:
            if exc.code:
                errors.append(f"{name} check failed")
        except Exception as exc:
            errors.append(f"{name} check failed: {exc}")

    if errors:
        print("\nDoctor found errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("Doctor checks passed")
