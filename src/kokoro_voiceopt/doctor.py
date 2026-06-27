from __future__ import annotations


import torch

from .corpus import VoiceCorpus
from .data import check_target_dataset
from .profile import load_target_speaker_profile
from .speaker import WavLMXVectorSpeakerEncoder


def doctor(run) -> None:
    errors: list[str] = []

    print("Checking config")
    try:
        run.write_resolved_config()
    except Exception as exc:
        errors.append(f"resolved config write failed: {exc}")

    print("Checking prepared target data")
    try:
        check_target_dataset(run)
    except SystemExit as exc:
        if exc.code:
            errors.append("prepared target data check failed")
    except Exception as exc:
        errors.append(f"prepared target data check failed: {exc}")

    print("Checking prepared corpus")
    corpus = None
    try:
        corpus = VoiceCorpus.load(run)
        print(f"Corpus voices: {len(corpus.records)} shape=({corpus.T}, {corpus.D})")
    except Exception as exc:
        errors.append(f"prepared corpus check failed: {exc}")

    print("Checking target profile")
    try:
        profile = load_target_speaker_profile(run)
        print(f"Profile embedding dim: {profile.embedding.numel()}")
    except Exception as exc:
        errors.append(f"target profile check failed: {exc}")

    print("Checking device")
    if run.device == "cuda" and not torch.cuda.is_available():
        errors.append("device is cuda but CUDA is unavailable")

    print("Checking imports")
    try:
        import kokoro  # noqa: F401
    except Exception as exc:
        errors.append(f"kokoro import failed: {exc}")

    try:
        import transformers  # noqa: F401
    except Exception as exc:
        errors.append(f"transformers import failed: {exc}")

    print("Checking speaker encoder loading")
    try:
        WavLMXVectorSpeakerEncoder(run.speaker_encoder, run.device)
    except Exception as exc:
        errors.append(f"speaker encoder load failed: {exc}")

    print("Checking Kokoro pipeline construction")
    try:
        from kokoro import KPipeline

        KPipeline(
            lang_code=run.target.lang_code,
            repo_id=run.assets.repo_id,
            device=run.device,
        )
    except Exception as exc:
        errors.append(f"Kokoro pipeline construction failed: {exc}")

    if corpus is not None:
        for record in corpus.records[:1]:
            if record.tensor.ndim != 2 or record.tensor.shape[-1] != 256:
                errors.append(
                    f"voice tensor shape incompatible: {record.name} {tuple(record.tensor.shape)}"
                )

    if errors:
        print("\nDoctor found errors:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("Doctor checks passed")
