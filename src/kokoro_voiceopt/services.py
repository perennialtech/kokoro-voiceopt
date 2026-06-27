from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Context


def make_speaker_encoder(ctx: Context):
    from .speaker import WavLMXVectorSpeakerEncoder

    return WavLMXVectorSpeakerEncoder(ctx.cfg.speaker_encoder, ctx.cfg.device)


def make_kokoro_pipeline(ctx: Context):
    from kokoro import KokoroTRT

    if not ctx.cfg.assets.trt_artifact_dir:
        raise ValueError(
            "trt_artifact_dir must be configured in assets to use TensorRT Kokoro"
        )

    return KokoroTRT(str(ctx.cfg.assets.trt_artifact_dir))


def make_synthesizer(ctx: Context):
    from .synth import KokoroSynthesizer

    return KokoroSynthesizer(
        make_kokoro_pipeline(ctx),
        sample_rate=24000,
        lang_code=ctx.cfg.target.lang_code,
    )
