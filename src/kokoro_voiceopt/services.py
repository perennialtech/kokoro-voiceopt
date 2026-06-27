from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Context


def make_speaker_encoder(ctx: Context):
    from .speaker import WavLMXVectorSpeakerEncoder

    return WavLMXVectorSpeakerEncoder(ctx.cfg.speaker_encoder, ctx.cfg.device)


def make_kokoro_pipeline(ctx: Context):
    from kokoro import KPipeline

    return KPipeline(
        lang_code=ctx.cfg.target.lang_code,
        repo_id=ctx.cfg.assets.repo_id,
        device=ctx.cfg.device,
    )


def make_synthesizer(ctx: Context):
    from .synth import KokoroSynthesizer

    return KokoroSynthesizer(
        make_kokoro_pipeline(ctx),
        sample_rate=ctx.cfg.audio.kokoro_sample_rate,
    )
