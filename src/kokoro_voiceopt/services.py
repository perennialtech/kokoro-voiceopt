from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Context


class ServiceFactory:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def speaker_encoder(self):
        from .speaker import WavLMXVectorSpeakerEncoder

        return WavLMXVectorSpeakerEncoder(self.ctx.speaker_encoder, self.ctx.device)

    def kokoro_pipeline(self):
        from kokoro import KPipeline

        return KPipeline(
            lang_code=self.ctx.target.lang_code,
            repo_id=self.ctx.assets.repo_id,
            device=self.ctx.device,
        )

    def synthesizer(self):
        from .synth import KokoroSynthesizer

        return KokoroSynthesizer(
            self.kokoro_pipeline(),
            sample_rate=self.ctx.audio.kokoro_sample_rate,
        )
