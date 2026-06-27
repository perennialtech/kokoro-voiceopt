from __future__ import annotations

import torch

from .voice import as_kokoro_voice


class KokoroSynthesizer:
    def __init__(self, tts, sample_rate: int = 24000, lang_code: str = "a"):
        self.tts = tts
        self.sample_rate = sample_rate
        self.lang_code = lang_code

    def synthesize(
        self,
        text: str,
        voice: torch.Tensor,
        speed: float = 1.0,
    ) -> torch.Tensor:
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text")

        kokoro_voice = as_kokoro_voice(voice).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        chunks: list[torch.Tensor] = []

        with torch.no_grad():
            for result in self.tts.synthesize(
                text=text, voice=kokoro_voice, language=self.lang_code, speed=speed
            ):
                audio = getattr(result, "audio", None)
                if audio is None:
                    continue
                audio = torch.as_tensor(audio).detach().to(torch.float32).cpu()
                if audio.ndim == 2:
                    audio = audio.mean(dim=0)
                chunks.append(audio.contiguous())

        if not chunks:
            raise RuntimeError("Kokoro produced no audio")

        audio = torch.cat(chunks, dim=-1).to(torch.float32).cpu().contiguous()

        if audio.numel() == 0:
            raise RuntimeError("Kokoro produced empty audio")
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            raise RuntimeError("Kokoro produced NaN or Inf audio")
        if audio.numel() < int(self.sample_rate * 0.05):
            raise RuntimeError(
                f"Kokoro produced extremely short audio: {audio.numel()} samples"
            )

        return audio.contiguous()
