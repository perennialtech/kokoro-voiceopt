from __future__ import annotations

import torch


def to_kokoro_voice(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        t = t.unsqueeze(1)

    if t.ndim != 3 or t.shape[1] != 1 or t.shape[2] != 256:
        raise ValueError(f"Invalid Kokoro voice shape: {tuple(t.shape)}")

    return t.detach().to(torch.float32).contiguous()


class KokoroSynthesizer:
    def __init__(self, pipeline, sample_rate: int = 24000):
        self.pipeline = pipeline
        self.sample_rate = sample_rate

    def _model_device(self) -> torch.device:
        model = getattr(self.pipeline, "model", None)
        device = getattr(model, "device", None)
        if device is not None:
            return torch.device(device)

        if model is not None:
            try:
                return next(model.parameters()).device
            except StopIteration:
                pass
            except AttributeError:
                pass

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def synthesize(
        self, text: str, voice: torch.Tensor, speed: float = 1.0
    ) -> torch.Tensor:
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text")

        kokoro_voice = to_kokoro_voice(voice).to(self._model_device())
        chunks: list[torch.Tensor] = []

        with torch.no_grad():
            for result in self.pipeline(text=text, voice=kokoro_voice, speed=speed):
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
