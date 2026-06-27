from __future__ import annotations

from pathlib import Path

import torch

from .serde import sha256_tensor


def as_voice_2d(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")

    if tensor.ndim == 3:
        if tensor.shape[1] != 1:
            raise ValueError(
                f"Expected Kokoro voice middle dimension 1 for [T,1,256], got {tuple(tensor.shape)}"
            )
        tensor = tensor[:, 0, :]
    elif tensor.ndim != 2:
        raise ValueError(
            f"Expected voice shape [T,256] or [T,1,256], got {tuple(tensor.shape)}"
        )

    if tensor.shape[-1] != 256:
        raise ValueError(
            f"Expected Kokoro voice embedding dimension 256, got {tuple(tensor.shape)}"
        )

    return tensor.detach().cpu().to(torch.float32).contiguous()


def as_kokoro_voice(tensor: torch.Tensor) -> torch.Tensor:
    voice = as_voice_2d(tensor).unsqueeze(1).contiguous()
    if voice.ndim != 3 or voice.shape[1] != 1 or voice.shape[2] != 256:
        raise ValueError(
            f"Invalid Kokoro voice shape after normalization: {tuple(voice.shape)}"
        )
    return voice


def voice_hash(tensor: torch.Tensor) -> str:
    return sha256_tensor(as_voice_2d(tensor))


def save_voice(path: str | Path, tensor: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(as_kokoro_voice(tensor).cpu().to(torch.float32).contiguous(), path)
