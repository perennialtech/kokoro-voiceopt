from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .serde import fingerprint, sha256_file
from .voice import as_voice_2d


@dataclass
class VoiceManifold:
    center: torch.Tensor
    components: torch.Tensor
    sigma: torch.Tensor
    T: int
    D: int
    voice_names: list[str]
    config: object
    explained_variance_ratio: torch.Tensor
    metadata: dict

    @classmethod
    def fit(cls, corpus, config, metadata: dict | None = None) -> "VoiceManifold":
        voices = corpus.tensors().to(torch.float32)
        N, T, D = voices.shape
        flat = voices.reshape(N, T * D).contiguous()

        if config.center == "median":
            center = flat.median(dim=0).values
        elif config.center == "mean":
            center = flat.mean(dim=0)
        else:
            raise ValueError(f"Unsupported center type: {config.center}")

        centered = flat - center
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)

        variance = singular_values.square() / max(N - 1, 1)
        total_variance = variance.sum()
        if total_variance <= 1e-12:
            raise ValueError("Voice corpus has near-zero variance; cannot fit manifold")

        explained = variance / total_variance
        cumulative = torch.cumsum(explained, dim=0)
        d_coverage = int((cumulative < config.variance_coverage).sum().item()) + 1
        nonzero_rank = int((singular_values > 1e-8).sum().item())

        d = min(config.max_latent_dim, d_coverage, nonzero_rank, vh.shape[0])
        if d <= 0:
            raise ValueError("No usable latent dimensions found")

        components = vh[:d].contiguous()
        sigma = (singular_values[:d] / math.sqrt(max(N - 1, 1))).clamp_min(1e-8)

        return cls(
            center=center.cpu().contiguous(),
            components=components.cpu().contiguous(),
            sigma=sigma.cpu().contiguous(),
            T=T,
            D=D,
            voice_names=corpus.names(),
            config=config,
            explained_variance_ratio=explained[:d].cpu().contiguous(),
            metadata=dict(metadata or {}),
        )

    @property
    def latent_dim(self) -> int:
        return int(self.components.shape[0])

    def encode(self, voice: torch.Tensor) -> torch.Tensor:
        voice = as_voice_2d(voice)
        if tuple(voice.shape) != (self.T, self.D):
            raise ValueError(
                f"Expected voice shape {(self.T, self.D)}, got {tuple(voice.shape)}"
            )

        flat = voice.to(torch.float32).reshape(-1).cpu()
        centered = flat - self.center
        z = centered @ self.components.T
        z = z / self.sigma
        return z.to(torch.float32).contiguous()

    def decode(self, z: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        z = z.to(torch.float32).cpu()
        single = z.ndim == 1
        if single:
            z = z.unsqueeze(0)

        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected z shape [d] or [B,d] with d={self.latent_dim}, got {tuple(z.shape)}"
            )

        if clamp:
            z = self.clamp_z(z)

        flat = (
            self.center.unsqueeze(0) + (z * self.sigma.unsqueeze(0)) @ self.components
        )
        voices = flat.reshape(z.shape[0], self.T, self.D).contiguous()
        return voices[0] if single else voices

    def clamp_z(self, z: torch.Tensor) -> torch.Tensor:
        return (
            z.to(torch.float32)
            .clamp(-float(self.config.z_hard_bound), float(self.config.z_hard_bound))
            .contiguous()
        )

    def prior_loss(self, z: torch.Tensor) -> torch.Tensor:
        return z.to(torch.float32).square().mean()

    def soft_bound_loss(self, z: torch.Tensor) -> torch.Tensor:
        excess = (
            z.to(torch.float32).abs() - float(self.config.z_soft_bound)
        ).clamp_min(0.0)
        return excess.square().mean()

    def to_payload(self) -> dict[str, Any]:
        return {
            "center": self.center.cpu().to(torch.float32).contiguous(),
            "components": self.components.cpu().to(torch.float32).contiguous(),
            "sigma": self.sigma.cpu().to(torch.float32).contiguous(),
            "T": self.T,
            "D": self.D,
            "voice_names": self.voice_names,
            "config": asdict(self.config),
            "explained_variance_ratio": self.explained_variance_ratio.cpu()
            .to(torch.float32)
            .contiguous(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, data: dict, config_type) -> "VoiceManifold":
        return cls(
            center=data["center"].to(torch.float32).contiguous(),
            components=data["components"].to(torch.float32).contiguous(),
            sigma=data["sigma"].to(torch.float32).contiguous(),
            T=int(data["T"]),
            D=int(data["D"]),
            voice_names=list(data["voice_names"]),
            config=config_type(**data["config"]),
            explained_variance_ratio=data["explained_variance_ratio"]
            .to(torch.float32)
            .contiguous(),
            metadata=dict(data.get("metadata", {})),
        )

    def report(self) -> dict[str, Any]:
        return {
            "T": self.T,
            "D": self.D,
            "latent_dim": self.latent_dim,
            "voice_names": self.voice_names,
            "manifold_config": asdict(self.config),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "explained_variance_total": float(self.explained_variance_ratio.sum()),
            "build_metadata": self.metadata,
        }


def manifold_fingerprint(ctx, corpus) -> dict[str, Any]:
    return {
        "corpus_manifest_sha256": sha256_file(ctx.paths.corpus("corpus_manifest.json")),
        "corpus_sha256": corpus.corpus_sha256,
        "manifold_config_sha256": fingerprint(ctx.cfg.manifold),
        "voice_names": corpus.names(),
        "T": corpus.T,
        "D": corpus.D,
    }
