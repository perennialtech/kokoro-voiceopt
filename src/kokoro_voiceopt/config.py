from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .serde import jsonable

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunConfig:
    id: str
    root: Path


@dataclass(frozen=True)
class TargetConfig:
    id: str
    lang_code: str = "a"


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int = 16000
    trim_edges: bool = True
    trim_threshold: float = 0.015
    trim_pad_ms: int = 120
    fade_ms: int = 8
    dc_remove: bool = True
    peak_normalize: bool = True
    max_peak: float = 0.98


@dataclass(frozen=True)
class DataConfig:
    min_duration_s: float = 1.5
    max_duration_s: float = 12.0
    min_total_duration_s: float = 20.0
    min_target_clips: int = 1
    max_target_clips: int = 24
    max_clip_ratio: float = 0.0001
    hard_end_threshold: float = 0.2
    seed: int = 42


@dataclass(frozen=True)
class SpeakerEncoderConfig:
    backend: str = "wavlm_xvector"
    model_name: str = "microsoft/wavlm-base-plus-sv"
    batch_size: int = 16


@dataclass(frozen=True)
class VoiceCorpusConfig:
    repo_id: str = "hexgrad/Kokoro-82M"
    trt_artifact_dir: Path | None = None
    voices_dir: Path | None = None
    prepared_corpus_dir: Path | None = None
    voice_names: tuple[str, ...] | None = None
    include_cross_language_voices: bool = True
    dtype: str = "float32"


@dataclass(frozen=True)
class TextConfig:
    min_text_chars: int = 20
    max_text_chars: int = 220
    validation_texts_path: Path | None = None


@dataclass(frozen=True)
class ManifoldConfig:
    center: str = "median"
    max_latent_dim: int = 32
    variance_coverage: float = 0.98
    z_soft_bound: float = 2.5
    z_hard_bound: float = 4.0


@dataclass(frozen=True)
class ObjectiveConfig:
    speaker_loss_weight: float = 1.0
    prior_loss_weight: float = 0.02
    bound_loss_weight: float = 0.10
    silence_loss_weight: float = 0.05
    clipping_loss_weight: float = 0.10
    duration_loss_weight: float = 0.02
    max_silence_ratio: float = 0.25
    max_clip_ratio: float = 0.001
    invalid_audio_loss: float = 100.0


@dataclass(frozen=True)
class SearchConfig:
    seed: int = 1234
    top_k_for_blend: int = 8

    blend_iterations: int = 60
    blend_population_pairs: int = 8
    blend_sigma_initial: float = 0.75
    blend_sigma_final: float = 0.15
    blend_learning_rate: float = 0.08

    latent_iterations: int = 180
    latent_population_pairs: int = 12
    latent_sigma_initial: float = 0.35
    latent_sigma_final: float = 0.06
    latent_learning_rate: float = 0.04

    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    keep_every: int = 10


@dataclass(frozen=True)
class Config:
    schema_version: int
    run: RunConfig
    target: TargetConfig
    audio: AudioConfig
    data: DataConfig
    speaker_encoder: SpeakerEncoderConfig
    assets: VoiceCorpusConfig
    text: TextConfig
    manifold: ManifoldConfig
    objective: ObjectiveConfig
    search: SearchConfig
    device: str


@dataclass(frozen=True)
class PathLayout:
    root: Path
    corpus_root: Path

    def _join(self, base: Path, *parts: str | Path) -> Path:
        path = base
        for part in parts:
            path = path / part
        return path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def profile_dir(self) -> Path:
        return self.root / "profile"

    @property
    def corpus_dir(self) -> Path:
        return self.corpus_root

    @property
    def manifold_dir(self) -> Path:
        return self.root / "manifold"

    @property
    def optimize_dir(self) -> Path:
        return self.root / "optimize"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def preview_dir(self) -> Path:
        return self.root / "preview"

    def data(self, *parts: str | Path) -> Path:
        return self._join(self.data_dir, *parts)

    def profile(self, *parts: str | Path) -> Path:
        return self._join(self.profile_dir, *parts)

    def corpus(self, *parts: str | Path) -> Path:
        return self._join(self.corpus_dir, *parts)

    def manifold(self, *parts: str | Path) -> Path:
        return self._join(self.manifold_dir, *parts)

    def optimize(self, *parts: str | Path) -> Path:
        return self._join(self.optimize_dir, *parts)

    def export(self, *parts: str | Path) -> Path:
        return self._join(self.export_dir, *parts)

    def preview(self, *parts: str | Path) -> Path:
        return self._join(self.preview_dir, *parts)

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "data_dir": str(self.data_dir),
            "profile_dir": str(self.profile_dir),
            "corpus_dir": str(self.corpus_dir),
            "optimize_dir": str(self.optimize_dir),
            "export_dir": str(self.export_dir),
            "preview_dir": str(self.preview_dir),
        }


@dataclass(frozen=True)
class Context:
    cfg: Config
    paths: PathLayout
    project_root: Path
    config_path: Path

    @classmethod
    def load(
        cls,
        config_path: str | Path,
        project_root: str | Path | None = None,
    ) -> "Context":
        project_root_path = Path(project_root or Path.cwd()).resolve()

        config_path_obj = Path(config_path)
        if not config_path_obj.is_absolute():
            base = project_root_path if project_root is not None else Path.cwd()
            config_path_obj = base / config_path_obj
        config_path_obj = config_path_obj.resolve()

        with open(config_path_obj, encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a YAML mapping: {config_path_obj}")

        if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(
                f"Only schema_version: {SCHEMA_VERSION} configs are supported"
            )

        run_raw = dict(raw["run"])
        run_id = str(run_raw["id"])
        run_root = resolve_path(
            project_root_path, run_raw.get("root", f"runs/{run_id}")
        )

        assets_raw = dict(raw.get("assets", {}))
        trt_artifact_dir = assets_raw.get("trt_artifact_dir")
        voices_dir = assets_raw.get("voices_dir")
        prepared_corpus_dir = assets_raw.get("prepared_corpus_dir")
        voice_names = assets_raw.get("voice_names")

        assets = VoiceCorpusConfig(
            repo_id=str(assets_raw.get("repo_id", "hexgrad/Kokoro-82M")),
            trt_artifact_dir=(
                resolve_path(project_root_path, trt_artifact_dir)
                if trt_artifact_dir
                else None
            ),
            voices_dir=(
                resolve_path(project_root_path, voices_dir) if voices_dir else None
            ),
            prepared_corpus_dir=(
                resolve_path(project_root_path, prepared_corpus_dir)
                if prepared_corpus_dir
                else run_root / "corpus"
            ),
            voice_names=(
                None
                if voice_names is None
                else tuple(str(name) for name in voice_names)
            ),
            include_cross_language_voices=bool(
                assets_raw.get("include_cross_language_voices", True)
            ),
            dtype=str(assets_raw.get("dtype", "float32")),
        )

        text_raw = dict(raw.get("text", {}))
        validation_texts_path = text_raw.get("validation_texts_path")
        text_raw["validation_texts_path"] = (
            resolve_path(project_root_path, validation_texts_path)
            if validation_texts_path
            else None
        )

        cfg = Config(
            schema_version=SCHEMA_VERSION,
            run=RunConfig(id=run_id, root=run_root),
            target=TargetConfig(**dict(raw["target"])),
            audio=AudioConfig(**dict(raw.get("audio", {}))),
            data=DataConfig(**dict(raw.get("data", {}))),
            speaker_encoder=SpeakerEncoderConfig(
                **dict(raw.get("speaker_encoder", {}))
            ),
            assets=assets,
            text=TextConfig(**text_raw),
            manifold=ManifoldConfig(**dict(raw.get("manifold", {}))),
            objective=ObjectiveConfig(**dict(raw.get("objective", {}))),
            search=SearchConfig(**dict(raw.get("search", {}))),
            device=str(raw.get("device", "cuda")),
        )

        paths = PathLayout(
            root=run_root,
            corpus_root=assets.prepared_corpus_dir or run_root / "corpus",
        )

        return cls(
            cfg=cfg,
            paths=paths,
            project_root=project_root_path,
            config_path=config_path_obj,
        )

    @property
    def id(self) -> str:
        return self.cfg.run.id

    def resolve_path(self, path: str | Path) -> Path:
        return resolve_path(self.project_root, path)

    def write_resolved_config(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        with open(
            self.paths.root / "config.resolved.yaml", "w", encoding="utf-8"
        ) as file:
            yaml.safe_dump(self.to_dict(), file, sort_keys=False, allow_unicode=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.cfg.schema_version,
            "run": jsonable(asdict(self.cfg.run)),
            "target": jsonable(asdict(self.cfg.target)),
            "assets": jsonable(asdict(self.cfg.assets)),
            "audio": jsonable(asdict(self.cfg.audio)),
            "data": jsonable(asdict(self.cfg.data)),
            "speaker_encoder": jsonable(asdict(self.cfg.speaker_encoder)),
            "text": jsonable(asdict(self.cfg.text)),
            "manifold": jsonable(asdict(self.cfg.manifold)),
            "objective": jsonable(asdict(self.cfg.objective)),
            "search": jsonable(asdict(self.cfg.search)),
            "device": self.cfg.device,
            "project_root": str(self.project_root),
            "config_path": str(self.config_path),
            "resolved_paths": self.paths.as_dict(),
        }

    def require_manifests(self) -> None:
        require_paths(
            [
                ("target_manifest", self.paths.data("manifests/target.jsonl"), False),
            ],
            "Required prepared target data missing:",
        )

    def require_profile(self) -> None:
        require_paths(
            [
                ("target_profile_pt", self.paths.profile("target_profile.pt"), False),
                (
                    "target_profile_json",
                    self.paths.profile("target_profile.json"),
                    False,
                ),
            ],
            "Required target profile missing:",
        )

    def require_corpus(self) -> None:
        require_paths(
            [
                ("corpus_pt", self.paths.corpus("corpus.pt"), False),
                ("corpus_manifest", self.paths.corpus("corpus_manifest.json"), False),
            ],
            "Required prepared voice corpus missing:",
        )


def resolve_path(project_root: str | Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(project_root) / path


def require_paths(required: list[tuple[str, str | Path, bool]], message: str) -> None:
    missing = []
    for label, path, is_dir in required:
        path = Path(path)
        ok = path.is_dir() if is_dir else path.is_file()
        if not ok:
            missing.append(f"{label}: {path}")

    if missing:
        raise FileNotFoundError(f"{message}\n  - " + "\n  - ".join(missing))
