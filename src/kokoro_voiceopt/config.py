from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunPaths:
    root: Path
    data_dir: Path
    audio_dir: Path
    manifest_dir: Path
    target_manifest: Path
    rejected_manifest: Path
    dataset_report: Path
    text_plan: Path

    profile_dir: Path
    target_profile_pt: Path
    target_profile_json: Path

    corpus_dir: Path
    corpus_pt: Path
    corpus_manifest: Path

    manifold_dir: Path
    manifold_pt: Path
    manifold_report: Path

    optimize_dir: Path
    optimize_stage_dir: Path
    optimize_checkpoint_dir: Path
    optimize_candidate_dir: Path
    optimize_run_info: Path

    export_dir: Path
    export_voice: Path
    export_voice_meta: Path
    export_voice_best: Path
    export_voice_final: Path
    export_voice_best_optimization: Path

    preview_dir: Path


@dataclass(frozen=True)
class TargetConfig:
    id: str
    lang_code: str = "a"


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int = 16000
    kokoro_sample_rate: int = 24000
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
    require_spoken_form: bool = True
    seed: int = 42


@dataclass(frozen=True)
class SpeakerEncoderConfig:
    backend: str = "wavlm_xvector"
    model_name: str = "microsoft/wavlm-base-plus-sv"
    batch_size: int = 16
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class VoiceCorpusConfig:
    repo_id: str = "hexgrad/Kokoro-82M"
    voices_dir: Path | None = None
    prepared_corpus_dir: Path | None = None
    voice_names: tuple[str, ...] | None = None
    include_cross_language_voices: bool = True
    require_consistent_shape: bool = True
    dtype: str = "float32"


@dataclass(frozen=True)
class TextConfig:
    max_optimization_texts: int = 3
    max_validation_texts: int = 6
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
    save_manifold: bool = True


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

    save_every: int = 10
    validate_every: int = 20


@dataclass(frozen=True)
class Run:
    config_path: Path
    project_root: Path
    id: str
    config: dict[str, Any]
    paths: RunPaths
    target: TargetConfig
    audio: AudioConfig
    data: DataConfig
    speaker_encoder: SpeakerEncoderConfig
    assets: VoiceCorpusConfig
    text: TextConfig
    manifold: ManifoldConfig
    objective: ObjectiveConfig
    search: SearchConfig
    device: str = "cuda"

    @classmethod
    def load(
        cls, config_path: str | Path, project_root: str | Path | None = None
    ) -> "Run":
        config_path = Path(config_path).resolve()
        project_root = Path(project_root or Path.cwd()).resolve()

        with open(config_path, encoding="utf-8") as file:
            raw = yaml.safe_load(file)

        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a YAML mapping: {config_path}")

        if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(
                f"Only schema_version: {SCHEMA_VERSION} configs are supported"
            )

        run_cfg = dict(raw["run"])
        run_id = str(run_cfg["id"])
        root = resolve_path(project_root, run_cfg.get("root", f"runs/{run_id}"))

        corpus_dir = root / "corpus"
        paths = RunPaths(
            root=root,
            data_dir=root / "data",
            audio_dir=root / "data" / "audio",
            manifest_dir=root / "data" / "manifests",
            target_manifest=root / "data" / "manifests" / "target.jsonl",
            rejected_manifest=root / "data" / "manifests" / "rejected.jsonl",
            dataset_report=root / "data" / "report.json",
            text_plan=root / "data" / "text_plan.json",
            profile_dir=root / "profile",
            target_profile_pt=root / "profile" / "target_profile.pt",
            target_profile_json=root / "profile" / "target_profile.json",
            corpus_dir=corpus_dir,
            corpus_pt=corpus_dir / "corpus.pt",
            corpus_manifest=corpus_dir / "corpus_manifest.json",
            manifold_dir=root / "manifold",
            manifold_pt=root / "manifold" / "manifold.pt",
            manifold_report=root / "manifold" / "manifold_report.json",
            optimize_dir=root / "optimize",
            optimize_stage_dir=root / "optimize" / "stages",
            optimize_checkpoint_dir=root / "optimize" / "checkpoints",
            optimize_candidate_dir=root / "optimize" / "candidates",
            optimize_run_info=root / "optimize" / "run_info.json",
            export_dir=root / "export",
            export_voice=root / "export" / "voice.pt",
            export_voice_meta=root / "export" / "voice_meta.json",
            export_voice_best=root / "export" / "voice_best.pt",
            export_voice_final=root / "export" / "voice_final.pt",
            export_voice_best_optimization=root
            / "export"
            / "voice_best_optimization.pt",
            preview_dir=root / "preview",
        )

        target = TargetConfig(**dict(raw["target"]))

        assets_raw = dict(raw.get("assets", {}))
        voices_dir = assets_raw.get("voices_dir")
        prepared_corpus_dir = assets_raw.get("prepared_corpus_dir")
        voice_names = assets_raw.get("voice_names")

        assets = VoiceCorpusConfig(
            repo_id=str(assets_raw.get("repo_id", "hexgrad/Kokoro-82M")),
            voices_dir=resolve_path(project_root, voices_dir) if voices_dir else None,
            prepared_corpus_dir=(
                resolve_path(project_root, prepared_corpus_dir)
                if prepared_corpus_dir
                else paths.corpus_dir
            ),
            voice_names=(
                None
                if voice_names is None
                else tuple(str(name) for name in voice_names)
            ),
            include_cross_language_voices=bool(
                assets_raw.get("include_cross_language_voices", True)
            ),
            require_consistent_shape=bool(
                assets_raw.get("require_consistent_shape", True)
            ),
            dtype=str(assets_raw.get("dtype", "float32")),
        )

        text_raw = dict(raw.get("text", {}))
        validation_texts_path = text_raw.get("validation_texts_path")
        text_raw["validation_texts_path"] = (
            resolve_path(project_root, validation_texts_path)
            if validation_texts_path
            else None
        )

        return cls(
            config_path=config_path,
            project_root=project_root,
            id=run_id,
            config=raw,
            paths=paths,
            target=target,
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

    @property
    def corpus_dir(self) -> Path:
        return Path(self.assets.prepared_corpus_dir or self.paths.corpus_dir)

    @property
    def corpus_pt(self) -> Path:
        return self.corpus_dir / "corpus.pt"

    @property
    def corpus_manifest(self) -> Path:
        return self.corpus_dir / "corpus_manifest.json"

    def write_resolved_config(self) -> None:
        self.paths.root.mkdir(parents=True, exist_ok=True)
        with open(
            self.paths.root / "config.resolved.yaml", "w", encoding="utf-8"
        ) as file:
            yaml.safe_dump(self.to_dict(), file, sort_keys=False, allow_unicode=True)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.config)
        result["resolved_paths"] = {
            key: str(value) for key, value in asdict(self.paths).items()
        }
        result["resolved_assets"] = {
            "repo_id": self.assets.repo_id,
            "voices_dir": (
                str(self.assets.voices_dir) if self.assets.voices_dir else None
            ),
            "prepared_corpus_dir": str(self.corpus_dir),
            "voice_names": (
                list(self.assets.voice_names) if self.assets.voice_names else None
            ),
            "include_cross_language_voices": self.assets.include_cross_language_voices,
            "require_consistent_shape": self.assets.require_consistent_shape,
            "dtype": self.assets.dtype,
        }
        return result

    def require_manifests(self) -> None:
        require_paths(
            [
                ("target_manifest", self.paths.target_manifest, False),
            ],
            "Required prepared target data missing:",
        )

    def require_profile(self) -> None:
        require_paths(
            [
                ("target_profile_pt", self.paths.target_profile_pt, False),
                ("target_profile_json", self.paths.target_profile_json, False),
            ],
            "Required target profile missing:",
        )

    def require_corpus(self) -> None:
        require_paths(
            [
                ("corpus_pt", self.corpus_pt, False),
                ("corpus_manifest", self.corpus_manifest, False),
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


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash_json(value: Any) -> str:
    encoded = json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def recursive_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: recursive_namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [recursive_namespace(item) for item in value]
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
