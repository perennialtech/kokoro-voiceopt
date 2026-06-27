from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .serde import fingerprint as fingerprint_value
from .serde import read_json, sha256_file, write_json


class StaleArtifactError(RuntimeError):
    """Raised when an artifact exists but does not match the current fingerprint."""


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    data_path: Path
    meta_path: Path
    fingerprint: dict[str, Any]


def _metadata(spec: ArtifactSpec) -> dict[str, Any]:
    if not spec.meta_path.exists():
        raise FileNotFoundError(f"Missing artifact metadata: {spec.meta_path}")

    data = read_json(spec.meta_path)
    if not isinstance(data, dict):
        raise ValueError(f"Artifact metadata must be a JSON object: {spec.meta_path}")
    return data


def stale_reasons(spec: ArtifactSpec) -> list[str]:
    reasons: list[str] = []

    if not spec.data_path.exists():
        reasons.append(f"missing data: {spec.data_path}")
    if not spec.meta_path.exists():
        reasons.append(f"missing metadata: {spec.meta_path}")
        return reasons

    try:
        metadata = _metadata(spec)
    except Exception as exc:
        return [f"metadata read failed: {exc}"]

    if metadata.get("schema_version") != 1:
        reasons.append("unsupported metadata schema_version")
    if metadata.get("artifact") != spec.name:
        reasons.append(
            f"artifact name mismatch: expected {spec.name}, got {metadata.get('artifact')}"
        )
    if metadata.get("fingerprint") != spec.fingerprint:
        reasons.append("fingerprint mismatch")

    if spec.data_path.exists() and metadata.get("data_sha256"):
        actual = sha256_file(spec.data_path)
        if metadata.get("data_sha256") != actual:
            reasons.append("data_sha256 mismatch")

    return reasons


def require_current(spec: ArtifactSpec) -> dict[str, Any]:
    reasons = stale_reasons(spec)
    if reasons:
        raise StaleArtifactError(
            f"Artifact is stale or missing: {spec.name}\n  - " + "\n  - ".join(reasons)
        )
    return _metadata(spec)


def save_artifact(
    spec: ArtifactSpec,
    data: Any,
    writer: Callable[[Path, Any], None],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec.data_path.parent.mkdir(parents=True, exist_ok=True)
    spec.meta_path.parent.mkdir(parents=True, exist_ok=True)
    writer(spec.data_path, data)

    metadata = {
        "schema_version": 1,
        "artifact": spec.name,
        "fingerprint": spec.fingerprint,
        "fingerprint_sha256": fingerprint_value(spec.fingerprint),
        "data_path": str(spec.data_path.name),
        "data_sha256": sha256_file(spec.data_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        metadata.update(extra)

    write_json(spec.meta_path, metadata)
    return metadata


def load_artifact(
    spec: ArtifactSpec,
    reader: Callable[[Path], Any],
) -> Any:
    require_current(spec)
    return reader(spec.data_path)
