from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .serde import fingerprint as fingerprint_value
from .serde import load_pt, read_json, save_pt, sha256_file, write_json


class StaleArtifactError(RuntimeError):
    """Raised when an artifact exists but does not match the current fingerprint."""


@dataclass(frozen=True)
class Artifact:
    name: str
    data_path: Path
    meta_path: Path
    fingerprint: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing artifact metadata: {self.meta_path}")
        data = read_json(self.meta_path)
        if not isinstance(data, dict):
            raise ValueError(
                f"Artifact metadata must be a JSON object: {self.meta_path}"
            )
        return data

    def stale_reasons(self) -> list[str]:
        reasons: list[str] = []

        if not self.data_path.exists():
            reasons.append(f"missing data: {self.data_path}")
        if not self.meta_path.exists():
            reasons.append(f"missing metadata: {self.meta_path}")
            return reasons

        try:
            metadata = self.metadata()
        except Exception as exc:
            return [f"metadata read failed: {exc}"]

        if metadata.get("schema_version") != 1:
            reasons.append("unsupported metadata schema_version")
        if metadata.get("artifact") != self.name:
            reasons.append(
                f"artifact name mismatch: expected {self.name}, got {metadata.get('artifact')}"
            )
        if metadata.get("fingerprint") != self.fingerprint:
            reasons.append("fingerprint mismatch")

        if self.data_path.exists() and metadata.get("data_sha256"):
            actual = sha256_file(self.data_path)
            if metadata.get("data_sha256") != actual:
                reasons.append("data_sha256 mismatch")

        return reasons

    def is_current(self) -> bool:
        return not self.stale_reasons()

    def require_current(self) -> dict[str, Any]:
        reasons = self.stale_reasons()
        if reasons:
            raise StaleArtifactError(
                f"Artifact is stale or missing: {self.name}\n  - "
                + "\n  - ".join(reasons)
            )
        return self.metadata()

    def save(
        self,
        data: Any,
        writer: Callable[[Path, Any], None],
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        writer(self.data_path, data)

        metadata = {
            "schema_version": 1,
            "artifact": self.name,
            "fingerprint": self.fingerprint,
            "fingerprint_sha256": fingerprint_value(self.fingerprint),
            "data_path": str(self.data_path.name),
            "data_sha256": sha256_file(self.data_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            metadata.update(extra)

        write_json(self.meta_path, metadata)
        return metadata

    def load(self, reader: Callable[[Path], Any]) -> Any:
        self.require_current()
        return reader(self.data_path)

    def save_json(
        self, data: Any, *, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.save(data, write_json, extra=extra)

    def load_json(self) -> Any:
        return self.load(read_json)

    def save_pt(
        self, data: Any, *, extra: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return self.save(data, save_pt, extra=extra)

    def load_pt(self) -> Any:
        return self.load(load_pt)


class ArtifactStore:
    def spec(
        self,
        name: str,
        *,
        data: str | Path,
        meta: str | Path,
        fingerprint: dict[str, Any],
    ) -> Artifact:
        return Artifact(
            name=name,
            data_path=Path(data),
            meta_path=Path(meta),
            fingerprint=fingerprint,
        )
