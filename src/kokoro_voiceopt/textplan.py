from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import hash_file, read_jsonl, stable_hash_json
from .data import normalize_text

DEFAULT_VALIDATION_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Please call Stella and ask her to bring these things with her from the store.",
    "We were away a year ago, but now we are here again.",
    "The blue notebook was beside the small wooden chair.",
    "Every voice has its own rhythm, color, and quiet little habits.",
]


@dataclass
class TextPlan:
    optimization_texts: list[str]
    validation_texts: list[str]
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "optimization_texts": self.optimization_texts,
            "validation_texts": self.validation_texts,
            "metadata": self.metadata,
        }


def split_transcript_sentences(transcript: str) -> list[str]:
    transcript = re.sub(r"\s+", " ", transcript.strip())
    if not transcript:
        return []

    parts = re.split(r"(?<=[.!?。！？])\s+", transcript)
    return [p.strip() for p in parts if p.strip()] or [transcript]


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    chunks: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word]).strip()
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def merge_sentences(sentences: list[str], min_chars: int, max_chars: int) -> list[str]:
    merged: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = normalize_text(sentence)
        if not sentence:
            continue

        if not current:
            current = sentence
            continue

        candidate = f"{current} {sentence}".strip()
        if len(current) < min_chars or len(candidate) <= max_chars:
            current = candidate
        else:
            merged.extend(_split_long_text(current, max_chars))
            current = sentence

    if current:
        merged.extend(_split_long_text(current, max_chars))

    return [item.strip() for item in merged if item.strip()]


def load_validation_texts(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [
        normalize_text(line)
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def build_text_plan(run) -> TextPlan:
    rows = sorted(read_jsonl(run.paths.target_manifest), key=lambda row: row["id"])
    target_text = " ".join(normalize_text(row["text_normalized"]) for row in rows)
    sentences = split_transcript_sentences(target_text)

    optimization_texts = merge_sentences(
        sentences,
        min_chars=run.text.min_text_chars,
        max_chars=run.text.max_text_chars,
    )[: run.text.max_optimization_texts]

    if not optimization_texts:
        raise ValueError(
            "No optimization texts could be built from prepared target data"
        )

    if run.text.validation_texts_path is not None:
        validation_source = load_validation_texts(run.text.validation_texts_path)
    else:
        validation_source = [normalize_text(text) for text in DEFAULT_VALIDATION_TEXTS]

    optimization_set = {text.strip() for text in optimization_texts}
    validation_source = [
        text
        for text in validation_source
        if text.strip() and text.strip() not in optimization_set
    ]
    if not validation_source:
        validation_source = [normalize_text(text) for text in DEFAULT_VALIDATION_TEXTS]

    validation_texts = merge_sentences(
        validation_source,
        min_chars=run.text.min_text_chars,
        max_chars=run.text.max_text_chars,
    )[: run.text.max_validation_texts]

    metadata = {
        "schema_version": 1,
        "target_manifest_sha256": hash_file(run.paths.target_manifest),
        "text_config_sha256": stable_hash_json(run.text),
    }

    return TextPlan(
        optimization_texts=optimization_texts,
        validation_texts=validation_texts,
        metadata=metadata,
    )


def build_and_save_text_plan(run) -> TextPlan:
    plan = build_text_plan(run)
    run.paths.text_plan.parent.mkdir(parents=True, exist_ok=True)
    with open(run.paths.text_plan, "w", encoding="utf-8") as file:
        json.dump(plan.to_dict(), file, indent=2, ensure_ascii=False)
    return plan


def load_text_plan(run) -> TextPlan:
    if not run.paths.text_plan.exists():
        raise FileNotFoundError(
            f"Missing text plan: {run.paths.text_plan}. Run `kokoro-voiceopt prepare`."
        )

    with open(run.paths.text_plan, encoding="utf-8") as file:
        data = json.load(file)

    expected = {
        "schema_version": 1,
        "target_manifest_sha256": hash_file(run.paths.target_manifest),
        "text_config_sha256": stable_hash_json(run.text),
    }

    metadata = dict(data.get("metadata", {}))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"text_plan metadata mismatch for {key}; rerun prepare")

    return TextPlan(
        optimization_texts=list(data["optimization_texts"]),
        validation_texts=list(data["validation_texts"]),
        metadata=metadata,
    )
