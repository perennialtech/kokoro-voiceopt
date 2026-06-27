from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import TextConfig

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


def split_transcript_sentences(transcript: str) -> list[str]:
    transcript = re.sub(r"\s+", " ", transcript.strip())
    if not transcript:
        raise ValueError("target_transcript must be non-empty")

    parts = re.split(r"(?<=[.!?。！？])\s+", transcript)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences or [transcript]


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

    return [c for c in chunks if c]


def merge_sentences(sentences: list[str], min_chars: int, max_chars: int) -> list[str]:
    merged: list[str] = []
    current = ""

    for sentence in sentences:
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

    return [m.strip() for m in merged if m.strip()]


def load_validation_texts(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def build_text_plan(config: TextConfig) -> TextPlan:
    if not config.target_transcript or not config.target_transcript.strip():
        raise ValueError("target_transcript must be non-empty")

    sentences = split_transcript_sentences(config.target_transcript)
    optimization_texts = merge_sentences(
        sentences,
        min_chars=config.min_text_chars,
        max_chars=config.max_text_chars,
    )[: config.max_optimization_texts]

    if not optimization_texts:
        raise ValueError(
            "No optimization texts could be created from target_transcript"
        )

    if config.validation_texts_path is not None:
        validation_texts = load_validation_texts(config.validation_texts_path)
    else:
        validation_texts = list(DEFAULT_VALIDATION_TEXTS)

    optimization_set = {t.strip() for t in optimization_texts}
    validation_texts = [
        t for t in validation_texts if t.strip() and t.strip() not in optimization_set
    ]

    if not validation_texts:
        validation_texts = list(DEFAULT_VALIDATION_TEXTS)

    validation_texts = merge_sentences(
        validation_texts,
        min_chars=config.min_text_chars,
        max_chars=config.max_text_chars,
    )[: config.max_validation_texts]

    return TextPlan(
        optimization_texts=optimization_texts,
        validation_texts=validation_texts,
    )
