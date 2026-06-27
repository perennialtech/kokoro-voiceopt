from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serde import fingerprint, read_jsonl, sha256_file, write_json

URL_OR_EMAIL_RE = re.compile(r"https?://\S+|\S+@\S+")
SPOKEN_FORM_RE = re.compile(r"[\d$€£¥@]")

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
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimization_texts": self.optimization_texts,
            "validation_texts": self.validation_texts,
            "metadata": self.metadata,
        }


def read_transcripts(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []

    with open(path, encoding="utf-8") as file:
        for line_no, raw in enumerate(file, 1):
            line = raw.strip()
            if not line:
                continue

            if path.suffix.lower() == ".jsonl":
                obj = json.loads(line)
                audio = obj.get("audio") or obj.get("path")
                text = obj.get("text")
                start_s = obj.get("start_s")
                end_s = obj.get("end_s")
                row_id = obj.get("id")
            else:
                parts = line.split("|", 1)
                if len(parts) != 2:
                    raise ValueError(f"{path}:{line_no}: expected audio|text")
                audio, text = parts
                start_s = None
                end_s = None
                row_id = None

            audio = str(audio).strip() if audio is not None else ""
            text = str(text).strip() if text is not None else ""
            if audio.lower() in {"audio", "path", "filename"}:
                continue
            if not audio or not text:
                raise ValueError(f"{path}:{line_no}: empty audio or text")

            rows.append(
                {
                    "id": str(row_id).strip() if row_id is not None else None,
                    "audio_source": audio,
                    "text_original": text,
                    "start_s": start_s,
                    "end_s": end_s,
                }
            )

    if not rows:
        raise ValueError(f"No transcript rows found in {path}")

    return rows


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\[[^\]]*\]|\([^\)]*\)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def contains_url_or_email(text: str) -> bool:
    return bool(URL_OR_EMAIL_RE.search(text))


def is_spoken_form(text: str) -> bool:
    return not contains_url_or_email(text) and not SPOKEN_FORM_RE.search(text)


def split_transcript_sentences(transcript: str) -> list[str]:
    transcript = re.sub(r"\s+", " ", transcript.strip())
    if not transcript:
        return []

    parts = re.split(r"(?<=[.!?。！？])\s+", transcript)
    return [part.strip() for part in parts if part.strip()] or [transcript]


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


def text_plan_fingerprint(ctx) -> dict[str, Any]:
    validation_path = ctx.cfg.text.validation_texts_path
    return {
        "target_manifest_sha256": sha256_file(ctx.paths.data("manifests/target.jsonl")),
        "text_config_sha256": fingerprint(ctx.cfg.text),
        "validation_texts_file_sha256": (
            sha256_file(validation_path)
            if validation_path is not None and Path(validation_path).exists()
            else None
        ),
    }


def build_text_plan(ctx) -> TextPlan:
    rows = sorted(
        read_jsonl(ctx.paths.data("manifests/target.jsonl")), key=lambda row: row["id"]
    )
    target_text = " ".join(normalize_text(row["text_normalized"]) for row in rows)
    sentences = split_transcript_sentences(target_text)

    optimization_texts = merge_sentences(
        sentences,
        min_chars=ctx.cfg.text.min_text_chars,
        max_chars=ctx.cfg.text.max_text_chars,
    )[: ctx.cfg.text.max_optimization_texts]

    if not optimization_texts:
        raise ValueError(
            "No optimization texts could be built from prepared target data"
        )

    if ctx.cfg.text.validation_texts_path is not None:
        validation_source = load_validation_texts(ctx.cfg.text.validation_texts_path)
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
        min_chars=ctx.cfg.text.min_text_chars,
        max_chars=ctx.cfg.text.max_text_chars,
    )[: ctx.cfg.text.max_validation_texts]

    metadata = text_plan_fingerprint(ctx)

    return TextPlan(
        optimization_texts=optimization_texts,
        validation_texts=validation_texts,
        metadata=metadata,
    )


def save_text_plan(ctx, plan: TextPlan) -> None:
    write_json(ctx.paths.data("text_plan.json"), plan.to_dict())


def build_and_save_text_plan(ctx) -> TextPlan:
    plan = build_text_plan(ctx)
    save_text_plan(ctx, plan)
    return plan


def load_text_plan(ctx) -> TextPlan:
    return build_and_save_text_plan(ctx)
