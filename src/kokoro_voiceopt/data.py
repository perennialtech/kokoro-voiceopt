from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf
from tqdm import tqdm

from .audio import ClipReject, prepare_target_clip, save_wave
from .serde import read_jsonl, sha256_file, write_jsonl
from .transcript import (build_and_save_text_plan, is_spoken_form,
                         normalize_text, read_transcripts)


def reject(
    reason: str,
    row: dict[str, Any],
    rejected: list[dict[str, Any]],
    detail=None,
) -> None:
    item = dict(row)
    item["reject_reason"] = reason
    if detail is not None:
        item["reject_detail"] = str(detail)
    rejected.append(item)


def prepare_target_dataset(
    ctx,
    audio_dir: str | Path,
    transcripts: str | Path,
    force: bool = False,
) -> dict:
    out = ctx.paths.data_dir
    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit(f"{out} already exists, use --force to recreate it")
        shutil.rmtree(out)

    ctx.write_resolved_config()
    ctx.paths.data("audio").mkdir(parents=True, exist_ok=True)
    ctx.paths.data("manifests").mkdir(parents=True, exist_ok=True)

    audio_root = Path(audio_dir)
    accepted_pending: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, row in enumerate(
        tqdm(read_transcripts(transcripts), desc="Preparing target")
    ):
        row_id = row["id"] or f"{idx:06d}"
        base = {
            "id": row_id,
            "audio_source": row["audio_source"],
            "text_original": row["text_original"],
            "speaker": ctx.cfg.target.id,
            "lang_code": ctx.cfg.target.lang_code,
        }

        source = audio_root / row["audio_source"]
        if not source.exists():
            reject("missing_audio", base, rejected)
            continue

        text_normalized = normalize_text(row["text_original"])
        if not text_normalized:
            reject("empty_text", base, rejected)
            continue

        if ctx.cfg.data.require_spoken_form and not is_spoken_form(text_normalized):
            reject(
                "text_not_spoken_form",
                {**base, "text_normalized": text_normalized},
                rejected,
            )
            continue

        try:
            prepared = prepare_target_clip(
                source,
                row["start_s"],
                row["end_s"],
                ctx.cfg.audio,
                ctx.cfg.data,
            )
        except ClipReject as exc:
            reject(
                exc.reason,
                {**base, **exc.metrics},
                rejected,
                detail=exc.detail,
            )
            continue

        metrics = prepared.metrics.to_manifest()
        accepted_pending.append(
            {
                "id": row_id,
                "source_audio": row["audio_source"],
                "source_start_s": prepared.source_start_s,
                "source_end_s": prepared.source_end_s,
                "text_original": row["text_original"],
                "text_normalized": text_normalized,
                "speaker": ctx.cfg.target.id,
                "lang_code": ctx.cfg.target.lang_code,
                "duration_s": round(metrics["duration_s"], 4),
                "n_chars": len(text_normalized),
                "peak": metrics["peak"],
                "rms": metrics["rms"],
                "clip_ratio": metrics["clip_ratio"],
                "_wave": prepared.wave,
            }
        )

    if not accepted_pending:
        write_jsonl(ctx.paths.data("manifests/rejected.jsonl"), rejected)
        raise SystemExit("No usable target clips were prepared")

    if len(accepted_pending) > ctx.cfg.data.max_target_clips:
        accepted_pending = sorted(
            accepted_pending,
            key=lambda r: (r["duration_s"], r["id"]),
            reverse=True,
        )[: ctx.cfg.data.max_target_clips]

    accepted: list[dict[str, Any]] = []
    for item in accepted_pending:
        row_id = item["id"]
        audio_path = ctx.paths.data("audio", f"{row_id}.wav")
        wave = item.pop("_wave")

        try:
            save_wave(audio_path, wave)
        except ClipReject as exc:
            reject(
                exc.reason,
                {
                    "id": row_id,
                    "audio_source": item["source_audio"],
                    "text_original": item["text_original"],
                    "speaker": ctx.cfg.target.id,
                    "lang_code": ctx.cfg.target.lang_code,
                    **exc.metrics,
                },
                rejected,
                detail=exc.detail,
            )
            continue

        accepted.append(
            {
                **item,
                "audio": str(audio_path.relative_to(ctx.paths.data_dir)),
                "audio_sha256": sha256_file(audio_path),
            }
        )

    if not accepted:
        write_jsonl(ctx.paths.data("manifests/rejected.jsonl"), rejected)
        raise SystemExit("No usable target clips were saved")

    write_jsonl(ctx.paths.data("manifests/target.jsonl"), accepted)
    write_jsonl(ctx.paths.data("manifests/rejected.jsonl"), rejected)

    durations = [float(row["duration_s"]) for row in accepted]
    report = {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "duration_s": round(sum(durations), 4),
        "duration_h": round(sum(durations) / 3600, 4),
        "min_duration_s": min(durations),
        "max_duration_s": max(durations),
        "mean_duration_s": round(sum(durations) / len(durations), 4),
        "sample_rate": ctx.cfg.audio.target_sample_rate,
        "reject_reasons": dict(Counter(row["reject_reason"] for row in rejected)),
        "total_chars": sum(int(row["n_chars"]) for row in accepted),
        "unique_source_files": len({row["source_audio"] for row in accepted}),
    }

    ctx.paths.data("report.json").parent.mkdir(parents=True, exist_ok=True)
    with open(ctx.paths.data("report.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    build_and_save_text_plan(ctx)

    print(json.dumps(report, indent=2))
    return report


def check_target_dataset(ctx) -> None:
    errors: list[str] = []
    target_manifest = ctx.paths.data("manifests/target.jsonl")
    rejected_manifest = ctx.paths.data("manifests/rejected.jsonl")

    if not target_manifest.exists():
        raise SystemExit(f"missing target manifest: {target_manifest}")

    rows = read_jsonl(target_manifest)
    if not rows:
        errors.append(f"empty target manifest: {target_manifest}")

    seen_ids = set()
    total_duration = 0.0
    durations: list[float] = []

    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            errors.append(f"duplicate id: {row_id}")
        seen_ids.add(row_id)

        if row.get("speaker") != ctx.cfg.target.id:
            errors.append(f"{row_id}: speaker mismatch")
        if row.get("lang_code") != ctx.cfg.target.lang_code:
            errors.append(f"{row_id}: lang_code mismatch")

        text = str(row.get("text_normalized", "")).strip()
        if not text:
            errors.append(f"{row_id}: empty text_normalized")
        if ctx.cfg.data.require_spoken_form and text and not is_spoken_form(text):
            errors.append(f"{row_id}: text_normalized is not spoken-form compatible")

        audio = ctx.paths.data(row.get("audio", ""))
        if not audio.exists():
            errors.append(f"{row_id}: missing audio: {audio}")
            continue

        try:
            info = sf.info(audio)
        except Exception as exc:
            errors.append(f"{row_id}: audio read failed: {exc}")
            continue

        if info.samplerate != ctx.cfg.audio.target_sample_rate:
            errors.append(
                f"{row_id}: expected {ctx.cfg.audio.target_sample_rate} Hz, got {info.samplerate}"
            )
        if info.channels != 1:
            errors.append(f"{row_id}: expected mono, got {info.channels} channels")

        actual_hash = sha256_file(audio)
        if row.get("audio_sha256") != actual_hash:
            errors.append(f"{row_id}: audio_sha256 mismatch")

        duration = info.frames / float(info.samplerate)
        manifest_duration = float(row.get("duration_s", 0.0))
        if abs(duration - manifest_duration) > 0.05:
            errors.append(f"{row_id}: duration mismatch")

        if duration < ctx.cfg.data.min_duration_s:
            errors.append(f"{row_id}: too short")
        if duration > ctx.cfg.data.max_duration_s:
            errors.append(f"{row_id}: too long")

        total_duration += duration
        durations.append(duration)

    if len(rows) < ctx.cfg.data.min_target_clips:
        errors.append(
            f"target clip count {len(rows)} below minimum {ctx.cfg.data.min_target_clips}"
        )
    if total_duration < ctx.cfg.data.min_total_duration_s:
        errors.append(
            f"total duration {total_duration:.2f}s below minimum "
            f"{ctx.cfg.data.min_total_duration_s:.2f}s"
        )

    print(f"Rows: {len(rows):,}")
    print(f"Duration: {total_duration:.2f}s")
    if durations:
        print(f"Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
    print(f"Speaker: {ctx.cfg.target.id}")
    print(f"Language code: {ctx.cfg.target.lang_code}")
    if rejected_manifest.exists():
        print(f"Rejected rows: {len(read_jsonl(rejected_manifest)):,}")

    if errors:
        print("\nErrors:")
        for error in errors[:100]:
            print(f"  {error}")
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100} more")
        raise SystemExit(1)

    print("Target dataset check passed")
