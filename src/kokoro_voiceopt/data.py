from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from .config import hash_file, read_jsonl, write_jsonl

URL_OR_EMAIL_RE = re.compile(r"https?://\S+|\S+@\S+")
SPOKEN_FORM_RE = re.compile(r"[\d$€£¥@]")


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


def reject(
    reason: str, row: dict[str, Any], rejected: list[dict[str, Any]], detail=None
):
    item = dict(row)
    item["reject_reason"] = reason
    if detail is not None:
        item["reject_detail"] = str(detail)
    rejected.append(item)


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    waveform = torch.from_numpy(audio)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    elif waveform.ndim != 1:
        raise ValueError(f"Unsupported audio shape: {tuple(waveform.shape)}")
    return waveform.to(torch.float32).contiguous(), int(sr)


def slice_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    start_s,
    end_s,
) -> tuple[torch.Tensor, float | None, float | None]:
    if start_s is None and end_s is None:
        return waveform, None, None
    if start_s is None or end_s is None:
        raise ValueError("start_s and end_s must be provided together")

    start = float(start_s)
    end = float(end_s)
    if start < 0 or end <= start:
        raise ValueError(f"bad timestamp range: start_s={start}, end_s={end}")

    start_sample = int(round(start * sample_rate))
    end_sample = int(round(end * sample_rate))
    if start_sample < 0 or end_sample > waveform.numel() or end_sample <= start_sample:
        raise ValueError(f"timestamp range outside audio: start_s={start}, end_s={end}")

    return waveform[start_sample:end_sample].contiguous(), start, end


def resample_audio(waveform: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return waveform.to(torch.float32).contiguous()
    return (
        torchaudio.functional.resample(
            waveform.to(torch.float32).unsqueeze(0), orig_sr, new_sr
        )
        .squeeze(0)
        .contiguous()
    )


def remove_dc_offset(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.numel() == 0:
        return waveform
    return (waveform - waveform.mean()).contiguous()


def trim_edges(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float,
    pad_ms: int,
) -> tuple[torch.Tensor, bool]:
    if waveform.numel() == 0:
        return waveform, False

    mask = waveform.abs() > threshold
    if not mask.any():
        return waveform, True

    indices = mask.nonzero(as_tuple=False).flatten()
    pad = int(sample_rate * pad_ms / 1000)
    start = max(0, int(indices[0]) - pad)
    end = min(waveform.numel(), int(indices[-1]) + pad + 1)
    return waveform[start:end].contiguous(), False


def apply_fades(waveform: torch.Tensor, sample_rate: int, fade_ms: int) -> torch.Tensor:
    n = min(int(sample_rate * fade_ms / 1000), waveform.numel() // 2)
    if n <= 1:
        return waveform.contiguous()

    waveform = waveform.clone()
    waveform[:n] *= torch.linspace(0, 1, n, dtype=waveform.dtype)
    waveform[-n:] *= torch.linspace(1, 0, n, dtype=waveform.dtype)
    return waveform.contiguous()


def audio_stats(
    waveform: torch.Tensor, sample_rate: int, hard_end_threshold: float
) -> dict:
    if waveform.numel() == 0:
        return {
            "duration_s": 0.0,
            "peak": 0.0,
            "rms": 0.0,
            "clip_ratio": 1.0,
            "hard_end": True,
        }

    peak = float(waveform.abs().max().item())
    edge = max(1, min(int(0.02 * sample_rate), waveform.numel() // 10))
    return {
        "duration_s": round(waveform.numel() / sample_rate, 4),
        "peak": round(peak, 6),
        "rms": round(float(torch.sqrt(torch.mean(waveform.square())).item()), 6),
        "clip_ratio": round(float((waveform.abs() >= 0.999).float().mean().item()), 8),
        "hard_end": bool(waveform[-edge:].abs().max().item() > hard_end_threshold),
    }


def save_wav(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.cpu().numpy(), sample_rate, subtype="PCM_16")


def prepare_target_dataset(
    run,
    audio_dir: str | Path,
    transcripts: str | Path,
    force: bool = False,
) -> dict:
    out = run.paths.data_dir
    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit(f"{out} already exists, use --force to recreate it")
        shutil.rmtree(out)

    run.write_resolved_config()
    run.paths.audio_dir.mkdir(parents=True, exist_ok=True)
    run.paths.manifest_dir.mkdir(parents=True, exist_ok=True)

    audio_root = Path(audio_dir)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, row in enumerate(
        tqdm(read_transcripts(transcripts), desc="Preparing target")
    ):
        row_id = row["id"] or f"{idx:06d}"
        base = {
            "id": row_id,
            "audio_source": row["audio_source"],
            "text_original": row["text_original"],
            "speaker": run.target.id,
            "lang_code": run.target.lang_code,
        }

        source = audio_root / row["audio_source"]
        if not source.exists():
            reject("missing_audio", base, rejected)
            continue

        text_normalized = normalize_text(row["text_original"])
        if not text_normalized:
            reject("empty_text", base, rejected)
            continue

        if run.data.require_spoken_form and not is_spoken_form(text_normalized):
            reject(
                "text_not_spoken_form",
                {**base, "text_normalized": text_normalized},
                rejected,
            )
            continue

        try:
            waveform, source_sr = load_audio(source)
            waveform, source_start_s, source_end_s = slice_audio(
                waveform, source_sr, row["start_s"], row["end_s"]
            )
        except ValueError as exc:
            reject("bad_timestamp", base, rejected, detail=exc)
            continue
        except Exception as exc:
            reject("audio_decode_failed", base, rejected, detail=exc)
            continue

        if waveform.numel() == 0:
            reject("empty_audio", base, rejected)
            continue

        waveform = resample_audio(waveform, source_sr, run.audio.target_sample_rate)

        if run.audio.dc_remove:
            waveform = remove_dc_offset(waveform)

        if run.audio.trim_edges:
            waveform, all_silence = trim_edges(
                waveform,
                run.audio.target_sample_rate,
                run.audio.trim_threshold,
                run.audio.trim_pad_ms,
            )
            if all_silence:
                reject("all_silence", base, rejected)
                continue

        waveform = apply_fades(
            waveform, run.audio.target_sample_rate, run.audio.fade_ms
        )

        stats = audio_stats(
            waveform,
            run.audio.target_sample_rate,
            run.data.hard_end_threshold,
        )
        if stats["clip_ratio"] > run.data.max_clip_ratio:
            reject("clipping", {**base, **stats}, rejected)
            continue

        peak = waveform.abs().max().item() if waveform.numel() else 0.0
        if peak > run.audio.max_peak:
            waveform = waveform / peak * run.audio.max_peak
        elif run.audio.peak_normalize and peak > 1e-8:
            waveform = waveform / peak * run.audio.max_peak

        stats = audio_stats(
            waveform,
            run.audio.target_sample_rate,
            run.data.hard_end_threshold,
        )
        if stats["duration_s"] < run.data.min_duration_s:
            reject("too_short", {**base, **stats}, rejected)
            continue
        if stats["duration_s"] > run.data.max_duration_s:
            reject("too_long", {**base, **stats}, rejected)
            continue
        if stats["hard_end"]:
            reject("hard_end", {**base, **stats}, rejected)
            continue

        filename = f"{row_id}.wav"
        audio_path = run.paths.audio_dir / filename

        try:
            save_wav(audio_path, waveform, run.audio.target_sample_rate)
        except Exception as exc:
            reject("save_failed", {**base, **stats}, rejected, detail=exc)
            continue

        accepted.append(
            {
                "id": row_id,
                "audio": str(audio_path.relative_to(run.paths.data_dir)),
                "source_audio": row["audio_source"],
                "source_start_s": source_start_s,
                "source_end_s": source_end_s,
                "text_original": row["text_original"],
                "text_normalized": text_normalized,
                "speaker": run.target.id,
                "lang_code": run.target.lang_code,
                "duration_s": stats["duration_s"],
                "n_chars": len(text_normalized),
                "peak": stats["peak"],
                "rms": stats["rms"],
                "clip_ratio": stats["clip_ratio"],
                "audio_sha256": hash_file(audio_path),
            }
        )

    if not accepted:
        write_jsonl(run.paths.rejected_manifest, rejected)
        raise SystemExit("No usable target clips were prepared")

    if len(accepted) > run.data.max_target_clips:
        accepted = sorted(
            accepted, key=lambda r: (r["duration_s"], r["id"]), reverse=True
        )[: run.data.max_target_clips]

    write_jsonl(run.paths.target_manifest, accepted)
    write_jsonl(run.paths.rejected_manifest, rejected)

    durations = [float(row["duration_s"]) for row in accepted]
    report = {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "duration_s": round(sum(durations), 4),
        "duration_h": round(sum(durations) / 3600, 4),
        "min_duration_s": min(durations),
        "max_duration_s": max(durations),
        "mean_duration_s": round(sum(durations) / len(durations), 4),
        "sample_rate": run.audio.target_sample_rate,
        "reject_reasons": dict(Counter(row["reject_reason"] for row in rejected)),
        "total_chars": sum(int(row["n_chars"]) for row in accepted),
        "unique_source_files": len({row["source_audio"] for row in accepted}),
    }

    run.paths.dataset_report.parent.mkdir(parents=True, exist_ok=True)
    with open(run.paths.dataset_report, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    from .textplan import build_and_save_text_plan

    build_and_save_text_plan(run)

    print(json.dumps(report, indent=2))
    return report


def check_target_dataset(run) -> None:
    errors: list[str] = []

    if not run.paths.target_manifest.exists():
        raise SystemExit(f"missing target manifest: {run.paths.target_manifest}")

    rows = read_jsonl(run.paths.target_manifest)
    if not rows:
        errors.append(f"empty target manifest: {run.paths.target_manifest}")

    seen_ids = set()
    total_duration = 0.0
    durations: list[float] = []

    for row in rows:
        row_id = row.get("id")
        if row_id in seen_ids:
            errors.append(f"duplicate id: {row_id}")
        seen_ids.add(row_id)

        if row.get("speaker") != run.target.id:
            errors.append(f"{row_id}: speaker mismatch")
        if row.get("lang_code") != run.target.lang_code:
            errors.append(f"{row_id}: lang_code mismatch")

        text = str(row.get("text_normalized", "")).strip()
        if not text:
            errors.append(f"{row_id}: empty text_normalized")
        if run.data.require_spoken_form and text and not is_spoken_form(text):
            errors.append(f"{row_id}: text_normalized is not spoken-form compatible")

        audio = run.paths.data_dir / row.get("audio", "")
        if not audio.exists():
            errors.append(f"{row_id}: missing audio: {audio}")
            continue

        try:
            info = sf.info(audio)
        except Exception as exc:
            errors.append(f"{row_id}: audio read failed: {exc}")
            continue

        if info.samplerate != run.audio.target_sample_rate:
            errors.append(
                f"{row_id}: expected {run.audio.target_sample_rate} Hz, got {info.samplerate}"
            )
        if info.channels != 1:
            errors.append(f"{row_id}: expected mono, got {info.channels} channels")

        actual_hash = hash_file(audio)
        if row.get("audio_sha256") != actual_hash:
            errors.append(f"{row_id}: audio_sha256 mismatch")

        duration = info.frames / float(info.samplerate)
        manifest_duration = float(row.get("duration_s", 0.0))
        if abs(duration - manifest_duration) > 0.05:
            errors.append(f"{row_id}: duration mismatch")

        if duration < run.data.min_duration_s:
            errors.append(f"{row_id}: too short")
        if duration > run.data.max_duration_s:
            errors.append(f"{row_id}: too long")

        total_duration += duration
        durations.append(duration)

    if len(rows) < run.data.min_target_clips:
        errors.append(
            f"target clip count {len(rows)} below minimum {run.data.min_target_clips}"
        )
    if total_duration < run.data.min_total_duration_s:
        errors.append(
            f"total duration {total_duration:.2f}s below minimum "
            f"{run.data.min_total_duration_s:.2f}s"
        )

    print(f"Rows: {len(rows):,}")
    print(f"Duration: {total_duration:.2f}s")
    if durations:
        print(f"Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
    print(f"Speaker: {run.target.id}")
    print(f"Language code: {run.target.lang_code}")
    if run.paths.rejected_manifest.exists():
        print(f"Rejected rows: {len(read_jsonl(run.paths.rejected_manifest)):,}")

    if errors:
        print("\nErrors:")
        for error in errors[:100]:
            print(f"  {error}")
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100} more")
        raise SystemExit(1)

    print("Target dataset check passed")
