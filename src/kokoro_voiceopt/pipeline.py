from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from .config import Context
from .corpus import VoiceCorpus
from .manifold import VoiceManifold, manifold_fingerprint
from .objective import VoiceObjective
from .profile import load_target_speaker_profile
from .search import (Candidate, SearchResult, run_baseline_scan,
                     run_blend_stage, run_latent_stage, validate_candidates)
from .serde import (fingerprint, jsonable, load_pt, read_json, read_jsonl,
                    sha256_file, write_json)
from .services import make_speaker_encoder, make_synthesizer
from .transcript import build_and_save_text_plan, build_text_plan
from .voice import save_voice, voice_hash


@dataclass
class VoiceOptimizationResult:
    output_dir: Path
    export_voice_path: Path
    export_meta_path: Path
    best_validation_loss: float
    best_optimization_loss: float
    baseline_best_similarity: float
    blend_best_similarity: float
    latent_best_similarity: float
    selected_stage: str
    selected_candidate_id: str
    selected_voice_hash: str
    final_voice_hash: str
    best_optimization_voice_hash: str


class RunWriter:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.history_path = ctx.paths.optimize("history.jsonl")
        self.candidates_path = ctx.paths.optimize("candidates.jsonl")
        self.run_path = ctx.paths.optimize("run.json")
        self.voices_dir = ctx.paths.optimize("voices")
        self.records: list[dict[str, Any]] = []

        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.candidates_path.parent.mkdir(parents=True, exist_ok=True)

        self.history_path.write_text("", encoding="utf-8")
        self.candidates_path.write_text("", encoding="utf-8")

    def save_voice_once(self, candidate: Candidate) -> Path:
        path = self.voices_dir / f"{candidate.voice_hash}.pt"
        if not path.exists():
            save_voice(path, candidate.voice)
        return path

    def append_history(self, row: dict[str, Any]) -> None:
        with open(self.history_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")

    def append_history_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.append_history(row)

    def write_candidate(
        self,
        candidate: Candidate,
        *,
        role: str,
        eval_set: str,
    ) -> dict[str, Any]:
        voice_path = self.save_voice_once(candidate)
        record = {
            "role": role,
            "eval_set": eval_set,
            "voice_path": str(voice_path.relative_to(self.ctx.paths.optimize_dir)),
            **candidate.to_dict(),
        }
        with open(self.candidates_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
        self.records.append(record)
        return record

    def write_run_summary(self, summary: dict[str, Any]) -> None:
        write_json(self.run_path, summary)


class VoiceOptimizationPipeline:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def run(self) -> VoiceOptimizationResult:
        self._set_seed(self.ctx.cfg.search.seed)
        self._prepare_dirs()
        self.ctx.write_resolved_config()

        self.ctx.require_manifests()
        self.ctx.require_profile()
        self.ctx.require_corpus()

        text_plan = build_and_save_text_plan(self.ctx)
        target_profile = load_target_speaker_profile(self.ctx)
        corpus = VoiceCorpus.load(self.ctx)

        speaker = make_speaker_encoder(self.ctx)
        synthesizer = make_synthesizer(self.ctx)
        objective = VoiceObjective(
            synthesizer=synthesizer,
            speaker_encoder=speaker,
            target_profile=target_profile,
            audio_config=self.ctx.cfg.audio,
            objective_config=self.ctx.cfg.objective,
        )

        artifact_metadata = {
            "schema_version": 1,
            "text_plan_sha256": sha256_file(self.ctx.paths.data("text_plan.json")),
            "optimization_texts_sha256": fingerprint(text_plan.optimization_texts),
            "validation_texts_sha256": fingerprint(text_plan.validation_texts),
            "objective_config_sha256": fingerprint(self.ctx.cfg.objective),
            "search_config_sha256": fingerprint(self.ctx.cfg.search),
            "corpus_manifest_sha256": sha256_file(
                self.ctx.paths.corpus("corpus_manifest.json")
            ),
            "corpus_sha256": corpus.corpus_sha256,
            "target_profile_sha256": sha256_file(
                self.ctx.paths.profile("target_profile.pt")
            ),
        }

        writer = RunWriter(self.ctx)

        baseline = run_baseline_scan(
            corpus=corpus,
            objective=objective,
            texts=text_plan.optimization_texts,
            top_k=self.ctx.cfg.search.top_k_for_blend,
            metadata=artifact_metadata,
        )
        for candidate in baseline.candidates:
            writer.write_candidate(
                candidate,
                role=(
                    "baseline_best"
                    if candidate.candidate_id == baseline.best.candidate_id
                    else "baseline"
                ),
                eval_set="optimization",
            )

        blend = run_blend_stage(
            top_candidates=baseline.top,
            objective=objective,
            texts=text_plan.optimization_texts,
            config=self.ctx.cfg.search,
            metadata=artifact_metadata,
        )
        writer.append_history_rows(blend.history)
        self._write_search_result_candidates(
            writer, blend, prefix="blend", eval_set="optimization"
        )

        manifold = VoiceManifold.fit(
            corpus,
            self.ctx.cfg.manifold,
            metadata=manifold_fingerprint(self.ctx, corpus),
        )
        initial_z = manifold.encode(blend.best.voice)

        latent = run_latent_stage(
            manifold=manifold,
            initial_z=initial_z,
            objective=objective,
            texts=text_plan.optimization_texts,
            config=self.ctx.cfg.search,
            metadata=artifact_metadata,
        )
        writer.append_history_rows(latent.history)
        self._write_search_result_candidates(
            writer, latent, prefix="latent", eval_set="optimization"
        )

        optimization_candidates = self._unique_candidates(
            [
                baseline.best,
                blend.best,
                blend.final,
                *blend.checkpoints,
                latent.best,
                latent.final,
                *latent.checkpoints,
            ]
        )
        best_optimization = min(
            optimization_candidates, key=lambda candidate: candidate.eval.total_loss
        )

        validation = validate_candidates(
            candidates=optimization_candidates,
            objective=objective,
            texts=text_plan.validation_texts,
            metadata=artifact_metadata,
            include_search_penalties=False,
        )
        for candidate in validation.candidates:
            writer.write_candidate(
                candidate,
                role=(
                    "validation_best"
                    if candidate.candidate_id == validation.best.candidate_id
                    else "validation"
                ),
                eval_set="validation",
            )

        best_validation = validation.best
        final_candidate = latent.final
        if final_candidate is None:
            raise RuntimeError("Latent search did not produce a final candidate")

        selected = {
            "default": "best",
            "best": self._selection_record(best_validation),
            "final": self._selection_record(final_candidate),
            "best_optimization": self._selection_record(best_optimization),
        }

        summary = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fingerprints": artifact_metadata,
            "texts": {
                "optimization": text_plan.optimization_texts,
                "validation": text_plan.validation_texts,
                "metadata": text_plan.metadata,
            },
            "manifold": manifold.report(),
            "selected": selected,
            "metrics": {
                "best_validation_loss": best_validation.eval.total_loss,
                "best_optimization_loss": best_optimization.eval.total_loss,
                "baseline_best_similarity": baseline.best.eval.mean_similarity,
                "blend_best_similarity": blend.best.eval.mean_similarity,
                "latent_best_similarity": latent.best.eval.mean_similarity,
            },
            "files": {
                "history": "history.jsonl",
                "candidates": "candidates.jsonl",
                "voices_dir": "voices",
            },
        }
        writer.write_run_summary(summary)

        self._write_export(
            selector="best",
            candidate=best_validation,
            run_summary=summary,
            optimization_texts=text_plan.optimization_texts,
            validation_texts=text_plan.validation_texts,
        )

        return VoiceOptimizationResult(
            output_dir=self.ctx.paths.optimize_dir,
            export_voice_path=self.ctx.paths.export("voice.pt"),
            export_meta_path=self.ctx.paths.export("voice_meta.json"),
            best_validation_loss=best_validation.eval.total_loss,
            best_optimization_loss=best_optimization.eval.total_loss,
            baseline_best_similarity=baseline.best.eval.mean_similarity,
            blend_best_similarity=blend.best.eval.mean_similarity,
            latent_best_similarity=latent.best.eval.mean_similarity,
            selected_stage=best_validation.stage,
            selected_candidate_id=best_validation.candidate_id,
            selected_voice_hash=best_validation.voice_hash,
            final_voice_hash=final_candidate.voice_hash,
            best_optimization_voice_hash=best_optimization.voice_hash,
        )

    def _prepare_dirs(self) -> None:
        if self.ctx.paths.optimize_dir.exists():
            shutil.rmtree(self.ctx.paths.optimize_dir)
        self.ctx.paths.optimize("voices").mkdir(parents=True, exist_ok=True)
        self.ctx.paths.export_dir.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _unique_candidates(self, candidates: list[Candidate | None]) -> list[Candidate]:
        unique: dict[str, Candidate] = {}
        for candidate in candidates:
            if candidate is not None:
                unique[candidate.candidate_id] = candidate
        return list(unique.values())

    def _write_search_result_candidates(
        self,
        writer: RunWriter,
        result: SearchResult,
        *,
        prefix: str,
        eval_set: str,
    ) -> None:
        checkpoint_ids = {candidate.candidate_id for candidate in result.checkpoints}

        for candidate in result.candidates:
            roles: list[str] = []
            if candidate.candidate_id == result.best.candidate_id:
                roles.append(f"{prefix}_best")
            if (
                result.final is not None
                and candidate.candidate_id == result.final.candidate_id
            ):
                roles.append(f"{prefix}_final")
            if candidate.candidate_id in checkpoint_ids:
                roles.append(f"{prefix}_checkpoint")
            if not roles:
                roles.append(prefix)

            writer.write_candidate(
                candidate,
                role="+".join(roles),
                eval_set=eval_set,
            )

    def _selection_record(self, candidate: Candidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "voice_hash": candidate.voice_hash,
            "stage": candidate.stage,
            "iteration": candidate.iteration,
            "total_loss": candidate.eval.total_loss,
            "mean_similarity": candidate.eval.mean_similarity,
            "include_latent_penalties": candidate.eval.include_latent_penalties,
        }

    def _write_export(
        self,
        *,
        selector: str,
        candidate: Candidate,
        run_summary: dict[str, Any],
        optimization_texts: list[str],
        validation_texts: list[str],
    ) -> None:
        self.ctx.paths.export_dir.mkdir(parents=True, exist_ok=True)
        save_voice(self.ctx.paths.export("voice.pt"), candidate.voice)
        write_json(
            self.ctx.paths.export("voice_meta.json"),
            {
                "schema_version": 1,
                "selector": selector,
                "voice_hash": candidate.voice_hash,
                "candidate_id": candidate.candidate_id,
                "stage": candidate.stage,
                "iteration": candidate.iteration,
                "optimization_texts": optimization_texts,
                "validation_texts": validation_texts,
                "candidate": candidate.to_dict(),
                "run": {
                    "fingerprints": run_summary["fingerprints"],
                    "selected": run_summary["selected"],
                    "metrics": run_summary["metrics"],
                },
            },
        )


def _load_voice_by_hash(ctx: Context, requested_voice_hash: str) -> torch.Tensor:
    path = ctx.paths.optimize("voices", f"{requested_voice_hash}.pt")
    if path.exists():
        return load_pt(path).to(torch.float32).contiguous()

    exported = ctx.paths.export("voice.pt")
    if exported.exists():
        voice = load_pt(exported).to(torch.float32).contiguous()
        if voice_hash(voice) == requested_voice_hash:
            return voice

    raise FileNotFoundError(f"Voice hash not found: {requested_voice_hash}")


def _candidate_records(ctx: Context) -> list[dict[str, Any]]:
    path = ctx.paths.optimize("candidates.jsonl")
    if not path.exists():
        return []
    return read_jsonl(path)


def _find_candidate_record(
    records: list[dict[str, Any]],
    *,
    candidate_id: str | None = None,
    requested_voice_hash: str | None = None,
) -> dict[str, Any] | None:
    for record in records:
        if candidate_id is not None and record.get("candidate_id") == candidate_id:
            return record
    for record in records:
        if (
            requested_voice_hash is not None
            and record.get("voice_hash") == requested_voice_hash
        ):
            return record
    return None


def export_voice(
    ctx: Context,
    candidate: str = "best",
    voice_out: Path | None = None,
    meta_out: Path | None = None,
) -> None:
    ctx.paths.export_dir.mkdir(parents=True, exist_ok=True)

    run_path = ctx.paths.optimize("run.json")
    if not run_path.exists():
        raise FileNotFoundError(f"Missing optimization run summary: {run_path}")

    run = read_json(run_path)
    records = _candidate_records(ctx)
    selected = run.get("selected", {})
    texts = run.get("texts", {})
    optimization_texts = list(texts.get("optimization", []))
    validation_texts = list(texts.get("validation", []))

    selector = candidate
    candidate_id: str | None = None

    if candidate in {"best", "final", "best_optimization"}:
        selected_record = selected.get(candidate)
        if not isinstance(selected_record, dict):
            raise ValueError(f"Run summary has no selected candidate for {candidate!r}")
        requested_voice_hash = str(selected_record["voice_hash"])
        candidate_id = str(selected_record["candidate_id"])
    else:
        requested_voice_hash = candidate

    voice = _load_voice_by_hash(ctx, requested_voice_hash)
    record = _find_candidate_record(
        records,
        candidate_id=candidate_id,
        requested_voice_hash=requested_voice_hash,
    )

    meta = {
        "schema_version": 1,
        "selector": selector,
        "voice_hash": requested_voice_hash,
        "candidate_id": candidate_id,
        "optimization_texts": optimization_texts,
        "validation_texts": validation_texts,
        "candidate": record,
        "run": {
            "fingerprints": run.get("fingerprints", {}),
            "selected": selected,
            "metrics": run.get("metrics", {}),
        },
    }

    voice_out = voice_out or ctx.paths.export("voice.pt")
    meta_out = meta_out or ctx.paths.export("voice_meta.json")

    save_voice(voice_out, voice)
    write_json(meta_out, meta)
    print(f"Exported voice: {voice_out}")
    print(f"Exported metadata: {meta_out}")


def _preview_validation_texts(ctx: Context) -> list[str]:
    meta_path = ctx.paths.export("voice_meta.json")
    if meta_path.exists():
        meta = read_json(meta_path)
        texts = meta.get("validation_texts")
        if isinstance(texts, list) and texts:
            return [str(text) for text in texts]

    return build_text_plan(ctx).validation_texts


def preview(ctx: Context, voice_path: Path | None = None) -> None:
    voice_path = voice_path or ctx.paths.export("voice.pt")
    if not voice_path.exists():
        raise FileNotFoundError("No exported voice found; run optimize/export first")

    validation_texts = _preview_validation_texts(ctx)
    synthesizer = make_synthesizer(ctx)
    voice = load_pt(voice_path)

    ctx.paths.preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, text in enumerate(validation_texts):
        audio = synthesizer.synthesize(text, voice)
        wav_path = ctx.paths.preview(f"preview_{idx:02d}.wav")
        json_path = ctx.paths.preview(f"preview_{idx:02d}.json")
        torchaudio.save(
            str(wav_path),
            audio.cpu().to(torch.float32).unsqueeze(0),
            synthesizer.sample_rate,
        )
        write_json(
            json_path,
            {
                "text": text,
                "voice": str(voice_path),
                "sample_rate": synthesizer.sample_rate,
            },
        )

    print(f"Wrote previews to {ctx.paths.preview_dir}")
