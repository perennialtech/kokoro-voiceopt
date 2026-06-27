from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .config import Context
from .corpus import VoiceCorpus
from .manifold import load_or_build_manifold
from .objective import VoiceObjective
from .profile import load_target_speaker_profile
from .search import (run_baseline_scan, run_blend_stage, run_latent_stage,
                     save_candidate_artifacts, validate_candidates)
from .serde import fingerprint, load_pt, read_json, sha256_file, write_json
from .transcript import load_text_plan
from .voice import save_voice, voice_hash


@dataclass
class VoiceOptimizationResult:
    output_dir: Path
    best_voice_path: Path
    final_voice_path: Path
    best_validation_loss: float
    best_optimization_loss: float
    baseline_best_similarity: float
    blend_best_similarity: float
    latent_best_similarity: float
    selected_stage: str
    selected_candidate_id: str
    selected_voice_hash: str


class VoiceOptimizationPipeline:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def run(self) -> VoiceOptimizationResult:
        self._set_seed(self.ctx.search.seed)
        self._prepare_dirs()
        self.ctx.write_resolved_config()

        self.ctx.require_manifests()
        self.ctx.require_profile()
        self.ctx.require_corpus()

        text_plan = load_text_plan(self.ctx)
        target_profile = load_target_speaker_profile(self.ctx)
        corpus = VoiceCorpus.load(self.ctx)

        speaker = self.ctx.services.speaker_encoder()
        synthesizer = self.ctx.services.synthesizer()
        objective = VoiceObjective(
            synthesizer=synthesizer,
            speaker_encoder=speaker,
            target_profile=target_profile,
            audio_config=self.ctx.audio,
            objective_config=self.ctx.objective,
        )

        artifact_metadata = {
            "schema_version": 1,
            "text_plan_sha256": sha256_file(self.ctx.paths.data("text_plan.json")),
            "optimization_texts_sha256": fingerprint(text_plan.optimization_texts),
            "validation_texts_sha256": fingerprint(text_plan.validation_texts),
            "objective_config_sha256": fingerprint(self.ctx.objective),
            "search_config_sha256": fingerprint(self.ctx.search),
            "corpus_manifest_sha256": sha256_file(
                self.ctx.paths.corpus("corpus_manifest.json")
            ),
            "corpus_sha256": corpus.corpus_sha256,
            "target_profile_sha256": sha256_file(
                self.ctx.paths.profile("target_profile.pt")
            ),
        }

        baseline = run_baseline_scan(
            corpus=corpus,
            objective=objective,
            texts=text_plan.optimization_texts,
            top_k=self.ctx.search.top_k_for_blend,
            metadata=artifact_metadata,
        )
        write_json(
            self.ctx.paths.optimize("stages/baseline.json"),
            [candidate.to_dict() for candidate in baseline.candidates],
        )
        for candidate in baseline.candidates:
            save_candidate_artifacts(candidate, self.ctx.paths)

        blend = run_blend_stage(
            top_candidates=baseline.top,
            objective=objective,
            texts=text_plan.optimization_texts,
            config=self.ctx.search,
            metadata=artifact_metadata,
            history_path=self.ctx.paths.optimize("stages/blend_history.jsonl"),
            paths=self.ctx.paths,
            validation_texts=text_plan.validation_texts,
        )

        manifold = load_or_build_manifold(self.ctx, corpus)
        initial_z = manifold.encode(blend.best.voice)

        latent = run_latent_stage(
            manifold=manifold,
            initial_z=initial_z,
            objective=objective,
            texts=text_plan.optimization_texts,
            config=self.ctx.search,
            metadata=artifact_metadata,
            history_path=self.ctx.paths.optimize("stages/latent_history.jsonl"),
            paths=self.ctx.paths,
            validation_texts=text_plan.validation_texts,
        )

        optimization_candidates = [
            baseline.best,
            blend.best,
            latent.best,
            latent.final,
            *latent.checkpoints,
        ]
        optimization_candidates = [
            candidate for candidate in optimization_candidates if candidate is not None
        ]
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
        write_json(
            self.ctx.paths.optimize("stages/validation.json"),
            [candidate.to_dict() for candidate in validation.candidates],
        )
        for candidate in validation.candidates:
            save_candidate_artifacts(candidate, self.ctx.paths)

        best_validation = validation.best
        final_candidate = latent.final
        if final_candidate is None:
            raise RuntimeError("Latent search did not produce a final candidate")

        save_voice(self.ctx.paths.export("voice_best.pt"), best_validation.voice)
        save_voice(self.ctx.paths.export("voice_final.pt"), final_candidate.voice)
        save_voice(
            self.ctx.paths.export("voice_best_optimization.pt"), best_optimization.voice
        )

        write_json(
            self.ctx.paths.export("voice_best_meta.json"), best_validation.to_dict()
        )
        write_json(
            self.ctx.paths.export("voice_final_meta.json"), final_candidate.to_dict()
        )
        write_json(
            self.ctx.paths.export("voice_best_optimization_meta.json"),
            best_optimization.to_dict(),
        )

        save_voice(self.ctx.paths.export("voice.pt"), best_validation.voice)
        write_json(self.ctx.paths.export("voice_meta.json"), best_validation.to_dict())

        result = VoiceOptimizationResult(
            output_dir=self.ctx.paths.optimize_dir,
            best_voice_path=self.ctx.paths.export("voice_best.pt"),
            final_voice_path=self.ctx.paths.export("voice_final.pt"),
            best_validation_loss=best_validation.eval.total_loss,
            best_optimization_loss=best_optimization.eval.total_loss,
            baseline_best_similarity=baseline.best.eval.mean_similarity,
            blend_best_similarity=blend.best.eval.mean_similarity,
            latent_best_similarity=latent.best.eval.mean_similarity,
            selected_stage=best_validation.stage,
            selected_candidate_id=best_validation.candidate_id,
            selected_voice_hash=best_validation.voice_hash,
        )
        write_json(self.ctx.paths.optimize("run_info.json"), result)
        return result

    def _prepare_dirs(self) -> None:
        for path in [
            self.ctx.paths.optimize_dir,
            self.ctx.paths.optimize("stages"),
            self.ctx.paths.optimize("checkpoints"),
            self.ctx.paths.optimize("voices"),
            self.ctx.paths.optimize("evaluations"),
            self.ctx.paths.optimize("candidates"),
            self.ctx.paths.optimize("params"),
            self.ctx.paths.export_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _load_voice_by_hash(ctx: Context, requested_voice_hash: str) -> torch.Tensor:
    path = ctx.paths.optimize("voices", f"{requested_voice_hash}.pt")
    if path.exists():
        return load_pt(path).to(torch.float32).contiguous()

    for exported in [
        ctx.paths.export("voice.pt"),
        ctx.paths.export("voice_best.pt"),
        ctx.paths.export("voice_final.pt"),
        ctx.paths.export("voice_best_optimization.pt"),
    ]:
        if exported.exists():
            voice = load_pt(exported).to(torch.float32).contiguous()
            if voice_hash(voice) == requested_voice_hash:
                return voice

    raise FileNotFoundError(f"Voice hash not found: {requested_voice_hash}")


def export_voice(
    ctx: Context,
    candidate: str = "best",
    voice_out: Path | None = None,
    meta_out: Path | None = None,
) -> None:
    ctx.paths.export_dir.mkdir(parents=True, exist_ok=True)

    if candidate == "best":
        source_voice = ctx.paths.export("voice_best.pt")
        source_meta = ctx.paths.export("voice_best_meta.json")
        voice = load_pt(source_voice)
        meta = read_json(source_meta)
    elif candidate == "final":
        source_voice = ctx.paths.export("voice_final.pt")
        source_meta = ctx.paths.export("voice_final_meta.json")
        voice = load_pt(source_voice)
        meta = read_json(source_meta)
    elif candidate == "best_optimization":
        source_voice = ctx.paths.export("voice_best_optimization.pt")
        source_meta = ctx.paths.export("voice_best_optimization_meta.json")
        voice = load_pt(source_voice)
        meta = read_json(source_meta)
    else:
        voice = _load_voice_by_hash(ctx, candidate)
        meta = {
            "selector": "voice_hash",
            "voice_hash": candidate,
            "note": "Explicit voice-hash export. See optimize/candidates and optimize/evaluations for records.",
        }

    voice_out = voice_out or ctx.paths.export("voice.pt")
    meta_out = meta_out or ctx.paths.export("voice_meta.json")

    save_voice(voice_out, voice)
    write_json(meta_out, meta)
    print(f"Exported voice: {voice_out}")
    print(f"Exported metadata: {meta_out}")


def preview(ctx: Context, voice_path: Path | None = None) -> None:
    voice_path = voice_path or ctx.paths.export("voice.pt")
    if not voice_path.exists():
        if ctx.paths.export("voice_best.pt").exists():
            voice_path = ctx.paths.export("voice_best.pt")
        else:
            raise FileNotFoundError(
                "No exported voice found; run optimize/export first"
            )

    text_plan = load_text_plan(ctx)
    synthesizer = ctx.services.synthesizer()
    voice = load_pt(voice_path)

    ctx.paths.preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, text in enumerate(text_plan.validation_texts):
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
