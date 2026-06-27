from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .config import Run, hash_file, stable_hash_json
from .corpus import VoiceCorpus
from .io import save_json, save_voice
from .manifold import load_or_build_manifold
from .objective import VoiceObjective
from .profile import load_target_speaker_profile
from .search import (run_baseline_scan, run_blend_search,
                     run_latent_search, save_candidate_artifacts,
                     validate_candidates)
from .speaker import WavLMXVectorSpeakerEncoder
from .synth import KokoroSynthesizer
from .textplan import load_text_plan


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
    selected_candidate_hash: str


class VoiceOptimizationPipeline:
    def __init__(self, run: Run):
        self.run = run

    def run(self) -> VoiceOptimizationResult:
        self._set_seed(self.run.search.seed)
        self._prepare_dirs()
        self.run.write_resolved_config()

        self.run.require_manifests()
        self.run.require_profile()
        self.run.require_corpus()

        text_plan = load_text_plan(self.run)
        target_profile = load_target_speaker_profile(self.run)
        corpus = VoiceCorpus.load(self.run)

        speaker = WavLMXVectorSpeakerEncoder(self.run.speaker_encoder, self.run.device)
        synthesizer = self._create_synthesizer()
        objective = VoiceObjective(
            synthesizer=synthesizer,
            speaker_encoder=speaker,
            target_profile=target_profile,
            audio_config=self.run.audio,
            objective_config=self.run.objective,
        )

        artifact_metadata = {
            "schema_version": 1,
            "corpus_manifest_sha256": hash_file(self.run.corpus_manifest),
            "target_profile_sha256": hash_file(self.run.paths.target_profile_pt),
            "objective_config_sha256": stable_hash_json(self.run.objective),
            "search_config_sha256": stable_hash_json(self.run.search),
        }

        baseline = run_baseline_scan(
            corpus=corpus,
            objective=objective,
            texts=text_plan.optimization_texts,
            top_k=self.run.search.top_k_for_blend,
            metadata=artifact_metadata,
        )
        save_json(
            self.run.paths.optimize_stage_dir / "baseline.json",
            [candidate.to_dict() for candidate in baseline.candidates],
        )
        for candidate in baseline.candidates:
            save_candidate_artifacts(candidate, self.run.paths.optimize_candidate_dir)

        blend = run_blend_search(
            top_candidates=baseline.top_candidates,
            objective=objective,
            texts=text_plan.optimization_texts,
            config=self.run.search,
            metadata=artifact_metadata,
            history_path=self.run.paths.optimize_stage_dir / "blend_history.jsonl",
            checkpoint_dir=self.run.paths.optimize_checkpoint_dir,
            candidate_dir=self.run.paths.optimize_candidate_dir,
            validation_texts=text_plan.validation_texts,
        )

        manifold = load_or_build_manifold(self.run, corpus)
        initial_z = manifold.encode(blend.best_candidate.voice)

        latent = run_latent_search(
            manifold=manifold,
            initial_z=initial_z,
            objective=objective,
            texts=text_plan.optimization_texts,
            config=self.run.search,
            metadata=artifact_metadata,
            history_path=self.run.paths.optimize_stage_dir / "latent_history.jsonl",
            checkpoint_dir=self.run.paths.optimize_checkpoint_dir,
            candidate_dir=self.run.paths.optimize_candidate_dir,
            validation_texts=text_plan.validation_texts,
        )

        optimization_candidates = [
            baseline.best_candidate,
            blend.best_candidate,
            latent.best_candidate,
            latent.final_candidate,
            *latent.checkpoint_candidates,
        ]
        best_optimization = min(
            optimization_candidates, key=lambda c: c.eval.total_loss
        )

        validation = validate_candidates(
            candidates=optimization_candidates,
            objective=objective,
            texts=text_plan.validation_texts,
            metadata=artifact_metadata,
        )
        save_json(
            self.run.paths.optimize_stage_dir / "validation.json",
            [candidate.to_dict() for candidate in validation.candidates],
        )
        for candidate in validation.candidates:
            save_candidate_artifacts(candidate, self.run.paths.optimize_candidate_dir)

        best_validation = validation.best_candidate
        final_candidate = latent.final_candidate

        save_voice(self.run.paths.export_voice_best, best_validation.voice)
        save_voice(self.run.paths.export_voice_final, final_candidate.voice)
        save_voice(
            self.run.paths.export_voice_best_optimization, best_optimization.voice
        )

        save_json(
            self.run.paths.export_dir / "voice_best_meta.json",
            best_validation.to_dict(),
        )
        save_json(
            self.run.paths.export_dir / "voice_final_meta.json",
            final_candidate.to_dict(),
        )
        save_json(
            self.run.paths.export_dir / "voice_best_optimization_meta.json",
            best_optimization.to_dict(),
        )

        save_voice(self.run.paths.export_voice, best_validation.voice)
        save_json(self.run.paths.export_voice_meta, best_validation.to_dict())

        result = VoiceOptimizationResult(
            output_dir=self.run.paths.optimize_dir,
            best_voice_path=self.run.paths.export_voice_best,
            final_voice_path=self.run.paths.export_voice_final,
            best_validation_loss=best_validation.eval.total_loss,
            best_optimization_loss=best_optimization.eval.total_loss,
            baseline_best_similarity=baseline.best_candidate.eval.mean_similarity,
            blend_best_similarity=blend.best_candidate.eval.mean_similarity,
            latent_best_similarity=latent.best_candidate.eval.mean_similarity,
            selected_stage=best_validation.stage,
            selected_candidate_hash=best_validation.candidate_hash,
        )
        save_json(self.run.paths.optimize_run_info, result)
        return result

    def _create_synthesizer(self) -> KokoroSynthesizer:
        from kokoro import KPipeline

        pipeline = KPipeline(
            lang_code=self.run.target.lang_code,
            repo_id=self.run.assets.repo_id,
            device=self.run.device,
        )
        return KokoroSynthesizer(
            pipeline,
            sample_rate=self.run.audio.kokoro_sample_rate,
        )

    def _prepare_dirs(self) -> None:
        for path in [
            self.run.paths.optimize_dir,
            self.run.paths.optimize_stage_dir,
            self.run.paths.optimize_checkpoint_dir,
            self.run.paths.optimize_candidate_dir,
            self.run.paths.export_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _load_candidate_by_hash(run: Run, candidate_hash: str) -> tuple[torch.Tensor, dict]:
    path = run.paths.optimize_candidate_dir / f"{candidate_hash}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Candidate hash not found: {candidate_hash}")
    data = torch.load(path, map_location="cpu")
    return data["voice"].to(torch.float32).contiguous(), dict(data["candidate"])


def export_voice(
    run: Run,
    candidate: str = "best",
    voice_out: Path | None = None,
    meta_out: Path | None = None,
) -> None:
    run.paths.export_dir.mkdir(parents=True, exist_ok=True)

    if candidate == "best":
        source_voice = run.paths.export_voice_best
        source_meta = run.paths.export_dir / "voice_best_meta.json"
        voice = torch.load(source_voice, map_location="cpu")
        meta = json.loads(source_meta.read_text(encoding="utf-8"))
    elif candidate == "final":
        source_voice = run.paths.export_voice_final
        source_meta = run.paths.export_dir / "voice_final_meta.json"
        voice = torch.load(source_voice, map_location="cpu")
        meta = json.loads(source_meta.read_text(encoding="utf-8"))
    elif candidate == "best_optimization":
        source_voice = run.paths.export_voice_best_optimization
        source_meta = run.paths.export_dir / "voice_best_optimization_meta.json"
        voice = torch.load(source_voice, map_location="cpu")
        meta = json.loads(source_meta.read_text(encoding="utf-8"))
    else:
        voice, meta = _load_candidate_by_hash(run, candidate)

    voice_out = voice_out or run.paths.export_voice
    meta_out = meta_out or run.paths.export_voice_meta

    save_voice(voice_out, voice)
    save_json(meta_out, meta)
    print(f"Exported voice: {voice_out}")
    print(f"Exported metadata: {meta_out}")


def preview(run: Run, voice_path: Path | None = None) -> None:
    voice_path = voice_path or run.paths.export_voice
    if not voice_path.exists():
        if run.paths.export_voice_best.exists():
            voice_path = run.paths.export_voice_best
        else:
            raise FileNotFoundError(
                "No exported voice found; run optimize/export first"
            )

    text_plan = load_text_plan(run)

    from kokoro import KPipeline

    pipeline = KPipeline(
        lang_code=run.target.lang_code,
        repo_id=run.assets.repo_id,
        device=run.device,
    )
    synthesizer = KokoroSynthesizer(pipeline, sample_rate=run.audio.kokoro_sample_rate)
    voice = torch.load(voice_path, map_location="cpu")

    run.paths.preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, text in enumerate(text_plan.validation_texts):
        audio = synthesizer.synthesize(text, voice)
        wav_path = run.paths.preview_dir / f"preview_{idx:02d}.wav"
        json_path = run.paths.preview_dir / f"preview_{idx:02d}.json"
        torchaudio.save(
            str(wav_path),
            audio.cpu().to(torch.float32).unsqueeze(0),
            synthesizer.sample_rate,
        )
        save_json(
            json_path,
            {
                "text": text,
                "voice": str(voice_path),
                "sample_rate": synthesizer.sample_rate,
            },
        )

    print(f"Wrote previews to {run.paths.preview_dir}")
