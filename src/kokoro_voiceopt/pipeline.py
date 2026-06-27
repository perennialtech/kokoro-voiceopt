from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import VoiceOptConfig
from .corpus import VoiceCorpus
from .io import (save_candidate_report, save_candidate_samples, save_json,
                 save_target_profile, save_voice)
from .manifold import VoiceManifold
from .objective import VoiceObjective, build_target_speaker_profile
from .search import (run_baseline_scan, run_blend_search,
                     run_latent_search, validate_candidates)
from .speaker import WavLMXVectorSpeakerEncoder
from .synth import KokoroSynthesizer
from .textplan import build_text_plan


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
    def __init__(self, config: VoiceOptConfig):
        self.config = config

    def run(self) -> VoiceOptimizationResult:
        self._set_seed(self.config.search.seed)

        output_dir = Path(self.config.output.output_dir)
        self._prepare_dirs(output_dir)
        save_json(output_dir / "config.json", self.config)

        text_plan = build_text_plan(self.config.text)
        save_json(output_dir / "text_plan.json", text_plan)

        speaker = WavLMXVectorSpeakerEncoder(self.config.speaker, self.config.device)
        target_profile = build_target_speaker_profile(
            audio_path=self._target_audio_path,
            audio_config=self.config.audio,
            speaker_encoder=speaker,
        )
        save_target_profile(output_dir, target_profile)

        corpus = VoiceCorpus.load(self.config.corpus)
        save_json(output_dir / "corpus" / "corpus_manifest.json", corpus.manifest())

        synthesizer = self._create_synthesizer()
        objective = VoiceObjective(
            synthesizer=synthesizer,
            speaker_encoder=speaker,
            target_profile=target_profile,
            audio_config=self.config.audio,
            objective_config=self.config.objective,
        )

        baseline = run_baseline_scan(
            corpus=corpus,
            objective=objective,
            texts=text_plan.optimization_texts,
            top_k=self.config.search.top_k_for_blend,
        )
        save_json(
            output_dir / "stages" / "baseline.json",
            [c.to_dict() for c in baseline.candidates],
        )
        save_voice(
            output_dir / "voices" / "voice_best_baseline.pt",
            baseline.best_candidate.voice,
        )
        save_candidate_report(
            output_dir / "voices" / "voice_best_baseline_meta.json",
            baseline.best_candidate,
        )

        blend = run_blend_search(
            top_candidates=baseline.top_candidates,
            objective=objective,
            texts=text_plan.optimization_texts,
            config=self.config.search,
        )
        save_json(output_dir / "stages" / "blend_history.json", blend.result.history)
        save_candidate_report(
            output_dir / "voices" / "voice_best_blend_meta.json", blend.best_candidate
        )
        save_voice(
            output_dir / "voices" / "voice_best_blend.pt", blend.best_candidate.voice
        )

        manifold = VoiceManifold.fit(corpus, self.config.manifold)
        if self.config.manifold.save_manifold:
            manifold.save(output_dir / "manifold" / "manifold.pt")
        save_json(output_dir / "manifold" / "manifold_report.json", manifold.report())

        initial_z = manifold.encode(blend.best_candidate.voice)
        latent = run_latent_search(
            manifold=manifold,
            initial_z=initial_z,
            objective=objective,
            texts=text_plan.optimization_texts,
            config=self.config.search,
        )
        save_json(output_dir / "stages" / "latent_history.json", latent.result.history)

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
        )
        save_json(
            output_dir / "stages" / "validation.json",
            [c.to_dict() for c in validation.candidates],
        )

        best_validation = validation.best_candidate
        final_candidate = latent.final_candidate

        best_voice_path = output_dir / "voices" / "voice_best.pt"
        final_voice_path = output_dir / "voices" / "voice_final.pt"

        save_voice(best_voice_path, best_validation.voice)
        save_voice(final_voice_path, final_candidate.voice)
        save_voice(
            output_dir / "voices" / "voice_best_optimization.pt",
            best_optimization.voice,
        )

        save_candidate_report(
            output_dir / "voices" / "voice_best_meta.json", best_validation
        )
        save_candidate_report(
            output_dir / "voices" / "voice_final_meta.json", final_candidate
        )
        save_candidate_report(
            output_dir / "voices" / "voice_best_optimization_meta.json",
            best_optimization,
        )

        if self.config.output.save_generated_samples:
            save_candidate_samples(
                output_dir / "samples" / "validation",
                synthesizer,
                validation.candidates,
                text_plan.validation_texts,
                max_candidates=8,
            )
            save_candidate_samples(
                output_dir / "samples" / "optimization_bests",
                synthesizer,
                [
                    baseline.best_candidate,
                    blend.best_candidate,
                    latent.best_candidate,
                    best_optimization,
                ],
                text_plan.optimization_texts,
                max_candidates=None,
            )

        result = VoiceOptimizationResult(
            output_dir=output_dir,
            best_voice_path=best_voice_path,
            final_voice_path=final_voice_path,
            best_validation_loss=best_validation.eval.total_loss,
            best_optimization_loss=best_optimization.eval.total_loss,
            baseline_best_similarity=baseline.best_candidate.eval.mean_similarity,
            blend_best_similarity=blend.best_candidate.eval.mean_similarity,
            latent_best_similarity=latent.best_candidate.eval.mean_similarity,
            selected_stage=best_validation.stage,
            selected_candidate_hash=best_validation.candidate_hash,
        )
        save_json(output_dir / "run_info.json", result)
        return result

    @property
    def _target_audio_path(self) -> Path:
        path = getattr(self.config, "target_audio_path", None)
        if path is None:
            raise ValueError(
                "VoiceOptConfig must have target_audio_path set by the CLI or caller"
            )
        return Path(path)

    def _create_synthesizer(self) -> KokoroSynthesizer:
        from kokoro import KPipeline

        pipeline = KPipeline(
            lang_code=self.config.corpus.lang_code,
            repo_id=self.config.corpus.repo_id,
            device=self.config.device,
        )
        return KokoroSynthesizer(
            pipeline, sample_rate=self.config.audio.kokoro_sample_rate
        )

    def _prepare_dirs(self, output_dir: Path) -> None:
        for rel in [
            ".",
            "corpus",
            "manifold",
            "stages",
            "voices",
            "samples",
            "logs",
        ]:
            (output_dir / rel).mkdir(parents=True, exist_ok=True)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
