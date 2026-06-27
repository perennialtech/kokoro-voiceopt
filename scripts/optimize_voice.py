#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kokoro_voiceopt.config import (AudioConfig, CorpusConfig, ManifoldConfig,
                                    ObjectiveConfig, OutputConfig,
                                    SearchConfig, SpeakerConfig, TextConfig,
                                    VoiceOptConfig)
from kokoro_voiceopt.pipeline import VoiceOptimizationPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Black-box Kokoro voice optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--target-audio", required=True, type=Path)
    parser.add_argument("--target-transcript", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--voices-dir", type=Path, default=None)
    parser.add_argument("--repo-id", type=str, default="hexgrad/Kokoro-82M")
    parser.add_argument("--lang-code", type=str, default="a")
    parser.add_argument("--cross-language", action="store_true", default=True)
    parser.add_argument("--same-language-only", action="store_true")

    parser.add_argument(
        "--speaker-model", type=str, default="microsoft/wavlm-base-plus-sv"
    )
    parser.add_argument("--speaker-batch-size", type=int, default=16)

    parser.add_argument("--validation-texts", type=Path, default=None)
    parser.add_argument("--max-optimization-texts", type=int, default=3)
    parser.add_argument("--max-validation-texts", type=int, default=6)
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument("--max-text-chars", type=int, default=220)

    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--variance-coverage", type=float, default=0.98)
    parser.add_argument("--z-soft-bound", type=float, default=2.5)
    parser.add_argument("--z-hard-bound", type=float, default=4.0)

    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--blend-iterations", type=int, default=60)
    parser.add_argument("--latent-iterations", type=int, default=180)
    parser.add_argument("--blend-population-pairs", type=int, default=8)
    parser.add_argument("--latent-population-pairs", type=int, default=12)
    parser.add_argument("--population-pairs", type=int, default=None)
    parser.add_argument("--blend-learning-rate", type=float, default=0.08)
    parser.add_argument("--latent-learning-rate", type=float, default=0.04)
    parser.add_argument("--blend-sigma-initial", type=float, default=0.75)
    parser.add_argument("--blend-sigma-final", type=float, default=0.15)
    parser.add_argument("--latent-sigma-initial", type=float, default=0.35)
    parser.add_argument("--latent-sigma-final", type=float, default=0.06)

    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=20)
    parser.add_argument("--no-samples", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    parser.add_argument("--speaker-loss-weight", type=float, default=1.0)
    parser.add_argument("--prior-loss-weight", type=float, default=0.02)
    parser.add_argument("--bound-loss-weight", type=float, default=0.10)
    parser.add_argument("--silence-loss-weight", type=float, default=0.05)
    parser.add_argument("--clipping-loss-weight", type=float, default=0.10)
    parser.add_argument("--duration-loss-weight", type=float, default=0.02)

    args = parser.parse_args()

    if not args.target_audio.exists():
        raise FileNotFoundError(f"Target audio not found: {args.target_audio}")
    if not args.target_transcript.strip():
        raise ValueError("--target-transcript must be non-empty")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU", file=sys.stderr)
        args.device = "cpu"

    blend_pairs = (
        args.population_pairs
        if args.population_pairs is not None
        else args.blend_population_pairs
    )
    latent_pairs = (
        args.population_pairs
        if args.population_pairs is not None
        else args.latent_population_pairs
    )

    config = VoiceOptConfig(
        text=TextConfig(
            target_transcript=args.target_transcript,
            max_optimization_texts=args.max_optimization_texts,
            max_validation_texts=args.max_validation_texts,
            min_text_chars=args.min_text_chars,
            max_text_chars=args.max_text_chars,
            validation_texts_path=args.validation_texts,
        ),
        output=OutputConfig(
            output_dir=args.output_dir,
            save_generated_samples=not args.no_samples,
        ),
        audio=AudioConfig(),
        speaker=SpeakerConfig(
            model_name=args.speaker_model,
            batch_size=args.speaker_batch_size,
        ),
        corpus=CorpusConfig(
            voices_dir=args.voices_dir,
            repo_id=args.repo_id,
            lang_code=args.lang_code,
            include_cross_language_voices=not args.same_language_only,
        ),
        manifold=ManifoldConfig(
            max_latent_dim=args.latent_dim,
            variance_coverage=args.variance_coverage,
            z_soft_bound=args.z_soft_bound,
            z_hard_bound=args.z_hard_bound,
        ),
        objective=ObjectiveConfig(
            speaker_loss_weight=args.speaker_loss_weight,
            prior_loss_weight=args.prior_loss_weight,
            bound_loss_weight=args.bound_loss_weight,
            silence_loss_weight=args.silence_loss_weight,
            clipping_loss_weight=args.clipping_loss_weight,
            duration_loss_weight=args.duration_loss_weight,
        ),
        search=SearchConfig(
            seed=args.seed,
            top_k_for_blend=args.top_k,
            blend_iterations=args.blend_iterations,
            latent_iterations=args.latent_iterations,
            blend_population_pairs=blend_pairs,
            latent_population_pairs=latent_pairs,
            blend_learning_rate=args.blend_learning_rate,
            latent_learning_rate=args.latent_learning_rate,
            blend_sigma_initial=args.blend_sigma_initial,
            blend_sigma_final=args.blend_sigma_final,
            latent_sigma_initial=args.latent_sigma_initial,
            latent_sigma_final=args.latent_sigma_final,
            save_every=args.save_every,
            validate_every=args.validate_every,
        ),
        device=args.device,
    )
    setattr(config, "target_audio_path", args.target_audio)

    print("=" * 80)
    print("KOKORO BLACK-BOX VOICE OPTIMIZATION")
    print("=" * 80)
    print(f"Target audio:       {args.target_audio}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Language code:      {args.lang_code}")
    print(f"Device:             {args.device}")
    print(f"Speaker model:      {args.speaker_model}")
    print(f"Top-k blend voices: {args.top_k}")
    print(f"Latent dimension:   {args.latent_dim}")
    print("=" * 80)

    result = VoiceOptimizationPipeline(config).run()

    print("\nOptimization complete")
    print(f"Best voice:              {result.best_voice_path}")
    print(f"Final voice:             {result.final_voice_path}")
    print(f"Best validation loss:    {result.best_validation_loss:.6f}")
    print(f"Best optimization loss:  {result.best_optimization_loss:.6f}")
    print(f"Selected stage:          {result.selected_stage}")
    print(f"Selected candidate hash: {result.selected_candidate_hash}")


if __name__ == "__main__":
    main()
