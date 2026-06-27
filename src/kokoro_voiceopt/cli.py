from __future__ import annotations

import argparse
from pathlib import Path

from .config import Run


def add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default=None)


def load(args) -> Run:
    return Run.load(args.config, project_root=args.project_root)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Manifest-driven Kokoro voice optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    assets_cmd = sub.add_parser("assets", help="Prepare canonical Kokoro voice corpus")
    add_config(assets_cmd)
    assets_cmd.add_argument("--force", action="store_true")

    def run_assets(args) -> None:
        from .assets import prepare_voice_corpus

        run = load(args)
        run.write_resolved_config()
        prepare_voice_corpus(run, force=args.force)

    assets_cmd.set_defaults(func=run_assets)

    prepare_cmd = sub.add_parser("prepare", help="Prepare target clips and manifest")
    add_config(prepare_cmd)
    prepare_cmd.add_argument("--audio-dir", required=True)
    prepare_cmd.add_argument("--transcripts", required=True)
    prepare_cmd.add_argument("--force", action="store_true")

    def run_prepare(args) -> None:
        from .data import prepare_target_dataset

        run = load(args)
        run.write_resolved_config()
        prepare_target_dataset(
            run,
            Path(args.audio_dir),
            Path(args.transcripts),
            force=args.force,
        )

    prepare_cmd.set_defaults(func=run_prepare)

    check_cmd = sub.add_parser("check", help="Validate prepared target data")
    add_config(check_cmd)

    def run_check(args) -> None:
        from .data import check_target_dataset

        check_target_dataset(load(args))

    check_cmd.set_defaults(func=run_check)

    profile_cmd = sub.add_parser("profile", help="Build target speaker profile")
    add_config(profile_cmd)
    profile_cmd.add_argument("--force", action="store_true")

    def run_profile(args) -> None:
        from .profile import build_target_speaker_profile
        from .speaker import WavLMXVectorSpeakerEncoder

        run = load(args)
        run.write_resolved_config()
        speaker = WavLMXVectorSpeakerEncoder(run.speaker_encoder, run.device)
        build_target_speaker_profile(run, speaker, force=args.force)

    profile_cmd.set_defaults(func=run_profile)

    doctor_cmd = sub.add_parser("doctor", help="Run preflight checks")
    add_config(doctor_cmd)

    def run_doctor(args) -> None:
        from .doctor import doctor

        doctor(load(args))

    doctor_cmd.set_defaults(func=run_doctor)

    optimize_cmd = sub.add_parser("optimize", help="Run voice optimization")
    add_config(optimize_cmd)

    def run_optimize(args) -> None:
        from .pipeline import VoiceOptimizationPipeline

        run = load(args)
        run.write_resolved_config()
        result = VoiceOptimizationPipeline(run).run()
        print(f"Best voice: {result.best_voice_path}")
        print(f"Final voice: {result.final_voice_path}")
        print(f"Best validation loss: {result.best_validation_loss:.6f}")
        print(f"Best optimization loss: {result.best_optimization_loss:.6f}")
        print(f"Selected stage: {result.selected_stage}")
        print(f"Selected candidate hash: {result.selected_candidate_hash}")

    optimize_cmd.set_defaults(func=run_optimize)

    export_cmd = sub.add_parser("export", help="Export selected optimized voice")
    add_config(export_cmd)
    export_cmd.add_argument(
        "--candidate",
        default="best",
        help="best, final, best_optimization, or an explicit candidate hash",
    )
    export_cmd.add_argument("--voice-out", default=None)
    export_cmd.add_argument("--meta-out", default=None)

    def run_export(args) -> None:
        from .pipeline import export_voice

        run = load(args)
        run.write_resolved_config()
        export_voice(
            run,
            candidate=args.candidate,
            voice_out=Path(args.voice_out) if args.voice_out else None,
            meta_out=Path(args.meta_out) if args.meta_out else None,
        )

    export_cmd.set_defaults(func=run_export)

    preview_cmd = sub.add_parser("preview", help="Synthesize preview WAVs")
    add_config(preview_cmd)
    preview_cmd.add_argument("--voice", default=None)

    def run_preview(args) -> None:
        from .pipeline import preview

        run = load(args)
        run.write_resolved_config()
        preview(run, voice_path=Path(args.voice) if args.voice else None)

    preview_cmd.set_defaults(func=run_preview)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
