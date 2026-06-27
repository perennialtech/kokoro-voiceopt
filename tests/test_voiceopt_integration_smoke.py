import os
from pathlib import Path

import pytest

from kokoro_voiceopt.config import (AudioConfig, CorpusConfig, ManifoldConfig,
                                    OutputConfig, SpeakerConfig, TextConfig,
                                    VoiceOptConfig)
from kokoro_voiceopt.corpus import VoiceCorpus
from kokoro_voiceopt.manifold import VoiceManifold


@pytest.mark.skipif(
    os.environ.get("VOICEOPT_SMOKE") != "1", reason="set VOICEOPT_SMOKE=1 to run"
)
def test_integration_smoke_with_small_local_corpus(tmp_path):
    voices_dir = Path(os.environ["VOICEOPT_VOICES_DIR"])
    target_audio = Path(os.environ["VOICEOPT_TARGET_AUDIO"])

    assert voices_dir.exists()
    assert target_audio.exists()

    corpus = VoiceCorpus.load(
        CorpusConfig(
            voices_dir=voices_dir,
            include_cross_language_voices=True,
            require_consistent_shape=True,
        ),
        voice_names=["af_heart", "af_bella", "am_adam"],
    )
    assert len(corpus.records) >= 2

    manifold = VoiceManifold.fit(
        corpus,
        ManifoldConfig(max_latent_dim=2, variance_coverage=0.95),
    )
    z = manifold.encode(corpus.records[0].tensor)
    decoded = manifold.decode(z)
    assert decoded.shape == corpus.records[0].tensor.shape

    config = VoiceOptConfig(
        text=TextConfig(
            target_transcript="This is a short target clip for smoke testing.",
            max_optimization_texts=1,
            max_validation_texts=1,
        ),
        output=OutputConfig(output_dir=tmp_path / "run", save_generated_samples=False),
        audio=AudioConfig(max_target_segments=1),
        speaker=SpeakerConfig(batch_size=2),
        corpus=CorpusConfig(voices_dir=voices_dir),
        manifold=ManifoldConfig(max_latent_dim=2),
    )
    setattr(config, "target_audio_path", target_audio)
    assert config.output.output_dir == tmp_path / "run"
