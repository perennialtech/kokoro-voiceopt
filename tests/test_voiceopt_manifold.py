import torch

from kokoro_voiceopt.config import ManifoldConfig
from kokoro_voiceopt.corpus import VoiceCorpus, VoiceRecord
from kokoro_voiceopt.manifold import VoiceManifold


def _corpus():
    torch.manual_seed(0)
    base = torch.randn(6, 256) * 0.01
    records = []
    for idx in range(5):
        records.append(
            VoiceRecord(
                name=f"v{idx}",
                tensor=base + idx * 0.01,
                source_path=None,
                language_prefix="a",
            )
        )
    return VoiceCorpus(records=records, T=6, D=256)


def test_manifold_encode_decode_shape_and_sanity():
    corpus = _corpus()
    manifold = VoiceManifold.fit(
        corpus,
        ManifoldConfig(max_latent_dim=5, variance_coverage=1.0, center="median"),
    )

    voice = corpus.records[2].tensor
    z = manifold.encode(voice)
    decoded = manifold.decode(z)

    assert z.shape == (manifold.latent_dim,)
    assert decoded.shape == voice.shape
    assert torch.isfinite(decoded).all()
    assert torch.mean((decoded - voice).abs()) < 1e-4


def test_clamp_prior_and_soft_bound_loss():
    corpus = _corpus()
    manifold = VoiceManifold.fit(
        corpus,
        ManifoldConfig(max_latent_dim=3, z_soft_bound=1.0, z_hard_bound=2.0),
    )

    z = torch.tensor([-5.0, 0.5, 3.0])
    clamped = manifold.clamp_z(z)
    assert clamped.min() >= -2.0
    assert clamped.max() <= 2.0

    prior = manifold.prior_loss(torch.tensor([1.0, 2.0]))
    assert torch.isclose(prior, torch.tensor(2.5))

    bound = manifold.soft_bound_loss(torch.tensor([0.5, 1.0, 2.0]))
    assert torch.isclose(bound, torch.tensor(1.0 / 3.0))
