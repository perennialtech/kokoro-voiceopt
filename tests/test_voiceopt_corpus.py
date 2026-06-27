import pytest
import torch
from kokoro_voiceopt.corpus import canonicalize_voice_tensor


def test_canonicalize_2d_voice():
    t = torch.randn(10, 256)
    out = canonicalize_voice_tensor(t)
    assert out.shape == (10, 256)
    assert out.dtype == torch.float32
    assert out.is_contiguous()


def test_canonicalize_3d_voice():
    t = torch.randn(10, 1, 256)
    out = canonicalize_voice_tensor(t)
    assert out.shape == (10, 256)
    assert out.dtype == torch.float32
    assert out.is_contiguous()


def test_canonicalize_rejects_bad_middle_dimension():
    with pytest.raises(ValueError):
        canonicalize_voice_tensor(torch.randn(10, 2, 256))


def test_canonicalize_rejects_bad_feature_dimension():
    with pytest.raises(ValueError):
        canonicalize_voice_tensor(torch.randn(10, 128))


def test_canonicalize_rejects_bad_rank():
    with pytest.raises(ValueError):
        canonicalize_voice_tensor(torch.randn(256))
