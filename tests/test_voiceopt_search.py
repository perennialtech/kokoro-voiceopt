import torch
from kokoro.voiceopt.config import SearchConfig
from kokoro.voiceopt.objective import CandidateEval
from kokoro.voiceopt.search import AntitheticNES, voice_hash


def _eval(loss):
    return CandidateEval(
        total_loss=float(loss),
        speaker_loss=float(loss),
        prior_loss=0.0,
        bound_loss=0.0,
        audio_quality_loss=0.0,
        mean_similarity=1.0 - float(loss),
        per_text=[],
    )


def test_nes_update_shape_bounds_and_bookkeeping():
    torch.manual_seed(0)
    target = torch.tensor([0.7, -0.4])
    config = SearchConfig(
        blend_iterations=1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
    )
    nes = AntitheticNES(config)

    def decode(params):
        return params.detach().clone()

    def evaluate(voices, params_list):
        return [_eval(torch.sum((p - target) ** 2)) for p in params_list]

    result = nes.run(
        initial_params=torch.zeros(2),
        decode=decode,
        evaluate=evaluate,
        stage="toy",
        bounds=(-1.0, 1.0),
        iterations=5,
        population_pairs=4,
        sigma_initial=0.5,
        sigma_final=0.1,
        learning_rate=0.1,
        save_every=2,
    )

    assert result.final_params.shape == (2,)
    assert result.best_params.shape == (2,)
    assert result.final_params.min() >= -1.0
    assert result.final_params.max() <= 1.0
    assert result.best_candidate.candidate_hash == voice_hash(
        result.best_candidate.voice
    )
    assert result.best_candidate.eval.total_loss == result.best_loss
    assert len(result.checkpoint_candidates) >= 2
