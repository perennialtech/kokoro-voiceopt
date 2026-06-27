from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from .config import SearchConfig
from .corpus import VoiceCorpus
from .manifold import VoiceManifold
from .objective import CandidateEval, LatentInfo, VoiceObjective


@dataclass
class Candidate:
    stage: str
    params: torch.Tensor | None
    voice: torch.Tensor
    eval: CandidateEval
    iteration: int | None
    created_at: str
    candidate_hash: str

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "iteration": self.iteration,
            "created_at": self.created_at,
            "candidate_hash": self.candidate_hash,
            "params_shape": (
                list(self.params.shape) if self.params is not None else None
            ),
            "voice_shape": list(self.voice.shape),
            "eval": self.eval.to_dict(),
        }


@dataclass
class NESResult:
    best_params: torch.Tensor
    final_params: torch.Tensor
    best_loss: float
    history: list[dict]
    best_eval: CandidateEval
    best_candidate: Candidate
    final_candidate: Candidate
    checkpoint_candidates: list[Candidate]


@dataclass
class BaselineResult:
    candidates: list[Candidate]
    top_candidates: list[Candidate]
    best_candidate: Candidate


@dataclass
class BlendResult:
    result: NESResult
    best_candidate: Candidate
    final_candidate: Candidate


@dataclass
class LatentResult:
    result: NESResult
    best_candidate: Candidate
    final_candidate: Candidate
    checkpoint_candidates: list[Candidate]


@dataclass
class ValidationResult:
    candidates: list[Candidate]
    best_candidate: Candidate


def voice_hash(voice: torch.Tensor) -> str:
    array = (
        voice.detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
        .numpy()
        .astype(np.float32, copy=False)
    )
    return hashlib.sha256(array.tobytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_candidate(
    stage: str,
    voice: torch.Tensor,
    eval_result: CandidateEval,
    params: torch.Tensor | None = None,
    iteration: int | None = None,
) -> Candidate:
    voice = voice.detach().cpu().to(torch.float32).contiguous()
    params = (
        None if params is None else params.detach().cpu().to(torch.float32).contiguous()
    )
    return Candidate(
        stage=stage,
        params=params,
        voice=voice,
        eval=eval_result,
        iteration=iteration,
        created_at=_now(),
        candidate_hash=voice_hash(voice),
    )


def run_baseline_scan(
    corpus: VoiceCorpus,
    objective: VoiceObjective,
    texts: list[str],
    top_k: int,
) -> BaselineResult:
    voices = [record.tensor for record in corpus.records]
    evals = objective.evaluate_voices(voices, texts)

    candidates = [
        make_candidate(
            stage=f"baseline:{record.name}",
            voice=record.tensor,
            eval_result=eval_result,
            params=None,
            iteration=None,
        )
        for record, eval_result in zip(corpus.records, evals)
    ]
    candidates.sort(key=lambda c: c.eval.total_loss)

    return BaselineResult(
        candidates=candidates,
        top_candidates=candidates[:top_k],
        best_candidate=candidates[0],
    )


def convex_blend(logits: torch.Tensor, voices: list[torch.Tensor]) -> torch.Tensor:
    weights = torch.softmax(logits.to(torch.float32).cpu(), dim=0)
    stacked = torch.stack([v.detach().cpu().to(torch.float32) for v in voices], dim=0)
    return (weights.view(-1, 1, 1) * stacked).sum(dim=0).contiguous()


class AntitheticNES:
    def __init__(self, config: SearchConfig):
        self.config = config

    def _clamp(
        self, params: torch.Tensor, bounds: tuple[float, float] | None
    ) -> torch.Tensor:
        params = params.to(torch.float32).cpu()
        if bounds is None:
            return params.contiguous()
        lo, hi = bounds
        return params.clamp(float(lo), float(hi)).contiguous()

    def _sigma(
        self, iteration: int, iterations: int, initial: float, final: float
    ) -> float:
        if iterations <= 1:
            return float(final)
        t = iteration / float(iterations - 1)
        return float(initial * ((final / initial) ** t))

    def run(
        self,
        initial_params: torch.Tensor,
        decode: Callable[[torch.Tensor], torch.Tensor],
        evaluate: Callable[
            [list[torch.Tensor], list[torch.Tensor]], list[CandidateEval]
        ],
        stage: str,
        bounds: tuple[float, float] | None,
        iterations: int,
        population_pairs: int,
        sigma_initial: float,
        sigma_final: float,
        learning_rate: float,
        save_every: int,
    ) -> NESResult:
        params = self._clamp(initial_params.detach().cpu().to(torch.float32), bounds)
        dim = params.numel()

        current_voice = decode(params)
        current_eval = evaluate([current_voice], [params])[0]
        current_candidate = make_candidate(
            stage, current_voice, current_eval, params=params, iteration=0
        )

        best_candidate = current_candidate
        checkpoint_candidates: list[Candidate] = []
        history: list[dict] = []

        m = torch.zeros_like(params)
        v = torch.zeros_like(params)
        beta1 = self.config.adam_beta1
        beta2 = self.config.adam_beta2

        iterator = tqdm(range(iterations), desc=f"{stage} NES")
        for iteration_zero in iterator:
            iteration = iteration_zero + 1
            sigma = self._sigma(iteration_zero, iterations, sigma_initial, sigma_final)

            directions = torch.randn(population_pairs, dim, dtype=torch.float32)
            param_list: list[torch.Tensor] = []
            for direction in directions:
                eps = direction.reshape_as(params)
                param_list.append(self._clamp(params + sigma * eps, bounds))
                param_list.append(self._clamp(params - sigma * eps, bounds))

            voices = [decode(p) for p in param_list]
            evals = evaluate(voices, param_list)
            losses = torch.tensor([e.total_loss for e in evals], dtype=torch.float32)

            for p, voice, eval_result in zip(param_list, voices, evals):
                candidate = make_candidate(
                    stage, voice, eval_result, params=p, iteration=iteration
                )
                if candidate.eval.total_loss < best_candidate.eval.total_loss:
                    best_candidate = candidate

            grad = torch.zeros_like(params)
            for pair_idx, direction in enumerate(directions):
                loss_plus = losses[2 * pair_idx]
                loss_minus = losses[2 * pair_idx + 1]
                grad = grad + ((loss_plus - loss_minus) * direction.reshape_as(params))

            grad = grad / max(2.0 * population_pairs * sigma, 1e-8)
            loss_std = losses.std(unbiased=False).clamp_min(1e-8)
            grad = grad / loss_std

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad.square()
            m_hat = m / (1.0 - beta1**iteration)
            v_hat = v / (1.0 - beta2**iteration)

            params = params - learning_rate * m_hat / (
                v_hat.sqrt() + self.config.adam_eps
            )
            params = self._clamp(params, bounds)

            row = {
                "stage": stage,
                "iteration": iteration,
                "best_loss": best_candidate.eval.total_loss,
                "population_mean_loss": float(losses.mean()),
                "population_std_loss": float(loss_std),
                "sigma": sigma,
                "learning_rate": learning_rate,
                "best_candidate_hash": best_candidate.candidate_hash,
            }
            history.append(row)
            iterator.set_postfix(
                {
                    "best": f"{best_candidate.eval.total_loss:.5f}",
                    "sigma": f"{sigma:.4f}",
                }
            )

            if save_every > 0 and iteration % save_every == 0:
                checkpoint_voice = decode(params)
                checkpoint_eval = evaluate([checkpoint_voice], [params])[0]
                checkpoint = make_candidate(
                    stage,
                    checkpoint_voice,
                    checkpoint_eval,
                    params=params,
                    iteration=iteration,
                )
                checkpoint_candidates.append(checkpoint)
                if checkpoint.eval.total_loss < best_candidate.eval.total_loss:
                    best_candidate = checkpoint

        final_voice = decode(params)
        final_eval = evaluate([final_voice], [params])[0]
        final_candidate = make_candidate(
            stage, final_voice, final_eval, params=params, iteration=iterations
        )
        if final_candidate.eval.total_loss < best_candidate.eval.total_loss:
            best_candidate = final_candidate

        return NESResult(
            best_params=best_candidate.params.clone(),
            final_params=params.clone(),
            best_loss=best_candidate.eval.total_loss,
            history=history,
            best_eval=best_candidate.eval,
            best_candidate=best_candidate,
            final_candidate=final_candidate,
            checkpoint_candidates=checkpoint_candidates,
        )


def run_blend_search(
    top_candidates: list[Candidate],
    objective: VoiceObjective,
    texts: list[str],
    config: SearchConfig,
) -> BlendResult:
    if not top_candidates:
        raise ValueError("Blend search requires at least one baseline candidate")

    voices = [candidate.voice for candidate in top_candidates]
    logits = torch.full((len(voices),), -4.0, dtype=torch.float32)
    logits[0] = 4.0

    nes = AntitheticNES(config)

    def decode(params: torch.Tensor) -> torch.Tensor:
        return convex_blend(params, voices)

    def evaluate(
        candidate_voices: list[torch.Tensor], params_list: list[torch.Tensor]
    ) -> list[CandidateEval]:
        return objective.evaluate_voices(candidate_voices, texts)

    result = nes.run(
        initial_params=logits,
        decode=decode,
        evaluate=evaluate,
        stage="blend",
        bounds=(-8.0, 8.0),
        iterations=config.blend_iterations,
        population_pairs=config.blend_population_pairs,
        sigma_initial=config.blend_sigma_initial,
        sigma_final=config.blend_sigma_final,
        learning_rate=config.blend_learning_rate,
        save_every=config.save_every,
    )

    return BlendResult(
        result=result,
        best_candidate=result.best_candidate,
        final_candidate=result.final_candidate,
    )


def run_latent_search(
    manifold: VoiceManifold,
    initial_z: torch.Tensor,
    objective: VoiceObjective,
    texts: list[str],
    config: SearchConfig,
) -> LatentResult:
    nes = AntitheticNES(config)

    def decode(params: torch.Tensor) -> torch.Tensor:
        return manifold.decode(params, clamp=True)

    def evaluate(
        candidate_voices: list[torch.Tensor], params_list: list[torch.Tensor]
    ) -> list[CandidateEval]:
        infos = [
            LatentInfo(z=manifold.clamp_z(p), manifold=manifold) for p in params_list
        ]
        return objective.evaluate_voices(candidate_voices, texts, latent_info=infos)

    result = nes.run(
        initial_params=manifold.clamp_z(initial_z),
        decode=decode,
        evaluate=evaluate,
        stage="latent",
        bounds=(-manifold.config.z_hard_bound, manifold.config.z_hard_bound),
        iterations=config.latent_iterations,
        population_pairs=config.latent_population_pairs,
        sigma_initial=config.latent_sigma_initial,
        sigma_final=config.latent_sigma_final,
        learning_rate=config.latent_learning_rate,
        save_every=config.save_every,
    )

    return LatentResult(
        result=result,
        best_candidate=result.best_candidate,
        final_candidate=result.final_candidate,
        checkpoint_candidates=result.checkpoint_candidates,
    )


def validate_candidates(
    candidates: list[Candidate],
    objective: VoiceObjective,
    texts: list[str],
) -> ValidationResult:
    if not candidates:
        raise ValueError("No candidates provided for validation")

    voices = [candidate.voice for candidate in candidates]
    evals = objective.evaluate_voices(voices, texts)

    validated = [
        make_candidate(
            stage=f"validation:{candidate.stage}",
            voice=candidate.voice,
            eval_result=eval_result,
            params=candidate.params,
            iteration=candidate.iteration,
        )
        for candidate, eval_result in zip(candidates, evals)
    ]
    validated.sort(key=lambda c: c.eval.total_loss)

    return ValidationResult(
        candidates=validated,
        best_candidate=validated[0],
    )
