from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import torch
from tqdm import tqdm

from .config import SearchConfig
from .corpus import VoiceCorpus
from .objective import EvalResult, LatentInfo, VoiceObjective
from .serde import fingerprint, sha256_tensor
from .voice import as_voice_2d, voice_hash


@dataclass
class Candidate:
    stage: str
    params: torch.Tensor | None
    voice: torch.Tensor
    eval: EvalResult
    iteration: int | None
    created_at: str
    voice_hash: str
    candidate_id: str
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "iteration": self.iteration,
            "created_at": self.created_at,
            "voice_hash": self.voice_hash,
            "candidate_id": self.candidate_id,
            "params": (
                self.params.detach().cpu().to(torch.float32).tolist()
                if self.params is not None
                else None
            ),
            "params_shape": (
                list(self.params.shape) if self.params is not None else None
            ),
            "voice_shape": list(self.voice.shape),
            "metadata": self.metadata,
            "eval": self.eval.to_dict(),
        }


@dataclass
class SearchResult:
    stage: str
    candidates: list[Candidate]
    best: Candidate
    final: Candidate | None = None
    checkpoints: list[Candidate] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    top: list[Candidate] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def params_hash(params: torch.Tensor | None) -> str | None:
    return (
        None
        if params is None
        else sha256_tensor(params.detach().cpu().to(torch.float32).contiguous())
    )


def make_candidate(
    stage: str,
    voice: torch.Tensor,
    eval_result: EvalResult,
    params: torch.Tensor | None = None,
    iteration: int | None = None,
    metadata: dict | None = None,
) -> Candidate:
    normalized_voice = as_voice_2d(voice)
    normalized_params = (
        None if params is None else params.detach().cpu().to(torch.float32).contiguous()
    )
    metadata = dict(metadata or {})
    vhash = voice_hash(normalized_voice)
    candidate_id = fingerprint(
        {
            "stage": stage,
            "iteration": iteration,
            "voice_hash": vhash,
            "params_hash": params_hash(normalized_params),
            "metadata": metadata,
            "eval": eval_result.to_dict(),
        }
    )

    return Candidate(
        stage=stage,
        params=normalized_params,
        voice=normalized_voice,
        eval=eval_result,
        iteration=iteration,
        created_at=_now(),
        voice_hash=vhash,
        candidate_id=candidate_id,
        metadata=metadata,
    )


def run_baseline_scan(
    corpus: VoiceCorpus,
    objective: VoiceObjective,
    texts: list[str],
    top_k: int,
    metadata: dict | None = None,
) -> SearchResult:
    voices = [record.tensor for record in corpus.records]
    evals = objective.evaluate_voices(voices, texts, include_latent_penalties=False)

    candidates = [
        make_candidate(
            stage=f"baseline:{record.name}",
            voice=record.tensor,
            eval_result=eval_result,
            params=None,
            iteration=None,
            metadata=metadata,
        )
        for record, eval_result in zip(corpus.records, evals)
    ]
    candidates.sort(key=lambda candidate: candidate.eval.total_loss)

    return SearchResult(
        stage="baseline",
        candidates=candidates,
        top=candidates[:top_k],
        best=candidates[0],
        final=None,
        checkpoints=[],
        history=[],
    )


def convex_blend(logits: torch.Tensor, voices: list[torch.Tensor]) -> torch.Tensor:
    weights = torch.softmax(logits.to(torch.float32).cpu(), dim=0)
    stacked = torch.stack([as_voice_2d(voice) for voice in voices], dim=0)
    return (weights.view(-1, 1, 1) * stacked).sum(dim=0).contiguous()


class AntitheticNES:
    def __init__(self, config: SearchConfig):
        self.config = config

    def _clamp(
        self,
        params: torch.Tensor,
        bounds: tuple[float, float] | None,
    ) -> torch.Tensor:
        params = params.to(torch.float32).cpu()
        if bounds is None:
            return params.contiguous()
        lo, hi = bounds
        return params.clamp(float(lo), float(hi)).contiguous()

    def _sigma(
        self,
        iteration: int,
        iterations: int,
        initial: float,
        final: float,
    ) -> float:
        if iterations <= 1:
            return float(final)
        t = iteration / float(iterations - 1)
        return float(initial * ((final / initial) ** t))

    def run(
        self,
        initial_params: torch.Tensor,
        decode: Callable[[torch.Tensor], torch.Tensor],
        evaluate: Callable[[list[torch.Tensor], list[torch.Tensor]], list[EvalResult]],
        stage: str,
        bounds: tuple[float, float] | None,
        iterations: int,
        population_pairs: int,
        sigma_initial: float,
        sigma_final: float,
        learning_rate: float,
        keep_every: int,
        metadata: dict | None = None,
    ) -> SearchResult:
        params = self._clamp(initial_params.detach().cpu().to(torch.float32), bounds)
        dim = params.numel()

        current_voice = decode(params)
        current_eval = evaluate([current_voice], [params])[0]
        current_candidate = make_candidate(
            stage,
            current_voice,
            current_eval,
            params=params,
            iteration=0,
            metadata=metadata,
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

            voices = [decode(candidate_params) for candidate_params in param_list]
            evals = evaluate(voices, param_list)
            losses = torch.tensor(
                [eval_result.total_loss for eval_result in evals], dtype=torch.float32
            )

            for candidate_params, voice, eval_result in zip(param_list, voices, evals):
                candidate = make_candidate(
                    stage,
                    voice,
                    eval_result,
                    params=candidate_params,
                    iteration=iteration,
                    metadata=metadata,
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
                "best_candidate_id": best_candidate.candidate_id,
                "best_voice_hash": best_candidate.voice_hash,
            }
            history.append(row)

            iterator.set_postfix(
                {
                    "best": f"{best_candidate.eval.total_loss:.5f}",
                    "sigma": f"{sigma:.4f}",
                }
            )

            if keep_every > 0 and iteration % keep_every == 0:
                checkpoint_voice = decode(params)
                checkpoint_eval = evaluate([checkpoint_voice], [params])[0]
                checkpoint = make_candidate(
                    stage,
                    checkpoint_voice,
                    checkpoint_eval,
                    params=params,
                    iteration=iteration,
                    metadata=metadata,
                )
                checkpoint_candidates.append(checkpoint)
                if checkpoint.eval.total_loss < best_candidate.eval.total_loss:
                    best_candidate = checkpoint

        final_voice = decode(params)
        final_eval = evaluate([final_voice], [params])[0]
        final_candidate = make_candidate(
            stage,
            final_voice,
            final_eval,
            params=params,
            iteration=iterations,
            metadata=metadata,
        )
        if final_candidate.eval.total_loss < best_candidate.eval.total_loss:
            best_candidate = final_candidate

        candidates = [best_candidate, final_candidate, *checkpoint_candidates]
        unique: dict[str, Candidate] = {
            candidate.candidate_id: candidate for candidate in candidates
        }

        return SearchResult(
            stage=stage,
            candidates=list(unique.values()),
            best=best_candidate,
            final=final_candidate,
            checkpoints=checkpoint_candidates,
            history=history,
            top=[],
        )


def run_blend_stage(
    top_candidates: list[Candidate],
    objective: VoiceObjective,
    texts: list[str],
    config: SearchConfig,
    metadata: dict | None = None,
) -> SearchResult:
    if not top_candidates:
        raise ValueError("Blend search requires at least one baseline candidate")

    voices = [candidate.voice for candidate in top_candidates]
    logits = torch.full((len(voices),), -4.0, dtype=torch.float32)
    logits[0] = 4.0
    nes = AntitheticNES(config)

    def decode(params: torch.Tensor) -> torch.Tensor:
        return convex_blend(params, voices)

    def evaluate(candidate_voices, params_list):
        return objective.evaluate_voices(
            candidate_voices,
            texts,
            include_latent_penalties=False,
        )

    return nes.run(
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
        keep_every=config.keep_every,
        metadata=metadata,
    )


def run_latent_stage(
    manifold,
    initial_z: torch.Tensor,
    objective: VoiceObjective,
    texts: list[str],
    config: SearchConfig,
    metadata: dict | None = None,
) -> SearchResult:
    nes = AntitheticNES(config)

    def decode(params: torch.Tensor) -> torch.Tensor:
        return manifold.decode(params, clamp=True)

    def evaluate(candidate_voices, params_list):
        infos = [
            LatentInfo(z=manifold.clamp_z(params), manifold=manifold)
            for params in params_list
        ]
        return objective.evaluate_voices(
            candidate_voices,
            texts,
            latent_info=infos,
            include_latent_penalties=True,
        )

    return nes.run(
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
        keep_every=config.keep_every,
        metadata=metadata,
    )


def validate_candidates(
    candidates: list[Candidate],
    objective: VoiceObjective,
    texts: list[str],
    metadata: dict | None = None,
    *,
    include_search_penalties: bool = False,
    latent_info_resolver: Callable[[Candidate], LatentInfo | None] | None = None,
) -> SearchResult:
    if not candidates:
        raise ValueError("No candidates provided for validation")

    voices = [candidate.voice for candidate in candidates]

    latent_infos: list[LatentInfo | None] | None = None
    if include_search_penalties:
        if latent_info_resolver is None:
            raise ValueError(
                "include_search_penalties=True requires latent_info_resolver"
            )
        latent_infos = [latent_info_resolver(candidate) for candidate in candidates]

    evals = objective.evaluate_voices(
        voices,
        texts,
        latent_info=latent_infos,
        include_latent_penalties=include_search_penalties,
    )

    validated = [
        make_candidate(
            stage=f"validation:{candidate.stage}",
            voice=candidate.voice,
            eval_result=eval_result,
            params=candidate.params,
            iteration=candidate.iteration,
            metadata={
                **candidate.metadata,
                **(metadata or {}),
                "validation_includes_search_penalties": include_search_penalties,
            },
        )
        for candidate, eval_result in zip(candidates, evals)
    ]
    validated.sort(key=lambda candidate: candidate.eval.total_loss)

    return SearchResult(
        stage="validation",
        candidates=validated,
        best=validated[0],
        final=None,
        checkpoints=[],
        history=[],
        top=[],
    )
