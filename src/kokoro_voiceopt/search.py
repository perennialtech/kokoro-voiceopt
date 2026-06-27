from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from .config import SearchConfig
from .corpus import VoiceCorpus
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
    metadata: dict

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
            "metadata": self.metadata,
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
    metadata: dict | None = None,
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
        metadata=dict(metadata or {}),
    )


def save_candidate_artifacts(candidate: Candidate, candidate_dir: str | Path) -> None:
    candidate_dir = Path(candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    base = candidate_dir / candidate.candidate_hash

    with open(base.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump(candidate.to_dict(), file, indent=2, ensure_ascii=False)

    torch.save(
        {
            "voice": candidate.voice.cpu().to(torch.float32).contiguous(),
            "params": (
                None
                if candidate.params is None
                else candidate.params.cpu().to(torch.float32).contiguous()
            ),
            "candidate": candidate.to_dict(),
        },
        base.with_suffix(".pt"),
    )


def _append_jsonl(path: Path | None, row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_checkpoint(candidate: Candidate, checkpoint_dir: Path | None) -> None:
    if checkpoint_dir is None:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = candidate.stage.replace(":", "_").replace("/", "_")
    iteration = candidate.iteration if candidate.iteration is not None else 0
    torch.save(
        {
            "voice": candidate.voice.cpu().to(torch.float32).contiguous(),
            "params": (
                None
                if candidate.params is None
                else candidate.params.cpu().to(torch.float32).contiguous()
            ),
            "candidate": candidate.to_dict(),
        },
        checkpoint_dir
        / f"{safe_stage}_{iteration:06d}_{candidate.candidate_hash[:10]}.pt",
    )


def run_baseline_scan(
    corpus: VoiceCorpus,
    objective: VoiceObjective,
    texts: list[str],
    top_k: int,
    metadata: dict | None = None,
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
            metadata=metadata,
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
        metadata: dict | None = None,
        history_path: Path | None = None,
        checkpoint_dir: Path | None = None,
        candidate_dir: Path | None = None,
        validation_texts: list[str] | None = None,
        validation_evaluate: (
            Callable[[list[torch.Tensor]], list[CandidateEval]] | None
        ) = None,
    ) -> NESResult:
        params = self._clamp(initial_params.detach().cpu().to(torch.float32), bounds)
        dim = params.numel()

        if history_path is not None and history_path.exists():
            history_path.unlink()

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

            voices = [decode(p) for p in param_list]
            evals = evaluate(voices, param_list)
            losses = torch.tensor([e.total_loss for e in evals], dtype=torch.float32)

            for p, voice, eval_result in zip(param_list, voices, evals):
                candidate = make_candidate(
                    stage,
                    voice,
                    eval_result,
                    params=p,
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
                "best_candidate_hash": best_candidate.candidate_hash,
            }

            if (
                validation_texts
                and validation_evaluate is not None
                and self.config.validate_every > 0
                and iteration % self.config.validate_every == 0
            ):
                validation_eval = validation_evaluate([best_candidate.voice])[0]
                row["validation_loss"] = validation_eval.total_loss
                row["validation_similarity"] = validation_eval.mean_similarity

            history.append(row)
            _append_jsonl(history_path, row)

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
                    metadata=metadata,
                )
                checkpoint_candidates.append(checkpoint)
                _save_checkpoint(checkpoint, checkpoint_dir)
                if candidate_dir is not None:
                    save_candidate_artifacts(checkpoint, candidate_dir)
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

        if candidate_dir is not None:
            save_candidate_artifacts(best_candidate, candidate_dir)
            save_candidate_artifacts(final_candidate, candidate_dir)

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
    metadata: dict | None = None,
    history_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    candidate_dir: Path | None = None,
    validation_texts: list[str] | None = None,
) -> BlendResult:
    if not top_candidates:
        raise ValueError("Blend search requires at least one baseline candidate")

    voices = [candidate.voice for candidate in top_candidates]
    logits = torch.full((len(voices),), -4.0, dtype=torch.float32)
    logits[0] = 4.0
    nes = AntitheticNES(config)

    def decode(params: torch.Tensor) -> torch.Tensor:
        return convex_blend(params, voices)

    def evaluate(candidate_voices, params_list):
        return objective.evaluate_voices(candidate_voices, texts)

    def validation_evaluate(candidate_voices):
        return objective.evaluate_voices(candidate_voices, validation_texts or texts)

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
        metadata=metadata,
        history_path=history_path,
        checkpoint_dir=checkpoint_dir,
        candidate_dir=candidate_dir,
        validation_texts=validation_texts,
        validation_evaluate=validation_evaluate,
    )

    return BlendResult(
        result=result,
        best_candidate=result.best_candidate,
        final_candidate=result.final_candidate,
    )


def run_latent_search(
    manifold,
    initial_z: torch.Tensor,
    objective: VoiceObjective,
    texts: list[str],
    config: SearchConfig,
    metadata: dict | None = None,
    history_path: Path | None = None,
    checkpoint_dir: Path | None = None,
    candidate_dir: Path | None = None,
    validation_texts: list[str] | None = None,
) -> LatentResult:
    nes = AntitheticNES(config)

    def decode(params: torch.Tensor) -> torch.Tensor:
        return manifold.decode(params, clamp=True)

    def evaluate(candidate_voices, params_list):
        infos = [
            LatentInfo(z=manifold.clamp_z(p), manifold=manifold) for p in params_list
        ]
        return objective.evaluate_voices(candidate_voices, texts, latent_info=infos)

    def validation_evaluate(candidate_voices):
        return objective.evaluate_voices(candidate_voices, validation_texts or texts)

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
        metadata=metadata,
        history_path=history_path,
        checkpoint_dir=checkpoint_dir,
        candidate_dir=candidate_dir,
        validation_texts=validation_texts,
        validation_evaluate=validation_evaluate,
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
    metadata: dict | None = None,
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
            metadata=metadata or candidate.metadata,
        )
        for candidate, eval_result in zip(candidates, evals)
    ]
    validated.sort(key=lambda c: c.eval.total_loss)

    return ValidationResult(
        candidates=validated,
        best_candidate=validated[0],
    )
