"""Deterministic parameter search for fitting (MIT)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SearchResult:
    best_params: dict[str, float]
    best_score: float
    history: list[dict]


def coordinate_descent(
    initial: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    steps: dict[str, float],
    evaluate: Callable[[dict[str, float]], float],
    rounds: int = 3,
    shrink: float = 0.5,
    maximize: bool = False,
) -> SearchResult:
    """Coordinate descent with per-parameter bounds and step sizes.

    - Start from `initial`, clipped into `bounds`.
    - For each round and each parameter (sorted by name), try +step and -step;
      keep the best candidate; then multiply that parameter's step by `shrink`.
    - `evaluate` returns a score; lower is better unless `maximize=True`.
    - `history` records every evaluated candidate as
      {"params": dict, "score": float, "round": int}; the clipped start point
      is evaluated once and logged with round 0.
    - Raise ValueError when initial/bounds/steps keys differ or rounds < 1.
    """
    if set(initial) != set(bounds) or set(initial) != set(steps):
        raise ValueError("initial, bounds and steps must have identical key sets")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be an integer >= 1")
    if not 0.0 < shrink <= 1.0:
        raise ValueError("shrink must be within (0, 1]")
    for name, value in steps.items():
        if value <= 0:
            raise ValueError(f"step must be positive for {name!r}")

    def clip(name: str, value: float) -> float:
        low, high = bounds[name]
        return min(max(value, low), high)

    def better(score: float, reference: float) -> bool:
        return score > reference if maximize else score < reference

    for name, (low, high) in bounds.items():
        if low > high:
            raise ValueError(f"lower bound exceeds upper bound for {name!r}: {(low, high)!r}")

    names = sorted(initial)
    current: dict[str, float] = {name: clip(name, float(initial[name])) for name in names}
    step_size: dict[str, float] = {name: float(steps[name]) for name in names}

    history: list[dict] = []
    current_score = float(evaluate(dict(current)))
    history.append({"params": dict(current), "score": current_score, "round": 0})
    best_params: dict[str, float] = dict(current)
    best_score = current_score

    for rnd in range(1, rounds + 1):
        for name in names:
            base = current  # both trials step away from the same point
            for delta in (step_size[name], -step_size[name]):
                trial = dict(base)
                trial[name] = clip(name, base[name] + delta)
                score = float(evaluate(dict(trial)))
                history.append({"params": dict(trial), "score": score, "round": rnd})
                if better(score, current_score):  # ties keep the incumbent
                    current = trial
                    current_score = score
                    best_params = dict(trial)
                    best_score = score
            step_size[name] *= shrink

    return SearchResult(best_params=best_params, best_score=best_score, history=history)
