"""M2 orchestration: fit parameters to a target measurement vector (MIT)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from core.fit.measure import Measurement, measurement_distance
from core.fit.search import SearchResult, coordinate_descent


@dataclass(frozen=True)
class FitResult:
    best_params: dict[str, float]
    best_score: float
    history: list[dict]
    measurements: dict[str, Measurement]


def fit_measurements(
    target: Mapping[str, Measurement],
    predict: Callable[[dict[str, float]], Mapping[str, Measurement]],
    initial: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    steps: dict[str, float],
    rounds: int = 4,
    shrink: float = 0.5,
) -> FitResult:
    """Fit parameters so that predict(params) matches the target measurements.

    - Lower measurement_distance is better.
    - Delegates the search to core.fit.search.coordinate_descent.
    - Raise ValueError when the target is empty.
    - Return measurements = predict(best_params).
    """
    if not target:
        raise ValueError("target must not be empty")

    def evaluate(params: dict[str, float]) -> float:
        return measurement_distance(target, predict(params))

    search: SearchResult = coordinate_descent(
        initial,
        bounds,
        steps,
        evaluate,
        rounds=rounds,
        shrink=shrink,
        maximize=False,
    )
    return FitResult(
        best_params=search.best_params,
        best_score=search.best_score,
        history=search.history,
        measurements=dict(predict(search.best_params)),
    )
