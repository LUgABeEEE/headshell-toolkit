"""Landmark and measurement types plus the fitting objective (MIT)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Landmark:
    name: str
    x: float
    y: float
    weight: float = 1.0


def _index_by_name(landmarks: Iterable[Landmark]) -> dict[str, Landmark]:
    """Map landmark name -> landmark; duplicates are rejected."""
    index: dict[str, Landmark] = {}
    for lm in landmarks:
        if lm.name in index:
            raise ValueError(f"duplicate landmark name: {lm.name!r}")
        index[lm.name] = lm
    return index


def match_by_name(target: Iterable[Landmark], predicted: Iterable[Landmark]) -> list[tuple[Landmark, Landmark]]:
    """Pair landmarks by name. Raise ValueError on duplicate or missing names."""
    target_list = list(target)
    predicted_list = list(predicted)
    t_index = _index_by_name(target_list)
    p_index = _index_by_name(predicted_list)
    if set(t_index) != set(p_index):
        missing = sorted(set(t_index) - set(p_index))
        unexpected = sorted(set(p_index) - set(t_index))
        raise ValueError(f"landmark names differ: missing={missing}, unexpected={unexpected}")
    # Pairs follow target order.
    return [(lm, p_index[lm.name]) for lm in target_list]


def rmse(target: Iterable[Landmark], predicted: Iterable[Landmark]) -> float:
    """Weighted root-mean-square Euclidean distance between matched landmarks.

    Weights come from the target landmark. Raise ValueError on empty input.
    """
    pairs = match_by_name(target, predicted)
    if not pairs:
        raise ValueError("rmse needs at least one landmark pair")
    sq = 0.0
    w_sum = 0.0
    for t, p in pairs:
        if t.weight < 0.0:
            raise ValueError(f"landmark weight must be non-negative: {t.name!r}")
        d2 = (t.x - p.x) ** 2 + (t.y - p.y) ** 2
        sq += t.weight * d2
        w_sum += t.weight
    if w_sum <= 0.0:
        raise ValueError("total landmark weight must be positive")
    return math.sqrt(sq / w_sum)


def combined_score(
    target: Iterable[Landmark],
    predicted: Iterable[Landmark],
    silhouette_iou: float | None = None,
    iou_weight: float = 0.2,
) -> float:
    """Lower is better: rmse + iou_weight * (1 - silhouette_iou).

    silhouette_iou must be within [0, 1] when provided.
    """
    if silhouette_iou is not None and not 0.0 <= silhouette_iou <= 1.0:
        raise ValueError(f"silhouette_iou must be within [0, 1], got {silhouette_iou}")
    base = rmse(target, predicted)
    if silhouette_iou is None:
        return base
    return base + iou_weight * (1.0 - silhouette_iou)
