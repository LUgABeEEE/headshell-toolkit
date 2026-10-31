"""Measurement extraction and comparison for M2 fitting (MIT).

Landmarks are 2D points in image pixels; `scale_mm_per_px` converts to millimetres.
Names are our own; do not import naming from any GPL project.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

Point = tuple[float, float]

REQUIRED_LANDMARKS = ("eye_l", "eye_r", "jaw_l", "jaw_r", "mouth_l", "mouth_r", "brow_mid", "chin")


@dataclass(frozen=True)
class Measurement:
    name: str
    value_mm: float
    weight: float = 1.0


def measure_face(landmarks: Mapping[str, Point], scale_mm_per_px: float = 1.0) -> dict[str, Measurement]:
    """Compute the face measurement vector from named 2D landmarks.

    Required keys: eye_l, eye_r, jaw_l, jaw_r, mouth_l, mouth_r, brow_mid, chin.
    Returned measurements (all in mm, weight 1.0):
      ipd_mm           distance between the two eye centres
      face_width_mm    distance between jaw_l and jaw_r
      mouth_width_mm   distance between mouth_l and mouth_r
      face_height_mm   distance between brow_mid and chin
    Raise ValueError when a required landmark is missing or scale_mm_per_px <= 0.
    """
    if scale_mm_per_px <= 0.0:
        raise ValueError(f"scale_mm_per_px must be positive, got {scale_mm_per_px}")
    for key in REQUIRED_LANDMARKS:
        if key not in landmarks:
            raise ValueError(f"missing required landmark: {key!r}")

    def dist(a: str, b: str) -> float:
        (x1, y1), (x2, y2) = landmarks[a], landmarks[b]
        return math.hypot(x2 - x1, y2 - y1) * scale_mm_per_px

    return {
        "ipd_mm": Measurement("ipd_mm", dist("eye_l", "eye_r")),
        "face_width_mm": Measurement("face_width_mm", dist("jaw_l", "jaw_r")),
        "mouth_width_mm": Measurement("mouth_width_mm", dist("mouth_l", "mouth_r")),
        "face_height_mm": Measurement("face_height_mm", dist("brow_mid", "chin")),
    }


def measurement_distance(
    target: Mapping[str, Measurement],
    predicted: Mapping[str, Measurement],
    relative: bool = True,
) -> float:
    """Weighted RMS error between two measurement vectors (lower is better).

    - Names must match exactly; raise ValueError otherwise.
    - Relative error per measurement is |t - p| / max(t, 1e-9); absolute is |t - p|.
    - Weighted RMS: sqrt(sum(w * e^2) / sum(w)) with weights from the target.
    - Raise ValueError on empty input, negative weights, or non-positive total weight.
    """
    if not target:
        raise ValueError("measurement_distance needs at least one measurement")
    if set(target) != set(predicted):
        missing = sorted(set(target) - set(predicted))
        unexpected = sorted(set(predicted) - set(target))
        raise ValueError(
            f"measurement names differ: missing={missing}, unexpected={unexpected}"
        )
    sq = 0.0
    w_sum = 0.0
    for name, t in target.items():
        p = predicted[name]
        if t.weight < 0.0:
            raise ValueError(f"measurement weight must be non-negative: {name!r}")
        e = abs(t.value_mm - p.value_mm)
        if relative:
            e /= max(t.value_mm, 1e-9)
        sq += t.weight * e * e
        w_sum += t.weight
    if w_sum <= 0.0:
        raise ValueError("total measurement weight must be positive")
    return math.sqrt(sq / w_sum)
