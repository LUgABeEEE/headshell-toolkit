"""Conservative clearance checks between an outer shell and the inner proxy (MIT).

This is an axis-aligned bounding-box pre-check. Exact mesh distances belong in the
Blender kernel; this module catches gross sizing errors early and is unit-testable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class ClearanceReport:
    ok: bool
    gaps_mm: Vec3
    required_mm: float


def bbox_clearance(outer_bbox_mm: Vec3, inner_bbox_mm: Vec3, required_mm: float) -> ClearanceReport:
    """Per-axis gap = (outer - inner) / 2; ok when the smallest gap >= required_mm.

    Raise ValueError when any bbox dimension is non-positive or required_mm is negative.
    The check is conservative: passing does not prove a curved shell clears the head.
    """
    _validate_bbox(outer_bbox_mm, "outer_bbox_mm")
    _validate_bbox(inner_bbox_mm, "inner_bbox_mm")
    if isinstance(required_mm, bool) or not isinstance(required_mm, (int, float)):
        raise ValueError("required_mm must be a number")
    if required_mm < 0:
        raise ValueError("required_mm must be >= 0")
    gaps_mm = tuple((outer_bbox_mm[i] - inner_bbox_mm[i]) / 2 for i in range(3))
    ok = min(gaps_mm) >= required_mm
    return ClearanceReport(ok, gaps_mm, float(required_mm))


def _validate_bbox(bbox: Vec3, name: str) -> None:
    """Raise ValueError unless bbox has exactly 3 positive numeric entries."""
    try:
        entries = tuple(bbox)
    except TypeError as exc:
        raise ValueError(f"{name} must be a 3-tuple of positive floats") from exc
    if len(entries) != 3:
        raise ValueError(f"{name} must have exactly 3 entries")
    for value in entries:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} entries must be numbers")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} entries must be finite")
        if value <= 0:
            raise ValueError(f"{name} entries must be positive")
