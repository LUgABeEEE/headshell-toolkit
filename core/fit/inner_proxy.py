"""Inner wearer-head proxy for clearance checks (MIT).

The reference head is the CC0 Blender Studio head; measurements come from the wearer.
All values are millimetres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class HeadMeasurements:
    width_mm: float
    depth_mm: float
    face_length_mm: float
    ipd_mm: float
    liner_mm: float
    circumference_mm: float | None = None


@dataclass(frozen=True)
class InnerProxySpec:
    scale: Vec3
    bbox_mm: Vec3
    liner_mm: float


def _check_number(name: str, value: float) -> float:
    """Return value as float, or raise ValueError when it is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    """Raise ValueError when value falls outside the inclusive range."""
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}, got {value}")


def fit_reference_head(reference_bbox_mm: Vec3, measurements: HeadMeasurements) -> InnerProxySpec:
    """Scale a reference head bounding box to the wearer and add the liner on every axis.

    - scale = (width/ref_x, depth/ref_y, face_length/ref_z)
    - bbox_mm = (width + 2*liner, depth + 2*liner, face_length + 2*liner)
    Raise ValueError on non-positive reference dimensions or out-of-range measurements:
    Plausibility windows are deliberately generic (see the module constants).
      circumference, when provided, must be 480..650.
    """
    ref = tuple(reference_bbox_mm)
    if len(ref) != 3:
        raise ValueError(f"reference_bbox_mm must have exactly 3 dimensions, got {len(ref)}")
    ref_x, ref_y, ref_z = (
        _check_number(f"reference_bbox_mm[{i}]", v) for i, v in enumerate(ref)
    )
    if ref_x <= 0.0 or ref_y <= 0.0 or ref_z <= 0.0:
        raise ValueError(f"reference_bbox_mm must be positive, got {ref}")

    width = _check_number("width_mm", measurements.width_mm)
    depth = _check_number("depth_mm", measurements.depth_mm)
    face_length = _check_number("face_length_mm", measurements.face_length_mm)
    ipd = _check_number("ipd_mm", measurements.ipd_mm)
    liner = _check_number("liner_mm", measurements.liner_mm)

    _check_range("width_mm", width, 80.0, 260.0)
    _check_range("depth_mm", depth, 100.0, 320.0)
    _check_range("face_length_mm", face_length, 100.0, 400.0)
    _check_range("ipd_mm", ipd, 30.0, 100.0)
    _check_range("liner_mm", liner, 0.0, 50.0)
    if measurements.circumference_mm is not None:
        circumference = _check_number("circumference_mm", measurements.circumference_mm)
        _check_range("circumference_mm", circumference, 300.0, 900.0)

    scale = (width / ref_x, depth / ref_y, face_length / ref_z)
    bbox_mm = (width + 2 * liner, depth + 2 * liner, face_length + 2 * liner)
    return InnerProxySpec(scale, bbox_mm, liner)
