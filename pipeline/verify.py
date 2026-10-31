"""STL verification stage (MIT)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from core.geometry import mesh_stats, parse_binary_stl

@dataclass
class VerifyReport:
    ok: bool
    checks: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

def verify_stl(path: str | Path, expected_bbox_mm: tuple[float, float, float] | None = None,
               bbox_tolerance_mm: float = 1.0, require_watertight: bool = True) -> VerifyReport:
    """Check a binary STL: watertight, non-empty, positive volume, and optional bbox.

    Use core.geometry.parse_binary_stl + mesh_stats. Errors are human-readable strings.
    """
    path = Path(path)
    checks: dict = {}
    errors: list[str] = []
    try:
        stats = mesh_stats(parse_binary_stl(str(path)))
    except Exception as exc:  # noqa: BLE001 - boundary
        return VerifyReport(ok=False, checks=checks, errors=[f"cannot read STL: {exc}"])
    checks["triangles"] = stats.get("triangles")
    checks["watertight"] = bool(stats.get("watertight"))
    checks["boundary_edges"] = stats.get("boundary_edges")
    checks["volume_cm3"] = stats.get("volume_cm3")
    checks["bbox_mm"] = stats.get("bbox_mm")
    if require_watertight and not checks["watertight"]:
        errors.append("mesh is not watertight")
    if not checks["triangles"]:
        errors.append("mesh has no triangles")
    if (checks["volume_cm3"] or 0) <= 0:
        errors.append("mesh volume must be positive")
    if expected_bbox_mm is not None:
        actual = stats.get("bbox_mm") or [0, 0, 0]
        for axis, (got, want) in enumerate(zip(actual, expected_bbox_mm)):
            if abs(float(got) - float(want)) > bbox_tolerance_mm:
                errors.append(f"bbox axis {axis}: {got:.3f} != {want:.3f} (+-{bbox_tolerance_mm})")
    checks["expected_bbox_mm"] = list(expected_bbox_mm) if expected_bbox_mm else None
    return VerifyReport(ok=not errors, checks=checks, errors=errors)
