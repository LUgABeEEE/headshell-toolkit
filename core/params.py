"""HeadShell parameter contract (product kernel, MIT).

The names here are our own; they intentionally do not mirror any GPL project's field names.

No tuned value is published here. Every millimetre parameter arrives from the
caller (or from a private parameter set); what this module fixes is the *shape*
of the contract plus deliberately loose plausibility bounds, so the kernel can
reject nonsense without disclosing a design.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

SCHEMA = "headshell/params@0"
UNIT_PARAMS = ("jaw_width", "cheek_fullness", "face_length", "eye_openness", "mouth_open")
SPLIT_PLANES = {"coronal", "sagittal"}

# Loose sanity bounds only: wide enough for any plausible wearer, unrelated to
# the values we actually build with.
MM_BOUNDS = {
    "ipd_mm": (40.0, 90.0),
    "head_width_mm": (100.0, 400.0),
    "wall_mm": (0.5, 20.0),
    "liner_mm": (0.0, 50.0),
    "vent_diameter_mm": (0.5, 30.0),
    "vent_spacing_mm": (2.0, 100.0),
}


@dataclass(frozen=True)
class SplitSpec:
    plane: str = "coronal"
    offset_mm: float | None = None

    def validate(self) -> list[str]:
        errors = []
        if self.plane not in SPLIT_PLANES:
            errors.append(f"split.plane must be one of {sorted(SPLIT_PLANES)}")
        if self.offset_mm is None:
            errors.append("split.offset_mm must be configured")
        elif not -400.0 <= self.offset_mm <= 400.0:
            errors.append("split.offset_mm must be within [-400, 400]")
        return errors


@dataclass(frozen=True)
class Params:
    schema: str = SCHEMA
    face_variant: str = "neutral"
    jaw_width: float = 0.0
    cheek_fullness: float = 0.0
    face_length: float = 0.0
    eye_openness: float = 0.0
    mouth_open: float = 0.0
    ipd_mm: float | None = None
    head_width_mm: float | None = None
    wall_mm: float | None = None
    liner_mm: float | None = None
    vent_diameter_mm: float | None = None
    vent_spacing_mm: float | None = None
    split: SplitSpec = field(default_factory=SplitSpec)

    def validate(self) -> list[str]:
        errors = []
        if self.schema != SCHEMA:
            errors.append(f"schema must be {SCHEMA!r}")
        if not self.face_variant.strip():
            errors.append("face_variant must be non-empty")
        for name in UNIT_PARAMS:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{name} must be a number")
            elif not -1.0 <= float(value) <= 1.0:
                errors.append(f"{name} must be within [-1, 1]")
        for name, (lo, hi) in MM_BOUNDS.items():
            value = getattr(self, name)
            if value is None:
                errors.append(f"{name} must be configured (no default is shipped)")
            elif not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{name} must be a number")
            elif not lo <= float(value) <= hi:
                errors.append(f"{name} must be within [{lo:g}, {hi:g}]")
        errors.extend(self.split.validate())
        return errors

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["split"] = asdict(self.split)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Params":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown params: {sorted(unknown)}")
        raw = dict(data)
        split_raw = raw.pop("split", None)
        split = SplitSpec(**split_raw) if isinstance(split_raw, dict) else SplitSpec()
        params = cls(split=split, **raw)
        errors = params.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return params
