"""Pure-Python mesh utilities for the product kernel (MIT).

No third-party dependencies: binary STL parsing, watertightness, volume, bbox, slice width.
Units: STL coordinates are millimetres.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

Vec3 = tuple[float, float, float]
Triangle = tuple[Vec3, Vec3, Vec3]

_HEADER_SIZE = 80
_COUNT_SIZE = 4
_TRI_STRUCT = struct.Struct("<12fH")  # normal(3f) + vertices(9f) + attribute(u16) = 50 bytes


def parse_binary_stl(path: str | Path) -> list[Triangle]:
    """Read a binary STL and return triangles as triples of (x, y, z) floats in file units."""
    data = Path(path).read_bytes()
    if len(data) < _HEADER_SIZE + _COUNT_SIZE:
        raise ValueError(f"binary STL too short: {len(data)} bytes")
    (count,) = struct.unpack_from("<I", data, _HEADER_SIZE)
    expected = _HEADER_SIZE + _COUNT_SIZE + _TRI_STRUCT.size * count
    if len(data) < expected:
        raise ValueError(
            f"binary STL too short for count {count}: expected {expected} bytes, got {len(data)}"
        )
    # Trailing bytes are tolerated: some exporters append padding after the last triangle.
    triangles: list[Triangle] = []
    offset = _HEADER_SIZE + _COUNT_SIZE
    for _ in range(count):
        values = _TRI_STRUCT.unpack_from(data, offset)
        offset += _TRI_STRUCT.size
        triangles.append((values[3:6], values[6:9], values[9:12]))
    return triangles


def bbox_mm(triangles: Iterable[Triangle]) -> tuple[Vec3, Vec3, Vec3]:
    """Return (min, max, size) over every vertex, in millimetres (file units are mm)."""
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    for tri in triangles:
        for v in tri:
            for i in range(3):
                c = float(v[i])
                if c < mn[i]:
                    mn[i] = c
                if c > mx[i]:
                    mx[i] = c
    if mn[0] == float("inf"):
        raise ValueError("bbox_mm: no vertices")
    min_v: Vec3 = (mn[0], mn[1], mn[2])
    max_v: Vec3 = (mx[0], mx[1], mx[2])
    size: Vec3 = (mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])
    return min_v, max_v, size


def _vertex_key(v: Vec3) -> tuple[float, float, float]:
    """Rounded coordinates used to dedupe vertices and address edges."""
    return (round(float(v[0]), 9), round(float(v[1]), 9), round(float(v[2]), 9))


def mesh_stats(triangles: Iterable[Triangle]) -> dict:
    """Return a dict with:
    triangles (int), unique_vertices (int), boundary_edges (int), nonmanifold_edges (int),
    watertight (bool), volume_cm3 (float), bbox_mm (tuple[float, float, float])
    """
    tris = list(triangles)
    vertices: set[tuple[float, float, float]] = set()
    edge_uses: dict[tuple, int] = {}
    signed_x6 = 0.0
    for a, b, c in tris:
        ka, kb, kc = _vertex_key(a), _vertex_key(b), _vertex_key(c)
        vertices.update((ka, kb, kc))
        for x, y in ((ka, kb), (kb, kc), (kc, ka)):
            edge = (x, y) if x <= y else (y, x)
            edge_uses[edge] = edge_uses.get(edge, 0) + 1
        # 6 * signed tetrahedron volume of (origin, a, b, c)
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
        bx, by, bz = float(b[0]), float(b[1]), float(b[2])
        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        signed_x6 += (
            ax * (by * cz - bz * cy)
            - ay * (bx * cz - bz * cx)
            + az * (bx * cy - by * cx)
        )
    boundary = sum(1 for n in edge_uses.values() if n == 1)
    nonmanifold = sum(1 for n in edge_uses.values() if n > 2)
    volume_mm3 = abs(signed_x6) / 6.0
    return {
        "triangles": len(tris),
        "unique_vertices": len(vertices),
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "watertight": boundary == 0 and nonmanifold == 0,
        "volume_cm3": volume_mm3 / 1000.0,
        "bbox_mm": bbox_mm(tris)[2],
    }


def slice_width_mm(triangles: Iterable[Triangle], z_mm: float, tolerance_mm: float = 2.0) -> float:
    """Width of the mesh cross-section at z_mm, in millimetres (file units are mm).

    The span is computed from both:
      - vertices whose z is within tolerance_mm of z_mm, and
      - exact intersections of triangle edges with the plane z = z_mm.
    This lets a plane cut through faces report the true width even when no vertex lies
    near that height. Return 0.0 when neither source yields a point.
    """
    xs: list[float] = []
    for tri in triangles:
        for v in tri:
            if abs(float(v[2]) - z_mm) <= tolerance_mm:
                xs.append(float(v[0]))
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            za = float(a[2]) - z_mm
            zb = float(b[2]) - z_mm
            if za * zb < 0.0:  # strict straddle: shared endpoints are covered above
                t = za / (za - zb)
                xs.append(float(a[0]) + t * (float(b[0]) - float(a[0])))
    if not xs:
        return 0.0
    return max(xs) - min(xs)
