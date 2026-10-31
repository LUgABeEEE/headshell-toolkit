from __future__ import annotations

import struct
from pathlib import Path

VEC = tuple[float, float, float]


def write_binary_stl(path: str | Path, triangles: list[tuple[VEC, VEC, VEC]]) -> Path:
    path = Path(path)
    with path.open("wb") as f:
        f.write(b"headshell test".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for v in (a, b, c):
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))
    return path


def cube_triangles(size: float = 10.0) -> list[tuple[VEC, VEC, VEC]]:
    s = size
    v = [
        (0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
        (0, 0, s), (s, 0, s), (s, s, s), (0, s, s),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2),  # bottom
        (4, 5, 6), (4, 6, 7),  # top
        (0, 1, 5), (0, 5, 4),  # front
        (1, 2, 6), (1, 6, 5),  # right
        (2, 3, 7), (2, 7, 6),  # back
        (3, 0, 4), (3, 4, 7),  # left
    ]
    return [(v[a], v[b], v[c]) for a, b, c in faces]


def write_cube_stl(path: str | Path, size: float = 10.0) -> Path:
    return write_binary_stl(path, cube_triangles(size))
