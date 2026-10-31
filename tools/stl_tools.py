#!/usr/bin/env python3
"""numpy-only binary-STL tools: scan, weld, edge statistics, grid decimation.

Used by tools/run_baseline.py (outside Blender) so the heavy number crunching
never touches Blender's address-space cap, and so reported statistics come from
the raw delivered file rather than from a render proxy.

CLI:
    python3 tools/stl_tools.py stats  IN.stl [--json]
    python3 tools/stl_tools.py decimate IN.stl OUT.stl [--target-tris N] [--grid G]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

HDR = 84
REC = 50
REC_DTYPE = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def scan(path: str) -> dict:
    """Cheap header scan; raises if the file is not a plain binary STL."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(HDR)
    if len(head) < HDR:
        raise ValueError(f"{path}: too small to be a binary STL")
    n = int(np.frombuffer(head[80:84], dtype="<u4")[0])
    expected = HDR + n * REC
    if expected != size:
        # some exporters leave junk after the payload; tolerate that
        if size < expected:
            raise ValueError(f"{path}: truncated binary STL ({size} < {expected} B)")
    return {"triangles": n, "bytes": size}


def corners(path: str) -> np.ndarray:
    n = scan(path)["triangles"]
    mm = np.memmap(path, dtype=REC_DTYPE, mode="r", offset=HDR, shape=(n,))
    verts = np.asarray(mm["v"], dtype=np.float32).reshape(-1, 3)
    del mm
    return verts


def weld(verts: np.ndarray, tol: float = 0.002):
    """Merge coincident vertices on a tol-mm grid. Returns (unique_pos, inverse)."""
    lo = verts.min(0)
    q = np.rint((verts - lo) / tol).astype(np.int64)
    span = int(q.max()) + 2
    key = (q[:, 0] * span + q[:, 1]) * span + q[:, 2]
    uniq_key, inv = np.unique(key, return_inverse=True)
    del key, q
    cnt = np.bincount(inv, minlength=len(uniq_key))
    pos = np.empty((len(uniq_key), 3), dtype=np.float32)
    for axis in range(3):
        s = np.bincount(inv, weights=verts[:, axis], minlength=len(uniq_key))
        pos[:, axis] = (s / np.maximum(cnt, 1)).astype(np.float32)
    return pos, inv, cnt


def stats(path: str) -> dict:
    t0 = time.time()
    info = scan(path)
    n = info["triangles"]
    verts = corners(path)
    lo, hi = verts.min(0), verts.max(0)
    pos, inv, cnt = weld(verts)
    del verts
    tri = inv.reshape(-1, 3)
    degenerate = int(((tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])).sum())
    a = np.minimum(tri[:, 0], tri[:, 1])
    b = np.maximum(tri[:, 0], tri[:, 1])
    e0 = a * len(pos) + b
    a = np.minimum(tri[:, 1], tri[:, 2])
    b = np.maximum(tri[:, 1], tri[:, 2])
    e1 = a * len(pos) + b
    a = np.minimum(tri[:, 0], tri[:, 2])
    b = np.maximum(tri[:, 0], tri[:, 2])
    e2 = a * len(pos) + b
    edges, counts = np.unique(np.concatenate([e0, e1, e2]), return_counts=True)
    return {
        "src": path,
        "triangles": n,
        "welded_vertices": int(len(pos)),
        "duplicate_vertex_ratio": round(1.0 - len(pos) / (3.0 * n), 4),
        "degenerate_triangles": degenerate,
        "unique_edges": int(len(edges)),
        "boundary_edges": int((counts == 1).sum()),
        "non_manifold_edges": int((counts > 2).sum()),
        "watertight": bool((counts == 1).sum() == 0 and (counts > 2).sum() == 0),
        "bbox_min": [round(float(v), 2) for v in lo],
        "bbox_max": [round(float(v), 2) for v in hi],
        "dims": [round(float(hi[i] - lo[i]), 2) for i in range(3)],
        "seconds": round(time.time() - t0, 1),
    }


def decimate(src: str, dst: str, grid: float | None = None, target_tris: int = 2_500_000) -> dict:
    t0 = time.time()
    info = scan(src)
    n = info["triangles"]
    verts = corners(src)
    lo, hi = verts.min(0), verts.max(0)
    if grid is None:
        area = float(np.prod(hi - lo) ** (2.0 / 3.0))
        grid = max(0.2, float(np.sqrt(area / max(target_tris, 1) * 6.0)))
    pos, inv, _cnt = weld(verts, tol=grid)
    del verts
    tri = inv.reshape(-1, 3)
    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri = tri[keep]
    if len(tri) == 0:
        raise RuntimeError("decimation removed every triangle; grid too coarse")

    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, np.where(ln == 0, 1, ln)).astype(np.float32)
    out = np.empty(len(tri), dtype=REC_DTYPE)
    out["normal"] = nrm
    out["v"] = np.stack([a, b, c], axis=1)
    out["attr"] = 0
    header = f"decimated from {os.path.basename(src)} grid={grid:.3f}mm".encode()[:80].ljust(80, b" ")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(header)
        fh.write(np.uint32(len(tri)).tobytes())
        out.tofile(fh)
    return {
        "src": src,
        "dst": dst,
        "src_triangles": n,
        "dst_triangles": int(len(tri)),
        "grid_mm": round(grid, 3),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("src")
    d = sub.add_parser("decimate")
    d.add_argument("src")
    d.add_argument("dst")
    d.add_argument("--grid", type=float, default=None)
    d.add_argument("--target-tris", type=int, default=2_500_000)
    args = ap.parse_args()
    if args.cmd == "stats":
        print(json.dumps(stats(args.src), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(decimate(args.src, args.dst, args.grid, args.target_tris), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
