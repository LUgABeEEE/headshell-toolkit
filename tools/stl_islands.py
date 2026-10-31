#!/usr/bin/env python3
"""Connected-component audit for a binary STL (numpy only).

Answers "is this one clean shell, or a pile of pieces glued together": it welds
coincident vertices, groups triangles into connected islands, and reports each
island's size, bbox and share of the mesh.

    python3 tools/stl_islands.py IN.stl [--tol 0.002] [--json OUT.json]

Why numpy rather than Blender: the numbers must describe the delivered file, and
this must survive the 8 GiB Blender address-space cap. Run it with the Blender
bundled interpreter (only numpy is needed):

    .tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11 tools/stl_islands.py IN.stl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stl_tools import corners, scan, weld  # noqa: E402


def components(path: str, tol: float = 0.002) -> dict:
    t0 = time.time()
    info = scan(path)
    verts = corners(path)
    pos, inv, _cnt = weld(verts, tol=tol)
    del verts, _cnt
    tri = inv.reshape(-1, 3).astype(np.int64)
    nv = len(pos)

    parent = np.arange(nv, dtype=np.int64)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:      # path compression
            parent[a], a = root, parent[a]
        return root

    keep = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    tri = tri[keep]
    for a, b, c in tri.tolist():
        ra, rb, rc = find(a), find(b), find(c)
        if rb != ra:
            parent[rb] = ra
        if rc != ra:
            parent[rc] = ra
    roots = np.array([find(i) for i in range(nv)], dtype=np.int64)
    vroot = roots[tri]                      # (T, 3) roots
    # every triangle keeps the root of its first vertex
    tri_root = vroot[:, 0]
    uniq, tri_index = np.unique(tri_root, return_inverse=True)
    counts = np.bincount(tri_index, minlength=len(uniq))

    # per-island bbox from the welded positions of its triangles
    lo_all = np.full((len(uniq), 3), np.inf)
    hi_all = np.full((len(uniq), 3), -np.inf)
    for k in range(3):
        pts = pos[tri[:, k]]
        np.minimum.at(lo_all[:, 0], tri_index, pts[:, 0])
        np.minimum.at(lo_all[:, 1], tri_index, pts[:, 1])
        np.minimum.at(lo_all[:, 2], tri_index, pts[:, 2])
        np.maximum.at(hi_all[:, 0], tri_index, pts[:, 0])
        np.maximum.at(hi_all[:, 1], tri_index, pts[:, 1])
        np.maximum.at(hi_all[:, 2], tri_index, pts[:, 2])

    order = np.argsort(-counts)
    islands = []
    for i in order:
        d = hi_all[i] - lo_all[i]
        islands.append({
            "tris": int(counts[i]),
            "share_pct": round(100.0 * counts[i] / len(tri), 2),
            "dims_mm": [round(float(v), 2) for v in d],
            "bbox_min": [round(float(v), 2) for v in lo_all[i]],
            "bbox_max": [round(float(v), 2) for v in hi_all[i]],
            "volume_proxy_mm3": round(float(d[0] * d[1] * d[2]), 1),
        })
    dims = np.array([i["volume_proxy_mm3"] for i in islands])
    return {
        "src": str(Path(path).resolve()),
        "bytes": info["bytes"],
        "triangles": int(len(tri)),
        "welded_vertices": int(nv),
        "weld_tolerance_mm": tol,
        "islands": len(islands),
        "largest_share_pct": islands[0]["share_pct"] if islands else 0.0,
        "top5": islands[:5],
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--json")
    args = ap.parse_args()
    out = components(args.src, args.tol)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
