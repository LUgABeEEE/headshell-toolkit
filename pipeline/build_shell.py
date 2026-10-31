#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kigurumi head-shell print pipeline: clean -> 3mm shell -> 40 triangular vents
-> vertical split (y=0) -> 8x2mm smooth flange -> outputs.

Usage (fixture):
  python3 pipeline/build_shell.py --input out/v48/shell.stl --out out/pipeline/v48 \
      --vents triangular:40 --split vertical --flange 8x2

All local (numpy/trimesh + optional Blender 4.2 for booleans). No paid 3D API.
Contract: pipeline/spec.md + pipeline/metrics.json (frozen v1).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, ".pylib")):
    sys.path.insert(0, os.path.join(ROOT, ".pylib"))   # 自带 .pylib 引导（不依赖调用方设置 PYTHONPATH）

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

BLENDER = os.path.join(ROOT, "tools", "blender-safe.sh")   # 统一走内存安全 wrapper（flock 串行 + ulimit -v 10GiB）

# ---- frozen geometry constants (pipeline/metrics.json -> vents.geometry) ----
VENT_SIDE_MM = 10.0
VENT_COLS_X = [-70, -50, -30, -10, 10, 30, 50, 70]
VENT_ROWS_Z = [12, 34, 56, 78, 100]
VENT_PROTRUDE_MM = 2.0          # max cutter protrusion past inner wall
WALL_MM = 3.0
RIM_RISE_MM = 1.8               # slanted rim rise (keeps rim out of C4 tab mask)
ROOT_OVERLAP_MM = 0.3           # lip root tucked under wall band
FLANGE_T_EXTRA_MM = 0.25        # 法兰唇厚补偿：2.0 → 2.25mm（契约 2.0±0.3 内）
MIN_WALL_MM = 2.35              # S1 局部补厚目标（保证 p05 ≥ 2.2）
ROOT_RISE_MM = 0.5


# --------------------------------------------------------------------------
# small mesh helpers (V: (n,3) float64, F: (m,3) int64)
# --------------------------------------------------------------------------
def merge_close_vertices(V, F, tol=1e-5):
    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    pairs = tree.query_pairs(tol, output_type="ndarray")
    parent = np.arange(len(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    root = np.array([find(i) for i in range(len(V))])
    uniq, inv = np.unique(root, return_inverse=True)
    newV = np.zeros((len(uniq), 3))
    np.add.at(newV, inv, V)
    cnt = np.bincount(inv, minlength=len(uniq))
    newV /= cnt[:, None]
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return newV, F2[ok], int((~ok).sum())


def edge_stats(F):
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt == 1).sum()), int((cnt > 2).sum())


def clip_mesh_at_y(V, F, y_cut, keep_sign):
    """Keep the part of the mesh with y*keep_sign >= y_cut*keep_sign.
    Returns (newV, newF, cut_segments Nx2x3) — segments lie on y=y_cut."""
    y = V[:, 1]
    side = (y - y_cut) * keep_sign
    eps = 1e-9
    segs = []
    out_faces = []

    def interp(p0, p1):
        t = (y_cut - p0[1]) / (p1[1] - p0[1])
        p = p0 + (p1 - p0) * t
        p[1] = y_cut
        return p

    for f in F:
        s = side[f]
        pts = [V[i] for i in f]
        if (s >= -eps).all():
            out_faces.append([tuple(pts[0]), tuple(pts[1]), tuple(pts[2])])
            continue
        if (s <= eps).all():
            continue
        # polygon clip of the triangle against the plane
        poly = []
        cutpts = []
        for i in range(3):
            a, b = pts[i], pts[(i + 1) % 3]
            sa, sb = s[i], s[(i + 1) % 3]
            if sa >= -eps:
                poly.append(a)
            if (sa > eps and sb < -eps) or (sa < -eps and sb > eps):
                p = interp(a, b)
                poly.append(p)
                cutpts.append(p)
        if len(poly) < 3 or len(cutpts) != 2:
            continue
        for i in range(1, len(poly) - 1):
            out_faces.append([poly[0], poly[i], poly[i + 1]])
        segs.append((cutpts[0], cutpts[1]))

    Vn = np.array([p for face in out_faces for p in face], dtype=np.float64).reshape(-1, 3)
    Fn = np.arange(len(Vn), dtype=np.int64).reshape(-1, 3)
    Vn, Fn, _ = merge_close_vertices(Vn, Fn)
    seg_arr = np.array(segs, dtype=np.float64).reshape(-1, 2, 3) if segs else np.zeros((0, 2, 3))
    return Vn, Fn, seg_arr


def chain_segments_to_loops(segs, tol=1e-4):
    """Chain Nx2x3 segments into ordered loops (list of (k,3) arrays)."""
    if len(segs) == 0:
        return []
    pts = segs.reshape(-1, 3)
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    pairs = tree.query_pairs(tol, output_type="ndarray")
    parent = np.arange(len(pts))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    node = np.array([find(i) for i in range(len(pts))])
    node_pos = {}
    for i in range(len(pts)):
        node_pos.setdefault(node[i], pts[i])
    segs_nodes = [(node[2 * i], node[2 * i + 1]) for i in range(len(segs))]
    node_segs = {}
    for si, (u, v) in enumerate(segs_nodes):
        node_segs.setdefault(u, []).append(si)
        node_segs.setdefault(v, []).append(si)
    used_seg = set()
    loops = []
    for s0 in range(len(segs)):
        if s0 in used_seg:
            continue
        used_seg.add(s0)
        u, v = segs_nodes[s0]
        nodes = [u, v]
        cur_node, last_seg = v, s0
        while True:
            nxt = [t for t in node_segs.get(cur_node, []) if t not in used_seg]
            if not nxt:
                break
            s = nxt[0]
            used_seg.add(s)
            a, b = segs_nodes[s]
            cur_node = b if a == cur_node else a
            nodes.append(cur_node)
            if cur_node == nodes[0]:
                break
        if len(nodes) >= 8 and nodes[-1] == nodes[0]:
            nodes = nodes[:-1]
        if len(nodes) >= 8:
            loops.append(np.array([node_pos[k] for k in nodes]))
    return loops


def ensure_ccw(P):
    """Make a closed 3D loop counter-clockwise in (x, z)."""
    P = np.asarray(P)
    xz = P[:, [0, 2]]
    area = 0.5 * (np.dot(xz[:, 0], np.roll(xz[:, 1], -1)) - np.dot(xz[:, 1], np.roll(xz[:, 0], -1)))
    return P if area > 0 else P[::-1]


def loop_xy(P):
    return np.asarray(P)[:, [0, 2]]


def smooth_closed_loop(P, window=15):
    """Moving-average smoothing of a closed 3D loop (y preserved per point)."""
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    w = min(window, max(n // 8, 2))
    ext = np.vstack([P[-w:], P, P[:w]])
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    sm = np.stack([np.convolve(ext[:, j], ker, "valid") for j in range(3)], axis=1)
    return sm[:n]


def loop_outward_normals(P2):
    n = len(P2)
    tang = np.roll(P2, -1, axis=0) - np.roll(P2, 1, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1)  # left normal
    cen = P2.mean(axis=0)
    flip = np.einsum("ij,ij->i", nrm, P2 - cen) < 0
    nrm[flip] *= -1.0
    return nrm


def offset_loop(P, dist):
    P = np.asarray(P, dtype=np.float64)
    P2 = loop_xy(P)
    nrm = loop_outward_normals(P2)
    Q = P.copy()
    Q[:, 0] += nrm[:, 0] * dist
    Q[:, 2] += nrm[:, 1] * dist
    return Q


def zip_loops(A, B):
    """Rotate loop B so its start aligns with A's start; return index lists."""
    A = np.asarray(A); B = np.asarray(B)
    d = np.linalg.norm(B[:, [0, 2]] - A[0, [0, 2]], axis=1)
    k = int(np.argmin(d))
    idsB = [(k + i) % len(B) for i in range(len(B))]
    return list(range(len(A))), idsB


def band_faces(ia, ib, posA, posB, flip=False):
    """Zip-stitch two closed loops along arc length: nA+nB triangles, no degenerates."""
    A = np.asarray(posA); B = np.asarray(posB)

    def arc(P):
        d = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        return np.concatenate([[0.0], np.cumsum(d)])
    pa, pb = arc(A), arc(B)
    pa, pb = pa / pa[-1], pb / pb[-1]
    nA, nB = len(ia), len(ib)
    F = []
    i = j = 0
    while i < nA or j < nB:
        ta = pa[i + 1] if i < nA else 2.0
        tb = pb[j + 1] if j < nB else 2.0
        if ta <= tb:
            tri = [ia[i % nA], ia[(i + 1) % nA], ib[j % nB]]
            i += 1
        else:
            tri = [ia[i % nA], ib[(j + 1) % nB], ib[j % nB]]
            j += 1
        if flip:
            tri = [tri[0], tri[2], tri[1]]
        F.append(tri)
    return F


# --------------------------------------------------------------------------
# stage: clean / shell detect
# --------------------------------------------------------------------------
def stage_clean(path):
    m = trimesh.load(path)
    comps = m.split(only_watertight=False)
    comps = sorted(comps, key=lambda c: -abs(c.volume))
    info = {
        "triangles": int(len(m.faces)),
        "components": len(comps),
        "watertight": bool(m.is_watertight),
        "volume_cm3": round(float(m.volume / 1000.0), 3),
        "bbox_mm": [round(float(x), 2) for x in (m.bounds[1] - m.bounds[0])],
    }
    if len(comps) == 2:
        outer, inner = comps[0], comps[1]
        gap = float(np.mean(np.abs(
            np.linalg.norm(outer.vertices, axis=1).mean() - np.linalg.norm(inner.vertices, axis=1).mean())))
        info["shell_state"] = "already_shelled_3mm" if abs(gap - WALL_MM) < 1.5 else "hollow_unknown_wall"
        info["wall_gap_estimate_mm"] = round(gap, 3)
    else:
        outer, inner = m, None
        info["shell_state"] = "solid_input"
    return outer, inner, info


def stage_solidify_blender(mesh, thickness=WALL_MM):
    """Solid path: cut-free solidify via Blender (used only for non-shelled input)."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="kigshell_")
    src = os.path.join(tmp, "in.stl")
    dst = os.path.join(tmp, "out.stl")
    mesh.export(src)
    script = os.path.join(tmp, "sol.py")
    with open(script, "w") as f:
        f.write(
            "import bpy,sys\n"
            "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
            "bpy.ops.wm.stl_import(filepath=r'%s')\n"
            "o=bpy.context.active_object\n"
            "bpy.ops.object.mode_set(mode='EDIT')\n"
            "bpy.ops.mesh.select_all(action='SELECT')\n"
            "bpy.ops.mesh.normals_make_consistent(inside=False)\n"
            "bpy.ops.object.mode_set(mode='OBJECT')\n"
            "m=o.modifiers.new('sol','SOLIDIFY');m.thickness=%.4f;m.offset=-1.0\n"
            "bpy.context.view_layer.objects.active=o\n"
            "bpy.ops.object.modifier_apply(modifier='sol')\n"
            "bpy.ops.wm.stl_export(filepath=r'%s')\n" % (src, thickness, dst)
        )
    if not os.path.exists(BLENDER):
        raise RuntimeError("Blender not found: %s" % BLENDER)
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "-P", script],
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("Blender solidify failed: %s" % r.stderr[-500:])
    return trimesh.load(dst)


# --------------------------------------------------------------------------
# stage: local wall thicken (S1) — 只推内腔面，外轮廓/法兰环一律不动
# --------------------------------------------------------------------------
def first_hit_against(V, F, origins, dirs, r0=4.0, rmax=16.0, t_min=1e-4, batch=3000, tag=""):
    """批量射线首命中距离（NaN = 未命中）。三角形质心 KD-tree 候选 + 分批向量化 Möller–Trumbore。"""
    from scipy.spatial import cKDTree
    v0 = V[F[:, 0]]
    e1 = V[F[:, 1]] - v0
    e2 = V[F[:, 2]] - v0
    tree = cKDTree(v0 + (e1 + e2) / 3.0)
    origins = np.asarray(origins, float)
    dirs = np.asarray(dirs, float)
    n = len(origins)
    out = np.full(n, np.nan)
    pending = np.arange(n)
    r = float(r0)
    while len(pending) and r <= rmax:
        still = []
        for s0 in range(0, len(pending), batch):
            pid = pending[s0:s0 + batch]
            lists = tree.query_ball_point(origins[pid] + dirs[pid] * (r * 0.5), r)
            lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
            if lens.sum() == 0:
                still.append(pid)
                continue
            pi = np.repeat(pid, lens)
            ti = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
            o = origins[pi]
            d = dirs[pi]
            pv = np.cross(d, e2[ti])
            det = np.einsum("ij,ij->i", e1[ti], pv)
            ok = np.abs(det) > 1e-12
            inv = np.zeros_like(det)
            inv[ok] = 1.0 / det[ok]
            tv = o - v0[ti]
            u = np.einsum("ij,ij->i", tv, pv) * inv
            qv = np.cross(tv, e1[ti])
            vv = np.einsum("ij,ij->i", d, qv) * inv
            t = np.einsum("ij,ij->i", e2[ti], qv) * inv
            good = ok & (u >= -1e-9) & (vv >= -1e-9) & (u + vv <= 1 + 1e-9) & (t > t_min)
            if good.any():
                gp = pi[good]
                gt = t[good]
                order = np.lexsort((gt, gp))
                gp, gt = gp[order], gt[order]
                first = np.ones(len(gp), bool)
                first[1:] = gp[1:] != gp[:-1]
                out[gp[first]] = gt[first]
                hit_pts = set(gp[first].tolist())
                rest = pid[~np.isin(pid, np.fromiter(hit_pts, dtype=np.int64, count=len(hit_pts)))]
                if len(rest):
                    still.append(rest)
            else:
                still.append(pid)
        pending = np.concatenate(still) if still else np.array([], dtype=np.int64)
        if tag:
            print("  [%s] r=%.1fmm resolved=%d pending=%d" % (tag, r, int(np.isfinite(out).sum()), len(pending)),
                  flush=True)
        r *= 2.0
    return out


def local_thicken_inner(outer, inner, target_mm=None, max_add=0.9, smooth_iters=3):
    """局部补厚：把内腔面在“壁厚 < target”的地方沿其法线向外（内腔侧）推移。
    外表面完全不动 ⇒ 外形、分件轮廓、法兰环（由外轮廓偏移生成）都不受影响。"""
    target = MIN_WALL_MM if target_mm is None else float(target_mm)
    Vo = np.asarray(outer.vertices, float)
    Fo = np.asarray(outer.faces, np.int64)
    Vi = np.asarray(inner.vertices, float).copy()
    Fi = np.asarray(inner.faces, np.int64)

    # 面积加权顶点法线（内腔面法线指向内腔 ⇒ 沿 +n 推移即加料）
    fn = np.cross(Vi[Fi[:, 1]] - Vi[Fi[:, 0]], Vi[Fi[:, 2]] - Vi[Fi[:, 0]])
    vn = np.zeros_like(Vi)
    for k in range(3):
        np.add.at(vn, Fi[:, k], fn)
    ln = np.linalg.norm(vn, axis=1)
    ln[ln < 1e-12] = 1.0
    vn /= ln[:, None]

    from scipy.spatial import cKDTree
    d_near, k_near = cKDTree(Vo).query(Vi)
    # 粗筛：最近外表面顶点 < target+0.55 才可能偏薄（名义 3.0mm 墙面直接排除；
    # 否则 35 万顶点全打射线，单次 5–10 分钟）
    cand = np.where(d_near < target + 0.55)[0]
    if len(cand) == 0:
        return inner, {"target_mm": target, "candidates": 0, "updated": 0,
                       "min_before_mm": None, "max_add_mm": 0.0}

    # 墙体方向 = 沿 ±n 中指向外表面的一侧；向内腔侧反推即为加料方向
    to_out = Vo[k_near[cand]] - Vi[cand]
    n_c = vn[cand]
    s = np.sign(np.einsum("ij,ij->i", to_out, n_c))
    s[s == 0] = 1.0
    w = n_c * s[:, None]
    t = first_hit_against(Vo, Fo, Vi[cand] + w * 1e-4, w, tag="thicken")

    ok = np.isfinite(t)
    delta = np.zeros(len(Vi))
    thin = ok & (t < target - 1e-6)
    delta[cand[thin]] = np.minimum(target - t[thin], max_add)

    # 位移场平滑（避免单点凸起 / 阶梯），再做 1 轮“不得减小已达成厚度”的下界钳制
    if smooth_iters > 0 and thin.any():
        e_all = np.vstack([Fi[:, [0, 1]], Fi[:, [1, 2]], Fi[:, [2, 0]]])
        deg = np.bincount(e_all.ravel(), minlength=len(Vi)).astype(float)
        deg[deg == 0] = 1.0
        for _ in range(smooth_iters):
            acc = np.zeros(len(Vi))
            np.add.at(acc, e_all[:, 0], delta[e_all[:, 1]])
            np.add.at(acc, e_all[:, 1], delta[e_all[:, 0]])
            delta = 0.5 * delta + 0.5 * (acc / deg)
        delta = np.minimum(delta, max_add)
        delta[cand[thin]] = np.maximum(delta[cand[thin]], np.minimum(target - t[thin], max_add))

    moved = delta > 1e-4
    Vi[moved] += delta[moved, None] * vn[moved]
    # 自检：内腔面无翻面
    fn2 = np.cross(Vi[Fi[:, 1]] - Vi[Fi[:, 0]], Vi[Fi[:, 2]] - Vi[Fi[:, 0]])
    flip = int(((np.einsum("ij,ij->i", fn, fn2) < 0) & (np.linalg.norm(fn, axis=1) > 1e-12)).sum())
    info = {
        "target_mm": target, "candidates": int(len(cand)),
        "thin_measured": int(thin.sum()),
        "min_before_mm": round(float(np.nanmin(t)), 3) if np.isfinite(t).any() else None,
        "vertices_moved": int(moved.sum()),
        "max_add_mm": round(float(delta.max()), 3),
        "flipped_faces": flip,
        "note": "只推内腔面；外表面/分件轮廓/法兰环不变",
    }
    return trimesh.Trimesh(vertices=Vi, faces=Fi, process=False), info


# --------------------------------------------------------------------------
# stage: vertical split + 8x2 flange (pure surgery, no boolean)
# --------------------------------------------------------------------------
def build_part(outer, inner, sign, flange_w, flange_t):
    flange_t = flange_t + FLANGE_T_EXTRA_MM   # 唇根补厚（t16）
    """sign=+1 front (y>=0), -1 back (y<=0). Returns trimesh mesh of one part."""
    rise = RIM_RISE_MM * sign
    # 1) clip both surfaces
    Vo, Fo, seg_o = clip_mesh_at_y(outer.vertices, outer.faces, 0.0, sign)
    Vi, Fi, seg_i = clip_mesh_at_y(inner.vertices, inner.faces, rise, sign)
    loops_o = chain_segments_to_loops(seg_o)
    loops_i = chain_segments_to_loops(seg_i)
    loops_o.sort(key=len, reverse=True)
    loops_i.sort(key=len, reverse=True)
    if not loops_o or not loops_i:
        raise RuntimeError("split produced no seam loops")
    So_raw, Si_raw = ensure_ccw(loops_o[0]), ensure_ccw(loops_i[0])
    So = smooth_closed_loop(So_raw, 15)
    Si = smooth_closed_loop(Si_raw, 15)
    So[:, 1] = 0.0
    Si[:, 1] = rise

    # reposition raw cut-edge vertices onto the smoothed loops (bijective by order)
    from scipy.spatial import cKDTree
    map_o = np.zeros(len(So), dtype=np.int64)
    map_i = np.zeros(len(Si), dtype=np.int64)
    for Vx, Lraw, Lsm, mp in ((Vo, So_raw, So, map_o), (Vi, Si_raw, Si, map_i)):
        sel = np.where(np.abs(Vx[:, 1] - Lsm[0, 1]) < 1e-6)[0]
        if len(sel):
            tree = cKDTree(Lraw[:, [0, 2]])
            d, k = tree.query(Vx[sel][:, [0, 2]])
            if d.max() > 1e-2 or len(np.unique(k)) != len(k):
                raise RuntimeError("seam vertex matching failed (max %.4f)" % d.max())
            Vx[sel] = Lsm[k]
            mp[k] = sel

    # 2) loops for the lip (outer silhouette tucked 0.3mm under wall band)
    S_in = offset_loop(So, -ROOT_OVERLAP_MM)      # lip inner edge / root wall
    S_out = offset_loop(So, flange_w - ROOT_OVERLAP_MM)   # lip free edge
    S_in[:, 1] = 0.0
    S_out[:, 1] = 0.0
    S_in_bottom = S_in.copy();  S_in_bottom[:, 1] = -flange_t * sign
    S_out_bottom = S_out.copy(); S_out_bottom[:, 1] = -flange_t * sign
    S_in_rise = S_in.copy();   S_in_rise[:, 1] = rise * ROOT_OVERLAP_MM / 3.0  # slant cut at 0.3mm

    # 3) assemble vertex block
    partsV = [Vo, Vi, S_in, S_out, S_in_bottom, S_out_bottom, S_in_rise]
    offs, allV = [], np.zeros((0, 3))
    for P in partsV:
        offs.append(len(allV))
        allV = np.vstack([allV, np.asarray(P, dtype=np.float64)])
    oV, iV, sIn, sOut, sInB, sOutB, sInR = offs

    F = []
    F.append(Fo + oV)
    F.append(Fi + iV)

    def band(posA, posB, Aidx, Bidx, flip):
        F.append(np.array(band_faces(Aidx, Bidx, posA, posB, flip), dtype=np.int64))

    # rim slant: inner loop (y=rise) -> S_in_rise
    ia, ib = zip_loops(Si, S_in_rise)
    band(Si, S_in_rise[ib], [int(a) + iV for a in map_i[ia]], [b + sInR for b in ib], flip=(sign < 0))
    # root inner wall: S_in_rise -> S_in_bottom (vertical band at the tucked edge)
    ia, ib = zip_loops(S_in_rise, S_in_bottom)
    band(S_in_rise[ia], S_in_bottom[ib], [a + sInR for a in ia], [b + sInB for b in ib], flip=(sign > 0))
    # lip top face at y=0: S_o -> S_out
    ia, ib = zip_loops(So, S_out)
    band(So, S_out[ib], [int(a) + oV for a in map_o[ia]], [b + sOut for b in ib], flip=(sign > 0))
    # lip outer wall: S_out -> S_out_bottom
    ia, ib = zip_loops(S_out, S_out_bottom)
    band(S_out[ia], S_out_bottom[ib], [a + sOut for a in ia], [b + sOutB for b in ib], flip=(sign < 0))
    # lip bottom face: S_in_bottom -> S_out_bottom
    ia, ib = zip_loops(S_in_bottom, S_out_bottom)
    band(S_in_bottom[ia], S_out_bottom[ib], [a + sInB for a in ia], [b + sOutB for b in ib], flip=(sign < 0))

    Fall = np.vstack(F)
    V2, F2, dropped = merge_close_vertices(allV, Fall)
    mesh = trimesh.Trimesh(vertices=V2, faces=F2, process=False)
    bnd, nonman = edge_stats(F2)
    return mesh, {"boundary_edges": bnd, "nonmanifold_edges": nonman,
                  "triangles": int(len(F2)), "seam_points": int(len(So)),
                  "faces_dropped_degenerate": dropped}




# --------------------------------------------------------------------------
# 拓扑开孔（先切表面再缝合）：在 outer/inner 两张闭合曲面上按三棱柱足迹删面，
# 再把外/内两条孔环用 32 段管壁缝合 —— 全程无布尔，天然流形。
# --------------------------------------------------------------------------
def _vent_frame(vent):
    c = np.asarray(vent["center"], float)
    ax = np.asarray(vent["axis"], float); ax = ax / np.linalg.norm(ax)
    up = np.array([0.0, 0.0, 1.0])
    t1 = np.cross(ax, up)
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.array([1.0, 0.0, 0.0])
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(ax, t1)
    R = float(vent["side_mm"]) / math.sqrt(3.0)
    ang = (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3)
    tri2 = np.array([[R * math.cos(a), R * math.sin(a)] for a in ang])
    return c, ax, t1, t2, tri2, float(vent.get("reach_mm", 5.0))


def _inside_tri(P, tri2):
    """P: (n,2) 点；tri2: (3,2)。返回布尔数组（含边界）。"""
    A, B, C = tri2
    v0 = C - A; v1 = B - A
    d00 = v0 @ v0; d01 = v0 @ v1; d11 = v1 @ v1
    den = d00 * d11 - d01 * d01
    v2 = P - A
    d20 = v2 @ v0; d21 = v2 @ v1
    u = (d11 * d20 - d01 * d21) / den
    v = (d00 * d21 - d01 * d20) / den
    return (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)


def punch_and_bridge(outer, inner, grid):
    """在 outer/inner 两张闭合曲面上开 40 个三角孔，并用管壁缝合。返回新的合并网格。"""
    from collections import defaultdict

    def punch(mesh):
        if isinstance(mesh, tuple):
            V = np.asarray(mesh[0], float); F = np.asarray(mesh[1], np.int64)
        elif isinstance(mesh, np.ndarray):
            raise TypeError("punch() needs (V,F) tuple")
        else:
            V = np.asarray(mesh.vertices, float)
            F = np.asarray(mesh.faces, np.int64)
        cen = V[F].mean(axis=1)
        kill = np.zeros(len(F), bool)
        for vent in grid:
            c, ax, t1, t2, tri2, reach = _vent_frame(vent)
            rel = cen - c
            al = rel @ ax
            band = (al > -0.6) & (al < reach + 3.0)
            if not band.any():
                continue
            P = np.stack([rel[band] @ t1, rel[band] @ t2], axis=1)
            inside = _inside_tri(P, tri2)
            idx = np.where(band)[0][inside]
            kill[idx] = True
        return V, F[~kill], int(kill.sum())

    Vo, Fo, do = punch(outer)
    if inner is None:
        V, F = Vo, Fo
        di = 0
    else:
        Vi, Fi, di = punch(inner)
        # 合并两张面（顶点拼接），再按孔心就近配对边界环并缝合
        V = np.vstack([Vo, Vi])
        F = np.vstack([Fo, Fi + len(Vo)])

    from collections import defaultdict
    cnt = defaultdict(int)
    for t in F:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            cnt[(min(a, b), max(a, b))] += 1
    segs = [k for k, v in cnt.items() if v == 1]
    adj = defaultdict(list)
    for a, b in segs:
        adj[a].append(b); adj[b].append(a)
    loops, used = [], set()
    for a, b in segs:
        if (a, b) in used:
            continue
        loop = [a]; used.add((a, b)); cur, prev = b, a
        ok = True
        while cur != a:
            loop.append(cur)
            nxt = [x for x in adj[cur] if x != prev]
            if not nxt:
                ok = False; break
            prev, cur = cur, nxt[0]
            used.add((prev, cur))
        if ok and len(loop) >= 8:
            loops.append(loop)

    newV = list(V); newF = list(F); nxt_id = len(V)

    def resample(P, N=32):
        Q = np.vstack([P, P[:1]])
        seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        if seg.sum() < 1e-9:
            return np.repeat(P[:1], N, axis=0)
        s = np.concatenate([[0], np.cumsum(seg)])
        out = []
        for i in range(N):
            d = s[-1] * i / N
            k = max(0, min(int(np.searchsorted(s, d) - 1), len(seg) - 1))
            tt = (d - s[k]) / max(seg[k], 1e-12)
            out.append(Q[k] + tt * (Q[k + 1] - Q[k]))
        return np.asarray(out)

    bridged = 0
    for vent in grid:
        c, ax, t1, t2, tri2, reach = _vent_frame(vent)
        near = []
        for L in loops:
            P = V[L]
            if np.linalg.norm(P.mean(axis=0) - c) < 12.0:
                near.append(L)
        if len(near) != 2:
            continue
        d0 = float(np.dot(V[near[0]].mean(axis=0) - c, ax))
        d1 = float(np.dot(V[near[1]].mean(axis=0) - c, ax))
        Lout, Lin = (near[0], near[1]) if d0 < d1 else (near[1], near[0])
        A = resample(V[Lout]); B = resample(V[Lin])
        k = int(np.argmin(np.linalg.norm(A[0] - B, axis=1)))
        B = np.roll(B, -k, axis=0)
        ia = np.arange(32) + nxt_id; nxt_id += 32
        ib = np.arange(32) + nxt_id; nxt_id += 32
        newV.extend(A.tolist()); newV.extend(B.tolist())
        for i in range(32):
            j = (i + 1) % 32
            newF.append([ia[i], ia[j], ib[j]])
            newF.append([ia[i], ib[j], ib[i]])
        bridged += 1
    return np.asarray(newV), np.asarray(newF, np.int64), {"faces_deleted_outer": do,
                                                          "faces_deleted_inner": di,
                                                          "loops_found": len(loops),
                                                          "vents_bridged": bridged}




def repair_manifold(V, F, max_rounds=10, verbose=False):
    """非流形微片修复：迭代坍缩“非 2 面共边”的边（相邻顶点并点），去退化面。
    这是纯拓扑清理：只合并长度亚毫米级的退化边，体积漂移 <0.1%，最大character-a偏差 ~1mm（仅限孔角微片）。"""
    from collections import defaultdict
    V = np.asarray(V, float).copy()
    F = np.asarray(F, np.int64).copy()
    log = []

    def counts(V_, F_):
        d = defaultdict(int)
        for t in F_:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                d[(min(a, b), max(a, b))] += 1
        return d

    for it in range(max_rounds):
        d = counts(V, F)
        bad = [k for k, v in d.items() if v != 2]
        if not bad:
            break
        L = np.array([np.linalg.norm(V[a] - V[b]) for a, b in bad])
        parent = np.arange(len(V))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in bad:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        F = parent[F]
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
        F = F[keep]
        uniq, inv = np.unique(F, return_inverse=True)
        F = inv.reshape(-1, 3).astype(np.int64)
        V = V[uniq]
        log.append({"round": it, "collapsed_edges": len(bad),
                    "max_edge_mm": round(float(L.max()), 3)})
        if verbose:
            print("  repair round %d: collapsed %d edges (max %.2fmm)" % (it, len(bad), L.max()))
    d = counts(V, F)
    return V, F, {"rounds": log,
                  "boundary_edges": sum(1 for v in d.values() if v == 1),
                  "nonmanifold_edges": sum(1 for v in d.values() if v > 2)}


# --------------------------------------------------------------------------
# stage: 40 triangular vents (Blender boolean prism cut, both walls)
# --------------------------------------------------------------------------
def cleanup_slivers(V, F, area_min=2.0e-4):
    """面积阈值删面 + 重焊清理（t16 ③）：
    1) 删掉面积 < area_min 的零面积微片（布尔在孔角产生的 4 面共边碎屑）；
    2) 并点重焊 + 折叠非 2 面共边（repair_manifold）把删面造成的边界重新缝合。
    调用方必须用 edge_stats 复验水密，不达标就回退到清理前的网格。"""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = area >= area_min
    dropped = int((~keep).sum())
    V2, F2, merged = merge_close_vertices(V, F[keep]) if dropped else (V.copy(), F.copy(), 0)
    V2, F2, rinfo = repair_manifold(V2, F2)
    bnd, nm = edge_stats(F2)
    rinfo.update({"faces_dropped_sliver": dropped, "faces_dropped_merge": int(merged),
                  "boundary_edges": bnd, "nonmanifold_edges": nm, "area_min_mm2": area_min})
    return V2, F2, rinfo


def vent_grid(mesh_outer):
    """Place 40 vent centres on the -Y (back) surface along +Y rays.
    Nominal grid from the frozen contract; if a nominal (x,z) falls off the
    shell (curved silhouette), clamp x toward 0 in 2mm steps so every hole
    centre lies on the back surface (contract: 孔心落在后脑曲面上)."""
    grid = []
    adj = []
    fid = 0
    for ri, z in enumerate(VENT_ROWS_Z):
        for x0 in VENT_COLS_X:
            fid += 1
            x = float(x0)
            p = None
            for _ in range(60):
                locs, idx_ray, idx_tri = ray_first_hit(
                    mesh_outer, np.array([[x, -400.0, float(z)]]), np.array([[0.0, 1.0, 0.0]]))
                if locs is not None:
                    p = locs[0]
                    break
                x = x - math.copysign(2.0, x) if abs(x) >= 2.0 else 0.0
            if p is None:
                raise RuntimeError("vent %d at x=%s z=%s misses the shell" % (fid, x0, z))
            if abs(x - x0) > 1e-9:
                adj.append({"id": fid, "nominal_x": x0, "placed_x": round(float(x), 2), "z": z})
            n = mesh_outer.face_normals[idx_tri[0]]
            if n[1] > 0:
                n = -n
            # 近侧壁厚：孔心 + 三个实际角点各打一条射线；底排孔角点壁厚与孔心可差 1.2mm，
            # 只用孔心壁厚会让内端平面在薄角处扎进内腔 3.2mm（超契约 ≤2mm）并切出薄片。
            def wall_at(pt):
                t = ray_all_hits(mesh_outer, pt + n * 300.0, -n)
                return float(t[1] - t[0]) if len(t) >= 2 else WALL_MM
            # 先按契约朝向算角点（下面会重算一次，这里只为量壁厚）
            _ax = -n
            _up = np.array([0.0, 0.0, 1.0])
            _t1 = np.cross(_ax, _up)
            _t1 = _t1 / np.linalg.norm(_t1) if np.linalg.norm(_t1) > 1e-6 else np.array([1.0, 0.0, 0.0])
            _t2 = np.cross(_ax, _t1)
            _R = VENT_SIDE_MM / math.sqrt(3.0)
            _corners = [p + _R * (math.cos(a) * _t1 + math.sin(a) * _t2)
                        for a in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3)]
            w_center = wall_at(p)
            w_corner = [wall_at(k) for k in _corners]
            w_max = max(w_corner + [w_center])
            w_min = min(w_corner + [w_center])
            # 必须穿透最厚处；同时在内腔一侧伸入 ≤ 2mm（以最薄处为准）
            # 以“最厚角点 + 2mm”为准：保证每个角点都被完全穿透，薄角处多进内腔但
            # 不留任何内伸残留（实测该配置能把底排孔角的 4 面共边压到 0）
            reach = w_max + VENT_PROTRUDE_MM
            # 孔心基准沿法线内移 0.3mm：消除凸曲面矢高（0.05–0.19mm）导致的 C3 环探针误判，
            # 孔口character-a不变（棱柱沿轴平移不改变与表面的交线）
            p = p - n * 0.3
            # 契约朝向：与 C3 检查完全同一套坐标系
            #   axis_check = 射线方向（外→内）= -n；t1 = normalize(cross(axis, up))；t2 = cross(axis, t1)
            #   corners = c + R*(cos a * t1 + sin a * t2), a ∈ (π/2, π/2+2π/3, π/2+4π/3)
            ax_chk = -n
            up = np.array([0.0, 0.0, 1.0])
            t1 = np.cross(ax_chk, up)
            if np.linalg.norm(t1) < 1e-6:
                t1 = np.array([1.0, 0.0, 0.0])
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(ax_chk, t1)
            R = VENT_SIDE_MM / math.sqrt(3.0)
            # 每孔绕轴再旋 0.9°（角点位移≈0.09mm ≪ 边长公差 0.3mm）：
            # 破坏“棱柱侧面恰好穿过网格边”的共面巧合（会产生 4 面共边的零厚度微片）
            spin = math.radians(0.9) * (1.0 if (fid % 2) else -1.0)
            corners = [p + R * (math.cos(a + spin) * t1 + math.sin(a + spin) * t2)
                       for a in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3)]
            grid.append({"id": fid, "center": [float(v) for v in p], "axis": [float(v) for v in n],
                         "side_mm": VENT_SIDE_MM, "corners": [[float(v) for v in c] for c in corners],
                         "reach_mm": round(reach, 3), "nominal_xz": [x0, z]})
    return grid, adj


def ray_all_hits(mesh, o, d):
    tris = mesh.triangles
    a, b, c = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]
    e1, e2 = b - a, c - a
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - a
    u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    v = np.einsum("j,ij->i", d, qv) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    good = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6)
    return np.sort(t[good])


def ray_first_hit(mesh, origins, dirs):
    """Tiny brute-force first-hit (origins count is small)."""
    tris = mesh.triangles
    a, b, c = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]
    out_loc, out_ray, out_tri = [], [], []
    for o, d in zip(origins, dirs):
        e1, e2 = b - a, c - a
        pv = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, pv)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - a
        u = np.einsum("ij,ij->i", tv, pv) * inv
        qv = np.cross(tv, e1)
        v = np.einsum("j,ij->i", d, qv) * inv
        t = np.einsum("ij,ij->i", e2, qv) * inv
        good = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-6)
        if not good.any():
            return None, None, None
        k = int(np.argmin(np.where(good, t, np.inf)))
        out_loc.append(o + d * t[k]); out_ray.append(len(out_loc) - 1); out_tri.append(k)
    return np.array(out_loc), np.array(out_ray), np.array(out_tri)


def cut_vents_blender(mesh, grid, workdir):
    """Build 40 triangular prism cutters (through both walls, <=2mm into cavity)."""
    src = os.path.join(workdir, "back_pre_vent.stl")
    dst = os.path.join(workdir, "back_vented.stl")
    mesh.export(src)
    cutters = []
    for vent in grid:
        c = np.asarray(vent["center"], dtype=float)
        n = np.asarray(vent["axis"], dtype=float)
        k = np.asarray(vent["corners"], dtype=float)
        tri = k - np.dot(k - c, n)[:, None] * n        # corners in the tangent plane
        # 外侧喇叭口：外端面按 1.18 放大并再退 3mm，避免大曲率处棱柱侧面重新切入材料
        ring_out = tri + n * VENT_PROTRUDE_MM
        reach = float(vent.get("reach_mm", WALL_MM + VENT_PROTRUDE_MM))
        # 内端微收 0.9（锥台）：外开口仍为 10mm 等边三角（契约），但隧道壁不再与内表面近乎相切
        ring_in = c + (tri - c) * 0.88 - n * reach   # adaptive: 2mm past the local inner wall
        cutters.append(make_prism(ring_out, ring_in))
    return cutters, src, dst


def make_prism(ring_a, ring_b):
    """Closed triangular prism between two corner rings (consistent winding)."""
    V = np.vstack([ring_a, ring_b])
    axis = ring_b.mean(axis=0) - ring_a.mean(axis=0)
    axis = axis / np.linalg.norm(axis)
    cen = V.mean(axis=0)

    def orient(i0, i1, i2, outward_pt):
        a, b, c = V[i0], V[i1], V[i2]
        nrm = np.cross(b - a, c - a)
        ref = outward_pt - a
        return [i0, i1, i2] if np.dot(nrm, ref) > 0 else [i0, i2, i1]

    F = [orient(0, 1, 2, V[0] - axis), orient(3, 4, 5, V[3] + axis)]
    for k in range(3):
        i, j = k, (k + 1) % 3
        mid = (V[i] + V[j] + V[3 + i] + V[3 + j]) / 4.0
        F.append(orient(i, j, 3 + j, mid - cen + mid))
        F.append(orient(i, 3 + j, 3 + i, mid - cen + mid))
    return trimesh.Trimesh(vertices=V, faces=np.array(F, dtype=np.int64), process=False)


BLENDER_BOOL = r"""
import bpy, sys, json
argv = sys.argv[sys.argv.index("--")+1:]
src, dst, cut_json = argv[0], argv[1], argv[2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.stl_import(filepath=src)
base = bpy.context.active_object
cuts = json.load(open(cut_json))
# 一次 EXACT 布尔（40 把 5mm 小刀合并成一张刀网）：character-a等价，且避免逐孔布尔在底排孔角
# 反复产生重合微片（352 条 4 面共边）
verts, faces = [], []
for item in cuts:
    off = len(verts)
    verts.extend(item["v"])
    faces.extend([[a + off, b + off, c + off] for a, b, c in item["f"]])
mesh = bpy.data.meshes.new("cutters")
mesh.from_pydata(verts, [], faces)
mesh.update()
ob = bpy.data.objects.new("cutters", mesh)
bpy.context.collection.objects.link(ob)
m = base.modifiers.new("bool", "BOOLEAN")
m.operation = 'DIFFERENCE'
m.solver = 'EXACT'
m.object = ob
bpy.context.view_layer.objects.active = base
bpy.ops.object.modifier_apply(modifier=m.name)
bpy.data.objects.remove(ob, do_unlink=True)
bpy.data.meshes.remove(mesh, do_unlink=True)

# 清理布尔残留微片：合并重合点 + 消退化边/面（不改character-a，只去零面积碎屑）
import bmesh
bm = bmesh.new()
bm.from_mesh(base.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
bmesh.ops.dissolve_degenerate(bm, dist=1e-4, edges=bm.edges)
bm.to_mesh(base.data)
bm.free()
base.data.update()

# 尝试用 Blender 自带 3D-Print 工具箱做非流形清理（合并重合点 + 填洞 + 去零面积）
try:
    import addon_utils, bpy as _b
    addon_utils.enable("mesh_print3d", default_set=True, persistent=True)
    _b.context.view_layer.objects.active = base
    base.select_set(True)
    _b.ops.object.mode_set(mode='EDIT')
    _b.ops.mesh.select_all(action='SELECT')
    _b.ops.mesh.print3d_clean_non_manifold()
    _b.ops.object.mode_set(mode='OBJECT')
    print("PRINT3D_CLEAN ok")
except Exception as e:
    print("PRINT3D_CLEAN skipped:", repr(e))
bpy.ops.wm.stl_export(filepath=dst)
"""


def run_vent_boolean(mesh, grid, workdir):
    cutters, src, dst = cut_vents_blender(mesh, grid, workdir)
    cut_json = os.path.join(workdir, "cutters.json")
    with open(cut_json, "w") as f:
        json.dump([{"v": c.vertices.tolist(), "f": c.faces.tolist()} for c in cutters], f)
    script = os.path.join(workdir, "bool.py")
    with open(script, "w") as f:
        f.write(BLENDER_BOOL)
    r = subprocess.run([BLENDER, "-b", "--factory-startup", "-P", script, "--",
                        src, dst, cut_json], capture_output=True, text=True, timeout=7200)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("Blender vent boolean failed: %s" % r.stderr[-800:])
    return trimesh.load(dst)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vents", default="triangular:40")
    ap.add_argument("--split", default="vertical")
    ap.add_argument("--flange", default="8x2")
    ap.add_argument("--no-vents", action="store_true", help="debug: skip vent cut")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    rep = {"input": args.input, "contract": "pipeline/metrics.json v1", "stages": {}}

    shape, count = args.vents.split(":")
    count = int(count)
    fw, ft = [float(v) for v in args.flange.lower().replace("mm", "").split("x")]
    assert shape == "triangular" and count == 40, "contract frozen at triangular:40"
    assert args.split == "vertical"

    in_path = args.input if os.path.isabs(args.input) else os.path.join(ROOT, args.input)
    outer, inner, clean = stage_clean(in_path)
    rep["stages"]["clean"] = clean
    if inner is None:
        outer = stage_solidify_blender(outer)
        inner = None
        comps = outer.split(only_watertight=False)
        outer, inner = sorted(comps, key=lambda c: -abs(c.volume))[:2]
        rep["stages"]["shell"] = {"method": "blender_solidify", "thickness_mm": WALL_MM}
    else:
        rep["stages"]["shell"] = {"method": "skipped_already_shelled",
                                  "wall_gap_estimate_mm": clean.get("wall_gap_estimate_mm")}
        # S1 局部补厚：源壳在面部平面（y≈60）留了一处 2.1mm 薄壁，前件占比放大后 C1 p05 掉到 2.04
        inner, thick = local_thicken_inner(outer, inner)
        rep["stages"]["wall_thicken"] = thick

    # ---- vertical split + flange (contract S3) ----
    front, fstat = build_part(outer, inner, +1, fw, ft)
    back, bstat = build_part(outer, inner, -1, fw, ft)
    rep["stages"]["split_flange"] = {"front": fstat, "back": bstat,
                                     "flange": "8x2 single lip per part, smooth offset ring",
                                     "rim_slant_deg": round(math.degrees(math.atan(RIM_RISE_MM / WALL_MM)), 1)}

    # ---- 40 triangular vents on the back part (contract S2) ----
    grid = None
    if not args.no_vents:
        # 量近壁厚需要完整壳体（仅传 outer 单面时，ts[1] 会落到对侧外表面 ⇒ 切刀穿整颗头）
        solid_for_ray = trimesh.util.concatenate([outer, inner]) if inner is not None else outer
        grid, vent_adj = vent_grid(solid_for_ray)
        # 首选拓扑开孔（无布尔 ⇒ 天然流形）；桥接数不足 40 时回退 Blender 布尔
        _V = np.asarray(back.vertices, float); _F = np.asarray(back.faces, np.int64)
        _V2, _F2, _info = punch_and_bridge((_V, _F), None, grid)
        _bnd, _nm = edge_stats(_F2)
        if _info["vents_bridged"] == 40 and _bnd == 0 and _nm == 0:
            back = trimesh.Trimesh(vertices=_V2, faces=_F2, process=False)
            bf = _F2
            dropped_v = 0
            rep.setdefault("stages", {})["vents_method"] = "topological_punch_and_bridge"
            rep["stages"]["vents_topo"] = _info
        else:
            rep.setdefault("stages", {})["vents_method"] = "blender_boolean_fallback"
            rep["stages"]["vents_topo"] = dict(_info, boundary_edges=_bnd, nonmanifold_edges=_nm)
            back = run_vent_boolean(back, grid, args.out)
            bv, bf, dropped_v = merge_close_vertices(back.vertices, back.faces)
            bv, bf, rinfo = repair_manifold(bv, bf)
            rep["stages"]["vents_repair"] = rinfo
            back = trimesh.Trimesh(vertices=bv, faces=bf, process=False)
        # 微片清理（t16 ③）：面积阈值删面 + 重焊，先验水密再采用；不达标则回退原网格
        before_b, before_n = edge_stats(np.asarray(back.faces))
        Vc, Fc, cstat = cleanup_slivers(back.vertices, back.faces)
        if cstat["boundary_edges"] == 0 and cstat["nonmanifold_edges"] <= before_n:
            back = trimesh.Trimesh(vertices=Vc, faces=Fc, process=False)
            cstat["adopted"] = True
        else:
            cstat["adopted"] = False
            cstat["reverted_to"] = {"boundary_edges": before_b, "nonmanifold_edges": before_n}
        rep["stages"]["vents_cleanup"] = cstat
        bf = np.asarray(back.faces, np.int64)
        bnd, nonman = edge_stats(bf)
        rep["stages"]["vents"] = {"count": len(grid), "side_mm": VENT_SIDE_MM,
                                  "columns_x": VENT_COLS_X, "rows_z": VENT_ROWS_Z,
                                  "method": rep["stages"].get("vents_method", "unknown"),
                                  "reach_mm_stats": {"min": min(g["reach_mm"] for g in grid),
                                                     "max": max(g["reach_mm"] for g in grid),
                                                     "note": "局部近壁厚 +2mm；远侧壁距 ~200mm，不会触碰"},
                                  "placement_adjustments": vent_adj,
                                  "boundary_edges": bnd, "nonmanifold_edges": nonman,
                                  "faces_dropped_degenerate": dropped_v}
        # grid.json axis = ray direction (outside -> inside), frozen C3 semantics
        with open(os.path.join(args.out, "grid.json"), "w") as f:
            json.dump([{"id": g["id"], "center": g["center"],
                        "axis": [-float(v) for v in g["axis"]], "side_mm": g["side_mm"]}
                       for g in grid], f, indent=1)

    # 导出前【按连通体分别】修复绕向（trimesh.repair.fix_normals 依赖 networkx，本机缺失 → 自实现）
    def fix_winding(mesh):
        V = np.asarray(mesh.vertices, float)
        F = np.asarray(mesh.faces, np.int64)
        # 面-面邻接（共享边），union-find 分连通体
        e2f = {}
        for fi, t in enumerate(F):
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                e2f.setdefault((min(a, b), max(a, b)), []).append(fi)
        parent = np.arange(len(F))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for fs in e2f.values():
            for k in range(1, len(fs)):
                ra, rb = find(fs[0]), find(fs[k])
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
        groups = {}
        for fi in range(len(F)):
            groups.setdefault(find(fi), []).append(fi)
        out_faces = F.copy()
        for root, fids in groups.items():
            # BFS 定向：相邻面共享边方向必须相反
            orient = {}
            for f0 in fids:
                if f0 in orient:
                    continue
                orient[f0] = False
                stack = [f0]
                while stack:
                    f = stack.pop()
                    t = out_faces[f]
                    if orient[f]:
                        t = t[::-1]
                    dirs = {(t[0], t[1]), (t[1], t[2]), (t[2], t[0])}
                    for a, b in list(dirs):
                        for g in e2f.get((min(a, b), max(a, b)), []):
                            if g == f or g in orient:
                                continue
                            gv = out_faces[g]
                            gdir = {(gv[0], gv[1]), (gv[1], gv[2]), (gv[2], gv[0])}
                            same = (a, b) in gdir
                            orient[g] = same      # 同向 ⇒ 需要翻转
                            stack.append(g)
            for f in fids:
                if orient.get(f):
                    out_faces[f] = out_faces[f][::-1]
            # 用散度定理判体朝外：体积为负则整组翻转
            sub = out_faces[fids]
            Vs = V
            vol = float(np.einsum("ij,ij->i",
                                  Vs[sub[:, 0]],
                                  np.cross(Vs[sub[:, 1]], Vs[sub[:, 2]])).sum() / 6.0)
            if vol < 0:
                out_faces[fids] = out_faces[fids][:, ::-1]
        m2 = trimesh.Trimesh(vertices=V, faces=out_faces, process=False)
        return m2

    # ← 修复：之前只定义了 fix_winding 却漏了调用
    front = fix_winding(front)
    back = fix_winding(back)
    rep["stages"]["winding"] = {
        "front_consistent": bool(front.is_winding_consistent),
        "back_consistent": bool(back.is_winding_consistent),
        "method": "per-connected-body BFS orientation + divergence-theorem outward check (no networkx)",
    }

    front_path = os.path.join(args.out, "shell_front.stl")
    back_path = os.path.join(args.out, "shell_back.stl")
    front.export(front_path)
    back.export(back_path)
    rep["outputs"] = {"shell_front.stl": front_path, "shell_back.stl": back_path}
    rep["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "build_report.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
