# -*- coding: utf-8 -*-
"""M1b 头壳打印验证（纯标准库，Python 3.9+，Windows 可用）

用法:
    python verify_print_m1b.py <shell_full.stl> [--json out.json]

测什么:
  1) 拓扑: V/E/F、欧拉示性数 χ、连通分量、边使用统计（是否闭合/流形/绕向一致）
  2) 开口清单: 由 χ 推出的贯通开口总数
  3) 体积/面积/悬垂（悬垂用character-a法线，不信任 STL 里存的法线）
  4) 正面开口图（眼孔/嘴孔）与背面开口图（通风孔阵列），并给出尺寸与中心
  5) 壁厚: 沿 -Y 的"两跳"射线（含斜切样本，属上偏测量）
  6) 0.2 mm 分层扫掠: 空层/断层/轮廓奇偶

方法说明:
  所有射线沿 ±Y，用 (x,z) 二维桶加速——沿 Y 的射线只可能命中 (x,z) 包围盒
  覆盖该点的三角形。判断"某处壁面是否存在"用的是**最外表面判据**：
  从正面外侧沿 -Y 打，若最外一次命中都在 y<0（中平面之后）→ 前壁缺失；
  从背面外侧沿 +Y 打，若最外一次命中都在 y>0 → 后壁缺失（即通风孔贯通）。
"""
from __future__ import annotations

import json
import math
import struct
import sys
from array import array

TRI = struct.Struct("<12fH")

# ---- 报告用的基准（本包 manifest_full.json / README_HANDOFF.md 的声明值）----
# The declared figures of a build are part of the design, not of this script:
# pass them in (--declared path.json) instead of reading them from here.
DECLARED: dict = {}


def load_declared(path: str | None) -> dict:
    """Read a build's declared figures from a file supplied by the caller."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class Mesh:
    def __init__(self, path):
        with open(path, "rb") as f:
            data = f.read()
        (self.n,) = struct.unpack_from("<I", data, 80)
        if len(data) < 84 + 50 * self.n:
            raise ValueError("不是二进制 STL 或文件被截断")
        keys = ("ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz", "nx", "ny", "nz")
        self.v = {k: array("d") for k in keys}
        app = {k: self.v[k].append for k in keys}
        for i in range(self.n):
            f_ = TRI.unpack_from(data, 84 + 50 * i)
            app["nx"](f_[0]); app["ny"](f_[1]); app["nz"](f_[2])
            app["ax"](f_[3]); app["ay"](f_[4]); app["az"](f_[5])
            app["bx"](f_[6]); app["by"](f_[7]); app["bz"](f_[8])
            app["cx"](f_[9]); app["cy"](f_[10]); app["cz"](f_[11])
        v = self.v
        self.xmin = min(min(v["ax"]), min(v["bx"]), min(v["cx"]))
        self.xmax = max(max(v["ax"]), max(v["bx"]), max(v["cx"]))
        self.ymin = min(min(v["ay"]), min(v["by"]), min(v["cy"]))
        self.ymax = max(max(v["ay"]), max(v["by"]), max(v["cy"]))
        self.zmin = min(min(v["az"]), min(v["bz"]), min(v["cz"]))
        self.zmax = max(max(v["az"]), max(v["bz"]), max(v["cz"]))
        # (x,z) 桶，服务沿 Y 的射线
        self.cs = 8.0
        self.nxc = int((self.xmax - self.xmin) / self.cs) + 2
        self.grid = {}
        for i in range(self.n):
            ix0 = int((min(v["ax"][i], v["bx"][i], v["cx"][i]) - self.xmin) / self.cs)
            ix1 = int((max(v["ax"][i], v["bx"][i], v["cx"][i]) - self.xmin) / self.cs)
            iz0 = int((min(v["az"][i], v["bz"][i], v["cz"][i]) - self.zmin) / self.cs)
            iz1 = int((max(v["az"][i], v["bz"][i], v["cz"][i]) - self.zmin) / self.cs)
            for ix in range(ix0, ix1 + 1):
                for iz in range(iz0, iz1 + 1):
                    self.grid.setdefault(ix * self.nxc + iz, []).append(i)

    def ray_y(self, x, z, y0, dy):
        """射线 (x,y0,z)+t*(0,dy,0) 的全部命中 y。返回按 y 升序、去重后的列表。"""
        v = self.v
        hits = []
        for i in self.grid.get(int((x - self.xmin) / self.cs) * self.nxc + int((z - self.zmin) / self.cs), ()):
            ax, az = v["ax"][i], v["az"][i]
            e1x, e1z = v["bx"][i] - ax, v["bz"][i] - az
            e2x, e2z = v["cx"][i] - ax, v["cz"][i] - az
            det = e1x * e2z - e2x * e1z
            if -1e-9 < det < 1e-9:      # 三角形投影退化（与 Y 平行）
                continue
            wx, wz = x - ax, z - az
            u = (wx * e2z - e2x * wz) / det
            t = (e1x * wz - wx * e1z) / det
            if u < -1e-9 or t < -1e-9 or u + t > 1 + 1e-9:
                continue
            yh = (1 - u - t) * v["ay"][i] + u * v["by"][i] + t * v["cy"][i]
            if (dy > 0 and yh <= y0 + 1e-7) or (dy < 0 and yh >= y0 - 1e-7):
                continue
            hits.append(yh)
        hits.sort()
        out = []
        for h in hits:
            if not out or abs(h - out[-1]) > 1e-4:
                out.append(h)
        return out


def topology(m):
    vid = {}

    def gid(x, y, z):
        k = (x, y, z)                     # 精确比较：STL 里重复顶点是逐位相同的
        j = vid.get(k)
        if j is None:
            j = len(vid); vid[k] = j
        return j

    und, dirc = {}, {}
    for i in range(m.n):
        v = m.v
        a = gid(v["ax"][i], v["ay"][i], v["az"][i])
        b = gid(v["bx"][i], v["by"][i], v["bz"][i])
        c = gid(v["cx"][i], v["cy"][i], v["cz"][i])
        for p, q in ((a, b), (b, c), (c, a)):
            dirc[(p, q)] = dirc.get((p, q), 0) + 1
            k = (p, q) if p < q else (q, p)
            und[k] = und.get(k, 0) + 1
    V, E, F = len(vid), len(und), m.n
    chi = V - E + F
    return {
        "V": V, "E": E, "F": F, "chi": chi,
        "boundary_edges": sum(1 for n in und.values() if n == 1),
        "nonmanifold_edges": sum(1 for n in und.values() if n > 2),
        "flipped_triangles_edge_reuse": sum(1 for n in dirc.values() if n > 1),
        "implied_through_openings": (2 - chi) / 2.0,
    }


def clusters(cells, step):
    pts, seen, out = set(cells), set(), []
    for p in sorted(cells):
        if p in seen:
            continue
        stack, comp = [p], []
        seen.add(p)
        while stack:
            cx, cz = stack.pop()
            comp.append((cx, cz))
            for dx in (-step, 0, step):
                for dz in (-step, 0, step):
                    q = (round(cx + dx, 3), round(cz + dz, 3))
                    if q in pts and q not in seen:
                        seen.add(q); stack.append(q)
        out.append(comp)
    return out


def opening_map(m, x0, x1, z0, z1, step, side):
    """side='front': 从正面外侧沿 -Y，最外命中 y<0 → 前壁缺失（眼/嘴/贯穿缝）
       side='back' : 从背面外侧沿 +Y，最外命中 y>0 → 后壁缺失（通风孔贯通）"""
    y0 = (m.ymax + 15.0, -1.0) if side == "front" else (m.ymin - 15.0, 1.0)
    cells, empty = [], []
    xs = []
    x = x0
    while x <= x1 + 1e-9:
        xs.append(round(x, 3)); x += step
    z = z0
    while z <= z1 + 1e-9:
        for xx in xs:
            h = m.ray_y(xx, round(z, 3), y0[0], y0[1])
            if not h:
                empty.append((xx, round(z, 3)))
            elif (max(h) < 0.0) if side == "front" else (min(h) > 0.0):
                cells.append((xx, round(z, 3)))
        z += step
    return cells, empty, xs


def main(path):
    r = {"file": path}
    m = Mesh(path)
    r["triangles"] = m.n
    r["bbox_mm"] = [round(m.xmax - m.xmin, 3), round(m.ymax - m.ymin, 3), round(m.zmax - m.zmin, 3)]
    r["orientation"] = "-Y = 正面(鼻/嘴) / +Y = 后脑 / +Z = 头顶"

    r["topology"] = topology(m)

    # 体积（散度定理；已证绕向一致、闭合）
    vol = 0.0
    for i in range(m.n):
        v = m.v
        ax, ay, az = v["ax"][i], v["ay"][i], v["az"][i]
        bx, by, bz = v["bx"][i], v["by"][i], v["bz"][i]
        cx, cy, cz = v["cx"][i], v["cy"][i], v["cz"][i]
        vol += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6.0
    r["volume_cm3"] = round(vol / 1000.0, 1)
    r["pla_grams"] = round(vol / 1000.0 * 1.24, 0)

    # 面积 + 悬垂（character-a法线）
    area = over_area = 0.0
    for i in range(m.n):
        v = m.v
        ux, uy, uz = v["bx"][i] - v["ax"][i], v["by"][i] - v["ay"][i], v["bz"][i] - v["az"][i]
        wx, wy, wz = v["cx"][i] - v["ax"][i], v["cy"][i] - v["ay"][i], v["cz"][i] - v["az"][i]
        nx, ny, nz = uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        if not ln:
            continue
        a = 0.5 * ln
        area += a
        if nz / ln < -0.70710678:
            over_area += a
    r["surface_area_cm2"] = round(area / 100.0, 1)
    r["surface_area_one_side_cm2"] = round(area / 200.0, 1)
    r["overhang_area_ratio"] = round(over_area / area, 4)
    r["mean_wall_from_volume_mm"] = round(vol / (area / 2.0), 2)   # 体积 / 单侧面积 ≈ 平均壁厚

    # 正面开口（眼/嘴/贯穿缝）
    cells, empty, _ = opening_map(m, -80.0, 80.0, -115.0, 8.0, 1.0, "front")
    r["front_openings"] = []
    for c in clusters(cells, 1.0):
        if len(c) < 4:
            continue
        xs = [p[0] for p in c]; zs = [p[1] for p in c]
        r["front_openings"].append({
            "x_range": [min(xs), max(xs)], "z_range": [min(zs), max(zs)],
            "size_mm": [round(max(xs) - min(xs), 1), round(max(zs) - min(zs), 1)],
            "center": [round((min(xs) + max(xs)) / 2, 1), round((min(zs) + max(zs)) / 2, 1)],
            "cells": len(c),
        })
    r["front_openings"].sort(key=lambda d: d["center"][0])
    r["front_no_material"] = [[min(p[0] for p in empty), max(p[0] for p in empty)],
                              [min(p[1] for p in empty), max(p[1] for p in empty)]] if empty else None

    # 背面开口（通风孔）
    cells, _, _ = opening_map(m, -85.0, 70.0, 0.0, 135.0, 0.5, "back")
    vents, slivers = [], []
    for c in clusters(cells, 0.5):
        xs = [p[0] for p in c]; zs = [p[1] for p in c]
        w, h = max(xs) - min(xs), max(zs) - min(zs)
        rec = {"center": [round((min(xs) + max(xs)) / 2, 1), round((min(zs) + max(zs)) / 2, 1)],
               "opening_mm": [round(w, 1), round(h, 1)], "cells": len(c)}
        if len(c) < 4:
            continue
        (vents if w <= 9.0 and h <= 9.0 else slivers).append(rec)
    r["vent_count_measured"] = len(vents)
    r["vents"] = sorted(vents, key=lambda d: (round(d["center"][0]), round(d["center"][1])))
    r["vent_columns_x"] = sorted({round(d["center"][0]) for d in vents})
    r["vent_rows_z"] = sorted({round(d["center"][1]) for d in vents})
    r["vent_partial_or_odd"] = slivers

    # 壁厚（-Y 两跳，含斜切样本 → 偏厚）
    th = []
    z = m.zmin + 2.0
    while z < m.zmax - 2.0:
        x = m.xmin + 2.0
        while x < m.xmax - 2.0:
            h = m.ray_y(x, z, m.ymax + 15.0, -1.0)
            for k in range(0, len(h) - 1, 2):
                d = h[k + 1] - h[k]
                if d > 0:
                    th.append(d)
            x += 3.0
        z += 3.0
    th.sort()

    def pc(p):
        if not th:
            return None
        k = (len(th) - 1) * p
        lo = int(k); hi = min(lo + 1, len(th) - 1)
        return round(th[lo] + (th[hi] - th[lo]) * (k - lo), 2)

    r["wall_two_hop_ray"] = {"samples": len(th), "min": pc(0), "p05": pc(.05),
                             "median": pc(.5), "p95": pc(.95), "max": pc(1),
                             "note": "沿 Y 的斜切样本会让读数偏大，只能作为上界参考"}

    # 0.2 mm 分层扫掠
    h_ = 0.2
    nlay = int((m.zmax - m.zmin) / h_) + 1
    buckets = [[] for _ in range(nlay)]
    v = m.v
    for i in range(m.n):
        k0 = int((min(v["az"][i], v["bz"][i], v["cz"][i]) - m.zmin) / h_)
        k1 = int((max(v["az"][i], v["bz"][i], v["cz"][i]) - m.zmin) / h_)
        for k in range(k0, k1 + 1):
            buckets[k].append(i)
    first = last = None
    empty_inside = odd = 0
    for k in range(nlay):
        zl = m.zmin + (k + 0.5) * h_
        pts = 0
        for i in buckets[k]:
            z1, z2, z3 = v["az"][i], v["bz"][i], v["cz"][i]
            pts += (z1 - zl) * (z2 - zl) < 0
            pts += (z2 - zl) * (z3 - zl) < 0
            pts += (z3 - zl) * (z1 - zl) < 0
        if not pts:
            if first is not None:
                empty_inside += 1
            continue
        if first is None:
            first = k
        last = k
        odd += pts % 2
    r["layers_0.2mm"] = {"count": nlay,
                         "span_mm": [round(m.zmin + (first + .5) * h_, 2), round(m.zmin + (last + .5) * h_, 2)],
                         "empty_layers_inside": empty_inside,
                         "layers_with_odd_crossings": odd}

    declared = None
    if "--declared" in sys.argv:
        declared = sys.argv[sys.argv.index("--declared") + 1]
    r["declared"] = load_declared(declared)
    return r


if __name__ == "__main__":
    res = main(sys.argv[1])
    js = json.dumps(res, ensure_ascii=False, indent=1)
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w", encoding="utf-8") as f:
            f.write(js)
    print(js)
