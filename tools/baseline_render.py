#!/usr/bin/env python3
"""Turn one modeler head-shell package into measurements + preview renders.

Runs inside Blender (headless):

    tools/blender-safe.sh -b --factory-startup -P tools/baseline_render.py -- \
        --src assets/incoming/<batch>/character-a \
        --out out/baseline/character-a

What it does, per character:
  1. import 前头.stl (front piece) and, if present, 后头.stl (back piece)
  2. measure: verts, tris, bbox (mm), watertightness / boundary edges, symmetry error
  3. render an alpha silhouette with an ortho front camera; the two enclosed holes
     are the eye openings -> eye-hole centres in model mm
  4. locate the painted irises in the modeler preview image and solve a
     scale+offset mapping from model (x, z) to preview pixels, then bake that
     mapping as UVs and re-render the front piece with the painted texture
     (STL carries no UVs / no paint, so this is a planar projection of the
     modeler's own preview - geometry is untouched, only appearance)
  5. write clay turntable views (front / three-quarter / side / back) + report.json

Every write stays under --out. Geometry is never modified: the mesh is only
read, and the projection lives in a UV layer used by a display material.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

SOCKET_FRAME_BEHIND_MM = 60.0   # camera framing offset, not a design value

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


# ---------------------------------------------------------------- utilities
def log(msg: str) -> None:
    print(f"[baseline] {msg}", flush=True)


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_stl(path: str) -> list:
    before = set(bpy.data.objects)
    bpy.ops.wm.stl_import(filepath=path)
    return [o for o in bpy.data.objects if o not in before and o.type == "MESH"]


def mesh_stats(ob) -> dict:
    me = ob.data
    nv, ne = len(me.vertices), len(me.edges)
    loop_edge = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("edge_index", loop_edge)
    counts = np.bincount(loop_edge, minlength=ne)
    boundary = int((counts == 1).sum())
    non_manifold = int((counts > 2).sum())
    tris = sum(len(p.vertices) - 2 for p in me.polygons)
    co = np.empty(nv * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(nv, 3)
    lo, hi = co.min(0), co.max(0)
    return {
        "verts": nv,
        "edges": ne,
        "tris": int(tris),
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
        "watertight": boundary == 0,
        "bbox_min": [round(float(v), 2) for v in lo],
        "bbox_max": [round(float(v), 2) for v in hi],
        "dims": [round(float(hi[i] - lo[i]), 2) for i in range(3)],
    }


def world_bbox(objs: list) -> tuple[Vector, Vector]:
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return lo, hi


def make_material(name: str, color=(0.80, 0.78, 0.76), rough=0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    return mat


def setup_world(strength: float = 1.0, color=(0.30, 0.32, 0.36)) -> None:
    w = bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = strength


def add_lights() -> None:
    for name, rot, energy in [
        ("key", (math.radians(58), 0, math.radians(-32)), 4.0),
        ("fill", (math.radians(70), 0, math.radians(120)), 1.7),
        ("rim", (math.radians(-58), 0, math.radians(160)), 2.2),
    ]:
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = energy
        ld.angle = math.radians(12)
        ob = bpy.data.objects.new(name, ld)
        bpy.context.scene.collection.objects.link(ob)
        ob.rotation_euler = rot


def camera(name: str, ortho: bool = False):
    cd = bpy.data.cameras.new(name)
    # 'AUTO' maps sensor_width to whichever resolution axis is LONGER, which
    # silently broke every mm<->px conversion for portrait renders. Pin the
    # horizontal axis so ortho_scale / FOV always mean the same thing.
    cd.sensor_fit = "HORIZONTAL"
    cd.lens = 72
    cd.clip_start = 1.0
    cd.clip_end = 1e6
    if ortho:
        cd.type = "ORTHO"
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def aim(cam, target: Vector, direction, distance: float, up=Vector((0, 0, 1))) -> None:
    z = Vector(direction).normalized()
    cam.location = Vector(target) + z * distance
    right = up.cross(z).normalized()
    tru = z.cross(right)
    cam.rotation_euler = Matrix([right, tru, z]).transposed().to_4x4().to_quaternion().to_euler()


def cam_fov(cam, res) -> tuple[float, float]:
    """(horizontal, vertical) FOV in radians for a HORIZONTAL-fit camera."""
    cd = cam.data
    fov_h = 2.0 * math.atan(cd.sensor_width / 2.0 / cd.lens)
    fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) * res[1] / res[0])
    return fov_h, fov_v


def fit_distance(cam, dims, res, margin: float = 1.04) -> float:
    """Camera distance that keeps the whole bbox inside the frame at any angle."""
    w, d, h = dims
    radius = 0.5 * math.sqrt(w * w + d * d + h * h)
    fov_h, fov_v = cam_fov(cam, res)
    return radius / math.sin(min(fov_h, fov_v) / 2.0) * margin


def ortho_scale_for(dims, res, margin: float = 1.04) -> float:
    """Ortho scale (horizontal extent in mm) that fits width and height."""
    w, d, h = dims
    aspect = res[0] / res[1]
    return max(w, h * aspect) * margin


def render_to(path: Path, res=(800, 1000), samples=32, denoise=False) -> float:
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.cycles.samples = samples
    sc.cycles.use_denoising = denoise
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(path)
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    return round(time.time() - t0, 1)


# ---------------------------------------------------------------- silhouettes
def silhouette(cam_ob, target: Vector, half_w: float, out_dir: Path, res_x: int = 480,
               res_y: int = 600, hide=()) -> dict:
    """Render the front piece as a binary alpha mask with an ortho camera.

    half_w is the HORIZONTAL half-extent in mm (sensor_fit is pinned to HORIZONTAL).
    `hide` lets the caller exclude the back shell: matching or painting against a
    mask that includes it makes the fit wrong."""
    sc = bpy.context.scene
    sc.render.film_transparent = True
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    sc.render.resolution_x, sc.render.resolution_y = res_x, res_y
    cam_ob.data.sensor_fit = "HORIZONTAL"
    cam_ob.data.ortho_scale = half_w * 2.0
    aim(cam_ob, target, (0, -1, 0), 900.0)
    sc.camera = cam_ob
    tmp = out_dir / "_silhouette.png"
    sc.render.filepath = str(tmp)
    for ob in hide:
        ob.hide_render = True
    bpy.ops.render.render(write_still=True)
    for ob in hide:
        ob.hide_render = False
    img = bpy.data.images.load(str(tmp))
    px = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], 4)
    bpy.data.images.remove(img)
    mask = px[..., 3] > 0.5
    sc.render.film_transparent = False
    return {
        "mask": mask,  # row 0 = bottom (Blender image), z increases with row
        "ortho_scale": half_w * 2.0,
        "res": (res_x, res_y),
        "target": target,
    }


def enclosed_components(mask: np.ndarray) -> list[dict]:
    """Connected zero-regions that do not touch the border = holes."""
    holes = ~mask
    h, w = holes.shape
    seen = np.zeros_like(holes)
    out = []
    for i in range(h):
        for j in range(w):
            if not holes[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            pts = []
            border = False
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                if y == 0 or x == 0 or y == h - 1 or x == w - 1:
                    border = True
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and holes[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if not border and len(pts) > 30:
                out.append({"pts": pts, "size": len(pts)})
    out.sort(key=lambda d: -d["size"])
    return out


def mask_px_to_model(sil: dict, py: float, px: float) -> tuple[float, float]:
    """Convert silhouette pixel to model (x, z) in mm."""
    res_x, res_y = sil["res"]
    scale = sil["ortho_scale"] / res_x  # mm per pixel
    tx, _, tz = sil["target"]
    x = tx + (px + 0.5 - res_x / 2.0) * scale
    z = tz + (py + 0.5 - res_y / 2.0) * scale
    return float(x), float(z)


# ---------------------------------------------------------------- painting
def preview_candidates(src: Path) -> list[Path]:
    DECAL_PREFIXES = ("左眼", "右眼")
    DECAL_PARTS = ("睫毛", "眼片", "眼球")
    return [
        p
        for p in sorted(src.iterdir())
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        and not p.name.startswith(DECAL_PREFIXES)
        and not any(part in p.stem for part in DECAL_PARTS)
    ]


def find_preview(src: Path) -> Path | None:
    """Pick the preview that actually carries paint.

    Deliveries often ship two images: a painted design sheet and a plain clay
    screenshot of the same head. Painting from the clay shot is meaningless, so
    score each candidate by how much of it is saturated colour and take the best.
    """
    best, best_score = None, -1.0
    for p in preview_candidates(src):
        try:
            rgb, _alpha = load_preview(p)
        except Exception:
            continue
        mx = rgb.max(axis=2)
        mn = rgb.min(axis=2)
        sat = (mx - mn) / np.maximum(mx, 1e-4)
        score = float((sat > 0.30).mean())
        if score > best_score:
            best, best_score = p, score
    return best


def load_preview(path: Path):
    """Return (rgb float array HxWx3 bottom-up, alpha or None, w, h)."""
    img = bpy.data.images.load(str(path))
    w, h = img.size
    ch = img.channels
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, ch)
    bpy.data.images.remove(img)
    rgb = px[..., :3]
    alpha = px[..., 3] if ch >= 4 else None
    return rgb, alpha


def preview_mask(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray | None:
    """Binary silhouette of the painted head in the preview image."""
    if alpha is not None and float(alpha.min()) < 0.5:
        return alpha > 0.5
    # No usable alpha (jpg / ZBrush screenshot). A plain border-colour distance
    # keeps viewport grid lines and drop shadows, which inflates the bbox and
    # therefore the fitted scale. Threshold on luminance relative to the border
    # instead: only the bright head survives.
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    bg = float(np.median(0.2126 * border[:, 0] + 0.7152 * border[:, 1] + 0.0722 * border[:, 2]))
    peak = float(np.percentile(lum, 99.5))
    thr = bg + 0.45 * max(peak - bg, 1e-3)
    m = lum > thr
    if m.mean() < 0.01 or m.mean() > 0.98:
        return None
    # keep the largest connected blob (drops stray marks and the grid)
    h, w = m.shape
    step = max(1, int(max(h, w) / 220))
    small = m[::step, ::step]
    best_lab, best_size, seen = None, 0, np.zeros_like(small)
    sh, sw = small.shape
    for i in range(sh):
        for j in range(sw):
            if not small[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            pts = []
            while stack:
                y, x = stack.pop()
                pts.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < sh and 0 <= nx < sw and small[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(pts) > best_size:
                best_size, best_lab = len(pts), pts
    if not best_lab or best_size < 20:
        return None
    out = np.zeros_like(m)
    ys = [p[0] * step for p in best_lab]
    xs = [p[1] * step for p in best_lab]
    y0, y1, x0, x1 = min(ys), min(max(ys) + step, h), min(xs), min(max(xs) + step, w)
    out[y0:y1, x0:x1] = m[y0:y1, x0:x1]
    return out


def register(model_mask: np.ndarray, mm_per_px: float, tl: tuple[float, float],
             model_bbox: tuple[float, float, float, float], pmask: np.ndarray, iters: int = 3):
    """Similarity fit (scale + x/z offset) of the model silhouette onto the preview
    silhouette, initialised from both bounding boxes and refined by IoU.

    Returns (s px/mm, tx, tz, iou, meta)."""
    ph, pw = pmask.shape
    ys, xs = np.nonzero(pmask)
    if len(xs) < 500:
        return None
    px0, px1, py0, py1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mx0, mx1, mz0, mz1 = model_bbox
    sx = (px1 - px0) / max(mx1 - mx0, 1e-6)
    sz = (py1 - py0) / max(mz1 - mz0, 1e-6)
    s = 0.5 * (sx + sz)
    tx = px0 - s * mx0
    ty = py0 - s * mz0

    mh, mw = model_mask.shape
    step = max(1, int(min(pw, ph) / 320))
    gy, gx = np.mgrid[0:ph:step, 0:pw:step]
    pp = pmask[gy, gx]

    def iou_of(ss, txx, tyy):
        mx = (gx - txx) / ss
        my = (gy - tyy) / ss
        j = np.rint((mx - tl[0]) / mm_per_px).astype(np.int32)
        i = np.rint((my - tl[1]) / mm_per_px).astype(np.int32)
        ok = (i >= 0) & (i < mh) & (j >= 0) & (j < mw)
        mm_ = np.zeros(gx.shape, dtype=bool)
        mm_[ok] = model_mask[i[ok], j[ok]]
        union = np.logical_or(mm_, pp).sum()
        return (np.logical_and(mm_, pp).sum() / union) if union else 0.0

    best = (s, tx, ty, iou_of(s, tx, ty))
    s_limits = (0.85 * s, 1.15 * s)
    lim_x = pw * 0.08
    lim_y = ph * 0.08
    for it in range(iters):
        span = 0.04 / (2 ** it)
        cand = []
        for ds in (0.0, span, -span, span / 2, -span / 2):
            for dtx in (0.0, span * pw / 5, -span * pw / 5):
                for dty in (0.0, span * ph / 5, -span * ph / 5):
                    ss = best[0] * (1 + ds)
                    cand.append((ss, best[1] + dtx, best[2] + dty))
        for ss, txx, tyy in cand:
            if not (s_limits[0] <= ss <= s_limits[1]):
                continue
            if abs(txx - tx) > lim_x or abs(tyy - ty) > lim_y:
                continue
            v = iou_of(ss, txx, tyy)
            if v > best[3]:
                best = (ss, txx, tyy, v)
    meta = {
        "preview_mask_bbox_px": [int(px0), int(py0), int(px1), int(py1)],
        "model_bbox_mm": [round(v, 2) for v in model_bbox],
        "sx": round(sx, 4),
        "sz": round(sz, 4),
        "init_s": round(s, 4),
    }
    return (*best, meta)


def iris_centres(rgb: np.ndarray) -> list[tuple[float, float]] | None:
    """Two painted-iris centroids (pixel coords, row 0 = bottom)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    blue = (b > 0.25) & (b - r > 0.10) & (b > g * 0.9)
    if blue.sum() < 500:
        return None
    ys, xs = np.nonzero(blue)
    left = xs < xs.mean()
    if left.sum() < 200 or (~left).sum() < 200:
        return None
    return [
        (float(xs[left].mean()), float(ys[left].mean())),
        (float(xs[~left].mean()), float(ys[~left].mean())),
    ]


def eye_registration(holes: list[dict], irises: list[tuple[float, float]],
                     model_width: float, img_w: int):
    """Fit scale+offset from eye openings to painted irises.

    Only pairs whose implied head width fits inside the image are considered
    (the preview may have margins; it may also be cropped, in which case the
    bounding box is the wrong anchor and the eyes are the reliable one)."""
    best = None
    nm = len(holes)
    if nm < 2:
        return None
    for i in range(nm):
        for j in range(i + 1, nm):
            a, b = holes[i]["centre_mm"], holes[j]["centre_mm"]
            if abs(a[0] + b[0]) > 10.0 or abs(a[1] - b[1]) > 8.0:
                continue
            dh = math.dist(a, b)
            if dh < 20.0:
                continue
            di = math.dist(irises[0], irises[1])
            sc = di / dh
            ratio = sc * model_width / img_w
            if not (0.45 <= ratio <= 1.02):
                continue
            tx = (irises[0][0] + irises[1][0]) / 2 - sc * (a[0] + b[0]) / 2
            ty = (irises[0][1] + irises[1][1]) / 2 - sc * (a[1] + b[1]) / 2
            cand = (ratio, sc, tx, ty, [a, b])
            if best is None or cand[1] > best[1]:
                best = cand
    return best



def find_eye_decals(src: Path) -> dict[str, Path]:
    """(character-left, character-right) printed eye images, first version only."""
    out: dict[str, Path] = {}
    for p in sorted(src.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        stem = p.stem
        if "左眼" in stem and "left" not in out:
            out["left"] = p
        elif "右眼" in stem and "right" not in out:
            out["right"] = p
    return out



def decal_uv_box(img_path: Path, pad: float = 0.02):
    """UV rectangle of the printed eye sheet's opaque area (drops the transparent margin)."""
    img = bpy.data.images.load(str(img_path))
    w, h = img.size
    ch = img.channels
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, ch)
    bpy.data.images.remove(img)
    alpha = px[..., 3] if ch >= 4 else np.ones((h, w), dtype=np.float32)
    ys, xs = np.nonzero(alpha > 0.5)
    if len(xs) < 100:
        return (0.0, 0.0, 1.0, 1.0)
    u0, u1 = xs.min() / w, (xs.max() + 1) / w
    v0, v1 = ys.min() / h, (ys.max() + 1) / h
    du, dv = (u1 - u0) * pad, (v1 - v0) * pad
    return (max(0.0, u0 - du), max(0.0, v0 - dv), min(1.0, u1 + du), min(1.0, v1 + dv))


def build_eye_inserts(scene, holes: list[dict], decals: dict[str, Path], lo: Vector, tree=None) -> list:
    """Two quads behind the eye openings, textured with the printed eye sheets.

    A through-hole gets the sheet just behind the rim. A recessed eye (no hole)
    gets it on the recess floor, found by a ray so it cannot float over the face."""
    if len(holes) != 2 or not decals:
        return []
    made = []
    depth_y = lo.y + SOCKET_FRAME_BEHIND_MM   # just behind the socket rim, so the opening frames it
    sides = ["right", "left"]      # model -x = the character's own right
    for hole, side in zip(sorted(holes, key=lambda h: h["centre_mm"][0]), sides):
        if hole.get("derived_from"):
            # Recessed eye: there is no opening to frame a flat sheet, and a ray
            # aimed at the recess floor just hits the face. Leave it to the clay
            # views instead of faking an eye that is not in the geometry.
            continue
        img_path = decals.get(side) or decals.get("left" if side == "right" else "right")
        if img_path is None:
            continue
        (bx0, bz0), (bx1, bz1) = hole["bbox_mm"]
        cx, cz = hole["centre_mm"]
        quad_y = depth_y
        w = max((bx1 - bx0) * 1.0, 8.0)
        h = max((bz1 - bz0) * 1.0, 8.0)
        verts = [
            (cx - w / 2, quad_y, cz - h / 2),
            (cx + w / 2, quad_y, cz - h / 2),
            (cx + w / 2, quad_y, cz + h / 2),
            (cx - w / 2, quad_y, cz + h / 2),
        ]
        me = bpy.data.meshes.new(f"eye_{side}")
        me.from_pydata(verts, [], [(0, 1, 2, 3)])
        me.update()
        layer = me.uv_layers.new(name="eye_uv")
        u0, v0, u1, v1 = decal_uv_box(img_path)
        corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        for i, loop in enumerate(me.loops):
            layer.data[i].uv = corners[loop.vertex_index]
        ob = bpy.data.objects.new(f"eye_{side}", me)
        scene.collection.objects.link(ob)

        mat = bpy.data.materials.new(f"eye_mat_{side}")
        mat.use_nodes = True
        nt = mat.node_tree
        for nd in list(nt.nodes):
            if nd.type != "OUTPUT_MATERIAL":
                nt.nodes.remove(nd)
        outn = nt.nodes["Material Output"]
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(img_path))
        tex.extension = "CLIP"
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.45
        emis = nt.nodes.new("ShaderNodeEmission")
        emis.inputs["Strength"].default_value = 0.35
        add = nt.nodes.new("ShaderNodeAddShader")
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(tex.outputs["Color"], emis.inputs["Color"])
        nt.links.new(bsdf.outputs[0], add.inputs[0])
        nt.links.new(emis.outputs[0], add.inputs[1])
        nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
        nt.links.new(transp.outputs[0], mix.inputs[1])
        nt.links.new(add.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], outn.inputs["Surface"])
        ob.data.materials.append(mat)
        ob.visible_shadow = False  # a flat sheet must not cast a shadow onto the face
        ob.hide_render = True      # visible only in the painted views
        made.append(ob)
    return made


def printability(ob, scene, sample: int = 20000) -> dict:
    """Thickness / overhang / area statistics for a part that is going to a printer.

    Thickness uses the same two-hop ray test as face/check_mesh.py: cast inward
    along the vertex normal, then back along the reverse normal to the opposite
    wall, so a grazing hit on the same wall is discarded."""
    import bmesh

    me = ob.data
    me.calc_loop_triangles()
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    V = co.reshape(nv, 3).astype(np.float64)
    nrm = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("normal", nrm)
    N = nrm.reshape(nv, 3)

    tris = len(me.loop_triangles)
    F = np.empty(tris * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", F)
    F = F.reshape(tris, 3)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(fn, axis=1)
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    unit = np.divide(fn, np.where(ln == 0, 1, ln))
    total_area = float(area.sum())
    # area-weighted fraction facing down more than 45 deg from the vertical
    down = unit[:, 2] < -0.7071
    overhang_area = float(area[down].sum())

    # signed tetra sum: same answer as bmesh volume, without building a bmesh
    # over millions of triangles (that alone blew the 8 GiB address-space cap)
    volume = abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))

    from mathutils.bvhtree import BVHTree

    tree = BVHTree.FromObject(ob, bpy.context.evaluated_depsgraph_get())
    rng = np.random.default_rng(0)
    idx = rng.choice(nv, size=min(sample, nv), replace=False)
    walls = []
    for k in idx:
        n = N[k]
        ln = math.sqrt(float(n[0]) ** 2 + float(n[1]) ** 2 + float(n[2]) ** 2)
        if ln < 1e-9:
            continue
        inward = Vector((-float(n[0]) / ln, -float(n[1]) / ln, -float(n[2]) / ln))
        origin = Vector((float(V[k][0]), float(V[k][1]), float(V[k][2])))
        # nudge along the ray: a ray starting exactly on the surface reports t=0
        hit = tree.ray_cast(origin + inward * 0.05, inward, 25.0)
        if hit[0] is None or hit[1] is None:
            continue
        # entering the hollow means we crossed to the inner skin: its normal
        # points back at us. A grazing hit on the same skin does not.
        if float(hit[1].dot(Vector(n))) > -0.30:
            continue
        t = (hit[0] - origin).length
        if 0.2 < t < 15.0:
            walls.append(t)

    del a, b, c, fn, unit, area, down
    gc.collect()
    walls.sort()
    def pct(q: float):
        return round(walls[min(len(walls) - 1, int(len(walls) * q))], 2) if walls else None

    return {
        "volume_cm3": round(volume / 1000.0, 2),
        "area_cm2": round(total_area / 100.0, 2),
        "wall_samples": len(walls),
        "wall_median_mm": pct(0.50),
        "wall_p05_mm": pct(0.05),
        "wall_min_mm": round(walls[0], 2) if walls else None,
        "wall_below_2mm_ratio": round(sum(1 for t in walls if t < 2.0) / max(1, len(walls)), 4),
        "overhang_area_ratio": round(overhang_area / total_area, 4) if total_area else None,
    }

def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--front-override", default=None, help="decimated proxy to render instead of 前头.stl")
    ap.add_argument("--back-override", default=None)
    ap.add_argument("--raw-metrics", default=None, help="json file with raw-file statistics")
    ap.add_argument("--no-paint", action="store_true")
    ap.add_argument("--with-paint", action="store_true",
                    help="project the modeler preview onto the front plate (cosmetic; off for print review)")
    ap.add_argument("--paint-only", action="store_true", help="skip the clay views (keep existing files)")
    ap.add_argument("--turntable", type=int, default=0, help="render N evenly spaced yaw views (+ top/bottom)")
    ap.add_argument("--turntable-only", action="store_true", help="only render the turntable set")
    ap.add_argument("--samples", type=int, default=40)
    args = ap.parse_args(argv)

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "headshell/baseline-model@1", "character": src.name, "src": str(src)}
    if args.turntable_only and (out / "report.json").exists():
        # keep the measurements/paint info from the full run, add the new views
        try:
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    t_start = time.time()

    clear_scene()
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    setup_world()
    add_lights()

    front_p = Path(args.front_override).resolve() if args.front_override else src / "前头.stl"
    back_p = Path(args.back_override).resolve() if args.back_override else src / "后头.stl"
    report["render_proxy"] = bool(args.front_override)
    if args.raw_metrics and Path(args.raw_metrics).exists():
        report["raw_stats"] = json.loads(Path(args.raw_metrics).read_text(encoding="utf-8"))
    if not front_p.exists():
        report["status"] = "error"
        report["error"] = "前头.stl missing"
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log("ERROR: 前头.stl missing")
        return 1

    front = import_stl(str(front_p))
    log(f"front imported: {[o.name for o in front]}")
    back = import_stl(str(back_p)) if back_p.exists() else []
    report["front_stats"] = [dict(name=o.name, **mesh_stats(o)) for o in front]
    if back:
        report["back_stats"] = [dict(name=o.name, **mesh_stats(o)) for o in back]
    report["total_tris"] = int(
        sum(o["tris"] for o in report["front_stats"]) + sum(o["tris"] for o in report.get("back_stats", []))
    )
    log(f"stats: {report['total_tris']} tris")

    lo, hi = world_bbox(front)
    centre = (lo + hi) / 2.0
    report["front_bbox"] = {
        "min": [round(v, 2) for v in lo],
        "max": [round(v, 2) for v in hi],
        "dims": [round(hi[i] - lo[i], 2) for i in range(3)],
    }

    clay = make_material("clay")
    for o in front + back:
        o.data.materials.clear()
        o.data.materials.append(clay)
        for p in o.data.polygons:
            p.use_smooth = False

    cam_p = camera("cam_persp", ortho=False)
    cam_o = camera("cam_ortho", ortho=True)

    # --- eye openings from the silhouette
    width = hi.x - lo.x
    height = hi.z - lo.z
    depth = hi.y - lo.y
    sil_res = (560, 700)
    sil_half_w = max(width, height * sil_res[0] / sil_res[1]) / 2.0 * 1.06
    sil = silhouette(cam_o, centre, sil_half_w, out, res_x=sil_res[0], res_y=sil_res[1], hide=back)
    raw_holes = enclosed_components(sil["mask"])[:14]
    hole_list = []
    for hole in raw_holes:
        ys = [p[0] for p in hole["pts"]]
        xs = [p[1] for p in hole["pts"]]
        hx, hz = mask_px_to_model(sil, float(np.mean(ys)), float(np.mean(xs)))
        bx0, bz0 = mask_px_to_model(sil, min(ys), min(xs))
        bx1, bz1 = mask_px_to_model(sil, max(ys), max(xs))
        hole_list.append(
            {
                "size_px": hole["size"],
                "centre_mm": [round(hx, 2), round(hz, 2)],
                "bbox_mm": [[round(bx0, 1), round(bz0, 1)], [round(bx1, 1), round(bz1, 1)]],
            }
        )
    report["eye_holes_candidates"] = hole_list

    def pick_eye_holes(cands: list[dict]) -> list[dict]:
        """Pick the symmetric eye-opening pair (ears / hair gaps also show up as holes)."""
        if len(cands) < 2:
            return []
        # an eye opening is a sizeable, wide oval - reject pin-holes and gaps
        def plausible(h: dict) -> bool:
            (x0, z0), (x1, z1) = h["bbox_mm"]
            hw, hh = abs(x1 - x0), abs(z1 - z0)
            if not (0.05 * width <= hw <= 0.22 * width):
                return False
            if not (0.02 * height <= hh <= 0.16 * height):
                return False
            return 0.8 <= hw / max(hh, 1e-6) <= 3.5

        filtered = [h for h in cands if plausible(h)]
        cands = filtered or cands
        biggest = max((c["size_px"] for c in cands), default=0)
        pairs = []
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i]["centre_mm"], cands[j]["centre_mm"]
                if abs(a[0] + b[0]) > 8.0:          # symmetric about x = 0
                    continue
                if abs(a[1] - b[1]) > 6.0:          # same height
                    continue
                ratio = max(cands[i]["size_px"], cands[j]["size_px"]) / max(
                    min(cands[i]["size_px"], cands[j]["size_px"]), 1
                )
                if ratio > 2.5:
                    continue
                cx = abs(a[0])
                cz = (a[1] + b[1]) / 2.0
                size = min(cands[i]["size_px"], cands[j]["size_px"])
                # A decorative pinhole is not an eye: only pairs comparable in size
                # to the biggest enclosed opening are eligible, then position decides.
                if size < 0.25 * biggest:
                    continue
                score = abs(cx - 0.28 * width) / width + abs(cz - 0.45 * height) / height
                pairs.append((score, i, j))
        if not pairs:
            return []
        pairs.sort()
        _, i, j = pairs[0]
        return [cands[i], cands[j]]

    picked = pick_eye_holes(hole_list)

    # Some heads have no through-hole at the eye: the eye sits in a recess, so the
    # silhouette shows solid surface and hole detection finds nothing usable.
    # Fall back to the painted eye positions from the preview, mapped into model
    # space with the bounding-box transform.
    if not picked or min(h["size_px"] for h in picked) < 300:
        prev_for_eyes = find_preview(src)
        derived = None
        if prev_for_eyes:
            rgb_e, alpha_e = load_preview(prev_for_eyes)
            pmask_e = preview_mask(rgb_e, alpha_e)
            irises_e = iris_centres(rgb_e)
            if pmask_e is not None and irises_e:
                ys_e, xs_e = np.nonzero(pmask_e)
                s_e = (xs_e.max() - xs_e.min()) / max(hi.x - lo.x, 1e-6)
                tx_e = xs_e.min() - s_e * lo.x
                ty_e = ys_e.min() - s_e * lo.z
                r_e, g_e, b_e = rgb_e[..., 0], rgb_e[..., 1], rgb_e[..., 2]
                blue = (b_e > 0.25) & (b_e - r_e > 0.10) & (b_e > g_e * 0.9)
                cols = np.arange(rgb_e.shape[1])[None, :]
                derived = []
                for k in (0, 1):
                    half = cols < rgb_e.shape[1] / 2 if k == 0 else cols >= rgb_e.shape[1] / 2
                    yy, xx = np.nonzero(blue & half)
                    if len(xx) < 50:
                        derived = None
                        break
                    cx_px, cy_px = float(xx.mean()), float(yy.mean())
                    w_mm = (xx.max() - xx.min()) / s_e * 2.0
                    h_mm = (yy.max() - yy.min()) / s_e * 2.0
                    cx_mm = (cx_px - tx_e) / s_e
                    cz_mm = (cy_px - ty_e) / s_e
                    derived.append(
                        {
                            "size_px": 900,
                            "centre_mm": [round(cx_mm, 2), round(cz_mm, 2)],
                            "bbox_mm": [
                                [round(cx_mm - w_mm / 2, 1), round(cz_mm - h_mm / 2, 1)],
                                [round(cx_mm + w_mm / 2, 1), round(cz_mm + h_mm / 2, 1)],
                            ],
                            "derived_from": "painted iris in preview",
                        }
                    )
        if derived:
            picked = derived
            report["eye_holes_source"] = "preview iris fallback (no through-hole)"
        else:
            report["eye_holes_source"] = "silhouette holes"
    else:
        report["eye_holes_source"] = "silhouette holes"

    report["eye_holes"] = picked or hole_list[:2]
    report["eye_holes_confident"] = bool(picked)
    report["eye_holes"].sort(key=lambda d: d["centre_mm"][0])
    # Eye sheets are a display part, not a printed one: skipped by default and
    # only built with --with-paint.
    if args.with_paint:
        eye_objs = build_eye_inserts(bpy.context.scene, report["eye_holes"], find_eye_decals(src), lo)
    else:
        eye_objs = []
    report["eye_inserts"] = [o.name for o in eye_objs]
    log(f"eye holes: {[h['centre_mm'] for h in report['eye_holes']]}")

    # --- clay renders (shape check, no texture)
    sc.render.film_transparent = False
    size = max(hi - lo)
    scene_dims = tuple(hi[i] - lo[i] for i in range(3))
    clay_res = (760, 980)
    dist = fit_distance(cam_p, scene_dims, clay_res)
    shots = {
        "clay_front": (0, -1, 0),
        "clay_three_quarter": (-0.55, -1, 0.10),
        "clay_side": (-1, -0.05, 0.05),
    }
    report["renders"] = {}
    skip_clay = args.paint_only or args.turntable_only
    # The two delivered parts overlap in Y (the back shell's rim sits in front of
    # the nose), so superimposing them hides the ears and makes the face look
    # recessed inside a hood. Render each part on its own: the front plate is the
    # face, the back shell is a separate view.
    for name, dirv in ([] if skip_clay else shots.items()):
        aim(cam_p, centre, dirv, dist)
        sc.camera = cam_p
        for o in back:
            o.hide_render = True
        report["renders"][name] = render_to(out / f"{name}.png", res=clay_res, samples=args.samples)
        for o in back:
            o.hide_render = False

    if back and not skip_clay:
        blo, bhi = world_bbox(back)
        bdims = tuple(bhi[i] - blo[i] for i in range(3))
        aim(cam_p, (blo + bhi) / 2, (0, 1, 0), fit_distance(cam_p, bdims, clay_res))
        sc.camera = cam_p
        for o in front:
            o.hide_render = True
        report["renders"]["clay_back"] = render_to(out / "clay_back.png", res=clay_res, samples=args.samples)
        for o in front:
            o.hide_render = False

    # --- multi-view turntable (even yaw spacing + top/bottom)
    if args.turntable:
        n = max(3, args.turntable)
        elev = math.radians(8.0)
        tt_res = (640, 820)
        tdist = fit_distance(cam_p, tuple(hi[i] - lo[i] for i in range(3)), tt_res)
        report["turntable"] = []
        for k in range(n):
            yaw = 2.0 * math.pi * k / n
            d = Vector((math.sin(yaw) * math.cos(elev), -math.cos(yaw) * math.cos(elev), math.sin(elev)))
            aim(cam_p, centre, d, tdist)
            sc.camera = cam_p
            for o in back:
                o.hide_render = True
            name = f"view_{int(round(math.degrees(yaw))) % 360:03d}"
            report["turntable"].append(name)
            report["renders"][name] = render_to(out / f"{name}.png", res=tt_res, samples=args.samples)
            for o in back:
                o.hide_render = False
        for name, d in (("view_top", Vector((0, -0.001, 1))), ("view_bottom", Vector((0, -0.001, -1)))):
            aim(cam_p, centre, d, tdist, up=Vector((0, 1, 0)))
            sc.camera = cam_p
            report["turntable"].append(name)
            report["renders"][name] = render_to(out / f"{name}.png", res=tt_res, samples=args.samples)
        log(f"turntable: {len(report['turntable'])} views")

    # --- painted front (planar projection of the modeler preview)
    preview = find_preview(src)
    report["preview"] = str(preview) if preview else None
    painted_ok = False
    if preview and args.with_paint and not args.no_paint and not args.turntable_only:
        rgb, alpha = load_preview(preview)
        img_h, img_w = rgb.shape[0], rgb.shape[1]
        pmask = preview_mask(rgb, alpha)
        mm_per_px = sil["ortho_scale"] / sil["res"][0]
        tl = (
            sil["target"].x - sil["res"][0] / 2.0 * mm_per_px,
            sil["target"].z - sil["res"][1] / 2.0 * mm_per_px,
        )
        naive_s = img_w / max(width, 1e-6)
        fit = (
            register(sil["mask"], mm_per_px, (tl[0], tl[1]), (lo.x, hi.x, lo.z, hi.z), pmask)
            if pmask is not None
            else None
        )
        # The preview and the STL describe the same object, so a good fit means the
        # two silhouettes overlap; that is also what makes the painted projection
        # land in the right place. Iris matching was removed: the painted iris is
        # smaller than the socket opening, so it biased the scale by ~30%.
        if fit and fit[3] >= 0.80:
            s_fit, tx, ty, iou, meta = fit
            report["paint_registration"] = {
                "method": "silhouette fit (bbox init + IoU refinement)",
                "iou": round(iou, 4),
                "accepted": True,
                "scale_px_per_mm": round(s_fit, 4),
                "naive_bbox_scale": round(naive_s, 4),
                **meta,
            }
        else:
            s_fit = naive_s
            tx = -s_fit * lo.x
            ty = -s_fit * lo.z
            report["paint_registration"] = {
                "method": "fallback: bounding-box mapping",
                "iou": round(fit[3], 4) if fit else None,
                "accepted": False,
                "scale_px_per_mm": round(s_fit, 4),
                "naive_bbox_scale": round(naive_s, 4),
            }
        ob = front[0]
        me = ob.data
        n = len(me.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(n, 3)
        u = (tx + s_fit * co[:, 0]) / img_w
        v = (ty + s_fit * co[:, 2]) / img_h
        li = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", li)
        uv = np.stack([u[li], v[li]], axis=1).astype(np.float32)
        layer = me.uv_layers.get("preview_proj") or me.uv_layers.new(name="preview_proj")
        layer.data.foreach_set("uv", uv.ravel())

        mat = bpy.data.materials.new("painted")
        mat.use_nodes = True
        nt = mat.node_tree
        for nd in list(nt.nodes):
            if nd.type != "OUTPUT_MATERIAL":
                nt.nodes.remove(nd)
        outn = nt.nodes["Material Output"]
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.55
        # Paint only the surfaces that face the camera. Without this the planar
        # projection smears across the temples and the sides of the head, which is
        # what made the earlier renders look warped.
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = (0.0, -1.0, 0.0)
        rng = nt.nodes.new("ShaderNodeMapRange")
        rng.inputs["From Min"].default_value = 0.05
        rng.inputs["From Max"].default_value = 0.45
        rng.inputs["To Min"].default_value = 1.0
        rng.inputs["To Max"].default_value = 0.0
        clay_tint = nt.nodes.new("ShaderNodeMixRGB")
        clay_tint.inputs["Color2"].default_value = (0.80, 0.78, 0.76, 1.0)
        nt.links.new(geo.outputs["Normal"], dot.inputs[0])
        nt.links.new(dot.outputs["Value"], rng.inputs["Value"])
        nt.links.new(rng.outputs["Result"], clay_tint.inputs["Fac"])
        timg = nt.nodes.new("ShaderNodeTexImage")
        timg.image = bpy.data.images.load(str(preview))
        uvmap = nt.nodes.new("ShaderNodeUVMap")
        uvmap.uv_map = "preview_proj"
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(uvmap.outputs["UV"], timg.inputs["Vector"])
        nt.links.new(timg.outputs["Color"], clay_tint.inputs["Color1"])
        nt.links.new(clay_tint.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(timg.outputs["Alpha"], mix.inputs["Fac"])
        nt.links.new(transp.outputs[0], mix.inputs[1])
        nt.links.new(bsdf.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], outn.inputs["Surface"])
        ob.data.materials.clear()
        ob.data.materials.append(mat)

        for o in eye_objs:
            o.hide_render = False
        for o in back:
            o.hide_render = True
        cam_o.data.sensor_fit = "HORIZONTAL"
        cam_o.data.ortho_scale = ortho_scale_for((width, 0.0, height), (820, 1000))
        aim(cam_o, centre, (0, -1, 0), 900.0)
        sc.camera = cam_o
        report["renders"]["face_painted_front"] = render_to(
            out / "face_painted_front.png", res=(820, 1000), samples=args.samples
        )
        for o in eye_objs:
            o.hide_render = True
        aim(cam_p, centre, (-0.55, -1, 0.10),
            fit_distance(cam_p, tuple(hi[i] - lo[i] for i in range(3)), clay_res))
        sc.camera = cam_p
        report["renders"]["face_painted_three_quarter"] = render_to(
            out / "face_painted_three_quarter.png", res=(760, 980), samples=args.samples
        )
        for o in back:
            o.hide_render = False
        painted_ok = True
    report["painted"] = painted_ok
    for tag, objs in (("front", front), ("back", back)):
        if not objs:
            continue
        try:
            report[f"printability_{tag}"] = printability(objs[0], bpy.context.scene, sample=3000)
        except Exception as exc:  # noqa: BLE001 - never fail the job on this
            report[f"printability_{tag}"] = {"error": f"{type(exc).__name__}: {exc}"}
    report["status"] = "ok"
    report["seconds"] = round(time.time() - t_start, 1)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done -> {out}  ({report['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
