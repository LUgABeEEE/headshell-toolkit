#!/usr/bin/env python3
"""Clay-review renders for a delivered STL (one model, several turntable views).

Runs inside Blender (headless):

    tools/blender-safe.sh -b --factory-startup -P tools/stl_review_render.py -- \
        --src in/front.stl [--src in/back.stl] --out out/review/m1 --name m1

Why it exists: the package pipeline (`tools/baseline_render.py`) starts from a
whole delivery directory (front + back + preview image, UV projection, eye-hole
solving). When all you have is one or two STLs and the question is "is this mesh
any good", you want a neutral clay sheet and nothing else. Geometry is never modified.

Per model it writes <out>/<prefix>_<view>.png for front / three_quarter / side /
back, plus review.json holding the mesh facts measured in Blender (verts, tris,
boundary & non-manifold edges, bbox in mm, symmetry error, triangle-edge
percentiles) and the camera setup used, so the sheet is reproducible.

Units: STL is unitless. The shell is authored in millimetres (bbox ~250 mm), so
1 Blender unit = 1 mm throughout; nothing rescales the imported mesh.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def log(msg: str) -> None:
    print(f"[review] {msg}", flush=True)


# ------------------------------------------------------------------ scene
def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_stl(path: str) -> list:
    before = set(bpy.data.objects)
    bpy.ops.wm.stl_import(filepath=path)
    return [o for o in bpy.data.objects if o not in before and o.type == "MESH"]


def make_material(name="clay", color=(0.80, 0.78, 0.76), rough=0.55):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    return mat


def setup_world(strength: float = 0.55, color=(0.30, 0.32, 0.36)) -> None:
    w = bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = strength


def light_rig():
    """Three suns parented to an empty, so the key stays camera-left as the
    camera orbits. Returns the empty (rotate around Z by azimuth)."""
    rig = bpy.data.objects.new("light_rig", None)
    bpy.context.scene.collection.objects.link(rig)
    for name, rot, energy in [
        ("key", (math.radians(52), 0, math.radians(-38)), 4.6),
        ("fill", (math.radians(74), 0, math.radians(118)), 0.8),
        ("rim", (math.radians(-46), 0, math.radians(168)), 1.6),
    ]:
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = energy
        ld.angle = math.radians(14)
        ob = bpy.data.objects.new(name, ld)
        bpy.context.scene.collection.objects.link(ob)
        ob.rotation_euler = rot
        ob.parent = rig
    return rig


def camera(name: str = "cam", lens: float = 72.0):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "HORIZONTAL"      # pin the axis so fov_h means the same thing
    cd.lens = lens
    cd.clip_start = 1.0
    cd.clip_end = 1e6
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def look_at(cam, target: Vector, direction, distance: float) -> None:
    z = Vector(direction).normalized()
    up = Vector((0, 0, 1))
    if abs(z.dot(up)) > 0.999:
        up = Vector((0, 1, 0))
    cam.location = Vector(target) + z * distance
    right = up.cross(z).normalized()
    tru = z.cross(right)
    cam.rotation_euler = Matrix([right, tru, z]).transposed().to_4x4().to_quaternion().to_euler()


def fovs(cam, res) -> tuple[float, float]:
    cd = cam.data
    fov_h = 2.0 * math.atan(cd.sensor_width / 2.0 / cd.lens)
    fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) * res[1] / res[0])
    return fov_h, fov_v


def fit_distance(cam, corners: list[Vector], target: Vector, direction, res,
                 margin: float = 1.05) -> float:
    """Tightest camera distance that keeps every bbox corner inside the frame."""
    z = Vector(direction).normalized()
    up = Vector((0, 0, 1)) if abs(z.dot(Vector((0, 0, 1)))) < 0.999 else Vector((0, 1, 0))
    right = up.cross(z).normalized()
    tru = z.cross(right)
    fov_h, fov_v = fovs(cam, res)
    th, tv = math.tan(fov_h / 2.0), math.tan(fov_v / 2.0)
    need = 0.0
    for c in corners:
        p = Vector(c) - Vector(target)
        x, y, dz = p.dot(right), p.dot(tru), p.dot(z)
        need = max(need, abs(x) / th + dz, abs(y) / tv + dz)
    return max(need * margin, 1.0)


def aim_view(cam, bbox_lo, bbox_hi, target, direction, res, margin=1.05) -> float:
    corners = [Vector((x, y, z))
               for x in (bbox_lo.x, bbox_hi.x)
               for y in (bbox_lo.y, bbox_hi.y)
               for z in (bbox_lo.z, bbox_hi.z)]
    d = fit_distance(cam, corners, target, direction, res, margin)
    look_at(cam, target, direction, d)
    return d


# ------------------------------------------------------------------ measure
def mesh_facts(objs: list) -> dict:
    facts = {}
    verts = tris = boundary = nonmanifold = 0
    lo = Vector((1e18, 1e18, 1e18))
    hi = Vector((-1e18, -1e18, -1e18))
    co_all = []
    edge_lens = []
    for ob in objs:
        me = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        mw = ob.matrix_world
        nv, ne = len(me.vertices), len(me.edges)
        co = np.empty(nv * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(nv, 3)
        # world-space (STL import normally leaves an identity matrix)
        if not all(abs(mw[i][j] - (1.0 if i == j else 0.0)) < 1e-9 for i in range(4) for j in range(4)):
            co = np.array([tuple(mw @ Vector(tuple(c))) for c in co], dtype=np.float64)
        co_all.append(co)
        loop_edge = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("edge_index", loop_edge)
        counts = np.bincount(loop_edge, minlength=ne)
        boundary += int((counts == 1).sum())
        nonmanifold += int((counts > 2).sum())
        tris += sum(len(p.vertices) - 2 for p in me.polygons)
        verts += nv
        l, h = co.min(0), co.max(0)
        lo = Vector((min(lo.x, l[0]), min(lo.y, l[1]), min(lo.z, l[2])))
        hi = Vector((max(hi.x, h[0]), max(hi.y, h[1]), max(hi.z, h[2])))
        # edge length percentiles, for "is the tessellation even" questions
        ev = np.empty(ne * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(ne, 2)
        seg = co[ev[:, 0]] - co[ev[:, 1]]
        edge_lens.append(np.linalg.norm(seg, axis=1))
        ob.to_mesh_clear()
    all_co = np.concatenate(co_all, axis=0) if co_all else np.zeros((0, 3))
    el = np.concatenate(edge_lens) if edge_lens else np.zeros(1)
    dims = [round(hi[i] - lo[i], 2) for i in range(3)]
    facts.update(
        objects=[o.name for o in objs],
        verts=verts,
        tris=int(tris),
        boundary_edges=boundary,
        non_manifold_edges=nonmanifold,
        watertight=bool(boundary == 0 and nonmanifold == 0),
        bbox_min=[round(v, 2) for v in lo],
        bbox_max=[round(v, 2) for v in hi],
        dims_mm=dims,
        centroid_mm=[round(float(v), 2) for v in all_co.mean(0)],
        edge_mm={
            "p01": round(float(np.percentile(el, 1)), 3),
            "p10": round(float(np.percentile(el, 10)), 3),
            "median": round(float(np.percentile(el, 50)), 3),
            "p90": round(float(np.percentile(el, 90)), 3),
            "p99": round(float(np.percentile(el, 99)), 3),
            "max": round(float(el.max()), 3),
        },
    )
    # left/right symmetry: mirror in X about the bbox centre plane and measure how
    # close each vertex lands to the nearest vertex on the other side
    cx = 0.5 * (lo.x + hi.x)
    origins = np.array([lo.x, lo.y, lo.z], dtype=np.float64)
    step = max(float(np.median(el)) * 2.0, 0.05)
    # coarse grid hashing keeps this affordable at 500k verts
    qa = np.rint((all_co - origins) / step).astype(np.int64)
    qb = np.rint((np.column_stack([2.0 * cx - all_co[:, 0], all_co[:, 1], all_co[:, 2]]) - origins)
                 / step).astype(np.int64)
    set_a = {tuple(k) for k in qa.tolist()}
    hit = sum(1 for k in qb.tolist() if tuple(k) in set_a)
    facts["symmetry"] = {
        "grid_mm": round(step, 3),
        "matched_ratio": round(hit / max(len(all_co), 1), 4),
        "note": "fraction of mirrored vertices that land on an existing vertex in a "
                f"{step:.2f} mm grid; 1.0 = vertex-level symmetric, low = sculpted/asymmetric",
    }
    # material volume / area -> wall-thickness proxy; the sign of the signed
    # volume also says whether the triangle winding is consistently outward
    import bmesh
    vol = area = 0.0
    for ob in objs:
        tmp = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        bm = bmesh.new()
        bm.from_mesh(tmp)
        vol += float(bm.calc_volume(signed=True))
        area += float(sum(f.calc_area() for f in bm.faces))
        bm.free()
        ob.to_mesh_clear()
    facts["shell"] = {
        "signed_volume_mm3": round(vol, 1),
        "surface_area_mm2": round(area, 1),
        "thickness_est_mm": round(2.0 * vol / area, 3) if area > 0 else None,
        "note": "closed thin shell: V = (A/2)*t -> t = 2V/A. Proxy only: a solid "
                "blob or a doubled wall shows up as an implausible value, and a "
                "negative signed volume means inconsistent winding.",
    }
    return facts


# ------------------------------------------------------------------ render
def render_views(out_dir: Path, prefix: str, views: dict, res, samples: int,
                 margins: float = 1.05) -> dict:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 25
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.exposure = 0.0
    sc.render.threads_mode = "AUTO"
    times = {}
    for name, (direction, elev) in views.items():
        path = out_dir / f"{prefix}_{name}.png"
        t0 = time.time()
        d = (direction[0] * math.cos(math.radians(elev)),
             direction[1] * math.cos(math.radians(elev)),
             math.sin(math.radians(elev)))
        cam = bpy.data.objects["cam"]
        aim_view(cam, CACHE["lo"], CACHE["hi"], CACHE["target"], d, res, margins)
        # lights follow the camera so every view is lit the same way
        CACHE["rig"].rotation_euler[2] = math.atan2(d[0], -d[1]) * -1.0
        sc.camera = cam
        sc.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        times[name] = round(time.time() - t0, 1)
        log(f"{name}: {path.name} {times[name]}s")
    return times


CACHE: dict = {}


def silhouette_symmetry(out_dir: Path, res=(640, 800), margin: float = 1.04) -> dict:
    """Mirror the front silhouette and measure overlap.

    Vertex-level symmetry is meaningless on a messy scan-derived mesh, but a
    silhouette is what the wearer and the viewer actually see: IoU of the front
    silhouette against its own mirror is a robust "does this face read as
    symmetric, and is it centred" answer.
    """
    sc = bpy.context.scene
    keep = (sc.render.film_transparent, sc.cycles.samples,
            sc.render.resolution_x, sc.render.resolution_y)
    cam = bpy.data.objects["cam"]
    lo, hi, target = CACHE["lo"], CACHE["hi"], CACHE["target"]
    dims = hi - lo
    aspect = res[0] / res[1]
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(dims.x, dims.z * aspect) * margin
    look_at(cam, target, (0.0, -1.0, 0.0), max(dims.length * 3.0, 900.0))
    sc.render.film_transparent = True
    sc.cycles.samples = 1
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.camera = cam
    tmp = out_dir / "_silhouette_sym.png"
    sc.render.filepath = str(tmp)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(tmp))
    w, h = img.size
    ch = img.channels
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, ch)
    bpy.data.images.remove(img)
    mask = px[..., 3] > 0.5
    flip = mask[:, ::-1]

    def iou(a, b):
        u = float((a | b).sum())
        return float((a & b).sum()) / u if u else 0.0

    zero = iou(mask, flip)
    best, best_shift = zero, 0
    span = max(1, int(w * 0.02))
    for s in range(-span, span + 1):
        shifted = np.roll(flip, s, axis=1)
        v = iou(mask, shifted)
        if v > best:
            best, best_shift = v, s
    # restore
    (sc.render.film_transparent, sc.cycles.samples,
     sc.render.resolution_x, sc.render.resolution_y) = keep
    cam.data.type = "PERSP"
    return {
        "iou_mirrored": round(zero, 4),
        "iou_best_shift": round(best, 4),
        "best_shift_px": best_shift,
        "mm_per_px": round(cam.data.ortho_scale / res[0], 3),
        "note": "front-silhouette IoU against its mirror. iou_best_shift allows a "
                "small lateral slide: a high iou_best_shift with a low iou_mirrored "
                "means a symmetric face that is not centred on x=0.",
    }


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="STL path (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="model")
    ap.add_argument("--views", default="front,three_quarter,side,back")
    ap.add_argument("--res", default="900x1200")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--margin", type=float, default=1.05)
    ap.add_argument("--flat", action="store_true",
                    help="flat shading: exposes the triangulation and facets the "
                         "smooth-shaded clay hides")
    ap.add_argument("--no-symmetry", action="store_true",
                    help="skip the front-silhouette symmetry probe")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    res = tuple(int(v) for v in args.res.lower().split("x"))

    clear_scene()
    objs = []
    for src in args.src:
        p = Path(src).resolve()
        log(f"import {p}")
        objs += import_stl(str(p))
    if not objs:
        raise SystemExit("no mesh imported")

    facts = mesh_facts(objs)

    # one clay for a single part; when several parts are loaded together, tint
    # them apart so an overlay render says which surface belongs to which file
    tints = [(0.80, 0.78, 0.76), (0.52, 0.58, 0.66), (0.78, 0.60, 0.52)]
    mats = [make_material(f"clay{i}", tints[i % len(tints)]) for i in range(len(objs))]
    for i, ob in enumerate(objs):
        ob.data.materials.append(mats[i])
        me = ob.data
        me.polygons.foreach_set("use_smooth", [not args.flat] * len(me.polygons))
        me.update()
    if not args.flat:
        try:
            bpy.ops.object.select_all(action="DESELECT")
            for ob in objs:
                ob.select_set(True)
            bpy.context.view_layer.objects.active = objs[0]
            bpy.ops.object.shade_smooth_by_angle(angle=math.radians(38))
        except Exception as exc:  # cosmetic only
            log(f"shade_smooth_by_angle skipped: {exc}")

    setup_world()
    rig = light_rig()
    cam = camera()
    bpy.context.scene.camera = cam

    lo = Vector(facts["bbox_min"])
    hi = Vector(facts["bbox_max"])
    CACHE.update(lo=lo, hi=hi, target=(lo + hi) / 2.0, rig=rig)

    ALL_VIEWS = {
        "front": ((0.0, -1.0, 0.0), 0.0),
        "three_quarter": ((-0.62, -0.78, 0.0), 12.0),
        "side": ((-1.0, 0.0, 0.0), 0.0),
        "back": ((0.0, 1.0, 0.0), 8.0),
        "top": ((0.0, -0.02, 1.0), 85.0),
        "bottom": ((0.0, -0.02, -1.0), -85.0),
        "three_quarter_back": ((0.72, 0.69, 0.0), 10.0),
    }
    wanted = [v.strip() for v in args.views.split(",") if v.strip()]
    views = {k: ALL_VIEWS[k] for k in wanted}
    log("parts: " + ", ".join(f"{o.name}->clay{i % 3}" for i, o in enumerate(objs)))
    if not args.no_symmetry:
        facts["silhouette_symmetry"] = silhouette_symmetry(out_dir)
    times = render_views(out_dir, args.prefix, views, res, args.samples, args.margin)

    report = {
        "schema": "headshell/stl-review@1",
        "sources": [str(Path(s).resolve()) for s in args.src],
        "prefix": args.prefix,
        "render": {"res": list(res), "samples": args.samples, "engine": "CYCLES/CPU",
                   "views": list(views), "seconds": times, "margin": args.margin},
        "mesh": facts,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / f"{args.prefix}_review.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
