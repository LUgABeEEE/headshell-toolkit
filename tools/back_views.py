"""Render the back shell on its own: 3/4 and side views (runs inside Blender).

    tools/blender-safe.sh -b --factory-startup -P tools/back_views.py -- --only "character-a"

The delivered back piece overlaps the front plate in Y, so it is rendered alone;
this adds the two views that the per-character pass does not produce.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "assets" / "incoming" / "modeler-20261020"
OUT_ROOT = ROOT / "out" / "modeler-baseline"


def setup(res=(760, 980)):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 32
    sc.cycles.use_denoising = False
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    w = bpy.data.worlds.new("w")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.30, 0.32, 0.36, 1)
    for name, rot, energy in (
        ("key", (math.radians(58), 0, math.radians(-32)), 4.0),
        ("fill", (math.radians(70), 0, math.radians(120)), 1.7),
        ("rim", (math.radians(-58), 0, math.radians(160)), 2.2),
    ):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = energy
        ld.angle = math.radians(12)
        ob = bpy.data.objects.new(name, ld)
        sc.collection.objects.link(ob)
        ob.rotation_euler = rot
    cd = bpy.data.cameras.new("cam")
    cd.lens = 72
    cd.sensor_fit = "HORIZONTAL"
    cd.clip_start = 1.0
    cd.clip_end = 1e7
    cam = bpy.data.objects.new("cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    return cam


def render_one(cam, char: str) -> dict:
    src = SRC_ROOT / char / "后头.stl"
    out = OUT_ROOT / char
    if not src.exists():
        return {"character": char, "status": "no-back"}
    sc = bpy.context.scene
    before = set(bpy.data.objects)
    bpy.ops.wm.stl_import(filepath=str(src))
    ob = [o for o in bpy.data.objects if o not in before and o.type == "MESH"][0]
    mat = bpy.data.materials.new("clay")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.80, 0.78, 0.76, 1)
    bsdf.inputs["Roughness"].default_value = 0.6
    ob.data.materials.append(mat)
    for poly in ob.data.polygons:
        poly.use_smooth = False

    pts = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    ctr = (lo + hi) / 2
    dims = tuple(hi[i] - lo[i] for i in range(3))
    radius = 0.5 * math.sqrt(sum(d * d for d in dims))
    cd = cam.data
    res = (sc.render.resolution_x, sc.render.resolution_y)
    fov_h = 2 * math.atan(cd.sensor_width / 2 / cd.lens)
    fov_v = 2 * math.atan(math.tan(fov_h / 2) * res[1] / res[0])
    dist = radius / math.sin(min(fov_h, fov_v) / 2) * 1.04
    made = []
    for name, direction in (
        ("back_three_quarter", (0.6, 1, 0.12)),
        ("back_side", (1, 0.05, 0.05)),
    ):
        z = Vector(direction).normalized()
        up = Vector((0, 0, 1))
        cam.location = ctr + z * dist
        right = up.cross(z).normalized()
        tru = z.cross(right)
        cam.rotation_euler = Matrix([right, tru, z]).transposed().to_4x4().to_quaternion().to_euler()
        sc.render.filepath = str(out / f"{name}.png")
        bpy.ops.render.render(write_still=True)
        made.append(name)
    return {"character": char, "status": "ok", "views": made}


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    chars = (
        args.only
        if args.only
        else sorted(p.name for p in SRC_ROOT.iterdir() if p.is_dir() and (p / "后头.stl").exists())
    )
    results = []
    for char in chars:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        cam = setup()
        results.append(render_one(cam, char))
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
