"""Normalize a reconstructed head mesh into a printable, watertight blank (MIT).

Fixes what image-to-3D output usually gets wrong: a neck stub, non-manifold
edges, open boundaries and an unknown origin. Uses a voxel remesh so the result
is guaranteed manifold, then decimates to a triangle budget.

  blender -b --factory-startup -P normalize_head.py -- --config cfg.json
"""
import bpy, bmesh, json, os, sys, datetime
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(n, d=None): return argv[argv.index(n) + 1] if n in argv else d

cfg = json.load(open(arg("--config", "cfg.json"), encoding="utf-8"))
src = os.path.abspath(cfg["src"])
out = os.path.abspath(cfg.get("out", "out/normalized/head_blank.stl"))
os.makedirs(os.path.dirname(out), exist_ok=True)

bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=src)
obj = bpy.context.selected_objects[0]
obj.name = "head"
me = obj.data
# Winding from image-to-3D output is unreliable; fix it before any normal-based offset.
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode="OBJECT")
before = {"triangles": sum(len(p.vertices) - 2 for p in me.polygons), "vertices": len(me.vertices)}

def _signed_volume(o):
    bm = bmesh.new(); bm.from_mesh(o.data)
    v = bm.calc_volume(signed=True); bm.free()
    return v


def bbox(o):
    vs = [o.matrix_world @ v.co for v in o.data.vertices]
    mn = Vector((min(v[i] for v in vs) for i in range(3)))
    mx = Vector((max(v[i] for v in vs) for i in range(3)))
    return mn, mx

mn, mx = bbox(obj)

# 1. Remove the neck stub: cut everything below the jaw plane.
cut_z = cfg.get("cut_z")
if cut_z is None:
    cut_z = mn.z + (mx.z - mn.z) * float(cfg.get("cut_ratio", 0.15))
bm = bmesh.new(); bm.from_mesh(me)
geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-4, plane_co=(0, 0, cut_z), plane_no=(0, 0, 1),
                       clear_inner=True)
bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if len(e.link_faces) == 1])
bm.to_mesh(me); bm.free()

# 2. Voxel remesh: turns a messy scan into a guaranteed-manifold surface.
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
remesh_tris = cfg.get("remesh_triangles", 70000)
try:
    obj.data.remesh_voxel_size = float(cfg.get("voxel_mm", 1.2))
    obj.data.remesh_voxel_adaptivity = float(cfg.get("voxel_adaptivity", 0.0))
    bpy.ops.object.voxel_remesh()
    remesh_ok = True
except Exception as exc:  # noqa: BLE001 - report and continue with the raw mesh
    print("REMESH_FAILED", exc)
    remesh_ok = False

# 3. Decimate to the triangle budget.
tris = sum(len(p.vertices) - 2 for p in obj.data.polygons)
if tris > remesh_tris:
    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.ratio = float(remesh_tris) / float(tris)
    bpy.ops.object.modifier_apply(modifier=mod.name)

# 3b. Hollow the solid into a single-layer shell with a uniform wall.
wall_mm = float(cfg.get("wall_mm", 0.0))
if wall_mm > 0:
    pre_mn, pre_mx = bbox(obj)
    # A closed solid cannot be hollowed by Solidify (it would thicken the surface both
    # ways). Re-open the bottom first: an open surface + Solidify(use_rim) gives a real
    # single-layer shell with a wall and a rim, and booleans later behave normally.
    bm = bmesh.new(); bm.from_mesh(obj.data)
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-4, plane_co=(0, 0, cut_z), plane_no=(0, 0, 1),
                           clear_inner=True)
    bm.to_mesh(obj.data); bm.free()
    obj.data.update()

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    md = obj.modifiers.new("shell", "SOLIDIFY")
    md.thickness = wall_mm
    md.offset = -1.0            # keep the outer surface where it is
    md.use_rim = True           # close the bottom edge into a 3 mm rim
    md.use_even_offset = False  # even offset spikes on noisy remeshed surfaces
    md.use_quality_normals = True
    bpy.ops.object.modifier_apply(modifier=md.name)

    post_mn, post_mx = bbox(obj)
    if max(abs(post_mn[i] - pre_mn[i]) for i in range(3)) > 1.0:
        raise SystemExit("SHELL_EXPLODED pre=%s post=%s" % (
            [round(x, 1) for x in (pre_mx - pre_mn)],
            [round(x, 1) for x in (post_mx - post_mn)]))
    hollow_mode = "open-bottom solidify"

    # Wall thickness on the finished shell: two-hop ray along -normal.
    dg3 = bpy.context.evaluated_depsgraph_get()
    sbvh = BVHTree.FromObject(obj, dg3)
    samples = []
    step = max(1, len(obj.data.vertices) // 8000)
    for i in range(0, len(obj.data.vertices), step):
        v = obj.data.vertices[i]
        n = v.normal
        co = obj.matrix_world @ v.co
        first = sbvh.ray_cast(co + n * 1.0, -n, 40.0)
        if first[0] is None:
            continue
        second = sbvh.ray_cast(first[0] - n * 0.02, -n, 40.0)
        if second[0] is None:
            continue
        t = second[3] + 0.02
        if 0.2 < t < 20.0:
            samples.append(t)
    samples.sort()
    wall_stats = None
    if samples:
        wall_stats = {
            "min_mm": round(samples[0], 2),
            "p05_mm": round(samples[int(len(samples) * 0.05)], 2),
            "median_mm": round(samples[len(samples) // 2], 2),
            "samples": len(samples),
        }

# 4. Centre on the origin.
mn, mx = bbox(obj)
centre = (mn + mx) / 2.0
for v in obj.data.vertices:
    v.co -= obj.matrix_world.inverted() @ centre if False else (obj.matrix_world.inverted() @ centre)
obj.data.update()

# 5. Optional uniform scale so the head width matches the target.
target_w = cfg.get("target_width_mm")
target_d = cfg.get("target_depth_mm")
scale = 1.0
if target_w or target_d:
    mn, mx = bbox(obj)
    sx = float(target_w) / (mx.x - mn.x) if target_w else 1.0
    sy = float(target_d) / (mx.y - mn.y) if target_d else 1.0
    for v in obj.data.vertices:
        v.co.x *= sx
        v.co.y *= sy
    obj.data.update()
    scale = round(sx, 5)

# 6. Recalculate normals and export. Never run this after the cavity join: it would
# flip the inner surface outwards and destroy the hollow.
if wall_mm <= 0:
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True)
bpy.ops.wm.stl_export(filepath=out, export_selected_objects=True, global_scale=1.0, ascii_format=False)

bm = bmesh.new(); bm.from_mesh(obj.data)
boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
nonmanifold = sum(1 for e in bm.edges if len(e.link_faces) > 2)
vol = abs(bm.calc_volume(signed=False)); bm.free()
mn, mx = bbox(obj)
manifest = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "kernel": "headshell.core.print.normalize_head v0.1 (MIT)",
    "config": cfg, "file": os.path.basename(out), "bytes": os.path.getsize(out),
    "before": before,
    "triangles": sum(len(p.vertices) - 2 for p in obj.data.polygons),
    "boundary_edges": boundary, "nonmanifold_edges": nonmanifold,
    "watertight": boundary == 0 and nonmanifold == 0,
    "volume_cm3": round(vol / 1000.0, 1),
    "bbox_mm": [round(mx[i] - mn[i], 1) for i in range(3)],
    "cut_z_mm": round(cut_z, 1), "scale": round(scale, 5), "remesh_ok": remesh_ok,
    "wall_mm": wall_mm, "wall_stats": wall_stats,
    "hollow_mode": hollow_mode if wall_mm > 0 else None,
}
open(os.path.join(os.path.dirname(out), "manifest_normalize.json"), "w", encoding="utf-8").write(
    json.dumps(manifest, ensure_ascii=False, indent=1))
print("NORMALIZED", json.dumps(manifest, ensure_ascii=False))
