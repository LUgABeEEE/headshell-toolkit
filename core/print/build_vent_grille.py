"""M1b: printable vent grille for a head shell (MIT).

The grille is a separate part: it follows the shell's back curvature and drops into a
window cut in the shell, so the shell never needs a 40-hole boolean array.

  blender -b --factory-startup -P build_vent_grille.py -- --config cfg.json
"""
import bpy, bmesh, json, math, os, sys, datetime
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(n, d=None): return argv[argv.index(n) + 1] if n in argv else d

cfg = json.load(open(arg("--config", "cfg.json"), encoding="utf-8"))
shell_path = os.path.abspath(cfg["shell"])
out = os.path.abspath(cfg.get("out", "out/v9/vent_grille.stl"))
os.makedirs(os.path.dirname(out), exist_ok=True)

bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
bpy.ops.wm.stl_import(filepath=shell_path)
shell = bpy.context.selected_objects[0]
from mathutils.bvhtree import BVHTree
dg = bpy.context.evaluated_depsgraph_get()
bvh = BVHTree.FromObject(shell, dg)

# ---- measure the back surface -------------------------------------------------
field_x = float(cfg.get("field_x_mm", 116.0))
field_z = float(cfg.get("field_z_mm", 86.0))
z_center = float(cfg.get("z_center_mm", 55.0))
half_x, half_z = field_x / 2.0, field_z / 2.0
back_positive = bool(cfg.get("back_positive_y", True))
def probe_dir():
    return Vector((0.0, -1.0, 0.0)) if back_positive else Vector((0.0, 1.0, 0.0))
def probe_start(x, z):
    return Vector((x, 400.0 if back_positive else -400.0, z))

samples = []
nx = int(cfg.get("samples_x", 9)); nz = int(cfg.get("samples_z", 9))
for i in range(nx):
    x = -half_x + field_x * i / (nx - 1)
    for j in range(nz):
        z = z_center - half_z + field_z * j / (nz - 1)
        hit = bvh.ray_cast(probe_start(x, z), probe_dir(), 1600.0)
        if hit[0] is not None:
            samples.append((x, z, hit[0].y))
if len(samples) < 9:
    raise SystemExit("GRILLE_NO_SURFACE samples=%d" % len(samples))

# Least-squares fit y = a + b x^2 + c z^2 over the window.
import itertools
A = [[1.0, x * x, (z - z_center) ** 2] for x, z, _ in samples]
b = [y for _, _, y in samples]
def solve3(A, b):
    n = 3
    M = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] + [sum(A[k][i] * b[k] for k in range(len(A)))] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i])); M[i], M[p] = M[p], M[i]
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n + 1):
                M[r][c] -= f * M[i][c]
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x
a, bq, cq = solve3(A, b)

def surf_y(x, z):
    return a + bq * x * x + cq * (z - z_center) ** 2

resid = max(abs(surf_y(x, z) - y) for x, z, y in samples)
plate_t = float(cfg.get("plate_mm", 2.5))
# ---- sample the surface directly on the plate grid ----------------------------
res = int(cfg.get("grid_res", 28))
plate_t = float(cfg.get("plate_mm", 2.5))

grid = [[None] * (res + 1) for _ in range(res + 1)]
missed = 0
for i in range(res + 1):
    x = -half_x + field_x * i / res
    for j in range(res + 1):
        z = z_center - half_z + field_z * j / res
        hit = bvh.ray_cast(probe_start(x, z), probe_dir(), 1600.0)
        if hit[0] is not None:
            grid[i][j] = hit[0].y
        else:
            grid[i][j] = surf_y(x, z)
            missed += 1

# 3x3 smoothing keeps the plate from following ray noise on the voxel-remeshed surface.
smoothed = [[grid[i][j] for j in range(res + 1)] for i in range(res + 1)]
for i in range(res + 1):
    for j in range(res + 1):
        acc, cnt = 0.0, 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i + di, j + dj
                if 0 <= ii <= res and 0 <= jj <= res:
                    acc += grid[ii][jj]; cnt += 1
        smoothed[i][j] = acc / cnt

def plate_y(i, j):
    return smoothed[i][j] - float(cfg.get("offset_mm", 0.6))

plate_t = float(cfg.get("plate_mm", 2.5))
# ---- build the plate mesh -----------------------------------------------------
bm = bmesh.new()
outer = []
for i in range(res + 1):
    x = -half_x + field_x * i / res
    row = []
    for j in range(res + 1):
        z = z_center - half_z + field_z * j / res
        row.append(bm.verts.new((x, plate_y(i, j), z)))
    outer.append(row)
for i in range(res):
    for j in range(res):
        bm.faces.new((outer[i][j], outer[i + 1][j], outer[i + 1][j + 1], outer[i][j + 1]))
bm.normal_update()
me = bpy.data.meshes.new("grille"); bm.to_mesh(me); bm.free()
plate = bpy.data.objects.new("grille", me); bpy.context.collection.objects.link(plate)
bpy.context.view_layer.objects.active = plate; plate.select_set(True)
md = plate.modifiers.new("sol", "SOLIDIFY")
md.thickness = plate_t; md.offset = 1.0; md.use_rim = True; md.use_even_offset = False
bpy.ops.object.modifier_apply(modifier=md.name)

# ---- punch the hole grid ------------------------------------------------------
holes = []
pitch = float(cfg.get("pitch_mm", 16.0))
dia = float(cfg.get("hole_mm", 5.0))
margin = float(cfg.get("hole_margin_mm", 10.0))
inner_x = half_x - margin; inner_z = half_z - margin
nx_h = int(inner_x * 2 // pitch); nz_h = int(inner_z * 2 // pitch)
for i in range(nx_h + 1):
    x = -inner_x + i * pitch
    for j in range(nz_h + 1):
        z = z_center - inner_z + j * pitch
        y = smoothed[min(res, max(0, int((x + half_x) / field_x * res)))][min(res, max(0, int((z - z_center + half_z) / field_z * res)))]
        bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=dia / 2.0,
                                           depth=plate_t * 6.0, location=(x, y, z),
                                           rotation=(math.radians(90), 0, 0))
        holes.append(bpy.context.active_object)
bpy.ops.object.select_all(action="DESELECT")
for h in holes:
    h.select_set(True)
bpy.context.view_layer.objects.active = holes[0]
bpy.ops.object.join()
cutter = bpy.context.active_object
md = plate.modifiers.new("holes", "BOOLEAN")
md.operation = "DIFFERENCE"; md.solver = "EXACT"; md.object = cutter
bpy.context.view_layer.objects.active = plate
bpy.ops.object.modifier_apply(modifier=md.name)
bpy.data.objects.remove(cutter, do_unlink=True)

# Weld and clean: the boolean leaves a few near-duplicate vertices that make the STL
# readback non-manifold even though Blender sees a clean mesh.
bm = bmesh.new(); bm.from_mesh(plate.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=float(cfg.get("weld_mm", 0.01)))
bmesh.ops.dissolve_degenerate(bm, dist=float(cfg.get("weld_mm", 0.01)), edges=bm.edges[:])
bm.to_mesh(plate.data); bm.free()
plate.data.update()

bpy.ops.object.select_all(action="DESELECT"); plate.select_set(True)
bpy.ops.wm.stl_export(filepath=out, export_selected_objects=True, global_scale=1.0, ascii_format=False)

bm = bmesh.new(); bm.from_mesh(plate.data)
boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
nonmanifold = sum(1 for e in bm.edges if len(e.link_faces) > 2)
vol = abs(bm.calc_volume(signed=False)); bm.free()
vs = [plate.matrix_world @ v.co for v in plate.data.vertices]
mn = [min(v[i] for v in vs) for i in range(3)]; mx = [max(v[i] for v in vs) for i in range(3)]
manifest = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "kernel": "headshell.core.print.build_vent_grille v0.1 (MIT)",
    "shell": shell_path, "file": os.path.basename(out), "bytes": os.path.getsize(out),
    "fit": {"a": round(a, 3), "b_x2": round(bq, 6), "c_z2": round(cq, 6),
            "quadratic_residual_mm": round(resid, 2), "missed_probes": missed,
            "direct_sampling": True},
    "plate_mm": plate_t, "holes": len(holes), "hole_mm": dia, "pitch_mm": pitch,
    "triangles": sum(len(p.vertices) - 2 for p in plate.data.polygons),
    "boundary_edges": boundary, "nonmanifold_edges": nonmanifold,
    "watertight": boundary == 0 and nonmanifold == 0,
    "weld_mm": float(cfg.get("weld_mm", 0.01)),
    "volume_cm3": round(vol / 1000.0, 2),
    "bbox_mm": [round(mx[i] - mn[i], 1) for i in range(3)],
}
open(os.path.join(os.path.dirname(out), "manifest_grille.json"), "w", encoding="utf-8").write(
    json.dumps(manifest, ensure_ascii=False, indent=1))
print("GRILLE", json.dumps(manifest, ensure_ascii=False))
