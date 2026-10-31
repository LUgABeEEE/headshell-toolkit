"""M1b: full single-layer shell with exact clearance, vents and eye openings (MIT).

Pipeline:
  outer STL + inner head STL -> align -> measure min clearance (BVH)
  -> dilate inner by liner -> boolean subtract -> eye/mouth cuts -> vent array
  -> export STL + manifest

Run with Blender headless:
  blender -b --factory-startup -P build_shell_full.py -- --config cfg.json
"""
import bpy, bmesh, json, math, os, sys, datetime
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

def _req(cfg: dict, key: str):
    """Design values are supplied by the caller; nothing buildable ships here."""
    if key not in cfg or cfg[key] is None:
        raise KeyError(f"config is missing '{key}' (no default is published)")
    return cfg[key]



argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
def arg(n, d=None): return argv[argv.index(n)+1] if n in argv else d
cfg = json.load(open(arg("--config", "cfg.json"), encoding="utf-8"))
out = os.path.abspath(cfg.get("out", "out/m1b/shell_full.stl"))
os.makedirs(os.path.dirname(out), exist_ok=True)

bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)

def import_stl(path, name):
    bpy.ops.wm.stl_import(filepath=os.path.abspath(path))
    o = bpy.context.selected_objects[0]; o.name = name
    return o

outer = import_stl(cfg["outer"], "outer")
inner = import_stl(cfg["inner"], "inner")

def bbox(o):
    vs = [o.matrix_world @ v.co for v in o.data.vertices]
    mn = Vector((min(v[i] for v in vs) for i in range(3)))
    mx = Vector((max(v[i] for v in vs) for i in range(3)))
    return mn, mx

# Align inner bbox centre to outer bbox centre.
omn, omx = bbox(outer); imn, imx = bbox(inner)
inner.location += (omn + omx) / 2.0 - (imn + imx) / 2.0
bpy.context.view_layer.objects.active = inner; inner.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def shape_adjust(o, cfg):
    """Local shape changes on the shell (mm): nose forward, chin down, back depth."""
    nose = float(cfg.get("nose_forward_mm", 0.0))
    chin = float(cfg.get("chin_down_mm", 0.0))
    back = float(cfg.get("back_depth_mm", 0.0))
    if nose == 0.0 and chin == 0.0 and back == 0.0:
        return
    for v in o.data.vertices:
        x, y, z = v.co
        if nose:
            g = math.exp(-((x / 35.0) ** 2 + ((y - 95.0) / 25.0) ** 2 + ((z - 5.0) / 30.0) ** 2))
            v.co.y += nose * g
        if chin:
            t = smoothstep((-30.0 - z) / 70.0)
            if t > 0.0 and y > -40.0:
                v.co.z -= chin * t * math.exp(-(x / 70.0) ** 2)
        if back:
            t = smoothstep((-40.0 - y) / 60.0)
            if t > 0.0:
                v.co.y -= back * t
    o.data.update()

shape_adjust(outer, cfg)

if cfg.get("decimate_outer"):
    mod = outer.modifiers.new("dec", "DECIMATE"); mod.ratio = float(cfg["decimate_outer"])
    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.modifier_apply(modifier=mod.name)

# Exact min clearance: nearest outer surface for every inner vertex.
dg = bpy.context.evaluated_depsgraph_get()
bvh = BVHTree.FromObject(outer, dg)
inner_eval = inner.evaluated_get(dg)
clearances = []
for v in inner_eval.data.vertices:
    co = inner.matrix_world @ v.co
    hit = bvh.find_nearest(co)
    if hit and hit[3] is not None:
        clearances.append((hit[3], tuple(round(c, 1) for c in co)))
clearances.sort(key=lambda item: item[0])
min_clear, min_at = clearances[0] if clearances else (0.0, ())
p05 = clearances[int(len(clearances) * 0.05)][0] if clearances else 0.0
print("CLEARANCE_MM", json.dumps({"min": round(min_clear, 2), "p05": round(p05, 2),
      "at_mm": min_at, "samples": len(clearances)}))

# Dilate the inner proxy by the liner thickness (outward along normals).
liner = float(_req(cfg, "liner_mm")) / 1000.0
if liner > 0:
    mod = inner.modifiers.new("dilate", "DISPLACE"); mod.strength = liner; mod.mid_level = 0.0
    bpy.context.view_layer.objects.active = inner
    bpy.ops.object.modifier_apply(modifier=mod.name)

def box(name, center, size):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    o = bpy.context.active_object; o.name = name; o.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return o

def ellipse(name, center, rx, rz, depth, verts=64):
    """Deterministic ellipse cylinder along +Y: X half-axis rx, Z half-axis rz, Y length depth."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=verts,
                          radius1=1.0, radius2=1.0, depth=1.0)
    mat = Matrix.Rotation(math.radians(90), 4, "X") @ Matrix.Diagonal((rx, rz, depth, 1.0))
    bmesh.ops.transform(bm, matrix=mat, verts=bm.verts)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(o)
    o.location = center
    return o

def cyl(name, center, radius, depth, axis="Y", verts=64):
    rot = {"Y": (math.radians(90),0,0), "X": (0,math.radians(90),0), "Z": (0,0,0)}[axis]
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=center, rotation=rot)
    o = bpy.context.active_object; o.name = name
    return o

def _mesh_volume_mm3(o):
    bm = bmesh.new(); bm.from_mesh(o.data)
    vol = abs(bm.calc_volume(signed=False)); bm.free()
    return vol


def boolean_op(target, cutter, operation, first_solver="EXACT"):
    """Apply a boolean, falling back from EXACT to FAST if the result degenerates.

    EXACT is preferable but fails on some remeshed/solidified shells (it returns a
    handful of stray triangles). We keep a mesh backup, validate the result, and
    retry with FAST when the solver produced garbage.
    """
    last = None
    pre_vol = _mesh_volume_mm3(target)
    order = ("FAST", "EXACT") if first_solver == "FAST" else ("EXACT", "FAST")
    for solver in order:
        backup = target.data.copy()
        mod = target.modifiers.new("b", "BOOLEAN")
        mod.operation = operation
        mod.solver = solver
        mod.object = cutter
        bpy.context.view_layer.objects.active = target
        bpy.ops.object.modifier_apply(modifier=mod.name)
        tris = sum(len(p.vertices) - 2 for p in target.data.polygons)
        vol = _mesh_volume_mm3(target)
        # A DIFFERENCE/INTERSECT cannot grow the volume. EXACT sometimes "fills" a
        # two-surface hollow into a solid, which passes a naive non-empty check.
        sane_vol = vol <= pre_vol * 1.02 if operation in ("DIFFERENCE", "INTERSECT") else vol >= pre_vol * 0.98
        print("BOOLEAN_STEP", solver, operation, "tris", tris, "vol_cm3", round(vol / 1000.0, 1),
              "pre_cm3", round(pre_vol / 1000.0, 1), "cut", getattr(cutter, "name", "?"))
        if tris >= 100 and vol > 1000.0 and sane_vol:
            bpy.data.meshes.remove(backup)
            bpy.data.objects.remove(cutter, do_unlink=True)
            if solver != "EXACT":
                print("BOOLEAN_SOLVER", solver, operation)
            return
        print("BOOLEAN_DEGENERATE", solver, operation, "tris", tris, "vol_cm3", round(vol / 1000.0, 2))
        last = (tris, vol)
        target.data = backup
    bpy.data.objects.remove(cutter, do_unlink=True)
    raise SystemExit("BOOLEAN_FAILED %s last=%s" % (operation, last))


def _mesh_quality(o):
    """Triangles, volume and edge statistics of a mesh."""
    bm = bmesh.new(); bm.from_mesh(o.data)
    boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    nonmanifold = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    vol = abs(bm.calc_volume(signed=False))
    bm.free()
    tris = sum(len(p.vertices) - 2 for p in o.data.polygons)
    return {"tris": tris, "vol": vol, "boundary": boundary, "nonmanifold": nonmanifold}


def _extent(o):
    vs = [o.matrix_world @ v.co for v in o.data.vertices]
    if not vs:
        return (0.0, 0.0, 0.0)
    return tuple(max(v[i] for v in vs) - min(v[i] for v in vs) for i in range(3))


def _repair_mesh(o, merge_mm=0.01, hole_sides=8):
    """Local repair: weld doubles, close small holes, unify normals."""
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=merge_mm)
    boundary = [e for e in bm.edges if len(e.link_faces) == 1]
    if boundary:
        try:
            bmesh.ops.holes_fill(bm, edges=boundary, sides=hole_sides)
        except Exception as exc:  # noqa: BLE001 - repair is best effort
            print("REPAIR_FILL_FAILED", exc)
    bm.to_mesh(o.data); bm.free()
    o.data.update()


def cut_with_repair(target, cutter, operation="DIFFERENCE", label="cut", cfg=None):
    """Boolean with solver fallback, local repair and rollback. Never leaves a broken mesh.

    A cut is accepted only when the result keeps a sane volume, does not grow the
    bounding box, and does not make the edge statistics much worse. Otherwise the mesh
    is rolled back, repaired and retried; if every attempt fails the cut is skipped and
    recorded in CUT_LEDGER instead of corrupting the shell.
    """
    cfg = cfg or {}
    guard = float(cfg.get("bbox_guard_mm", 1.0))
    edge_tol = int(cfg.get("cut_edge_tolerance", 40))
    nm_tol = int(cfg.get("cut_nonmanifold_tolerance", 400))
    max_removal = float(cfg.get("cut_max_removal_fraction", 0.25))
    pre = _mesh_quality(target)
    pre_ext = _extent(target)
    attempts = []
    for order in (("EXACT", "FAST"), ("FAST", "EXACT")):
        for solver in order:
            backup = target.data.copy()
            mod = target.modifiers.new("cut", "BOOLEAN")
            mod.operation = operation; mod.solver = solver; mod.object = cutter
            bpy.context.view_layer.objects.active = target
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception as exc:  # noqa: BLE001 - solver can hard fail
                target.data = backup
                attempts.append({"solver": solver, "result": "exception", "detail": str(exc)[:120]})
                continue
            q = _mesh_quality(target)
            ext = _extent(target)
            grew = max(ext[i] - pre_ext[i] for i in range(3))
            sane_vol = q["vol"] <= pre["vol"] * 1.02 if operation in ("DIFFERENCE", "INTERSECT") \
                else q["vol"] >= pre["vol"] * 0.98
            ok = (q["tris"] >= pre["tris"] * 0.5 and q["vol"] > pre["vol"] * (1.0 - max_removal)
                  and sane_vol and grew <= guard
                  and q["boundary"] <= max(pre["boundary"], 0)
                  and q["nonmanifold"] <= pre["nonmanifold"] + nm_tol)
            attempts.append({"solver": solver, "result": "ok" if ok else "rejected",
                             "tris": q["tris"], "vol_cm3": round(q["vol"] / 1000.0, 1),
                             "bbox_growth_mm": round(grew, 2),
                             "boundary": q["boundary"], "nonmanifold": q["nonmanifold"]})
            if ok:
                bpy.data.meshes.remove(backup)
                bpy.data.objects.remove(cutter, do_unlink=True)
                CUT_LEDGER.append({"label": label, "status": "cut", "solver": solver, "attempts": attempts})
                if solver != "EXACT":
                    print("CUT_SOLVER", label, solver)
                return True
            # try a local repair before giving up on this attempt
            _repair_mesh(target, float(cfg.get("repair_merge_mm", 0.01)),
                         int(cfg.get("repair_hole_sides", 8)))
            q2 = _mesh_quality(target); ext2 = _extent(target)
            grew2 = max(ext2[i] - pre_ext[i] for i in range(3))
            ok2 = (q2["tris"] >= pre["tris"] * 0.5 and q2["vol"] > pre["vol"] * (1.0 - max_removal)
                   and q2["vol"] <= pre["vol"] * 1.02
                   and grew2 <= guard and q2["boundary"] <= max(pre["boundary"], 0)
                   and q2["nonmanifold"] <= pre["nonmanifold"] + nm_tol)
            attempts.append({"solver": solver + "+repair", "result": "ok" if ok2 else "rejected",
                             "tris": q2["tris"], "vol_cm3": round(q2["vol"] / 1000.0, 1),
                             "bbox_growth_mm": round(grew2, 2),
                             "boundary": q2["boundary"], "nonmanifold": q2["nonmanifold"]})
            if ok2:
                bpy.data.meshes.remove(backup)
                bpy.data.objects.remove(cutter, do_unlink=True)
                CUT_LEDGER.append({"label": label, "status": "cut+repaired", "solver": solver + "+repair",
                                   "attempts": attempts})
                print("CUT_REPAIRED", label, solver)
                return True
            target.data = backup
    bpy.data.objects.remove(cutter, do_unlink=True)
    CUT_LEDGER.append({"label": label, "status": "skipped", "attempts": attempts})
    print("CUT_SKIPPED", label, json.dumps(attempts[-2:], ensure_ascii=False))
    return False


CUT_LEDGER: list = []


def boolean(target, cutter, first_solver="EXACT"):
    boolean_op(target, cutter, "DIFFERENCE", first_solver=first_solver)

# Cavity is only meaningful for a solid outer. A printed single-layer shell is already hollow.
inner_bvh = BVHTree.FromObject(inner, dg)
if cfg.get("carve_cavity", False):
    boolean(outer, inner)
else:
    bpy.data.objects.remove(inner, do_unlink=True)

# Base blanks may already carry openings; plug them before cutting your own.
for px, pz in cfg.get("existing_eye_centers_mm", []):
    # Rounded patch instead of a box: fills the old 47x30 hole with a curved plug.
    plug = ellipse("plug", (float(px), float(_req(cfg, "eye_y_mm")), float(pz)),
                   27.0, 18.0, 10.0, verts=48)
    boolean_op(outer, plug, "UNION")

# Face openings.
dg_open = bpy.context.evaluated_depsgraph_get()
outer_bvh = BVHTree.FromObject(outer, dg_open)
face_negative = bool(cfg.get("face_negative_y", False))


def surface_y(x, z, probes=None):
    """Y of the outermost surface near (x, z), taken from the face side.

    A single probe can fall through an existing opening and hit an interior surface,
    which would place the cutter inside the cavity. Probing a small ring and keeping the
    outermost hit makes the placement robust next to holes.
    """
    sign = -1.0 if face_negative else 1.0
    offsets = probes if probes is not None else ((0, 0), (18, 0), (-18, 0), (0, 16), (0, -16))
    hits = []
    for dx, dz in offsets:
        start = Vector((x + dx, sign * 400.0, z + dz))
        hit = outer_bvh.ray_cast(start, Vector((0.0, -sign, 0.0)), 1600.0)
        if hit[0] is not None:
            hits.append(hit[0].y)
    if not hits:
        return None
    return max(hits) if sign > 0 else min(hits)


def cutter_center_y(surf_y, depth, protrusion=None):
    """Place a through-cutter so it protrudes only a hair past the outer surface.

    Centring the cutter on the surface would push half its length outside the head; the
    solver then tends to leave the far cap in the mesh (bbox grows). Instead the cutter
    starts just outside the surface and runs into the cavity.
    """
    if protrusion is None:
        protrusion = float(cfg.get("cut_protrusion_mm", 0.5))
    if face_negative:
        return surf_y + depth / 2.0 - protrusion
    return surf_y - depth / 2.0 + protrusion


ipd = float(_req(cfg, "ipd_mm"))
eye_rx = float(_req(cfg, "eye_radius_x_mm")); eye_rz = float(_req(cfg, "eye_radius_z_mm"))
seat = float(cfg.get("eye_seat_mm", 4.0))
eye_y = float(_req(cfg, "eye_y_mm")); eye_z = float(_req(cfg, "eye_z_mm"))
for sign in (1, -1):
    cx = sign * ipd / 2.0
    if cfg.get("place_by_ray", False):
        y_hit = surface_y(cx, eye_z)
        if y_hit is not None:
            depth = float(cfg.get("eye_cut_depth_mm", 60.0))
            eye_y = cutter_center_y(y_hit, depth)
            print("EYE_PLACED", round(cx, 1), round(eye_z, 1), round(eye_y, 1), "surf", round(y_hit, 1))
    cut_with_repair(outer, ellipse("eye", (cx, eye_y, eye_z), eye_rx, eye_rz,
                                   float(cfg.get("eye_cut_depth_mm", 60.0))),
                    label="eye%d" % (1 if sign > 0 else 2), cfg=cfg)
    if seat > 0:
        # Shallow full ring on the outer surface: leaves the wall and the bridge intact.
        seat_depth = float(cfg.get("eye_seat_depth_mm", 1.5))
        seat_y = eye_y + 4.0
        big = ellipse("seat_big", (cx, seat_y, eye_z), eye_rx + seat, eye_rz + seat, seat_depth)
        small = ellipse("seat_small", (cx, seat_y, eye_z), eye_rx, eye_rz, seat_depth + 2.0)
        boolean(big, small)
        boolean(outer, big)
mw = float(_req(cfg, "mouth_width_mm")); mh = float(_req(cfg, "mouth_height_mm"))
mouth_z = float(cfg.get("mouth_z_mm", -85.0))
mouth_y = float(cfg.get("mouth_y_mm", 100.0))
if cfg.get("place_by_ray", False):
    y_mouth = surface_y(0.0, mouth_z)
    if y_mouth is not None:
        mouth_y = cutter_center_y(y_mouth, float(cfg.get("mouth_cut_depth_mm", 60.0)))
        print("MOUTH_PLACED", round(mouth_z, 1), round(mouth_y, 1), "surf", round(y_mouth, 1))
cut_with_repair(outer, box("mouth", (0, mouth_y, mouth_z),
                           (mw, float(cfg.get("mouth_cut_depth_mm", 60.0)), mh)), label="mouth", cfg=cfg)

# Neck / bottom opening: a box that cuts through the bottom of the shell.
neck = cfg.get("neck_opening_mm")
if neck:
    zmin = min((outer.matrix_world @ v.co).z for v in outer.data.vertices)
    nw, nd = float(neck[0]), float(neck[1])
    zc = zmin + float(cfg.get("neck_opening_depth_mm", 12.0)) / 2.0
    yc = cfg.get("neck_opening_y_mm")
    if yc is None:
        yc = sum((outer.matrix_world @ v.co).y for v in outer.data.vertices) / len(outer.data.vertices)
    boolean(outer, box("neck", (0.0, yc, zc), (nw, nd, float(cfg.get("neck_opening_depth_mm", 12.0)))))

# Vent window: one large opening for a separately printed grille.
window = cfg.get("vent_window_mm")
if window:
    wz = float(_req(cfg, "vent_window_z_mm"))
    wz0 = float(_req(cfg, "vent_window_z_span_mm"))
    # The window goes on the side opposite the face.
    ys = [(outer.matrix_world @ v.co).y for v in outer.data.vertices]
    back_y = max(ys) if cfg.get("face_negative_y", False) else min(ys)
    toward = 1.0 if cfg.get("face_negative_y", False) else -1.0
    half = float(_req(cfg, "vent_window_cut_depth_mm")) / 2.0
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0.0, back_y + toward * (half - 0.5), wz))
    win = bpy.context.active_object
    win.dimensions = (float(window), float(_req(cfg, "vent_window_cut_depth_mm")), wz0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    cut_with_repair(outer, win, label="vent_window", cfg=cfg)

# Vent array on the back: cylinders along Y.
vd = float(_req(cfg, "vent_diameter_mm")) / 2.0
vs = float(_req(cfg, "vent_spacing_mm"))
vz0, vz1 = cfg.get("vent_z_mm", [10.0, 120.0]); vx = cfg.get("vent_x_mm", 70.0)
# Outer STL is already in millimetres.
ymn = omn.y
cutters = []
skipped = 0
z = vz0
while z <= vz1:
    x = -vx
    while x <= vx:
        hit = inner_bvh.find_nearest(Vector((x, ymn, z)))
        if hit and hit[3] is not None and hit[3] < vd + 2.0:
            skipped += 1
        else:
            cutters.append(cyl("vent", (x, ymn - 10.0, z), vd,
                               float(cfg.get("vent_cut_depth_mm", 80.0)), verts=24))
        x += vs
    z += vs
if cutters:
    if cfg.get("vent_individual", True):
        # One boolean per hole: far more reliable than subtracting a 35-cylinder compound.
        for i, c in enumerate(cutters):
            cut_with_repair(outer, c, label="vent%d" % (i + 1), cfg=cfg)
    else:
        bpy.ops.object.select_all(action="DESELECT")
        for c in cutters: c.select_set(True)
        bpy.context.view_layer.objects.active = cutters[0]
        bpy.ops.object.join()
        vents = bpy.context.active_object
        boolean(outer, vents)

# Optional repair: merge coincident vertices and close small holes left by the solver.
if cfg.get("repair", False):
    bm = bmesh.new(); bm.from_mesh(outer.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=float(cfg.get("repair_merge_mm", 0.05)))
    boundary = [e for e in bm.edges if len(e.link_faces) == 1]
    if boundary:
        try:
            bmesh.ops.holes_fill(bm, edges=boundary, sides=int(cfg.get("repair_hole_sides", 32)))
        except Exception as exc:  # noqa: BLE001 - report and continue
            print("REPAIR_FILL_FAILED", exc)
    bm.to_mesh(outer.data); bm.free()

# Boolean/decimate can flip normals; make them consistent before export.
# Skip this for a two-surface hollow: it would flip the cavity outwards and destroy it.
if not cfg.get("preserve_hollow", False):
    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

# Print checks: wall thickness, overhang area, vents skipped.
dg2 = bpy.context.evaluated_depsgraph_get()
final_bvh = BVHTree.FromObject(outer, dg2)
me = outer.data
thickness = []
step = max(1, len(me.vertices) // 20000)
if cfg.get("preserve_hollow", False):
    # For a two-surface hollow the two-hop probe hits the cavity; skip it and rely on
    # the wall measurement recorded by normalize_head.
    step = len(me.vertices) + 1
for i in range(0, len(me.vertices), step):
    v = me.vertices[i]
    n = v.normal
    origin = (outer.matrix_world @ v.co) + n * 1.0
    first = final_bvh.ray_cast(origin, -n, 20.0)
    if first[0] is None:
        continue
    second = final_bvh.ray_cast(first[0] - n * 0.02, -n, 20.0)
    if second[0] is None:
        continue
    t = second[3] + 0.02
    if 0.2 < t < 20.0:
        thickness.append(t)
thickness.sort()
wall_min = round(thickness[0], 2) if thickness else None
wall_p05 = round(thickness[int(len(thickness) * 0.05)], 2) if thickness else None
wall_med = round(thickness[len(thickness) // 2], 2) if thickness else None
area = sum(p.area for p in me.polygons)
overhang = sum(p.area for p in me.polygons if p.normal.z < -0.3)
overhang_ratio = round(overhang / area, 4) if area else None
print("PRINT_CHECK", json.dumps({"wall_min_mm": wall_min, "wall_p05_mm": wall_p05, "wall_median_mm": wall_med,
      "samples": len(thickness), "overhang_ratio": overhang_ratio,
      "vents_skipped": skipped}, ensure_ascii=False))

# Export + stats.
bpy.ops.object.select_all(action="DESELECT"); outer.select_set(True)
bpy.ops.wm.stl_export(filepath=out, export_selected_objects=True, apply_modifiers=True,
                      ascii_format=False, global_scale=1.0)
me = outer.data
bm = bmesh.new(); bm.from_mesh(me)
boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
nonmanifold = sum(1 for e in bm.edges if len(e.link_faces) > 2)
vol = abs(bm.calc_volume(signed=False)); bm.free()
vs2 = [outer.matrix_world @ v.co for v in me.vertices]
mn = [min(v[i] for v in vs2) for i in range(3)]; mx = [max(v[i] for v in vs2) for i in range(3)]
manifest = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "kernel": "headshell.core.print.build_shell_full v0.1 (MIT)",
            "config": cfg, "file": os.path.basename(out), "bytes": os.path.getsize(out),
            "triangles": sum(len(p.vertices) - 2 for p in me.polygons),
            "polygons": len(me.polygons), "boundary_edges": boundary,
            "nonmanifold_edges": nonmanifold, "watertight": boundary == 0 and nonmanifold == 0,
            "volume_cm3": round(vol / 1000.0, 1),
            "bbox_mm": [round(mx[i]-mn[i], 1) for i in range(3)],
            "clearance_mm": {"min": round(min_clear, 2), "p05": round(p05, 2),
                             "at_mm": min_at, "liner_ok": min_clear >= _req(cfg, "liner_mm")},
            "cuts": CUT_LEDGER,
            "print_check": {"wall_min_mm": wall_min, "wall_p05_mm": wall_p05, "wall_median_mm": wall_med,
                            "overhang_ratio": overhang_ratio, "vents_skipped": skipped,
                            "wall_samples": len(thickness)},
            "liner_mm": _req(cfg, "liner_mm")}
open(os.path.join(os.path.dirname(out), "manifest_full.json"), "w", encoding="utf-8").write(
    json.dumps(manifest, ensure_ascii=False, indent=1))
print("MANIFEST", json.dumps(manifest, ensure_ascii=False))
