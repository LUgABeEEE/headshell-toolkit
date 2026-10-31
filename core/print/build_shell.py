"""HeadShell print kernel (MIT): split a head-shell mesh and cut face openings.

This module is deliberately independent of any GPL base-model generator. It consumes a
plain STL/OBJ base mesh and applies our own fabrication operations:

  - split into a front and a back half at a plane
  - cut two eye openings at the requested IPD
  - cut a mouth opening
  - export one STL per part plus a manifest with printability checks

Run with Blender's bundled Python (headless):
  blender -b --factory-startup -P build_shell.py -- --config cfg.json
"""
import bpy, bmesh, json, math, os, sys, datetime


def _req(cfg: dict, key: str):
    """Design values are supplied by the caller; nothing buildable ships here."""
    if key not in cfg or cfg[key] is None:
        raise KeyError(f"config is missing '{key}' (no default is published)")
    return cfg[key]

argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
def arg(n, d=None): return argv[argv.index(n)+1] if n in argv else d
cfg = json.load(open(arg("--config", "cfg.json"), encoding="utf-8"))
out_dir = os.path.abspath(cfg.get("out_dir", "out/m1")); os.makedirs(out_dir, exist_ok=True)

# ---- clean scene -----------------------------------------------------------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for block in (bpy.data.meshes, bpy.data.materials):
    for item in list(block):
        if item.users == 0: block.remove(item)

# ---- import base -----------------------------------------------------------
base_path = os.path.abspath(cfg["base_stl"])
bpy.ops.wm.stl_import(filepath=base_path)
base = bpy.context.selected_objects[0]
base.name = "base"
print("IMPORT", base_path, "verts", len(base.data.vertices), "polys", len(base.data.polygons))

def duplicate(obj, name):
    new = obj.copy(); new.data = obj.data.copy(); new.name = name
    bpy.context.collection.objects.link(new)
    return new

def apply_boolean(target, cutter, operation="DIFFERENCE"):
    mod = target.modifiers.new("bool", "BOOLEAN")
    mod.operation = operation; mod.solver = "EXACT"; mod.object = cutter
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)

def box(name, center, size):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    o = bpy.context.active_object; o.name = name
    o.dimensions = size
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return o

def cylinder(name, center, radius, depth, axis="Y"):
    rot = {"Y": (math.radians(90), 0, 0), "X": (0, math.radians(90), 0), "Z": (0, 0, 0)}[axis]
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=depth, location=center, rotation=rot)
    o = bpy.context.active_object; o.name = name
    return o

# ---- split into front / back ----------------------------------------------
# Product convention: all mesh coordinates are millimetres.
split_y = _req(cfg, "split_y_mm")
bbox_size = max(base.dimensions) * 2.0
front = duplicate(base, "shell_front")
back = duplicate(base, "shell_back")
# front keeps Y > split_y: subtract a box covering Y < split_y
front_cutter = box("cut_front", (0, split_y - bbox_size / 2.0, 0), (bbox_size, bbox_size, bbox_size))
apply_boolean(front, front_cutter)
# back keeps Y < split_y
back_cutter = box("cut_back", (0, split_y + bbox_size / 2.0, 0), (bbox_size, bbox_size, bbox_size))
apply_boolean(back, back_cutter)

# ---- face openings --------------------------------------------------------
ipd = float(_req(cfg, "ipd_mm"))
eye_z = _req(cfg, "eye_z_mm")
eye_r = cfg.get("eye_radius_mm", 22.0)
eye_y = cfg.get("eye_y_mm", 88.0)
for sign, label in ((1, "l"), (-1, "r")):
    cutter = cylinder("eye_" + label, (sign * ipd / 2.0, eye_y, eye_z), eye_r, 140.0)
    apply_boolean(front, cutter)

mouth_z = cfg.get("mouth_z_mm", -85.0)
mouth_y = cfg.get("mouth_y_mm", 100.0)
mouth_w = cfg.get("mouth_width_mm", 36.0)
mouth_h = cfg.get("mouth_height_mm", 16.0)
mouth = box("mouth", (0, mouth_y, mouth_z), (mouth_w, 120.0, mouth_h))
apply_boolean(front, mouth)

# ---- export + checks ------------------------------------------------------
def stats(obj):
    me = obj.data
    bm = bmesh.new(); bm.from_mesh(me)
    boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    nonmanifold = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    try: vol = abs(bm.calc_volume(signed=False))
    except Exception: vol = 0.0
    bm.free()
    coords = [obj.matrix_world @ v.co for v in me.vertices]
    mn = [min(c[i] for c in coords) for i in range(3)]
    mx = [max(c[i] for c in coords) for i in range(3)]
    return {"verts": len(me.vertices), "polys": len(me.polygons), "boundary_edges": boundary,
            "nonmanifold_edges": nonmanifold, "watertight": boundary == 0 and nonmanifold == 0,
            "volume_cm3": round(vol / 1000.0, 1),
            "bbox_mm": [round(mx[i] - mn[i], 1) for i in range(3)]}

manifest = {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "kernel": "headshell.core.print.build_shell v0.1 (MIT)",
            "base": os.path.basename(base_path), "config": cfg, "parts": []}
for obj, part_id, kind in ((front, "shell_front", "shell"), (back, "shell_back", "shell")):
    path = os.path.join(out_dir, part_id + ".stl")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True, apply_modifiers=True, ascii_format=False)
    item = {"id": part_id, "kind": kind, "file": os.path.basename(path), "bytes": os.path.getsize(path)}
    item.update(stats(obj)); manifest["parts"].append(item)
    print("PART", json.dumps(item, ensure_ascii=False))

with open(os.path.join(out_dir, "manifest_kernel.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)
print("MANIFEST", os.path.join(out_dir, "manifest_kernel.json"))
