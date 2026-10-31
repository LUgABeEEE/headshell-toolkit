"""Build a wearer inner-head proxy from the CC0 Blender Studio head (MIT adapter).

  blender -b human_base_meshes_bundle.blend -P build_inner_proxy.py -- \
      --width 150 --depth 190 --length 230 --out ../out/inner/inner_proxy.stl

Output units are millimetres. The reference head is CC0; this script only scales it.
"""
import bpy, json, sys, os, math
from mathutils import Vector

argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
def arg(n, d=None): return float(argv[argv.index(n)+1]) if n in argv else d
width = arg("--width", 150.0); depth = arg("--depth", 190.0); length = arg("--length", 230.0)
out = argv[argv.index("--out")+1] if "--out" in argv else "inner_proxy.stl"
out = os.path.abspath(out)

head = bpy.data.objects.get("GEO-head_stylized")
if head is None:
    raise SystemExit("GEO-head_stylized not found")

# Reference bbox in Blender units (metres in this bundle).
ref_vs = [head.matrix_world @ v.co for v in head.data.vertices]
ref_mn = Vector((min(v[i] for v in ref_vs) for i in range(3)))
ref_mx = Vector((max(v[i] for v in ref_vs) for i in range(3)))

def eye_centre(name):
    e = bpy.data.objects.get(name)
    if e is None: return None
    pts = [e.matrix_world @ v.co for v in e.data.vertices]
    if not pts: return None
    return Vector((sum(p[i] for p in pts) / len(pts) for i in range(3)))

ref_eyes = [eye_centre("GEO-head_stylized.eye.L"), eye_centre("GEO-head_stylized.eye.R")]
vs = [head.matrix_world @ v.co for v in head.data.vertices]
mn = Vector((min(v[i] for v in vs) for i in range(3)))
mx = Vector((max(v[i] for v in vs) for i in range(3)))
size = mx - mn
scale = Vector((width / (size.x * 1000.0), depth / (size.y * 1000.0), length / (size.z * 1000.0)))
head.scale = (head.scale.x * scale.x, head.scale.y * scale.y, head.scale.z * scale.z)
bpy.context.view_layer.objects.active = head
head.select_set(True)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Centre at the origin.
vs = [head.matrix_world @ v.co for v in head.data.vertices]
mn = Vector((min(v[i] for v in vs) for i in range(3)))
mx = Vector((max(v[i] for v in vs) for i in range(3)))
# Bake the world transform into the mesh, then centre by subtracting the bbox centre.
from mathutils import Matrix
mw = head.matrix_world.copy()
for v in head.data.vertices:
    v.co = mw @ v.co
head.matrix_world = Matrix.Identity(4)
head.data.update()
vs = [v.co for v in head.data.vertices]
mn2 = Vector((min(v[i] for v in vs) for i in range(3)))
mx2 = Vector((max(v[i] for v in vs) for i in range(3)))
centre = (mn2 + mx2) / 2.0
for v in head.data.vertices:
    v.co -= centre
# The reference head faces -Y; rotate 180 deg so it faces +Y like the shell.
rot = Matrix.Rotation(math.pi, 4, "Z")
for v in head.data.vertices:
    v.co = rot @ v.co
head.data.update()

for o in bpy.data.objects:
    if o.type == "MESH" and o.name != head.name:
        o.hide_render = True
bpy.ops.object.select_all(action="DESELECT")
head.select_set(True)
bpy.ops.wm.stl_export(filepath=out, export_selected_objects=True, apply_modifiers=True,
                      ascii_format=False, global_scale=1000.0)

vs = [head.matrix_world @ v.co for v in head.data.vertices]
mn = [min(v[i] for v in vs) * 1000.0 for i in range(3)]
mx = [max(v[i] for v in vs) * 1000.0 for i in range(3)]
new_mn = Vector(mn); new_size = Vector((mx[i]-mn[i] for i in range(3)))
ref_size = ref_mx - ref_mn
eye_centres_mm = []
for e in ref_eyes:
    if e is None:
        eye_centres_mm.append(None); continue
    rel = Vector(((e[i]-ref_mn[i])/ref_size[i] for i in range(3)))
    p = new_mn + Vector((rel[i]*new_size[i] for i in range(3)))
    p = Matrix.Rotation(math.pi, 4, "Z") @ p
    eye_centres_mm.append([round(p[i], 2) for i in range(3)])
meta = {"file": out, "bytes": os.path.getsize(out),
        "bbox_mm": [round(new_size[i], 1) for i in range(3)],
        "bbox_min_mm": [round(new_mn[i], 1) for i in range(3)],
        "eye_centres_mm": eye_centres_mm,
        "scale": [round(s, 6) for s in scale], "target_mm": [width, depth, length]}
meta_path = os.path.splitext(out)[0] + ".json"
open(meta_path, "w", encoding="utf-8").write(json.dumps(meta, ensure_ascii=False, indent=1))
print("INNER_PROXY", json.dumps(meta, ensure_ascii=False))
