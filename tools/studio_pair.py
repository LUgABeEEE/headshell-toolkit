#!/usr/bin/env python3
"""Studio product shot: N STL parts side by side on a soft floor.

Runs inside Blender (headless), one call per sheet:

    tools/blender-safe.sh -b --factory-startup -P tools/studio_pair.py -- \
        --src a.stl --src b.stl --out shot.png --res 2200x1250 --samples 160 --gap 45

Unlike tools/stl_review_render.py (four orthographic-ish turntable views for
inspection), this is the product-shot path: parts are laid out left to right in X,
snapped to a shared floor, lit by a sun rig plus a real plane so they cast
contact shadows, and framed by a perspective camera that fits the whole row.

Caveats learned the hard way: the camera clip range must be raised (STL is in
millimetres, so a 1000-unit default clip_end silently cuts the entire scene
away), and object matrices need a view_layer update before the moved bbox is
read back.
"""
import sys, math
import bpy
from mathutils import Vector, Matrix

argv = sys.argv[sys.argv.index("--")+1:]
args = {}
srcs = []
i = 0
while i < len(argv):
    if argv[i] == "--src": srcs.append(argv[i+1]); i += 2
    elif argv[i] == "--out": args["out"] = argv[i+1]; i += 2
    elif argv[i] == "--res": args["res"] = argv[i+1]; i += 2
    elif argv[i] == "--samples": args["samples"] = int(argv[i+1]); i += 2
    elif argv[i] == "--gap": args["gap"] = float(argv[i+1]); i += 2
    else: i += 1
res = tuple(int(v) for v in args.get("res","2000x1250").split("x"))
gap = args.get("gap", 45.0)

bpy.ops.wm.read_factory_settings(use_empty=True)

def import_stl(p):
    before = set(bpy.data.objects)
    bpy.ops.wm.stl_import(filepath=p)
    return [o for o in bpy.data.objects if o not in before and o.type=="MESH"]

def bbox(obs):
    pts=[o.matrix_world @ Vector(c) for o in obs for c in o.bound_box]
    lo=Vector([min(p[i] for p in pts) for i in range(3)])
    hi=Vector([max(p[i] for p in pts) for i in range(3)])
    return lo,hi

groups=[import_stl(s) for s in srcs]
boxes=[bbox(g) for g in groups]
# place left-to-right on x, aligned on the same floor height
cursor = 0.0
zmin = min(b[0].z for b in boxes)
centres=[]
for g,(lo,hi) in zip(groups,boxes):
    w = hi.x-lo.x
    cx_target = cursor + w/2.0
    cx_now = (lo.x+hi.x)/2.0
    g[0].location.x += cx_target - cx_now
    g[0].location.z += zmin - lo.z
    centres.append(cx_target)
    cursor += w + gap
bpy.context.view_layer.update()
total_lo, total_hi = bbox([o for g in groups for o in g])
target = (total_lo+total_hi)/2.0
dims = total_hi-total_lo

# material
mat = bpy.data.materials.new("clay"); mat.use_nodes=True
b = mat.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value=(0.80,0.78,0.755,1.0)
b.inputs["Roughness"].default_value=0.52
b.inputs["Specular IOR Level"].default_value=0.35
for g in groups:
    for ob in g:
        ob.data.materials.append(mat)
        me=ob.data
        me.polygons.foreach_set("use_smooth",[True]*len(me.polygons)); me.update()
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
try:
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(38))
except Exception as e:
    print("smooth skip", e)

# floor
bpy.ops.mesh.primitive_plane_add(size=max(dims.x,dims.y)*4, location=(target.x,target.y,zmin-3))
floor=bpy.context.object
fm=bpy.data.materials.new("floor"); fm.use_nodes=True
fb=fm.node_tree.nodes["Principled BSDF"]
fb.inputs["Base Color"].default_value=(0.34,0.35,0.37,1.0)
fb.inputs["Roughness"].default_value=0.75
floor.data.materials.append(fm)

# world + lights
w=bpy.data.worlds.new("w"); bpy.context.scene.world=w; w.use_nodes=True
bg=w.node_tree.nodes["Background"]
bg.inputs["Color"].default_value=(0.34,0.36,0.40,1.0); bg.inputs["Strength"].default_value=0.45
rig=bpy.data.objects.new("rig",None); bpy.context.scene.collection.objects.link(rig)
for name,rot,energy in [("key",(math.radians(48),0,math.radians(-34)),4.2),
                        ("fill",(math.radians(72),0,math.radians(112)),0.9),
                        ("rim",(math.radians(-40),0,math.radians(172)),2.2)]:
    ld=bpy.data.lights.new(name,"SUN"); ld.energy=energy; ld.angle=math.radians(16)
    ob=bpy.data.objects.new(name,ld); bpy.context.scene.collection.objects.link(ob)
    ob.rotation_euler=rot; ob.parent=rig
rig.rotation_euler[2]=math.radians(28)

# camera
cd=bpy.data.cameras.new("cam"); cd.sensor_fit="HORIZONTAL"; cd.lens=70
cd.clip_start=1.0; cd.clip_end=1e6   # mm units: the default 1000 clips the whole scene away
cam=bpy.data.objects.new("cam",cd); bpy.context.scene.collection.objects.link(cam)
d=Vector((0.38,-0.74,0.55)).normalized()          # from front-right-above
up=Vector((0,0,1)); right=up.cross(d).normalized(); tru=d.cross(right)
cam.rotation_euler=Matrix([right,tru,d]).transposed().to_4x4().to_quaternion().to_euler()
fov_h=2*math.atan(cd.sensor_width/2/cd.lens); fov_v=2*math.atan(math.tan(fov_h/2)*res[1]/res[0])
th,tv=math.tan(fov_h/2),math.tan(fov_v/2)
need=0.0
for x in (total_lo.x,total_hi.x):
    for y in (total_lo.y,total_hi.y):
        for z in (total_lo.z,total_hi.z):
            p=Vector((x,y,z))-target
            need=max(need, abs(p.dot(right))/th+p.dot(d), abs(p.dot(tru))/tv+p.dot(d))
cam.location=target+d*(need*1.05)
bpy.context.scene.camera=cam
print("DEBUG target",tuple(round(v,1) for v in target),"dims",tuple(round(v,1) for v in dims))
print("DEBUG lo",tuple(round(v,1) for v in total_lo),"hi",tuple(round(v,1) for v in total_hi))
print("DEBUG need",round(need,1),"cam",tuple(round(v,1) for v in cam.location))
print("DEBUG objects",[ (o.name, o.type) for o in bpy.context.scene.objects ])
print("DEBUG camera set", bpy.context.scene.camera.name if bpy.context.scene.camera else None)

sc=bpy.context.scene
sc.render.engine="CYCLES"; sc.cycles.device="CPU"
sc.cycles.samples=args.get("samples",120); sc.cycles.use_denoising=True
sc.cycles.use_adaptive_sampling=True
sc.render.resolution_x,sc.render.resolution_y=res
sc.render.image_settings.file_format="PNG"; sc.render.image_settings.color_mode="RGB"
sc.view_settings.view_transform="AgX"; sc.view_settings.exposure=0.25
sc.render.filepath=args["out"]
bpy.ops.render.render(write_still=True)
print("HERO", args["out"], res)
