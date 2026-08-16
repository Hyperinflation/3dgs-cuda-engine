import bpy
import os
import struct
import numpy as np

# Reset scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Path to model.ply
ply_path = os.path.abspath("model.ply")
blend_out = os.path.abspath("Mimari_3DGS_Model.blend")

print(f"[*] Importing 3DGS model into Blender: {ply_path}")

# Read PLY binary vertices
verts = []
colors = []

with open(ply_path, "rb") as f:
    # Parse header
    header_end = False
    num_verts = 0
    while True:
        line = f.readline().decode('ascii', errors='ignore').strip()
        if line.startswith("element vertex"):
            num_verts = int(line.split()[-1])
        if line == "end_header":
            break
    
    print(f"[*] Loading {num_verts:,} Gaussians from PLY...")
    # Step sampling if huge for ultra fast viewport
    step = 1 if num_verts < 1500000 else 2
    raw_bytes = f.read(num_verts * 17 * 4)
    
    data = np.frombuffer(raw_bytes, dtype=np.float32).reshape((num_verts, 17))
    
    # Filter floaters
    pos = data[::step, 0:3]
    # Invert Y for Blender coordinate system (Z-Up)
    # COLMAP (X-Right, Y-Down, Z-Forward) -> Blender (X-Right, Y-Forward, Z-Up)
    bx = pos[:, 0]
    by = pos[:, 2]
    bz = -pos[:, 1]
    
    center = np.mean(np.column_stack([bx, by, bz]), axis=0)
    dist_sq = (bx - center[0])**2 + (by - center[1])**2 + (bz - center[2])**2
    valid_mask = dist_sq < 160.0
    
    bx = bx[valid_mask]
    by = by[valid_mask]
    bz = bz[valid_mask]
    
    f_dc = data[::step, 6:9][valid_mask]
    # Convert SH DC to sRGB: clamp(f_dc * 0.28209 + 0.5, 0, 1)
    rgb = np.clip(f_dc * 0.28209479177387814 + 0.5, 0.0, 1.0)

# Create Blender Mesh
mesh = bpy.data.meshes.new("3DGS_Gaussian_PointCloud")
obj = bpy.data.objects.new("3DGS_Model", mesh)
bpy.context.collection.objects.link(obj)

mesh_verts = [ (bx[i], by[i], bz[i]) for i in range(len(bx)) ]
mesh.from_pydata(mesh_verts, [], [])
mesh.update()

# Add Vertex Colors
col_attr = mesh.color_attributes.new(name="Col", type='FLOAT_COLOR', domain='POINT')
for i in range(len(rgb)):
    col_attr.data[i].color = (rgb[i, 0], rgb[i, 1], rgb[i, 2], 1.0)

# Material for Point Splats
mat = bpy.data.materials.new(name="3DGS_Splat_Material")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

node_attr = nodes.new(type='ShaderNodeAttribute')
node_attr.attribute_name = "Col"
node_attr.location = (-300, 0)

node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
node_bsdf.location = (0, 0)
node_bsdf.inputs['Roughness'].default_value = 0.8

node_out = nodes.new(type='ShaderNodeOutputMaterial')
node_out.location = (300, 0)

links.new(node_attr.outputs['Color'], node_bsdf.inputs['Base Color'])
links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])

obj.data.materials.append(mat)

# Add Camera & Sun Lamp
cam_data = bpy.data.cameras.new("MainCamera")
cam_obj = bpy.data.objects.new("MainCamera", cam_data)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = (center[0] + 4.5, center[1] - 5.5, center[2] + 3.5)
cam_obj.rotation_euler = (np.radians(65), 0, np.radians(40))
bpy.context.scene.camera = cam_obj

# Sun Light
light_data = bpy.data.lights.new(name="Sun", type='SUN')
light_data.energy = 3.5
light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
bpy.context.collection.objects.link(light_obj)
light_obj.rotation_euler = (np.radians(45), np.radians(30), np.radians(60))

# Save blend
bpy.ops.wm.save_as_mainfile(filepath=blend_out)
print(f"[OK] Blender Projesi Başarıyla Kaydedildi: {blend_out}")
