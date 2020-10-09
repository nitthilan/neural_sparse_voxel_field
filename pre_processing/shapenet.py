import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"#"egl" #opengl seems to only work with TPU
# print(os.environ['PYOPENGL_PLATFORM'])

import OpenGL.GL as gl
print(gl.glGetString(gl.GL_VERSION))
print(gl.glGetString(gl.GL_VENDOR))

import pickle
import json



# https://github.com/JonathanLehner/Colab-collection/blob/master/pyrender_example.ipynb

"""Examples of using pyrender for viewing and offscreen rendering.
"""
import pyglet
pyglet.options['shadow_window'] = False
import os
import numpy as np
import trimesh
from PIL import Image
import random
import sklearn.preprocessing


from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     OffscreenRenderer, RenderFlags


def get_object_to_name(BASE_PATH):
	obj_id_list = []
	obj_name_list = []
	name_to_id = {}
	with open('./shapenet_synset_list.txt') as f:
		for line in f:
			# print(line, line.strip().split(" "))
			obj_id, obj_name = line.strip().split(" ")
			# object_to_name[obj_id] = obj_name
			obj_id_list.append(obj_id)
			obj_name_list.append(obj_name)
			name_to_id[obj_name] = obj_id

	return obj_id_list, obj_name_list, name_to_id

def get_all_filelist(BASE_PATH, obj_id_list):
	name_to_filelist = {}
	# for obj_id in obj_id_list:
	# 	name_to_filelist[obj_id] = []

	for obj_id in obj_id_list:
		# print(os.path.join(BASE_PATH, obj_id))
		filelist = next(os.walk(os.path.join(BASE_PATH, obj_id)))[1]
		name_to_filelist[obj_id] = filelist

	return name_to_filelist

def get_random_obj_filepath(BASE_PATH, object_id, object_filelist, index):
	if index == -1:
		object_file = object_filelist[random.randint(0, len(object_filelist)-1)]
	else:
		object_file = object_filelist[index]
	input_path = os.path.join(BASE_PATH, object_id, object_file, "model.obj")
	# output_path = object_id+"_"+object_file
	return input_path


def get_all_folder(BASE_PATH):
	# output_directory = [x[0] for x in os.walk(BASE_PATH)]
	output_directory = next(os.walk(BASE_PATH))[1]
	with open(os.path.join(BASE_PATH, "taxonomy.json")) as f:
		load_obj_type = json.load(f)

	folder_to_name = {}
	for obj in load_obj_type:
		object_name = obj['name'].split(",")[0]
		# print(object_name)
		folder_to_name[obj['synsetId']] = {'name':object_name, 'numInstances':obj['numInstances']}

	# print(output_directory)
	# print(load_obj_type)
	folder_desc = {}
	for folder in output_directory:
		# print(folder, folder_to_name[folder])
		folder_desc[folder] = folder_to_name[folder]
		folder_name = os.path.join(BASE_PATH, folder)
		folder_desc[folder]['subfolder'] = next(os.walk(os.path.join(BASE_PATH, folder)))[1]
		# print(len(folder_desc[folder]['subfolder']))

	return folder_desc


def dump_rendered_scene(input_path, output_path, cam_pose, width, height, focal):
	#==============================================================================
	# Mesh creation
	#==============================================================================

	#------------------------------------------------------------------------------
	# Creating textured meshes from trimeshes
	#------------------------------------------------------------------------------
	object_trimesh = trimesh.load(input_path)
	# https://trimsh.org/trimesh.html#trimesh.PointCloud.bounds
	print("Object extents ", object_trimesh.bounds)
	print("Input path ", input_path)

	#==============================================================================
	# Camera creation
	#==============================================================================
	cam_angle = focal
	cam = PerspectiveCamera(yfov=cam_angle)
	# cam_pose = np.array([
	#     [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
	#     [1.0, 0.0,           0.0,           0.0],
	#     [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
	#     [0.0,  0.0,           0.0,          1.0]
	# ])

	#==============================================================================
	# Scene creation
	#==============================================================================

	# scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
	scene = Scene.from_trimesh_scene(object_trimesh, 
		bg_color = np.array([0.0, 0.0, 0.0, 1.0]),
		ambient_light=np.array([1.0, 1.0, 1.0, 1.0]))
	#==============================================================================
	# Rendering offscreen from that camera
	#==============================================================================

	cam_node = scene.add(cam, pose=cam_pose)
	r = OffscreenRenderer(viewport_width=width, viewport_height=height)

	flags = RenderFlags.RGBA
	# color, depth = r.render(scene, flags=flags)
	color, depth = r.render(scene)
	r.delete()

	depth_value = depth.copy()
	img_output = color.copy()
	# depth_value[depth_value <= 0.0001] = 1.5
	check_output = np.sum(color, axis=-1)
	print(color.shape, depth_value.shape, np.min(color), np.max(color), 
		np.min(depth_value), np.max(depth_value), check_output.shape)
	print(color[check_output==0].shape)
	# for i in range(width):
	# 	for j in range(height):
	# 		if(np.sum(color[j,i,:])==0):
	# 			img_output[j,i,0] = 255 - img_output[j,i,0]
	# 			img_output[j,i,1] = 255 - img_output[j,i,1]
	# 			img_output[j,i,2] = 255 - img_output[j,i,2]

	# import matplotlib.pyplot as plt
	# plt.figure(figsize=(20,20))
	# plt.imshow(color)

	img = Image.fromarray(img_output, 'RGB')
	img.save(output_path)

	return cam_angle

def camera_position(param):
	theta = np.deg2rad(param[0])
	phi = np.deg2rad(param[1])

	camY = param[3]*np.sin(phi)
	temp = param[3]*np.cos(phi)
	camX = temp * np.cos(theta)    
	camZ = temp * np.sin(theta)        
	cam_pos = np.array([camX, camY, camZ])

	cam_pos.copy()
	return

def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm




# http://ksimek.github.io/2012/08/22/extrinsic/
# https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3]*np.sin(phi) #/0.57
    temp = param[3]*np.cos(phi) #/0.57
    camX = temp * np.cos(theta)    
    camZ = temp * np.sin(theta)        
    cam_pos = np.array([camX, camY, camZ])        

    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    # cam_mat = np.array([axisX, axisY, axisZ])
    # cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
    cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])
    cam_mat = cam_mat.transpose()

    # https://hackr.io/blog/numpy-matrix-multiplication
    disp = cam_pos 

    # print(axisX, axisY, axisZ, cam_mat)
    # print(cam_pos)
    # print(disp)
    cam_pose = np.hstack((cam_mat, np.array([disp]).T))
    cam_pose = np.vstack((cam_pose, np.array([0.0, 0.0, 0.0, 1.0])))
    # print(cam_pose, cam_mat, disp)

    return cam_mat, cam_pos, cam_pose


# print(name_to_id[object_type])
# print(name_to_filelist[name_to_id[object_type]])
	# print(obj_id, filelist)
# print()


# BASE_PATH = "/local/data/nitthilan/data/ShapeNet/ShapeNetCore.v1/"
BASE_PATH = "/nitthilan/data/ShapeNet/ShapeNetCore.v1/"

# object_path = "02942699/147183af1ba4e97b8a94168388287ad5/model.obj"

# object_path = "02808440/5703aca738f8d92b44221bef0fa3c36b/model.obj"

BASE_PATH = "/nitthilan/data/ShapeNet/ShapeNetCore.v2/"
object_path = "02691156/14453b29e139d81f6d07c55cf995503e/models/model_normalized.obj"


output_basepath = "../data/shapenet/02691156/"

object_type = "02691156" # "airplane" # "trash_bin" # "bathtub" #
num_objects = 55
obj_num = 2

# obj_id_list, obj_name_list, name_to_id = get_object_to_name(BASE_PATH)
# name_to_filelist = get_all_filelist(BASE_PATH, obj_id_list)


elevation =  0 #25.0
distance = 2 #1.0
start_angle = 0.0
step_angle =  7.5 # 3.725 #
width, height, focal = 320, 240, np.pi/10 #640, 480, np.pi/3


folder_desc = get_all_folder(BASE_PATH)
out_sample_basepath = "../data/shapenet/sample/"

param = [0, elevation, 0, distance, 25]
cam_mat, cam_pos, cam_pose = camera_info(param)
for name in folder_desc:
	desc = folder_desc[name]
	print(name, desc['name'])
	obj_num = random.randint(0, desc['numInstances']-1)
	path_for_model = os.path.join(BASE_PATH, name, desc['subfolder'][obj_num],
		"models/model_normalized.obj")
	output_img_path = os.path.join(out_sample_basepath, "images/", 
		"render_"+name+"_"+str(obj_num)+"_"+desc['subfolder'][obj_num]+".png")
	print(path_for_model, output_img_path)
	cam_angle = dump_rendered_scene(path_for_model, output_img_path, cam_pose,
		width, height, focal)

cam_params_store = {"width":width, "height":height, "focal":focal}
for out_idx, angle  in enumerate(np.arange(0, 360, step_angle)):
	param = [angle, elevation, 0, distance, 25]
	cam_mat, cam_pos, cam_pose = camera_info(param)
	# path_for_model = get_random_obj_filepath(BASE_PATH, name_to_id[object_type], 
	# 	name_to_filelist[name_to_id[object_type]], obj_num)
	path_for_model = os.path.join(BASE_PATH, object_path)
	output_img_path = os.path.join(output_basepath, "images/", 
		"render_"+object_type+"_"+str(obj_num)+"_"+str(out_idx)+".png")
	cam_angle = dump_rendered_scene(path_for_model, output_img_path, cam_pose,
		width, height, focal)
	cam_params_store[out_idx] = cam_pose


# json = json.dumps(dict)
a_file = open(os.path.join(output_basepath, "render_"+object_type+"_"+str(obj_num)+".pkl"), "wb")
pickle.dump(cam_params_store, a_file)
a_file.close()


# for i in range(num_objects):
# 	obj_num = i
# 	path_for_model = get_random_obj_filepath(BASE_PATH, name_to_id[object_type], 
# 		name_to_filelist[name_to_id[object_type]], obj_num)
# 	output_img_path = os.path.join("../../data/", "render_"+object_type+"_"+str(obj_num)+".png")
# 	print(path_for_model, output_img_path)

# 	dump_rendered_scene(path_for_model, output_img_path)


# object_mesh = Mesh.from_trimesh(object_trimesh)


#==============================================================================
# Light creation
#==============================================================================

# direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
# spot_l = SpotLight(color=np.ones(3), intensity=10.0,
#                    innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
# point_l = PointLight(color=np.ones(3), intensity=10.0)



# direc_l_node = scene.add(direc_l, pose=cam_pose)
# spot_l_node = scene.add(spot_l, pose=cam_pose)
# point_l_node = scene.add(point_l, pose=cam_pose)


# scene.add_node(object_mesh)


#==============================================================================
# Using the viewer with a default camera
#==============================================================================

# needs a screen
# v = Viewer(scene, shadows=True)

#==============================================================================
# Using the viewer with a pre-specified camera
#==============================================================================

# needs a screen
# cam_node = scene.add(cam, pose=cam_pose)
# v = Viewer(scene, central_node=drill_node)


# plt.show()



# # Fuze trimesh
# fuze_trimesh = trimesh.load('./models/fuze.obj')
# fuze_mesh = Mesh.from_trimesh(fuze_trimesh)

# # Drill trimesh
# drill_trimesh = trimesh.load('./models/drill.obj')
# drill_mesh = Mesh.from_trimesh(drill_trimesh)
# drill_pose = np.eye(4)
# drill_pose[0,3] = 0.1
# drill_pose[2,3] = -np.min(drill_trimesh.vertices[:,2])

# # Wood trimesh
# wood_trimesh = trimesh.load('./models/wood.obj')
# wood_mesh = Mesh.from_trimesh(wood_trimesh)

# # Water bottle trimesh
# bottle_gltf = trimesh.load('./models/WaterBottle.glb')
# bottle_trimesh = bottle_gltf.geometry[list(bottle_gltf.geometry.keys())[0]]
# bottle_mesh = Mesh.from_trimesh(bottle_trimesh)
# bottle_pose = np.array([
#     [1.0, 0.0,  0.0, 0.1],
#     [0.0, 0.0, -1.0, -0.16],
#     [0.0, 1.0,  0.0, 0.13],
#     [0.0, 0.0,  0.0, 1.0],
# ])

# #------------------------------------------------------------------------------
# # Creating meshes with per-vertex colors
# #------------------------------------------------------------------------------
# boxv_trimesh = trimesh.creation.box(extents=0.1*np.ones(3))
# boxv_vertex_colors = np.random.uniform(size=(boxv_trimesh.vertices.shape))
# boxv_trimesh.visual.vertex_colors = boxv_vertex_colors
# boxv_mesh = Mesh.from_trimesh(boxv_trimesh, smooth=False)

# #------------------------------------------------------------------------------
# # Creating meshes with per-face colors
# #------------------------------------------------------------------------------
# boxf_trimesh = trimesh.creation.box(extents=0.1*np.ones(3))
# boxf_face_colors = np.random.uniform(size=boxf_trimesh.faces.shape)
# boxf_trimesh.visual.face_colors = boxf_face_colors
# boxf_mesh = Mesh.from_trimesh(boxf_trimesh, smooth=False)

# #------------------------------------------------------------------------------
# # Creating meshes from point clouds
# #------------------------------------------------------------------------------
# points = trimesh.creation.icosphere(radius=0.05).vertices
# point_colors = np.random.uniform(size=points.shape)
# points_mesh = Mesh.from_points(points, colors=point_colors)


# #==============================================================================
# # Adding objects to the scene
# #==============================================================================

# #------------------------------------------------------------------------------
# # By manually creating nodes
# #------------------------------------------------------------------------------
# fuze_node = Node(mesh=fuze_mesh, translation=np.array([0.1, 0.15, -np.min(fuze_trimesh.vertices[:,2])]))
# scene.add_node(fuze_node)
# boxv_node = Node(mesh=boxv_mesh, translation=np.array([-0.1, 0.10, 0.05]))
# scene.add_node(boxv_node)
# boxf_node = Node(mesh=boxf_mesh, translation=np.array([-0.1, -0.10, 0.05]))
# scene.add_node(boxf_node)

# #------------------------------------------------------------------------------
# # By using the add() utility function
# #------------------------------------------------------------------------------
# drill_node = scene.add(drill_mesh, pose=drill_pose)
# bottle_node = scene.add(bottle_mesh, pose=bottle_pose)
# wood_node = scene.add(wood_mesh)


