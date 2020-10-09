

# Works only with blender 2.79 version
# https://download.blender.org/release/Blender2.79/blender-2.79-linux-glibc219-x86_64.tar.bz2

# blender --background --python render_blender.py -- --output_folder /tmp path_to_model.obj
# ./blender-2.79-linux-glibc219-x86_64/blender --background --python render_blender.py -- --output_folder /tmp /nitthilan/data/ShapeNet/ShapeNetCore.v2/02691156/14453b29e139d81f6d07c55cf995503e/models/model_normalized.obj 

import os
import json
import random

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

def execute_blender_rendering(input_obj_file, output_folder, out_tag, cam_pos):
	x, y, z = cam_pos
	exec_string = "./blender-2.79-linux-glibc219-x86_64/blender --background --python render_blender.py -- --output_folder "
	exec_string += output_folder
	exec_string += " "
	exec_string += "--views 1 "
	exec_string += "--out_tag "
	exec_string += out_tag
	exec_string += " "
	exec_string += "--cam_x "+str(x)+" --cam_y "+str(y)+" --cam_z "+str(z)+" "
	exec_string += input_obj_file
	os.system(exec_string)
	return

BASE_PATH = "/nitthilan/data/ShapeNet/ShapeNetCore.v2/"
folder_desc = get_all_folder(BASE_PATH)
out_sample_basepath = "../data/shapenet/sample/"

for name in folder_desc:
	desc = folder_desc[name]
	print(name, desc['name'])
	obj_num = random.randint(0, desc['numInstances']-1)
	path_for_model = os.path.join(BASE_PATH, name, desc['subfolder'][obj_num],
		"models/model_normalized.obj")
	output_img_path = os.path.join(out_sample_basepath, "images/") 
		# "render_"+name+"_"+str(obj_num)+"_"+desc['subfolder'][obj_num]+".png")
	print(path_for_model, output_img_path)
	print(os.path.split(path_for_model))
	out_tag = name+"_"+str(obj_num)+"_"+desc['subfolder'][obj_num]
	cam_pos = (0, 0, 1)
	execute_blender_rendering(path_for_model, output_img_path, out_tag, cam_pos)
	# cam_angle = dump_rendered_scene(path_for_model, output_img_path, cam_pose,
	# 	width, height, focal)
	# exit()
