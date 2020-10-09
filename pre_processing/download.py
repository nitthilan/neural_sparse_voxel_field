
import sys
import requests
import os
from PIL import Image 
import numpy as np
import pickle




print(sys.argv)

# http://www.pixologic01.com/turntable/turntables3/j_cody_lucido/ztg/1/image01.jpg


def download_images(images_url, output_image_path, num_images):
	split_url = images_url.split("/")
	folder_name = split_url[-3]+"_"+split_url[-1]
	output_basefolder = os.path.join(output_image_path, folder_name, "images")
	print(output_basefolder, split_url)
	os.makedirs(output_basefolder, exist_ok = True)

	for i in range(num_images):
		image_path = os.path.join(images_url, "image"+"{:02d}".format(i+1)+".jpg")
		output_path = os.path.join(output_basefolder, "cropped_image"+"{:02d}".format(i+1)+".jpg")
		print(image_path, output_path)

		response = requests.get(image_path)

		file = open(output_path, "wb")
		file.write(response.content)
		file.close()
	return

# python download.py http://www.pixologic01.com/turntable/turntables3/j_cody_lucido/ztg/1 ../data/pixologic/ 30
# images_url = sys.argv[1]
# output_image_path = sys.argv[2]
# num_images = int(sys.argv[3])
# download_images(images_url, output_image_path, num_images)
# exit()

basefolder = "./data/pixologic/pornsin_sooksanguan_1/images/"
def analyse_images(basefolder, num_images):
	for i in range(num_images):
		image_path = os.path.join(basefolder, "cropped_image"+"{:02d}".format(i+1)+".jpg")

		image = np.array(Image.open(image_path))
		print(image.shape, image.dtype, image.ndim)
		pixel_mean = np.mean(image, axis=-1)
		print(pixel_mean.shape)
		vert_hist = np.mean(pixel_mean, axis=0)
		print(vert_hist.shape)
		# print(vert_hist[:100], vert_hist[-100:])
		print(np.nonzero(vert_hist > 20))
	return
# analyse_images(basefolder, 30)
# exit()

def crop_images(basefolder, num_images):
	for i in range(num_images):
		image_path = os.path.join(basefolder, "cropped_image"+"{:02d}".format(i+1)+".jpg")
		cropped_image_path = os.path.join(basefolder, "cropped_image"+"{:02d}".format(i+1)+".jpg")
		image = Image.open(image_path)
		image_cropped = image.crop((240, 0, 750-240, 562))
		image_cropped.save(cropped_image_path, "jpeg")
		print(cropped_image_path)
	return

# crop_images(basefolder, 30)
# exit()

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

def generate_camera_pose(camera_param):
	width, height, focal, elevation, distance, step_angle = camera_param
	cam_params_store = {"width":width, "height":height, "focal":focal,
		"num_images": 30}
	for out_idx, angle  in enumerate(np.arange(0, 360, step_angle)):
		param = [angle, elevation, 0, distance, 25]
		cam_mat, cam_pos, cam_pose = camera_info(param)
		cam_params_store[out_idx] = cam_pose
		# print(out_idx)

	# # json = json.dumps(dict)
	# a_file = open(os.path.join(output_basepath, "render_"+object_type+"_"+str(obj_num)+".pkl"), "wb")
	# pickle.dump(cam_params_store, a_file)
	# a_file.close()
	return cam_params_store

step_angle = 360/30.0
elevation = 0.0
distance = 2.0
width = 270 #720 #385
height = 562 #560
focal = 50*np.pi/180
cam_params_store = generate_camera_pose((width, height, focal, elevation, distance, step_angle))
a_file = open(os.path.join("./data/pixologic/pornsin_sooksanguan_1/meta_info.pkl"), "wb")
pickle.dump(cam_params_store, a_file)
a_file.close()