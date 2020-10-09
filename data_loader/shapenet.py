import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *

import pickle

import pyrr.geometric_tests as gt
import time



class ShapeNet(Dataset):
    def __init__(self, root_dir, object_type='airplane', obj_num=2, split='train', img_wh=(320, 240)):
        self.root_dir = root_dir
        self.split = split
        # assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()

        self.read_meta(object_type, obj_num)
        self.white_back = True

        self.object_type = object_type
        self.obj_num = obj_num

    # https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    def get_min_max_depth(self, rays, bounding_box, grid_size):
        min_bound, max_bound = bounding_box
        min_bound = torch.FloatTensor(min_bound)
        max_bound = torch.FloatTensor(max_bound)
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        unit_d = rays_d + 0.0000001 # To avoid dived by zero
        min_bound_val = (min_bound - rays_o)/unit_d
        max_bound_val = (max_bound - rays_o)/unit_d
        min_z = torch.where(unit_d >= 0, min_bound_val, max_bound_val)
        max_z = torch.where(unit_d >= 0, max_bound_val, min_bound_val)
        min_z[min_z > 1000] = -1000
        max_z[max_z < -1000] = 1000
        near = torch.max(min_z, axis=-1)
        far = torch.min(max_z, axis=-1)
        # print(near.values.size(), rays.size(), near.values[:10])
        rays[:, 6] = near.values
        rays[:, 7] = far.values
        # print("Values ", rays[:15, :], min_bound, max_bound)

        grid_size = torch.LongTensor(grid_size)

        step_size = (max_bound - min_bound)/grid_size
        unit_step_size_t = rays_d + 0.0000001 # To avoid divde by zero
        # Step size along the ray
        step_size_t = torch.abs(step_size/unit_step_size_t) 
        # generate the t_vals intersection along each direction
        # print("Size of the values ", far.values.size())
        far_t_val = far.values.unsqueeze(-1).expand(far.values.size()[0], 3)

        # far_t_val = torch.ones(far.values.size()[0], 3)
        t_inter_array = []
        for i in range(torch.max(grid_size)):
            t_inter_val = min_z + i*step_size_t
            if((t_inter_val > far_t_val).all()):
                break
            # print(t_inter_val.size(), step_size_t[:3], min_z[:3], t_inter_val[:3])
            t_inter_array += [t_inter_val]

        t_inter_array = torch.cat(t_inter_array, -1)

        far_t_val = far.values.unsqueeze(-1).expand(far.values.size()[0], t_inter_array.size()[-1])
        near_t_val = near.values.unsqueeze(-1).expand(near.values.size()[0], t_inter_array.size()[-1])
        t_inter_array = torch.where(t_inter_array > far_t_val, far_t_val, t_inter_array)
        t_inter_array = torch.where(t_inter_array < near_t_val, near_t_val, t_inter_array)
        t_inter_array, _ = torch.sort(t_inter_array)
        # print(t_inter_array.size(), rays.size())
        # print(t_inter_array[:4])
        return rays


    def read_meta(self, object_type, obj_num):
        with open(os.path.join(self.root_dir, "render_"+object_type+"_"+str(obj_num)+".pkl"), "rb") as f:
            self.meta = pickle.load(f)
        w, h = self.meta["width"], self.meta["height"]
        self.img_wh = w, h
        self.focal = 0.5*h/np.tan(0.5*self.meta["focal"]) # original focal length
                                                                     # when W=800
        # self.focal = 0.5*h/np.tan(self.meta["focal"])

        # self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        # self.near = 0.7 #2.0
        # self.far = 3.0 #6.0
        self.near = 0.5 #2.0
        self.far = 3 #6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal) # (h, w, 3)
            
        # if self.split == 'train': # create buffer of all rays and rgb data
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_rays_1 = []
        self.all_rgbs_1 = []

        self.min_bound = [100, 100, 100]
        self.max_bound = [-100, -100, -100]

        # print("Total number of frames ", len(self.meta))
        for frame in range(96):#self.meta['frames']:
            pose = np.array(self.meta[frame])[:3,:4]
            # print("The pose ", frame, pose)
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)

            image_path = os.path.join(self.root_dir, "images/", 
                "render_"+object_type+"_"+str(obj_num)+"_"+str(frame)+".png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            # img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 4) RGBA
            # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            # self.all_rgbs += [img]

            check_output = torch.sum(img, axis=1)
            # print("Sum of all rgbs ", torch.sum(check_output != 0), check_output.shape)

            self.all_rgbs += [img[check_output != 0]]
            self.all_rgbs_1 += [img[check_output == 0]]

            # print(self.directions.shape, c2w.shape)
            rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

            ray_array = torch.cat([rays_o, rays_d, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_o[:, :1])],
                                1) # (h*w, 8)
            # print(bounding_ray)
            self.all_rays += [ray_array[check_output != 0]]
            self.all_rays_1 += [ray_array[check_output == 0]]

            def find_min_max(pt):
                for i in range(3):
                    if(self.min_bound[i] > pt[i]):
                        self.min_bound[i] = pt[i]
                    if(self.max_bound[i] < pt[i]):
                        self.max_bound[i] = pt[i]
                return
            def get_min_max_bounds(rays_o, rays_d, near, far, idx):
                min_point = rays_o[idx]+near*rays_d[idx]
                max_point = rays_o[idx]+far*rays_d[idx]
                return min_point, max_point
            for i in [0, w-1, h*w-w, h*w-1]:
                min_pt, max_pt = get_min_max_bounds(rays_o, rays_d, self.near, self.far, i)
                find_min_max(min_pt)
                find_min_max(max_pt)
            self.bounding_box = (self.min_bound, self.max_bound)
            # print(get_min_max_bounds(rays_o, rays_d, self.near,  self.far, 0))
            # print(get_min_max_bounds(rays_o, rays_d, self.near,  self.far, w-1))
            # print(get_min_max_bounds(rays_o, rays_d, self.near,  self.far, h*w-w))
            # print(get_min_max_bounds(rays_o, rays_d, self.near,  self.far, h*w-1))

        # Max and min bounds could be reduced volume by intersection of max volume and min volume
        print("Min, Max bounds ", self.min_bound, self.max_bound)


        self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_rays_1 = torch.cat(self.all_rays_1, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_rgbs_1 = torch.cat(self.all_rgbs_1, 0) # (len(self.meta['frames])*h*w, 3)

        # self.all_rays = self.get_min_max_depth(self.all_rays, self.bounding_box, (10, 10, 10))
        # self.all_rays_1 = self.get_min_max_depth(self.all_rays_1, self.bounding_box, (10, 10, 10))

        self.step_size = int(len(self.all_rgbs_1)/len(self.all_rgbs))
        print("Total number of rays ", len(self.all_rgbs), len(self.all_rgbs_1), self.step_size)
        # num_div_per_dir = 10
        # find_min_box(self.min_bound, self.max_bound, self.all_rays, num_div_per_dir)

        # print("Total length ", len(self.all_rays), len(self.all_rays_1))

    def get_bounding_box(self):
        return self.bounding_box
    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return 2*len(self.all_rays)
        if self.split == 'val':
            return 4 # only validate 8 images (to support <=8 gpus)
        return 96 # len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            out_idx = int(idx/2)
            if(idx%2 == 0):
                sample = {'rays': self.all_rays[out_idx],
                      'rgbs': self.all_rgbs[out_idx]}
            else:
                out_idx = out_idx*self.step_size
                sample = {'rays': self.all_rays_1[out_idx],
                      'rgbs': self.all_rgbs_1[out_idx]}

        else: # create data for each image separately
            frame = idx
            image_path = os.path.join(self.root_dir, "images/", 
                "render_"+self.object_type+"_"+str(self.obj_num)+"_"+str(frame)+".png")
            c2w = torch.FloatTensor(self.meta[frame])[:3, :4]

            img = Image.open(image_path)
            # img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(3, -1).permute(1, 0) # (H*W, 4) RGBA
            # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)

            valid_mask = np.ones(img.shape[0])
            img_sum = torch.sum(img, dim=1)
            # print(img.shape, valid_mask.shape, idx)

            # # print(valid_mask.shape)
            # print((img_sum < 1))
            valid_mask[img_sum < 10.0/(3*256)] = 0
            # print("Num valid ", np.sum(valid_mask), (img_sum < 10.0/(3*256)).sum())

            c2w = torch.FloatTensor(np.array(self.meta[frame])[:3,:])
            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}
            # sample = {'rays': self.all_rays[idx],
            #           'rgbs': self.all_rgbs[idx],
            #           'c2w': c2w,
            #           'valid_mask': valid_mask}

        return sample