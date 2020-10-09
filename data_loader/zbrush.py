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


class Zbrush(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(320, 240)):
        self.root_dir = root_dir
        self.split = split
        # assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.scale_img = 0.5
        self.white_back = True

        self.thresh = 0.0
        self.is_data_balanced = True
        if(self.is_data_balanced):
            self.thresh = -1.0

        self.define_transforms()

        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.root_dir, "meta_info.pkl"), "rb") as f:
            self.meta = pickle.load(f)

        w, h, num_images = int(self.meta["width"]*self.scale_img), int(self.meta["height"]*self.scale_img), self.meta["num_images"]
        self.img_wh = int(w), int(h)
        print("Image dimensions ", self.img_wh)
        self.num_images = num_images
        self.focal = 0.5*h/np.tan(0.5*self.meta["focal"]) # original focal length
                                                                     # when W=800
        # bounds, common for all scenes
        # self.near = 0.7 #2.0
        # self.far = 3.0 #6.0
        self.near = 0.5 #2.0
        self.far = 2.0 #6.0
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

        # print("Total number of frames ", len(self.meta))
        for frame in range(num_images):#self.meta['frames']:
            pose = np.array(self.meta[frame])[:3,:4]
            # print("The pose ", frame, pose)
            self.poses += [pose]
            c2w = torch.FloatTensor(pose)

            image_path = os.path.join(self.root_dir, "images/", 
                "cropped_image"+"{:02d}".format(frame+1)+".jpg")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 4) RGBA
            # img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            # self.all_rgbs += [img]

            check_output = torch.sum(img, axis=1)
            # print("Sum of all rgbs ", torch.sum(check_output != 0), check_output.shape)

            thresh = self.thresh #3*48/255.0
            self.all_rgbs += [img[check_output > thresh]]
            self.all_rgbs_1 += [img[check_output <= thresh]]

            # print(self.directions.shape, c2w.shape)
            rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

            ray_array = torch.cat([rays_o, rays_d, 
                                self.near*torch.ones_like(rays_o[:, :1]),
                                self.far*torch.ones_like(rays_o[:, :1])],
                                1) # (h*w, 8)
            # print(bounding_ray)
            self.all_rays += [ray_array[check_output > thresh]]
            self.all_rays_1 += [ray_array[check_output <= thresh]]

        self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_rays_1 = torch.cat(self.all_rays_1, 0) # (len(self.meta['frames])*h*w, 3)
        self.all_rgbs_1 = torch.cat(self.all_rgbs_1, 0) # (len(self.meta['frames])*h*w, 3)

        self.step_size = int(len(self.all_rgbs_1)/len(self.all_rgbs))
        print("Total number of rays ", len(self.all_rgbs), len(self.all_rgbs_1), self.step_size)
        # num_div_per_dir = 10
        # find_min_box(self.min_bound, self.max_bound, self.all_rays, num_div_per_dir)

        # print("Total length ", len(self.all_rays), len(self.all_rays_1))

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            if(self.is_data_balanced):
                return len(self.all_rays)
            else:
                return 2*len(self.all_rays)

        if self.split == 'val':
            return 4 # only validate 8 images (to support <=8 gpus)
        return self.num_images # len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            if(self.is_data_balanced):
                sample = {'rays': self.all_rays[idx],
                          'rgbs': self.all_rgbs[idx]}
            else:
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
                "cropped_image"+"{:02d}".format(frame+1)+".jpg")
            c2w = torch.FloatTensor(self.meta[frame])[:3, :4]

            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
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