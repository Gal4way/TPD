from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image, ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import bezier
from numpy import asarray

import random
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid
from shutil import copyfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import os

label_to_index={
    "background": 0,
    "hat": 1,
    "hair": 2,
    "glove": 3,
    "sunglasses": 4,
    "upperclothes": 5,
    "dress": 6,
    "coat": 7,
    "socks": 8,
    "pants": 9,
    "jumpsuits": 10,
    "scarf": 11,
    "skirt": 12,
    "face": 13,
    "leftArm": 14,
    "rightArm": 15,
    "leftLeg": 16,
    "rightLeg": 17,
    "leftShoe": 18,
    "rightShoe": 19
}

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    return torchvision.transforms.Compose(transform_list)


class try_on_dataset_VITONHD(data.Dataset):
    def __init__(self, state, order: str = 'paired', pairs_file: str = None, **args):
        self.state = state
        self.args = args
        self.kernel = np.ones((1, 1), np.uint8)
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.boundingbox_as_inpainting_mask_rate = 0.4

        self.order = order
        self.pairs_file = pairs_file



        self.source_dir = []
        self.segment_map_dir = []
        self.ref_dir = []
        self.pose_dir = []
        self.densepose_dir = []


        dataroot = os.path.join(args["dataset_dir"], self.state)

        if self.order == 'paired':
            filename = os.path.join(args["dataset_dir"], f"VITONHD_{self.state}_paired.txt")
        else:
            filename = os.path.join(args["dataset_dir"], f"VITONHD_{self.state}_unpaired.txt")
        
        if pairs_file:
            filename = pairs_file


        with open(filename) as f:
            pairs = f.readlines()
            for pair in pairs:
                im_name, c_name = pair.strip().split()
                self.source_dir.append(os.path.join(dataroot, "image", im_name))
                self.segment_map_dir.append(os.path.join(dataroot, "image-parse-v3", im_name.replace('.jpg', '.png')))
                self.ref_dir.append(os.path.join(dataroot, "cloth", c_name))
                self.pose_dir.append(os.path.join(dataroot, "openpose_img", im_name.replace('.jpg', '_rendered.png')))
                self.densepose_dir.append(os.path.join(dataroot, "image-densepose", im_name))


        self.length = len(self.source_dir)

    def __getitem__(self, index):
        source_path = self.source_dir[index]
        segment_map_path = self.segment_map_dir[index]
        ref_path = self.ref_dir[index]
        pose_path = self.pose_dir[index]
        densepose_path = self.densepose_dir[index]

        source_img = Image.open(source_path).convert("RGB")
        source_img = source_img.resize((384,512), Image.BILINEAR)
        image_tensor = get_tensor()(source_img)


        segment_map = Image.open(segment_map_path)
        segment_map = segment_map.resize((384,512), Image.NEAREST)
        parse_array = np.array(segment_map)


        garment_mask = (parse_array == 5).astype(np.float32) + \
                        (parse_array == 7).astype(np.float32)

        garment_mask_with_arms = (parse_array == 5).astype(np.float32) + \
                        (parse_array == 7).astype(np.float32) + \
                    (parse_array == 14).astype(np.float32) + \
                    (parse_array == 15).astype(np.float32)


        epsilon_randomness = random.uniform(0.001, 0.005)
        randomness_range = random.choice([ 80, 90, 100])
        kernel_size = random.choice([ 80, 100, 130, 150])


        # predict mask GT, inpainting mask to be dilated 
        garment_mask = 1 - garment_mask.astype(np.float32)
        garment_mask[garment_mask < 0.5] = 0
        garment_mask[garment_mask >= 0.5] = 1
        garment_mask_resized = cv2.resize(garment_mask, (384,512), interpolation=cv2.INTER_NEAREST)

    
        contours, _ = cv2.findContours(((1 - garment_mask_resized) * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            max_contour = max(contours, key = cv2.contourArea)
            epsilon = epsilon_randomness * cv2.arcLength(max_contour, closed=True)  
            approx_contour = cv2.approxPolyDP(max_contour, epsilon, closed=True)
            randomness = np.random.randint(-randomness_range, randomness_range, approx_contour.shape)
            approx_contour = approx_contour + randomness

            zero_mask = np.zeros((512, 384))
            contours = [approx_contour]

            cv2.drawContours(zero_mask, contours, -1, (255), thickness=cv2.FILLED)

            kernel = np.ones((kernel_size,kernel_size),np.uint8)
            garment_mask_inpainting = cv2.morphologyEx(zero_mask, cv2.MORPH_CLOSE, kernel)
            garment_mask_inpainting = garment_mask_inpainting.astype(np.float32) / 255.0
            garment_mask_inpainting[garment_mask_inpainting < 0.5] = 0
            garment_mask_inpainting[garment_mask_inpainting >= 0.5] = 1
            garment_mask_inpainting = garment_mask_resized * (1 - garment_mask_inpainting)
        else:
            garment_mask_inpainting = np.zeros((512, 384))

        garment_mask_GT = cv2.erode(garment_mask_resized, self.kernel_dilate, iterations=3)[None]
        garment_mask_inpainting = cv2.erode(garment_mask_inpainting, self.kernel_dilate, iterations=5)[None]

        garment_mask_GT_tensor = torch.from_numpy(garment_mask_GT)
        garment_mask_inpainting_tensor = torch.from_numpy(garment_mask_inpainting)


        # generate inpainting boundingbox, inpainting mask to be dilated, 
        garment_mask_with_arms = 1 - garment_mask_with_arms.astype(np.float32)
        garment_mask_with_arms[garment_mask_with_arms < 0.5] = 0
        garment_mask_with_arms[garment_mask_with_arms >= 0.5] = 1
        garment_mask_with_arms_resized = cv2.resize(garment_mask_with_arms, (384,512), interpolation=cv2.INTER_NEAREST)

        garment_mask_with_arms_boundingbox = cv2.erode(garment_mask_with_arms_resized, self.kernel_dilate, iterations=5)[None]


        # boundingbox
        _, y, x = np.where(garment_mask_with_arms_boundingbox == 0)
        if x.size > 0 and y.size > 0:
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            boundingbox = np.ones_like(garment_mask_with_arms_boundingbox)
            boundingbox[:, y_min:y_max, x_min:x_max] = 0
        else:
            boundingbox = np.zeros_like(garment_mask_with_arms_boundingbox)

        boundingbox_tensor = torch.from_numpy(boundingbox)


        # limit in the boundingbox
        garment_mask_inpainting_tensor = torch.where((garment_mask_inpainting_tensor==0) & (boundingbox_tensor==0), torch.zeros_like(garment_mask_inpainting_tensor), torch.ones_like(garment_mask_inpainting_tensor))


        # select inpainting mask
        if self.state != "test":
            mask_or_boundingbox = random.random()
            if mask_or_boundingbox < 1 - self.boundingbox_as_inpainting_mask_rate:
                inpainting_mask_tensor = garment_mask_inpainting_tensor
            else:
                inpainting_mask_tensor = boundingbox_tensor
        else:
            inpainting_mask_tensor = boundingbox_tensor


        ref_img_combine = Image.open(ref_path).convert("RGB")
        ref_img_combine = ref_img_combine.resize((384,512), Image.BILINEAR)
        ref_img_combine_tensor = get_tensor()(ref_img_combine)

        pose_img = Image.open(pose_path).convert("RGB")
        pose_img = pose_img.resize((384,512), Image.BILINEAR)
        poseimage_tensor = get_tensor()(pose_img)

        densepose_img = Image.open(densepose_path).convert("RGB")
        densepose_img = densepose_img.resize((384,512), Image.BILINEAR)
        denseposeimage_tensor = get_tensor()(densepose_img)

        inpaint_image = image_tensor * inpainting_mask_tensor

        ref_tensors = [ref_img_combine_tensor]

        # 768 * 512 GT_image
        GT_image_combined = torch.cat((image_tensor, ref_img_combine_tensor), dim=2)

        # 768 * 512 GT_mask
        GT_mask_combined = torch.cat((garment_mask_GT_tensor, torch.ones((1, 512, 384), dtype=torch.float32)), dim=2)

        # 768 * 512 inpaint_image
        inpaint_image_combined = torch.cat((inpaint_image, ref_img_combine_tensor), dim=2)

        # 768 * 512 inpainting mask
        inpainting_mask_combined = torch.cat((inpainting_mask_tensor, torch.ones((1, 512, 384), dtype=torch.float32)), dim=2)

        # 768 * 512 posemap
        pose_combined = torch.cat((poseimage_tensor, ref_img_combine_tensor), dim=2)

        # 768 * 512 densepose
        densepose_combined = torch.cat((denseposeimage_tensor, ref_img_combine_tensor), dim=2)
        
        # image_name
        image_name = os.path.split(source_path)[-1]



        return {
            "image_name": image_name,
            "GT_image" : GT_image_combined,
            "GT_mask" : GT_mask_combined,
            "inpaint_image": inpaint_image_combined,
            "inpaint_mask": inpainting_mask_combined,
            "posemap": pose_combined,
            "densepose": densepose_combined,
            "ref_list": ref_tensors,
        }

    def __len__(self):
        return self.length