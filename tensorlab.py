import torch

shape = (2,3,1)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


root = './data/PennFudanPed/'
# load all image files, sorting them to
# ensure that they are aligned

def printmask(idx):
    imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
    masks = list(sorted(os.listdir(os.path.join(root, "PedMasks")))) 
    print(idx)
    # 初始化原图片和掩码图片列表
    img_path = os.path.join(root, "PNGImages", imgs[idx])
    mask_path = os.path.join(root, "PedMasks", masks[idx])
    img = read_image(img_path)
    mask = read_image(mask_path)
    # instances are encoded as different colors
    # 使用unique提取不同的颜色来获取目标物体的数量(id)
    obj_ids = torch.unique(mask)
    # first id is the background, so remove it
    print(obj_ids)
    obj_ids = obj_ids[1:]
    num_objs = len(obj_ids)

    # split the color-encoded mask into a set
    # of binary masks
    masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

    print(img)
    print(mask)
    print(obj_ids[:, None, None])
    print(masks)
    print('\n')


printmask(6)