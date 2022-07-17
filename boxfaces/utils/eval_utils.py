import os
from collections import OrderedDict
import numpy as np
from utils.data_utils import make_dataset_txt
import pickle
from torchvision.utils import save_image, make_grid
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

transform = list()
transform.append(A.Resize(height=256, width=256))
transform.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform.append(ToTensorV2())
transforms = A.Compose(transform)


def process_img(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, [512, 512])
    image = transforms(image=image)['image'].unsqueeze(0)
    return image


def make_sure_dir(out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return


def _change_key(checkpoint):
    for k, v in checkpoint.items():
        if k.split('_')[0] == 'state':
            new_state_dict = OrderedDict()
            for p, s in v.items():
                new_state_dict[p[7:]] = s
            checkpoint[k] = new_state_dict


def save_img(img, path):
    if type(img) is list:
        img = make_grid(torch.cat(img, dim=0), nrow=len(img), padding=0, normalize=True, range=(-1, 1))
        save_image(img, path)
    else:
        save_image(img, path, normalize=True, padding=0, range=(-1, 1))
    return
