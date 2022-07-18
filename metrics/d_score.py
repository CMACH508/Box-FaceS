#!/usr/bin/python
#
# Copyright 2020 Helisa Dhamo, Iro Laina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from common_args import parser
from utils.eval_utils import load_d_model
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import tqdm
import pickle
import torch.nn.functional as F
import numpy as np

parser.add_argument('--src', type=str)
parser.add_argument('--edit', type=str)
parser.add_argument('--bbox',default='data/celebahq_bbox.pkl', type=str)
parser.add_argument('--index', default=0, type=int)
args = parser.parse_args()

GPU = 0
torch.cuda.set_device(GPU)
device = torch.device(GPU)

EXP_DIR = os.path.join(args.out_dir)

transform = list()

transform.append(A.Resize(height=256, width=256))
transform.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform.append(ToTensorV2())
transform = A.Compose(transform)


def get_node_box(fs, ori_size, boxes):
    scale = ori_size / fs
    rescaled_boxes = torch.ceil((boxes - scale / 2) / scale).int()
    rescaled_boxes = torch.clamp(rescaled_boxes, 0, fs - 1).to(torch.int32)
    return rescaled_boxes


def process_img(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image= Image.open(img_path).convert("RGB")
    image = transform(image=image)['image']
    image = image.unsqueeze(0).cuda()
    return image


def main():
    names = ['nose', 'mouth', 'l_brow', 'r_brow', 'l_eye', 'r_eye']
    d_ema = load_d_model(args, device)

    src_dir = args.src
    edited = args.edit

    score1 = 0
    score2 = 0

    with open(args.bbox, 'rb') as handle:
        bboxes = pickle.load(handle)

    mean_mse = 0
    img_lists = list(os.listdir(src_dir))

    with torch.no_grad():
        for batch_idx, filename in enumerate(tqdm.tqdm(img_lists)):
            # print(os.path.join(src_dir, filename))
            img1 = process_img(os.path.join(src_dir, filename))
            img2 = process_img(os.path.join(edited, names[args.index], filename))

            src_box = np.array(bboxes[filename.split('.')[0] + '.jpg'])
            src_box = torch.from_numpy(src_box)
            src_box = get_node_box(256, 512, src_box)

            mask = torch.ones_like(img1)

            i = args.index
            mask[:, :, src_box[i][0]:src_box[i][2], src_box[i][1]:src_box[i][3]] = 0

            mean_mse += F.mse_loss(img1 * mask, img2 * mask).item()
            score1 += d_ema(img1).item()
            score2 += d_ema(img2).item()

        print(mean_mse / len(img_lists))
        # print(score1 / len(img_lists))
        # print(score2 / len(img_lists))
        score_drop = score2 / len(img_lists) - score1 / len(img_lists)
        print(score_drop)

    with open('mse_score.txt', 'a') as f:
        f.write('%f' % (mean_mse / len(img_lists)))
        f.write('\n')

    with open('d_score_drop.txt', 'a') as f:
        f.write('%f' % score_drop)
        f.write('\n')


if __name__ == '__main__':
    main()
