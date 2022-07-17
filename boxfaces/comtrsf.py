import os
import torch
from tqdm import tqdm
from prepare import prepare_model
from utils.eval_utils import make_sure_dir, save_img
import argparse
from utils.eval_utils import process_img
import numpy as np
import pickle
from utils.data_utils import make_dataset_txt

GPU = 0
torch.cuda.set_device(GPU)
device = torch.device(GPU)
args = argparse.ArgumentParser(description='PyTorch Template')
args.add_argument('-c', '--config', default='configs/celebahq.json', type=str)
args.add_argument('-r', '--resume', default='checkpoints/celebahq/checkpoint-latest.pth', type=str)
args.add_argument('--index', default=[2], type=str)
args.add_argument('--bbox_path', default='data/bbox.pkl', type=str)
args = args.parse_args()

img_dir = '/home/huangwenjing/data/face_manipulation/CelebAMask-HQ/CelebA-HQ-img'
sources = make_dataset_txt('data/test.txt')
references = make_dataset_txt('data/shuffled_test.txt')

netE, netG = prepare_model(args, device)
with open(args.bbox_path, 'rb') as handle:
    bboxes = pickle.load(handle)

out_dir = os.path.join('output/', 'component-transfer')
make_sure_dir(out_dir)

with torch.no_grad():
    for idx, src_filename in enumerate(tqdm(sources)):
        ref_image = process_img(os.path.join(img_dir, references[idx]))
        ref_bbox = torch.from_numpy(np.array(bboxes[references[idx]])).unsqueeze(0)

        ref_face, ref_nodes = netE.reconstruct(ref_image.cuda(), ref_bbox)

        src_image = process_img(os.path.join(img_dir, src_filename))
        src_bbox = torch.from_numpy(np.array(bboxes[src_filename])).unsqueeze(0)

        out, nodes = netE.reconstruct(src_image.cuda(), src_bbox)

        nodes[:, args.index, :] = ref_nodes[:, args.index, :]
        recon = netG(out, nodes, src_bbox)

        save_img(recon, os.path.join(out_dir, src_filename.split('.')[0] + '.png'))
