import os
import torch
from tqdm import tqdm
from prepare import prepare_model
from utils.eval_utils import make_sure_dir, save_img
import argparse
from utils.eval_utils import process_img
import numpy as np
import pickle

GPU = 0
torch.cuda.set_device(GPU)
device = torch.device(GPU)
args = argparse.ArgumentParser(description='PyTorch Template')
args.add_argument('-c', '--config', default='configs/celebahq.json', type=str)
args.add_argument('-r', '--resume', default='checkpoints/celebahq/checkpoint-latest.pth', type=str)
args.add_argument('--index', default=[2], type=list)
args.add_argument('--bbox_path', default='data/bbox.pkl', type=str)
args = args.parse_args()

targets = 'samples/targets'
references = 'samples/eyebrows'

netE, netG = prepare_model(args, device)
with open(args.bbox_path, 'rb') as handle:
    bboxes = pickle.load(handle)

out_dir = os.path.join('output/', 'replace')
make_sure_dir(out_dir)

with torch.no_grad():
    for ref_idx, ref_filename in enumerate(tqdm(os.listdir(references))):
        ref_image = process_img(os.path.join(references, ref_filename))
        ref_bbox = torch.from_numpy(np.array(bboxes[ref_filename])).unsqueeze(0)

        ref_face, ref_nodes = netE.reconstruct(ref_image.cuda(), ref_bbox)
        for src_idx, src_filename in enumerate(tqdm(os.listdir(targets))):
            src_image = process_img(os.path.join(targets, src_filename))
            src_bbox = torch.from_numpy(np.array(bboxes[src_filename])).unsqueeze(0)

            out, nodes = netE.reconstruct(src_image.cuda(), src_bbox)

            nodes[:, args.index, :] = ref_nodes[:, args.index, :]
            recon = netG(out, nodes, src_bbox)

            ref_filename = ref_filename.split('.')[0]
            src_filename = src_filename.split('.')[0]

            save_img(recon, os.path.join(out_dir, '%s_%s.png' % (ref_filename, src_filename)))
