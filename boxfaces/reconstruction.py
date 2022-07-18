import os
import torch
from tqdm import tqdm
from prepare import prepare_model, prepare_data_loader
from utils.eval_utils import make_sure_dir, save_img
import argparse


GPU = 0
torch.cuda.set_device(GPU)
device = torch.device(GPU)
args = argparse.ArgumentParser(description='PyTorch Template')
args.add_argument('-r', '--resume', default='checkpoints/celeba_hq_256.pth', type=str)
args.add_argument('--data_path', default='samples/targets', type=str)
args.add_argument('--bbox_path', default='data/bbox.pkl', type=str)
args.add_argument('--img_dir', default='samples/targets', type=str)
args = args.parse_args()

netE, netG = prepare_model(args, device)
data_loader = prepare_data_loader(args.data_path, args.img_dir, args.bbox_path)

out_dir = os.path.join('output/', 'recon')
real_dir = os.path.join('output/', 'reals')
make_sure_dir(out_dir)
make_sure_dir(real_dir)
# #############################################
# # reconstruction
# #############################################

for batch_idx, batch in enumerate(tqdm(data_loader)):
    real_img, bbox, img_id = batch
    img_id = img_id[0].split('.')[0]
    real_img = real_img.cuda()

    out, nodes = netE.reconstruct(real_img, bbox)
    recon = netG(out, nodes, bbox)

    save_img(recon, out_dir + '/%s.png' % img_id)
    save_img(real_img,  real_dir + '/%s.png' % img_id)
