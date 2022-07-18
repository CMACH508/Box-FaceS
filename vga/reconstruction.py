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
args.add_argument('-r', '--resume', default='checkpoints/vga_256.pth', type=str)
args.add_argument('--data_path', default='data/sample.txt', type=str)
args.add_argument('--bbox_path', default='data/bbox.pkl', type=str)
args.add_argument('--img_dir', default='data/VGA-img', type=str)
args = args.parse_args()

netE, netG = prepare_model(args, device)
data_loader = prepare_data_loader(args.data_path, args.img_dir, args.bbox_path)
out_dir = os.path.join('output/', 'recon')
real_dir = os.path.join('output/', 'reals')
make_sure_dir(out_dir)
make_sure_dir(real_dir)
with torch.no_grad():
    for batch_idx, (image, objs, bbox, obj_to_name, path) in enumerate(tqdm(data_loader)):
        real_img = image.cuda()
        bbox = bbox.cuda()
        index = 0

        target_face, target_nodes, _ = netE(real_img, real_img, objs, bbox, obj_to_name)
        name = path[0].split('/')[-1].split('.')[0]

        save_img(real_img, os.path.join(real_dir, '%s.png' % name))
        save_img(netG(target_face, target_nodes, target_nodes, bbox, obj_to_name),
                 os.path.join(out_dir, '%s.png' % name))
