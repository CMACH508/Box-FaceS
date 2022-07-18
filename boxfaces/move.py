import os
import torch
from tqdm import tqdm
from prepare import prepare_model, prepare_data_loader
from utils.eval_utils import make_sure_dir, save_img
from utils.model_utils import get_node_box
import argparse
from utils.move_parameters import command_to_box

GPU = 0
torch.cuda.set_device(GPU)
device = torch.device(GPU)
args = argparse.ArgumentParser(description='PyTorch Template')
args.add_argument('-r', '--resume', default='checkpoints/celeba_hq_256.pth', type=str)
args.add_argument('--cmd', default='eyes_up', type=str)
args.add_argument('--data_path', default='samples/targets', type=str)
args.add_argument('--bbox_path', default='data/bbox.pkl', type=str)
args.add_argument('--img_dir', default='samples/targets', type=str)
args = args.parse_args()

netE, netG = prepare_model(args, device)
data_loader = prepare_data_loader(args.data_path, args.img_dir, args.bbox_path)


def get_mask(image, index, boxes):
    mask = torch.ones_like(image)
    bbox = get_node_box(256, 512, boxes)
    for i in index:
        box = bbox[0, i]
        mask[:, :, box[0]:box[2], box[1]:box[3]] = 0
    return mask


command = args.cmd
out_dir = os.path.join('output/', 'move', command)
make_sure_dir(out_dir)

with torch.no_grad():
    for batch_idx, (image, bbox, id) in enumerate(tqdm(data_loader)):
        real_img = image.cuda()
        tem_tab = command_to_box[command]
        mask2 = get_mask(real_img, tem_tab['index'], bbox).cuda()
        mask1 = torch.ones_like(real_img)
        target_face, target_nodes = netE(real_img, mask1, mask2, bbox)
        # save_img(real_img*mask, os.path.join(out_dir, '%s.png' % id[0].split('.')[0]))

        for i, axis in enumerate(tem_tab['axis']):
            if command not in ['eyes_sparse', 'eyes_tight', 'sparse', 'wink']:
                bbox[:, tem_tab['index'], axis] += tem_tab['offset'][i]
            else:
                for j, index in enumerate(tem_tab['index']):
                    bbox[:, index, axis] += tem_tab['offset'][j][i]
        generated = netG(target_face, target_nodes, bbox, randomize_noise=False)
        save_img(generated, os.path.join(out_dir, '%s.png' % id[0].split('.')[0]))
