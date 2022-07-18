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
args.add_argument('--cmd', default='left', type=str)
args.add_argument('--data_path', default='data/sample.txt', type=str)
args.add_argument('--bbox_path', default='data/bbox.pkl', type=str)
args.add_argument('--img_dir', default='data/VGA-img', type=str)
args = args.parse_args()

netE, netG = prepare_model(args, device)
data_loader = prepare_data_loader(args.data_path, args.img_dir, args.bbox_path)
cmd = args.cmd
out_dir = os.path.join('output/', cmd)
make_sure_dir(out_dir + '/move')
make_sure_dir(out_dir + '/real')


def get_left_right_top_bottom(box, height, width):
    """
    - box: Tensor of size [4]
    - height: scalar, image hight
    - width: scalar, image width
    return: left, right, top, bottom in image coordinates
    """
    left = (box[0] * width).type(torch.int32)
    right = (box[2] * width).type(torch.int32)
    top = (box[1] * height).type(torch.int32)
    bottom = (box[3] * height).type(torch.int32)
    return left, right, top, bottom


def get_mask(image, index, boxes):
    left, right, top, bottom = \
        get_left_right_top_bottom(boxes[index], image.size(2), image.size(3))

    mask[:, :, top:bottom, left:right] = 0
    return mask


with torch.no_grad():
    for batch_idx, (image, objs, bbox, obj_to_name, name) in enumerate(tqdm(data_loader)):
        real_img = image.cuda()
        bbox = bbox.cuda()

        min_box = 1
        index = 0
        for i, box in enumerate(bbox):
            if (box[3] - box[1]) * (box[2] - box[0]) < min_box:
                min_box = (box[3] - box[1]) * (box[2] - box[0])
                index = i

        mask = torch.ones_like(real_img)
        mask = get_mask(mask, index, bbox).cuda()

        target_face, target_nodes, _ = netE(real_img, real_img * mask, objs, bbox, obj_to_name)

        bbox2 = bbox
        bbox = bbox2[index]
        if cmd == 'bigger':

            cw = (bbox[0] + bbox[2]) / 2
            ch = (bbox[1] + bbox[3]) / 2

            w = (bbox[2] - bbox[0]) * 1.5 / 2
            h = (bbox[3] - bbox[1]) * 1.5 / 2

            bbox2[index, 0] = cw - w
            bbox2[index, 2] = cw + w
            bbox2[index, 1] = ch - h
            bbox2[index, 3] = ch + h
        elif cmd == 'smaller':
            cw = (bbox[0] + bbox[2]) / 2
            ch = (bbox[1] + bbox[3]) / 2

            w = (bbox[2] - bbox[0]) * 0.7 / 2
            h = (bbox[3] - bbox[1]) * 0.7 / 2

            bbox2[index, 0] = cw - w
            bbox2[index, 2] = cw + w
            bbox2[index, 1] = ch - h
            bbox2[index, 3] = ch + h
        elif cmd == 'right':

            bbox2[index, 0] += 0.15
            bbox2[index, 2] += 0.15

        elif cmd == 'left':

            bbox2[index, 0] -= 0.15
            bbox2[index, 2] -= 0.15

        elif cmd == 'fat':

            cw = (bbox[0] + bbox[2]) / 2
            ch = (bbox[1] + bbox[3]) / 2

            w = (bbox[2] - bbox[0]) * 2 / 2
            h = (bbox[3] - bbox[1]) * 2 / 2

            bbox2[index, 0] = cw - w
            bbox2[index, 2] = cw + w

        elif cmd == 'thinner':
            cw = (bbox[0] + bbox[2]) / 2
            ch = (bbox[1] + bbox[3]) / 2

            w = (bbox[2] - bbox[0]) * 0.5 / 2
            h = (bbox[3] - bbox[1]) * 0.5 / 2

            bbox2[index, 0] = cw - w
            bbox2[index, 2] = cw + w

        # save_img(mask*255, os.path.join(out_dir, 'mask_%d.png' % batch_idx))
        try:
            save_img(netG(target_face, target_nodes, target_nodes, bbox2, obj_to_name),
                     os.path.join(out_dir, 'move/%s.png' % name[0].split('.')[0]))
            save_img(real_img,
                     os.path.join(out_dir + '/real', '%s.png' % name[0].split('.')[0]))
        except:
            pass
