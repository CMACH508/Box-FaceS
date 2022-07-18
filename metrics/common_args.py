import argparse
import os

parser = argparse.ArgumentParser(
    description='Train semantic boundary with given latent codes and '
                'attribute scores.')

parser.add_argument('--ckpt', default='checkpoint/latest.pt', type=str)
parser.add_argument('--out_dir', default="output/", type=str)
parser.add_argument('--img_size', default=256, type=tuple)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--image_dir', default='/home/huangwenjing/data/face_manipulation/CelebAMask-HQ/CelebA-HQ-img/')
parser.add_argument('--test_path', default='/media/huangwenjing/disk/ICME-MM2022/data/stylegan2/test2601-latent.pkl', type=str)

