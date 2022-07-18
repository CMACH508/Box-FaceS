from utils.manipulator import linear_interpolate
import math
import os
import torch
from torch import optim
from torch.nn import functional as F
from model import Generator, Discriminator
from tqdm import tqdm
import time
import pickle
from torchvision.utils import save_image, make_grid


def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(path)

    return img_paths


def save_img(img, path):
    if type(img) is list:
        img = make_grid(torch.cat(img, dim=0), nrow=len(img), padding=0, normalize=True, range=(-1, 1))
        save_image(img, path)
    else:
        save_image(img, path, normalize=True, padding=0, range=(-1, 1))
    return


def load_model(args, device):
    g_ema = Generator(args.img_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    return g_ema


def load_d_model(args, device):
    d_ema = Discriminator(args.img_size, 2)
    d_ema.load_state_dict(torch.load(args.ckpt)["d"], strict=False)
    d_ema.eval()
    d_ema = d_ema.to(device)
    return d_ema


def save_pkl(table, data_dir, name):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    with open(data_dir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(table, f, pickle.HIGHEST_PROTOCOL)


def inter_code(latent_codes, boundary, args):
    old_codes = latent_codes.cpu().numpy()

    new_codes = linear_interpolate(old_codes, boundary,
                                   start_distance=args.start, end_distance=args.end, steps=args.steps)

    return new_codes


def make_sure_dir(out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to("cpu")
            .numpy()
    )


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def search(g_ema, latent_mean, latent_std, imgs, percept, args):
    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(args.step))
    st = time.time()

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        # batch, channel, height, width = img_gen.shape

        # if height > 256:
        #     factor = height // 256
        #
        #     img_gen = img_gen.reshape(
        #         batch, channel, height // factor, factor, width // factor, factor
        #     )
        #     img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)

        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        # if (i + 1) % 100 == 0:
        #     latent_path.append(latent_in.detach().clone())
        #
        # pbar.set_description(
        #     (
        #         f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
        #         f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
        #     )
        # )
    ed = time.time()

    return latent_in, noises, ed - st
