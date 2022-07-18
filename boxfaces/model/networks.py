import math

import torch
from torch import nn
from modules.nets_builders import EqualLinear, ConvLayer, ResBlock, StyledConv, ToRGB
from utils.model_utils import get_node_feats, get_node_box
import numpy as np


class Encoder(nn.Module):
    def __init__(
            self,
            channel,
            img_size=256,
            feats_dim=[128, 128, 128, 128, 128, 128]
    ):
        super().__init__()

        self.fs = 32

        stem = [ConvLayer(3, channel, 1)]

        num_block = int(math.log(img_size // self.fs, 2))
        in_channel = channel
        for i in range(1, num_block + 1):
            ch = channel * (2 ** i)
            stem.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.stem = nn.Sequential(*stem)

        stem2 = []
        stem2.append(ResBlock(ch, ch * 2, downsample=True, padding="reflect"))
        stem2.append(ResBlock(ch * 2, ch * 4, downsample=True, padding="reflect"))
        self.stem2 = nn.Sequential(*stem2)

        self.layers = nn.Sequential(EqualLinear(6272, feats_dim[0], activation="fused_lrelu"),
                                    EqualLinear(6912, feats_dim[1], activation="fused_lrelu"),
                                    EqualLinear(4096, feats_dim[2], activation="fused_lrelu"),
                                    EqualLinear(3072, feats_dim[3], activation="fused_lrelu"),
                                    )

    def forward(self, input, mask1, mask2, bbox=None):
        out = self.stem(input * mask1)
        boxes = get_node_box(self.fs, 512, bbox)
        obj_feats = get_node_feats(out, boxes, self.layers)
        out = self.stem2(self.stem(input * mask2))
        nodes = torch.cat([obj_feats], dim=1)
        return out, nodes

    def reconstruct(self, input, bbox=None):
        out = self.stem(input)
        boxes = get_node_box(self.fs, 512, bbox)
        obj_feats = get_node_feats(out, boxes, self.layers)
        out = self.stem2(out)
        nodes = torch.cat([obj_feats], dim=1)
        return out, nodes


class Generator(nn.Module):
    def __init__(
            self,
            img_size,
            feats_dim,
            in_fs=8,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1]
    ):
        super().__init__()

        self.size = img_size

        style_dim = np.array(feats_dim).sum()

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.conv1 = StyledConv(
            self.channels[in_fs], self.channels[in_fs], 3, feats_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[in_fs], style_dim, upsample=False)

        self.log_size = int(math.log(img_size, 2))
        start = int(math.log(in_fs, 2))
        self.num_layers = (self.log_size - start) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[in_fs]

        off_set = int(math.log(in_fs, 2)) * 2 + 1

        for layer_idx in range(self.num_layers):
            res = (layer_idx + off_set) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(start, self.log_size):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    feats_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, feats_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(
            self,
            latent,
            nodes,
            bbox,
            noise=None,
            randomize_noise=True,
    ):

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        style = torch.flatten(nodes, 1)

        out = self.conv1(latent, nodes, bbox, noise=noise[0])
        skip = self.to_rgb1(out, style)

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, nodes, bbox, noise=noise1)
            out = conv2(out, nodes, bbox, noise=noise2)
            skip = to_rgb(out, style, skip)

            i += 2

        image = skip

        return image


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, downsample=True))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)

        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out
