import torch.nn.functional as F
import torch
from utils.bilinear import bilinear_sample, tensor_linspace, _invperm


def crop_bbox_batch_cudnn(feats, bbox, bbox_to_feats, HH, WW=None):
    # print("here ========================================" ,feats.size())
    N, C, H, W = feats.size()
    B = bbox.size(0)
    if WW is None: WW = HH

    feats_flat, bbox_flat, all_idx = [], [], []
    for i in range(N):
        idx = (bbox_to_feats.data == i).nonzero()
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
        cur_bbox = bbox[idx]
        # print(cur_bbox.shape)

        feats_flat.append(cur_feats)
        bbox_flat.append(cur_bbox)
        all_idx.append(idx)

    feats_flat = torch.cat(feats_flat, dim=0)
    bbox_flat = torch.cat(bbox_flat, dim=0)
    crops = crop_bbox(feats_flat, bbox_flat, HH, WW, backend='cudnn')

    # If the crops were sequential (all_idx is identity permutation) then we can
    # simply return them; otherwise we need to permute crops by the inverse
    # permutation from all_idx.
    all_idx = torch.cat(all_idx, dim=0)

    eye = torch.arange(0, B).type_as(all_idx)
    if (all_idx == eye).all():
        return crops
    return crops[_invperm(all_idx)]


def crop_bbox(feats, bbox, HH, WW=None, backend='cudnn'):
    """
    Take differentiable crops of feats specified by bbox.

    Inputs:
    - feats: Tensor of shape (N, C, H, W)
    - bbox: Bounding box coordinates of shape (N, 4) in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space.
    - HH, WW: Size of the output crops.

    Returns:
    - crops: Tensor of shape (N, C, HH, WW) where crops[i] is the portion of
      feats[i] specified by bbox[i], reshaped to (HH, WW) using bilinear sampling.
    """
    N = feats.size(0)
    assert bbox.size(0) == N
    # print(bbox.shape)
    assert bbox.size(1) == 4
    if WW is None: WW = HH
    if backend == 'cudnn':
        # Change box from [0, 1] to [-1, 1] coordinate system
        bbox = 2 * bbox - 1

    x0, y0 = bbox[:, 0], bbox[:, 1]
    x1, y1 = bbox[:, 2], bbox[:, 3]

    X = tensor_linspace(x0, x1, steps=WW).view(N, 1, WW).expand(N, HH, WW)
    Y = tensor_linspace(y0, y1, steps=HH).view(N, HH, 1).expand(N, HH, WW)
    if backend == 'jj':
        return bilinear_sample(feats, X, Y)
    elif backend == 'cudnn':
        grid = torch.stack([X, Y], dim=3)
        return F.grid_sample(feats, grid)


def crop_bbox_batch(feats, bbox, bbox_to_feats, HH, WW=None, backend='cudnn'):
    """
    Inputs:
    - feats: FloatTensor of shape (N, C, H, W)
    - bbox: FloatTensor of shape (B, 4) giving bounding box coordinates
    - bbox_to_feats: LongTensor of shape (B,) mapping boxes to feature maps;
      each element is in the range [0, N) and bbox_to_feats[b] = i means that
      bbox[b] will be cropped from feats[i].
    - HH, WW: Size of the output crops

    Returns:
    - crops: FloatTensor of shape (B, C, HH, WW) where crops[i] uses bbox[i] to
      crop from feats[bbox_to_feats[i]].
    """
    if backend == 'cudnn':
        return crop_bbox_batch_cudnn(feats, bbox, bbox_to_feats, HH, WW)
    # print("here ========================================" ,feats.size())
    N, C, H, W = feats.size()
    B = bbox.size(0)
    if WW is None: WW = HH
    dtype, device = feats.dtype, feats.device
    crops = torch.zeros(B, C, HH, WW, dtype=dtype, device=device)
    for i in range(N):
        idx = (bbox_to_feats.data == i).nonzero()
        if idx.dim() == 0:
            continue
        idx = idx.view(-1)
        n = idx.size(0)
        cur_feats = feats[i].view(1, C, H, W).expand(n, C, H, W).contiguous()
        cur_bbox = bbox[idx]
        cur_crops = crop_bbox(cur_feats, cur_bbox, HH, WW)
        crops[idx] = cur_crops
    return crops
