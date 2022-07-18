from torchvision import transforms as T
import torch


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


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def celebahq_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=[2.0, 2.0, 2.0]),
        T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def celebahq_depreprocess_batch(imgs, rescale=True):
    """
        Input:
        - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

        Output:
        - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
          in the range [0, 255]
        """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = celebahq_deprocess(rescale_image=rescale)
    imgs_de = []
    for i in range(imgs.size(0)):
        netDe = deprocess_fn(imgs[i])[None]
        # netDe = netDe.mul(255).clamp(0, 255).byte()
        imgs_de.append(netDe)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de
