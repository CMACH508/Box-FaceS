import os
from torch.utils.data import Dataset
from reconstruction import make_dataset_txt
from PIL import Image
from torchvision import transforms as T
import torch


class CelebAHQDataset(Dataset):
    def __init__(self, real_path=None, fake_path=None, img_list=None, transform=None):
        super(CelebAHQDataset, self).__init__()
        if img_list is not None:
            self.image_ids = make_dataset_txt(img_list)
        else:
            self.image_ids = os.listdir(real_path)
        self.transform = transform
        self.real = real_path
        self.fake = fake_path

    def __len__(self):
        num = len(self.image_ids)
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (num_objs,)
        - boxes: FloatTensor of shape (num_objs, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (num_triples, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.

        """
        # name = str(int(self.image_ids[index].split('.')[0]))
        name = self.image_ids[index].split('.')[0]
        fake_filename = os.path.join(self.fake, f"{name}.png")
        real_filename = os.path.join(self.real, f"{name}.png")

        if os.path.isfile(fake_filename) and os.path.isfile(real_filename):
            fake_img = self.transform(Image.open(fake_filename).convert("RGB"))
            real_img = self.transform(Image.open(real_filename).convert("RGB"))
            # assert real_img.shape == (3, 256, 256)
        else:
            print(f"{fake_filename} or {real_filename} doesn't exists")

        return real_img, fake_img


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
