import os
from torch.utils.data import Dataset
from PIL import Image


class CelebAHQDataset(Dataset):
    def __init__(self, real_path=None, fake_path=None, transform=None):
        super(CelebAHQDataset, self).__init__()

        self.image_ids = os.listdir(real_path)
        self.transform = transform
        self.real = real_path
        self.fake = fake_path

    def __len__(self):
        num = len(self.image_ids)
        return num

    def __getitem__(self, index):
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
