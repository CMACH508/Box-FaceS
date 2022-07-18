import os
import cv2
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.data_utils import make_dataset
from model.networks import Encoder, Generator


def prepare_model(config, device):
    if not os.path.isfile(config.resume):
        print('ERROR: Checkpoint file "%s" not found' % config.resume)
        return
    map_location = 'cpu' if device == torch.device('cpu') else None
    checkpoint = torch.load(config.resume, map_location=map_location)
    config = checkpoint['config']
    #########################################
    # load models
    #########################################
    netE = Encoder(**config['encoder']['args'])
    netG = Generator(**config['generator']['args'])
    netE.load_state_dict(checkpoint['e_ema'])
    netG.load_state_dict(checkpoint['g_ema'])
    netE.eval()
    netG.eval()
    netE.to(device)
    netG.to(device)
    return netE, netG


def prepare_data_loader(data_path, img_dir, bbox_path):
    #########################################
    # build data loader
    #########################################
    init_kwargs = {
        "data_path": data_path,
        "image_dir": img_dir,
        "box_path": bbox_path,
        "resolution": 256,
    }
    val_dataset = CelebAHQDataset(**init_kwargs)
    init_kwargs = {
        'dataset': val_dataset,
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 1
    }
    data_loader = DataLoader(**init_kwargs)
    return data_loader


#########################################
# dataset for evaluations
#########################################
class CelebAHQDataset(Dataset):
    def __init__(self, data_path=None, box_path=None,
                 image_dir=None, resolution=128):
        super(CelebAHQDataset, self).__init__()
        self.resolution = resolution
        self.image_ids = make_dataset(data_path)
        with open(box_path, 'rb') as handle:
            self.bboxes = pickle.load(handle)
        self.image_dir = image_dir

        # define image transform
        transform = list()
        transform.append(A.Resize(height=resolution, width=resolution))
        transform.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform.append(ToTensorV2())
        self.transform = A.Compose(transform)

    def __len__(self):
        num = len(self.image_ids)
        return num

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_ids[index])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, [512, 512])
        bboxes = np.array(self.bboxes[self.image_ids[index]])
        image = self.transform(image=image)['image']
        boxes = torch.from_numpy(np.array(bboxes))
        return image, boxes, self.image_ids[index]
