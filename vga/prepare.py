from torch.utils.data import DataLoader
from model.networks import Encoder, Generator
import os
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import albumentations as A
import pickle
from PIL import Image
from utils.data_utils import make_dataset_txt


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
        "bbox_path": bbox_path,
        "resolution": 256,
    }
    val_dataset = VisualGenomeAnimals(**init_kwargs)
    init_kwargs = {
        'dataset': val_dataset,
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 1,
        'collate_fn': collate_fn_vg
    }
    data_loader = DataLoader(**init_kwargs)
    return data_loader


#########################################
# dataset for evaluations
#########################################

REF_OBJECTS = [129, 116, 137, 130, 105, 120, 122]  # animals


# class_labels = ['sheep', 'elephant', 'animal', 'zebra', 'horse', 'giraffe', 'cow']


class VisualGenomeAnimals(Dataset):
    def __init__(self, data_path, image_dir=None, bbox_path=None, resolution=256):
        super(VisualGenomeAnimals, self).__init__()

        with open(bbox_path, 'rb') as f:
            self.data = pickle.load(f)

        self.image_ids = make_dataset_txt(data_path)

        self.image_dir = image_dir

        self.random_crop_resize = A.Compose(
            [A.Resize(height=resolution, width=resolution, p=1)],
            bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']),
        )

        self.resize = A.Compose(
            [A.Resize(height=resolution, width=resolution)],
            bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']),
        )

        transform = list()
        transform.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform.append(ToTensorV2())
        self.transform = A.Compose(transform)

    def __len__(self):
        num = len(self.image_ids)
        return num

    def __getitem__(self, index):
        path = self.image_ids[index]
        img_path = os.path.join(self.image_dir, self.image_ids[index])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        objs, boxes = self.data[path]['obj'], self.data[path]['box']
        objs, boxes = torch.LongTensor(objs), torch.Tensor(boxes)
        bboxes = np.clip(np.array(boxes), 0, 1)
        target = self.resize(image=image, bboxes=bboxes, class_labels=objs)
        image = self.transform(image=target['image'])['image']
        boxes = torch.from_numpy(np.array(target['bboxes'])).float()
        return image, objs, boxes, path


def collate_fn_vg(batch):
    """
    Collate function to be used when wrapping a SceneGraphNoPairsDataset in a
    DataLoader. Returns a tuple of the following:
    - imgs: FloatTensor of shape (N, 3, H, W)
    - objs: LongTensor of shape (num_objs,) giving categories for all objects
    - boxes: FloatTensor of shape (num_objs, 4) giving boxes for all objects
    - obj_to_img: LongTensor of shape (num_objs,) mapping objects to images;
      obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - paths: List of image names (N,)
    """
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_imgs, all_mask1, all_objs, all_boxes, all_mask2 = [], [], [], [], []
    all_obj_to_img = []

    all_paths = []

    obj_offset = 0

    for i, (img, objs, boxes, path) in enumerate(batch):
        all_imgs.append(img[None])
        num_objs = objs.size(0)

        all_objs.append(objs)
        all_boxes.append(boxes)
        all_paths.append(path)

        all_obj_to_img.append(torch.LongTensor(num_objs).fill_(i))
        obj_offset += num_objs

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_obj_to_img = torch.cat(all_obj_to_img)

    return all_imgs, all_objs, all_boxes, \
           all_obj_to_img, all_paths
