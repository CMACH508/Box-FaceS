import os
import random
import torch
import pickle
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from utils.task import random_irregular_mask, random_regular_mask
from utils.data_utils import mask_image_in_bbox, make_dataset_txt
from PIL import Image
import numpy as np
import albumentations as A

OBJECTS = [129, 116, 137, 130, 105, 120, 122]


# class_labels = ['sheep', 'elephant', 'animal', 'zebra', 'horse', 'giraffe', 'cow']

class VisualGenome(Dataset):
    def __init__(self, split, data_path, bbox_path, image_dir, resolution, num_samples=-1):
        super(VisualGenome, self).__init__()

        self.split = split

        with open(os.path.join(bbox_path), 'rb') as f:
            self.data = pickle.load(f)

        self.image_ids = make_dataset_txt(data_path)
        if num_samples != -1:
            self.image_ids = self.image_ids[:num_samples]

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
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (num_objs,)
        - boxes: FloatTensor of shape (num_objs, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - mask1: FloatTensor of shape (C, H, W)
        - mask2: FloatTensor of shape (C, H, W)
        """
        path = self.image_ids[index]

        img_path = os.path.join(self.image_dir, self.image_ids[index])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        objs, boxes = self.data[path]['obj'], self.data[path]['box']
        objs, boxes = torch.LongTensor(objs), torch.Tensor(boxes)

        bboxes = np.clip(np.array(boxes), 0, 1)

        if self.split == 'train':
            target = self.random_crop_resize(image=image, bboxes=bboxes, class_labels=objs)
            image = self.transform(image=target['image'])['image']
            mask1 = random_irregular_mask(image, width=16)
            mask2 = random_irregular_mask(image, width=16)
            if random.random() > 0.1:
                mask2 = mask2 * random_regular_mask(image)
        else:
            target = self.resize(image=image, bboxes=bboxes, class_labels=objs)
            image = self.transform(image=target['image'])['image']
            mask1, mask2 = torch.ones_like(image), torch.ones_like(image)
        boxes = torch.from_numpy(np.array(target['bboxes'])).float()

        box_keep, mask3, box_index = self.get_masked_imgs(image, objs, boxes)
        if random.random() > 0.2 and self.split == 'train':
            mask2 = mask2 * mask3
        objs = objs[box_index]
        boxes = boxes[box_index]

        return image, objs, boxes, mask1, mask2

    def get_masked_imgs(self, in_img, objs, boxes_gt):
        num_objs = len(objs)
        mask = torch.ones_like(in_img)
        box_keep = torch.ones([num_objs, 1], dtype=boxes_gt.dtype, device=boxes_gt.device)
        box_index = self.choose_box(objs)
        box_keep[box_index] = 0
        mask = mask_image_in_bbox(mask, boxes_gt[box_index])
        return box_keep, mask, box_index

    def choose_box(self, objs):
        objs_index = list(range(0, len(objs)))
        box_index = []
        for index in objs_index:
            if objs[index] in OBJECTS:
                box_index.append(index)
        random.shuffle(box_index)

        return [box_index[0]]


def collate_fn_vg(batch):
    """
    Collate function to be used when wrapping a SceneGraphNoPairsDataset in a
    DataLoader. Returns a tuple of the following:
    - imgs: FloatTensor of shape (N, 3, H, W)
    - objs: LongTensor of shape (num_objs,) giving categories for all objects
    - boxes: FloatTensor of shape (num_objs, 4) giving boxes for all objects
    - obj_to_img: LongTensor of shape (num_objs,) mapping objects to images;
      obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - mask1: FloatTensor of shape (N, 3, H, W)
    - mask2: FloatTensor of shape (N, 3, H, W)
    """
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_imgs, all_mask1, all_objs, all_boxes, all_mask2 = [], [], [], [], []
    all_obj_to_img = []

    obj_offset = 0

    for i, (img, objs, boxes, mask1, mask2) in enumerate(batch):
        all_imgs.append(img[None])
        num_objs = objs.size(0)

        all_objs.append(objs)
        all_boxes.append(boxes)
        all_mask1.append(mask1[None])
        all_mask2.append(mask2[None])

        all_obj_to_img.append(torch.LongTensor(num_objs).fill_(i))
        obj_offset += num_objs

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_mask1 = torch.cat(all_mask1)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_mask2 = torch.cat(all_mask2)

    return all_imgs, all_objs, all_boxes, \
           all_obj_to_img, all_mask1, all_mask2
