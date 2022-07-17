import os
from torch.utils.data import Dataset
from utils.data_utils import make_dataset
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import random
import albumentations as A
import cv2
from utils.task import random_irregular_mask
import pickle
import copy
from utils.model_utils import get_node_box


class CelebAHQDataset(Dataset):
    def __init__(self, split, num_samples=-1, data_path=None, box_path=None,
                 image_dir=None, resolution=128):
        super(CelebAHQDataset, self).__init__()

        self.split = split
        self.resolution = resolution

        self.image_ids = make_dataset(os.path.join(data_path))
        with open(box_path, 'rb') as handle:
            self.bboxes = pickle.load(handle)

        random.shuffle(self.image_ids)
        if num_samples != -1:
            self.image_ids = self.image_ids[:num_samples]

        self.image_dir = image_dir

        # define common transform
        # self.flip = A.HorizontalFlip()  # bbox (x_min, y_min, x_max, y_max)

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
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - mask: LongTensor of shape (num_objs,)
        - boxes: FloatTensor of shape (num_objs, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        """

        img_path = os.path.join(self.image_dir, self.image_ids[index])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, [512, 512])
        bboxes = np.array(self.bboxes[self.image_ids[index]])
        boxes = copy.deepcopy(bboxes)
        if self.split == 'train' and random.random() > 0.5:
            image = cv2.flip(image, 1)
            boxes[:, 1], boxes[:, 3] = 512 - bboxes[:, 3], 512 - bboxes[:, 1]

        # mask = np.zeros_like(image)
        # for box in boxes[1:2]:
        #     mask[box[0]:box[2], box[1]:box[3]] = 1
        # cv2.imwrite('%s' % self.image_ids[index], (mask * image))

        image = self.transform(image=image)['image']
        boxes = torch.from_numpy(np.array(bboxes))
        box_256 = get_node_box(256, 512, boxes)

        if self.split == 'train':
            mask1 = random_irregular_mask(image, width=16)
            mask2 = random_irregular_mask(image, width=18)
            if random.random() > 0.3:
                num = random.randint(1, 6)
                nodes = random.sample([0, 1, 2, 3, 4, 5], num)
                for node in nodes:
                    mask2[:, box_256[node, 0]:box_256[node, 2], box_256[node, 1]:box_256[node, 3]] = 0
        else:
            mask1, mask2 = torch.ones_like(image), torch.ones_like(image)

        return image, mask1, mask2, boxes
