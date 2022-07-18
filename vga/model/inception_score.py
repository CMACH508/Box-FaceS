import numpy as np
import torch
import torch.utils.data
from scipy.stats import entropy
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from modules.layer_builders import Interpolate


class InceptionScore(nn.Module):
    def __init__(self, cuda=True, batch_size=32, resize=False):
        super(InceptionScore, self).__init__()
        assert batch_size > 0
        self.resize = resize
        self.batch_size = batch_size
        self.cuda = cuda
        # Set up dtype
        self.device = 'cuda' if cuda else 'cpu'
        if not cuda and torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")

        # Load inception model
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.inception_model.eval()
        self.up = Interpolate(size=(299, 299), mode='bilinear').to(self.device)
        self.clean()

    def clean(self):
        self.preds = np.zeros((0, 1000))

    def get_pred(self, x):
        if self.resize:
            x = self.up(x)
        x = self.inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    def forward(self, imgs):
        # Get predictions
        preds_imgs = self.get_pred(imgs.to(self.device))
        self.preds = np.append(self.preds, preds_imgs, axis=0)

    def compute_score(self, splits=1):
        # Now compute the mean kl-div
        split_scores = []
        preds = self.preds
        N = self.preds.shape[0]
        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)
