import os
import numpy as np
from sklearn.model_selection import train_test_split
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)

class model:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_training_data(self, x, y, thickness=1, bsize=48):
        noisy_train, noisy_test, phantom_train, phantom_test = train_test_split(self._make_patches(x, thickness) , self._make_patches(y, thickness))
        dataset = torch.utils.data.TensorDataset(self._torchify(phantom_train), self._torchify(noisy_train))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=True)
        dataset = torch.utils.data.TensorDataset(self._torchify(phantom_test), self._torchify(noisy_test))
        valloader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=True)
        return dataloader, valloader

    def _make_patches(self, x, thickness):
        if thickness == 1:
            return _make_patches2d(x)
        else:
            return _make_patches3d(x, thickness)

    def _make_patches3d(self, x, thickness):
        return x.reshape(x.shape[0], thickness, 8, 64, 8, 64).swapaxes(3, 4). \
            reshape(x.shape[0], thickness, -1, 64, 64).swapaxes(1, 2).reshape(-1, thickness, 64, 64)

    def _make_patches2d(self, x):
        return x.reshape(x.shape[0], 4, 64, 4, 64).swapaxes(2, 3).reshape( -1, 64, 64)

    def _torchify(self, x):
        return torch.from_numpy(x).float().unsqueeze(1)

    def infer(self, x):
        pass

    def train(self, x, y, epoch_number = 25):
        pass