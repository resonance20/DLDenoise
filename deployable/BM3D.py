#%%Imports
import time
import numpy as np
import pytorch_ssim
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import bm3d
import cv2

from deployable.model import model

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)

#Non learnable denoising
class bm3d_dl(model):

    def __init__(self):
        model.__init__(self)

    def run_init(self):
        self.gen = None
        print('BM3D has no trainable parameters!')

    class _gen(nn.Module):
        def __init__(self):
            pass

        def forward(self, x):
            pass

    def _infer(self, x, fname = 'deployable/WGAN_gen_30.pth'):

        denoised_patches = np.zeros_like(x)
        
        for i in range(0, x.shape[0]):
            x_filt = x[i] - cv2.pyrUp(cv2.pyrDown(x[i]))#High pass filtering
            x_var = np.median( np.abs(x_filt - np.median(x_filt)) )#Variance is estimated using median absolute deviation
            denoised_patches[i] = bm3d.bm3d(x[i], x_var)

        print('Inference complete!')
        return denoised_patches

    def train(self, train_dataset, val_dataset, epoch_number = 25, batch_size = 48, fname = 'WGAN'):

        raise NotImplementedError('BM3D has no trainable parameters and cannot be trained! Please only use inference mode!')