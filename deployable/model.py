import os
import numpy as np
import time
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision

from deployable.dicom_helpers import read_dicom_folder, write_dicom_folder

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)

class model(ABC):

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = None

    def _torchify(self, x):
        x = x.astype(np.float32)
        return torch.from_numpy(x).float().unsqueeze(1)

    #VGG features
    class _VGGLoss(nn.Module):
        def __init__(self):
            super(model._VGGLoss, self).__init__()
            model.__init__(self)
            vgg19_model = torchvision.models.vgg19(pretrained=True).eval()
            self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35]).to(self.device)
            self.feature_extractor.eval()

        def forward(self, x, y):
            if len(x.shape)==5:
                x = x.squeeze(2)
            if len(y.shape)==5:
                y = y.squeeze(2)
            feat_x = self.feature_extractor(x.repeat(1, 3, 1, 1))
            feat_y = self.feature_extractor(y.repeat(1, 3, 1, 1))
            return nn.L1Loss()(feat_x, feat_y)
    
    #External denoising function
    def denoise_dicom(self, in_folder, out_folder, series_description, fname):
        """! Function to conduct inference on a given DICOM series. Calls _infer internally.
        @param in_folder  A string containing the location of the given DICOM folder.
        @param out_folder  A string containing the location where the denoised DICOM series will be saved.
        @param series_description   A string with the series description of the new denoised DICOM series.
        @return  None. The denoised DICOM series is written into a new folder.
        """
        noisy_array = read_dicom_folder(in_folder)
        denoised_array = self._infer(noisy_array, fname)
        write_dicom_folder(folder=in_folder, new_volume=denoised_array, output_folder=out_folder, series_description=series_description)
        return None
    
    #Numpy array denoiser, to be implemented by each model
    @abstractmethod
    def _infer(self, x, fname):
        """! Function to conduct inference on a give NumPy array.
        @param x  The noisy NumPy array.
        @param fname   The location and name of the model file containing the weights. Is a string.
        @return  The denoised NumPy array.
        """
        pass
    
    #Train function, to be implemented by each model
    @abstractmethod
    def train(self, train_dataset, val_dataset, fname, batch_size = 48, epoch_number = 30):
        """! Function to train a given model.
        @param train_dataset   The training dataset. Expected as a pair of (input, GT). This is a PyTorch dataset.
        @param val_dataset   The validation dataset. Expected as a pair of (input, GT). This is a PyTorch dataset.
        @param fname   The output file location and name. Do not append .pth to the filename, this is done automatically.
        @param batch_size  Size of each training batch.
        @param epoch_number  Number of training epochs.
        @return  None. the output models are saved with the epoch numbers appended to the filename.
        """
        pass