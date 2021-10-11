import numpy as np
import os
import pydicom
import cv2
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit
import math
from scipy.spatial import distance
from pathos.pools import ParallelPool

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchvision import models, transforms as T
from GuidedFilter import ConvGuidedFilter

#Custom class for inference network
#%%Define IRQM in pytorch
class IRQM_net(nn.Module):

    def get_gaussian_kernel(self, kernel_size=5, sigma=2, channels=1):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        
        return gaussian_filter

    def __init__(self, requires_grad=False):
        super(IRQM_net, self).__init__()
        self.filt = self.get_gaussian_kernel()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = (3, 3), stride = 1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size = (3, 3), stride = 1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = (3, 3), stride = 1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 1)
        self.lin1 = nn.Linear(64, 64)
        self.lin2 = nn.Linear(64, 1)

    def forward(self, x):
        x -= self.filt(F.pad(x, (2, 2, 2, 2)))
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = F.elu(self.lin1(x))
        x = self.lin2(x)
        return x