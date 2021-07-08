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

    def __init__(self, requires_grad=False):
        super(IRQM_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = (3, 3), stride = 1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size = (3, 3), stride = 1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size = (3, 3), stride = 1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 1)
        self.lin1 = nn.Linear(64, 64)
        self.lin2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = F.elu(self.lin1(x))
        x = self.lin2(x)
        return torch.sigmoid(x)