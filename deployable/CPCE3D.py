#%%Imports
import os
import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data

from deployable.WGAN_VGG import wgan_vgg

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)
#torch.cuda.empty_cache()

#CPCE3D Shan et al.
class cpce3d(wgan_vgg):

    def __init__(self):
        wgan_vgg.__init__(self)

    def run_init(self):
        self.gen = self._gen().to(self.device)
        params = sum([np.prod(p.size()) for p in self.gen.parameters()])
        print('Number of params in CPCE3D: %d'%(params))

    class _gen(nn.Module):
        def __init__(self):
            super(cpce3d._gen, self).__init__()
            self.conv1 = nn.Conv3d(1, 32, kernel_size = (3, 3, 3), stride = 1)
            self.conv2 = nn.Conv3d(32, 32, kernel_size = (3, 3, 3), stride = 1)
            self.conv3 = nn.Conv3d(32, 32, kernel_size = (3, 3, 3), stride = 1)
            self.conv4 = nn.Conv3d(32, 32, kernel_size = (3, 3, 3), stride = 1)
            self.pc1 = nn.Conv3d(64, 32, kernel_size = (1, 1, 1), stride = 1)
            self.pc2 = nn.Conv3d(64, 32, kernel_size = (1, 1, 1), stride = 1)
            self.pc3 = nn.Conv3d(64, 32, kernel_size = (1, 1, 1), stride = 1)
            self.upconv4 = nn.ConvTranspose3d(32, 32, kernel_size = (1, 3, 3), stride = 1)
            self.upconv3 = nn.ConvTranspose3d(32, 32, kernel_size = (1, 3, 3), stride = 1)
            self.upconv2 = nn.ConvTranspose3d(32, 32, kernel_size = (1, 3, 3), stride = 1)
            self.upconv1 = nn.ConvTranspose3d(32, 1, kernel_size = (1, 3, 3), stride = 1)

        def forward(self, x):
            #Conv
            x = F.leaky_relu(self.conv1(x))
            im1 = x.clone()[:,:,3,:,:].unsqueeze(2)
            x = F.leaky_relu(self.conv2(x))
            im2 = x.clone()[:,:,2,:,:].unsqueeze(2)
            x = F.leaky_relu(self.conv3(x))
            im3 = x.clone()[:,:,1,:,:].unsqueeze(2)
            x = F.leaky_relu(self.conv4(x))
            #Deconv
            x = F.leaky_relu(self.upconv4(x))
            x = F.leaky_relu(self.pc1(torch.cat((x, im3), 1) ))
            x = F.leaky_relu(self.upconv3(x))
            x = F.leaky_relu(self.pc2(torch.cat((x, im2), 1) ))
            x = F.leaky_relu(self.upconv2(x))
            x = F.leaky_relu(self.pc3(torch.cat((x, im1), 1) ))
            out = F.relu(self.upconv1(x))

            return out

    def _infer(self, x, fname = 'deployable/CPCE3D_gen_30.pth'):

        self.gen.load_state_dict(torch.load(fname))

        for p in self.gen.parameters():
            p.requires_grad = False

        self.gen.eval()
        patches = self._torchify(x)
        patches = patches.squeeze().unsqueeze(0).unsqueeze(0)
        denoised_patches = torch.zeros_like(patches)
        patches = F.pad(patches, (0, 0, 0, 0, 4, 4), mode='replicate')

        for i in range(4, patches.shape[2]-4):
            with torch.no_grad():
                inp_patch = patches[:, :, i-4:i+5]
                denoised_patches[:, :, i - 4] = self.gen(inp_patch.to(self.device))

        denoised_patches = torch.clamp(torch.round(denoised_patches), 0, 4096)
        print('Inference complete!')
        return np.squeeze(denoised_patches.cpu().detach().numpy())