#%%Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.model_selection import train_test_split
from skimage.measure import compare_ssim as ssim
from skimage.external import tifffile as tif
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data

from deployable.model import model

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)

#%%QAE architecture
class _quadConv(nn.Module):
    def __init__(self, inp, out, pad=True):
        super(_quadConv, self).__init__()
        if(pad):
            self.conv1 = nn.Conv2d(inp, out, kernel_size = (5, 5), stride = 1, padding=(2, 2))
            self.conv2 = nn.Conv2d(inp, out, kernel_size = (5, 5), stride = 1, padding=(2, 2))
            self.conv3 = nn.Conv2d(inp, out, kernel_size = (5, 5), stride = 1, padding=(2, 2))
        else:
            self.conv1 = nn.Conv2d(inp, out, kernel_size = (5, 5), stride = 1)
            self.conv2 = nn.Conv2d(inp, out, kernel_size = (5, 5), stride = 1)
            self.conv3 = nn.Conv2d(inp, out, kernel_size = (5, 5), stride = 1)

    def forward(self, x):
        quad_conv = (self.conv1(x)*self.conv2(x)) + self.conv3(x*x)

        return quad_conv

class _quadDeconv(nn.Module):
    def __init__(self, inp, out, pad = True):
        super(_quadDeconv, self).__init__()
        if(pad):
            self.conv1 = nn.ConvTranspose2d(inp, out, kernel_size = (5, 5), stride = 1, padding=(2, 2))
            self.conv2 = nn.ConvTranspose2d(inp, out, kernel_size = (5, 5), stride = 1, padding=(2, 2))
            self.conv3 = nn.ConvTranspose2d(inp, out, kernel_size = (5, 5), stride = 1, padding=(2, 2))
        else:
            self.conv1 = nn.ConvTranspose2d(inp, out, kernel_size = (5, 5), stride = 1)
            self.conv2 = nn.ConvTranspose2d(inp, out, kernel_size = (5, 5), stride = 1)
            self.conv3 = nn.ConvTranspose2d(inp, out, kernel_size = (5, 5), stride = 1)

    def forward(self, x):
        quad_deconv = (self.conv1(x)*self.conv2(x)) + self.conv3(x*x)

        return quad_deconv

class _qae(nn.Module):
    def __init__(self):
        super(_qae, self).__init__()
        self.conv1 = _quadConv(1, 15)
        self.conv2 = _quadConv(15, 15)
        self.conv3 = _quadConv(15, 15)
        self.conv4 = _quadConv(15, 15)
        self.conv5 = _quadConv(15, 15, pad = False)
        self.deconv5 = _quadDeconv(15, 15, pad = False)
        self.deconv4 = _quadDeconv(15, 15)
        self.deconv3 = _quadDeconv(15, 15)
        self.deconv2 = _quadDeconv(15, 15)
        self.deconv1 = _quadDeconv(15, 1)

    def forward(self, x):
        #Conv
        im = x.clone()
        x = F.relu(self.conv1(x))
        im1 = x.clone()
        x = F.relu(self.conv2(x))
        im2 = x.clone()
        x = F.relu(self.conv3(x))
        im3 = x.clone()
        x = F.relu(self.conv4(x))
        im4 = x.clone()
        x = F.relu(self.conv5(x))
        #Deconv
        x = F.relu(self.deconv5(x) + im4)
        x = F.relu(self.deconv4(x) + im3)
        x = F.relu(self.deconv3(x) + im2)
        x = F.relu(self.deconv2(x) + im1)
        x = F.relu(self.deconv1(x) + im)
        return x

#%%
class quadratic_autoencoder(model):

    def __init__(self):
        model.__init__(self)
        self.q_ae = _qae().to(self.device)
        params = sum([np.prod(p.size()) for p in self.q_ae.parameters()])
        print('Number of params in QAE: %d'%(params))

    def infer(self, x):

        self.q_ae.load_state_dict(torch.load('deployable/QAE_25.pth'))#Please change this directory if you change the model file location!!

        for p in self.q_ae.parameters():
            p.requires_grad = False

        self.q_ae.eval()
        patches = self._torchify(x)
        print(patches.shape)
        denoised_patches = torch.zeros_like(patches)

        for i in range(0, patches.shape[0]):
            with torch.no_grad():
                inp_patch = patches[i:i+1]/4096
                denoised_patches[i] = self.q_ae(inp_patch.to(self.device))

        denoised_patches = denoised_patches*4096
        denoised_patches = torch.round(denoised_patches)
        print('Inference complete!')
        return np.squeeze(denoised_patches.cpu().detach().numpy(), axis = 1)


    def train(self, x, y, epoch_number = 25):

        for p in self.q_ae.parameters():
            p.requires_grad = True

        dataloader, valloader = self._prepare_training_data(x, y, thickness=1, bsize=128)
        optimiser = optim.Adam(self.q_ae.parameters(), lr=1e-4)        

        for epoch in range(epoch_number):
            running_loss = 0
            val_loss = 0
            self.q_ae.train() 

            start = time.time()

            #Train step
            for i, data in enumerate(dataloader, 0):

                #Load data and initialise optimisers
                phantom_in, noisy_in = data

                #Rescale
                phantom_in = phantom_in/4096
                noisy_in = noisy_in/4096

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                optimiser.zero_grad()

                #Forward pass
                outputs1 = self.q_ae(noisy_in)

                #Calculate perceptual loss
                main_loss = nn.MSELoss()(outputs1, phantom_in)
                regulariser = nn.L1Loss()(outputs1, phantom_in)
                loss = main_loss + 0.01*regulariser

                #Backprop and update
                loss.backward()
                optimiser.step()

                # print statistics
                running_loss += loss.item()

            #Val step
            self.q_ae.eval()

            for j, data in enumerate(valloader, 0):

                #Load validation data
                phantom_in, noisy_in = data

                #Rescale
                phantom_in = phantom_in/4096
                noisy_in = noisy_in/4096

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                #Forward pass in the validation phase
                optimiser.zero_grad()

                #Forward pass
                with torch.no_grad():
                    outputs1 = self.q_ae(noisy_in)

                #Calculate perceptual loss
                main_loss = nn.MSELoss()(outputs1, phantom_in)
                regulariser = nn.L1Loss()(outputs1, phantom_in)
                loss = main_loss + 0.01*regulariser

                # print statistics
                loss =  loss
                val_loss += loss.item()

            print('[%d, %5d] train_loss: %f val_loss: %f' %
                    (epoch + 1, i + 1, running_loss / int(i+1), val_loss / int(j+1) ))
            running_loss = 0.0
            val_loss = 0.0
            end = time.time()
            print('Time taken for epoch: '+str(end-start)+' seconds')

            if (epoch+1)%5 == 0:
                print('Saving model...')
                torch.save(self.q_ae.state_dict(), 'QAE_'+str(epoch+1)+'.pth')#Please change this directory if you want to save elsewhere!!

        print('Training complete!')