#%%Imports
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

from deployable.model import model

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)
#torch.cuda.empty_cache()

#%%CPCE Architecture
class _redcnn(nn.Module):
    def __init__(self):
        super(_redcnn, self).__init__()
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

class cpce3d(model):

    def __init__(self):
        model.__init__(self)
        self.redcnn = _redcnn().to(self.device)
        params = sum([np.prod(p.size()) for p in self.redcnn.parameters()])
        print('Number of params in CPCE3D: %d'%(params))    

    def infer(self, x):

        x_pad = np.pad(x, ((4, 4), (0, 0), (0, 0)), 'edge')
        
        self.redcnn.load_state_dict(torch.load('deployable/REDCNN_25.pth'))#Please change this directory if you change the model file location!!

        for p in self.redcnn.parameters():
            p.requires_grad = False

        self.redcnn.eval()
        patches = self._torchify(x_pad)
        print(patches.shape)
        denoised_patches = torch.from_numpy(np.zeros_like(x)).float().unsqueeze(1)

        for i in range(4, patches.shape[0] - 4):
            with torch.no_grad():
                inp_patch = patches[i-4:i+5, :, :, :].unsqueeze(0).transpose(1, 2)/4096
                denoised_patches[i - 4, :, :, :] = self.redcnn(inp_patch.to(self.device)).squeeze(0)

        denoised_patches = denoised_patches*4096
        denoised_patches = torch.round(denoised_patches)
        print('Inference complete!')
        return np.squeeze(denoised_patches.cpu().detach().numpy(), axis = 1)

    def train(self, x, y, epoch_number = 25):

        for p in self.redcnn.parameters():
            p.requires_grad = True

        dataloader, valloader = self._prepare_training_data(x, y, thickness=9, bsize=48)

        optimiser = optim.Adam(self.redcnn.parameters(), lr=1e-4)

        #%%Train branch
        for epoch in range(epoch_number):
            running_loss = 0
            val_loss = 0
            self.redcnn.train() 

            start = time.time()

            #Train step
            for i, data in enumerate(dataloader, 0):

                #Load data and initialise optimisers
                phantom_in, noisy_in = data

                #Rescale
                phantom_in = phantom_in/4096[:, :, 4:5, :, :]
                noisy_in = noisy_in/4096

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                optimiser.zero_grad()

                #Forward pass
                outputs1 = self.redcnn(noisy_in)

                #Calculate perceptual loss
                main_loss = nn.MSELoss()(outputs1, phantom_in)
                regulariser = nn.L1Loss()(outputs1, phantom_in)
                loss = main_loss + 0.01*regulariser

                #Backprop and update
                loss.backward()
                optimiser.step()

                # print statistics
                running_loss += loss.item()

                #Clear memory
                #del phantom_in
                #del noisy_in
                #torch.cuda.empty_cache()

                #print(i)

            #Val step
            self.redcnn.eval()
            #torch.cuda.empty_cache()

            for j, data in enumerate(valloader, 0):

                #Load validation data
                phantom_in, noisy_in = data

                #Rescale
                phantom_in = phantom_in/4096[:, :, 4:5, :, :]
                noisy_in = noisy_in/4096

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                #Forward pass in the validation phase
                optimiser.zero_grad()

                #Forward pass
                with torch.no_grad():
                    outputs1 = self.redcnn(noisy_in)

                #Calculate perceptual loss
                main_loss = nn.MSELoss()(outputs1, phantom_in)
                regulariser = nn.L1Loss()(outputs1, phantom_in)
                loss = main_loss + 0.01*regulariser

                # print statistics
                loss =  loss
                val_loss += loss.item()

                #Clear memory
                #del phantom_in
                #del noisy_in
                #torch.cuda.empty_cache()

            print('[%d, %5d] train_loss: %f val_loss: %f' %
                    (epoch + 1, i + 1, running_loss / int(i+1), val_loss / int(j+1) ))
            running_loss = 0.0
            val_loss = 0.0
            end = time.time()
            print('Time taken for epoch: '+str(end-start)+' seconds')
            #scheduler.step()

            if (epoch+1)%5 == 0:
                print('Saving model...')
                torch.save(self.red_cnn.state_dict(), 'CPCE3D_'+str(epoch+1)+'.pth')#Please change this directory if you want to save elsewhere!!

        print('Training complete!')

#%%
