#%%Imports
import os
from sklearn.model_selection import train_test_split
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad, Variable
import torch.nn.functional as F
import torch.utils.data

from model import model

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)

class _SAmodule(nn.Module):
    def __init__(self, channels):
        super(_SAmodule, self).__init__()
        self.depth_conv = nn.Conv3d(channels, channels, kernel_size = (1, 1, 1), stride = 1)
        self.plane_conv = nn.Conv3d(channels, channels, kernel_size = (1, 1, 1), stride = 1)
        self.base_conv = nn.Conv3d(channels, channels, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.scaling_factor = nn.Parameter(torch.rand(1, 1, 1, 1, 1))

    def forward(self, x):
        q = self.plane_conv(x)
        k = self.depth_conv(x)
        v = self.base_conv(x)

        #Plane attention
        q_flat = q.view(q.shape[0]* q.shape[1]* q.shape[2], -1, 1)
        k_flat = k.view(k.shape[0]* k.shape[1]* k.shape[2], 1, -1)
        v_flat = v.view(v.shape[0]* v.shape[1]* v.shape[2], -1)
        plane_mat = torch.matmul(q_flat, k_flat)
        v_flat = torch.sum(F.softmax(plane_mat) * v_flat.unsqueeze(2), dim = 1)
        p_att = v_flat.view(v.shape)
        
        #Depth attention
        q_flat = torch.transpose(q.view(q.shape[0]* q.shape[1], 1, q.shape[2], -1), 1, 3).reshape(-1, q.shape[2], 1)
        k_flat = torch.transpose(k.view(k.shape[0]* k.shape[1], k.shape[2], 1, -1), 1, 3).reshape(-1, 1, k.shape[2])
        v_flat = torch.transpose(v.view(v.shape[0]* v.shape[1], v.shape[2], -1), 2, 1).reshape(-1, v.shape[2])
        depth_mat = torch.matmul(q_flat, k_flat)
        v_flat = torch.sum(F.softmax(depth_mat) * v_flat.unsqueeze(2), dim = 1)
        d_att = torch.transpose(v_flat.view(v.shape), 2, 4)

        #Attention fusion
        out = x + self.scaling_factor * (p_att + d_att)

        return out

#%%SACNN Architecture
class _sacnn(nn.Module):
    def __init__(self):
        super(_sacnn, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.conv2 = nn.Conv3d(64, 32, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.sa1 = _SAmodule(32)
        self.conv3 = nn.Conv3d(32, 16, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.sa2 = _SAmodule(16)
        self.conv4 = nn.Conv3d(16, 32, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.sa3 = _SAmodule(32)
        self.conv5 = nn.Conv3d(32, 64, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))
        self.conv6 = nn.Conv3d(64, 1, kernel_size = (3, 3, 3), stride = 1, padding = (1, 1, 1))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.sa1(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.sa2(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.sa3(x)
        x = F.leaky_relu(self.conv5(x))
        out = F.relu(self.conv6(x))

        return out
        
class self_attention(model):

    def __init__(self):
        self.sacnn = _sacnn().to(self.device)

    def infer(self, x):

        pad = x.shape[0]%3
        if pad!=0:
            x = np.concatenate((x, np.zeros(pad, x.shape[1], x.shape[2])), dim = 0)

        self.sacnn.load_state_dict(torch.load('deployable/SACNN_15.pth'))#Please change this directory if you change the model file location!!

        for p in self.sacnn.parameters():
            p.requires_grad = False

        self.sacnn.eval()
        patches = self._torchify(x)
        print(patches.shape)
        denoised_patches = torch.zeros_like(patches)

        for i in range(0, patches.shape[0], 3):
            with torch.no_grad():
                inp_patch = patches[i:i+3, :, :].unsqueeze(0).transpose(1, 2)/4096
                print(inp_patch.shape)
                denoised_patches[i: i + 3, :, :] = self.sacnn(inp_patch.to(self.device))

        denoised_patches = denoised_patches*4096
        denoised_patches = torch.round(denoised_patches)
        denoised_patches = denoised_patches[:denoised_patches.shape[0] - pad, :, :]
        print('Inference complete!')
        return np.squeeze(denoised_patches.cpu().detach().numpy(), axis = 1)


    def train(self, x, y, epoch_number = 25):

        for p in self.sacnn.parameters():
            p.requires_grad = True

        dataloader, valloader = self._prepare_training_data(x, y, thickness=3, bsize=1)

        optimiser = optim.Adam(self.sacnn.parameters(), lr=1e-4) 

        for epoch in range(epoch_number):
            running_loss = 0
            val_loss = 0
            self.sacnn.train() 

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
                outputs1 = self.sacnn(noisy_in)
                
                loss = nn.MSELoss()(outputs1, phantom_in) 

                #Backprop and update
                loss.backward(retain_graph=True)
                optimiser.step()

                # print statistics
                running_loss += loss.item()

            #Val step
            self.sacnn.eval()

            for j, data in enumerate(valloader, 0):

                #Load validation data
                phantom_in, noisy_in = data

                #Rescale
                phantom_in = phantom_in/4096
                noisy_in = noisy_in/4096

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                #Forward pass
                with torch.no_grad():
                    outputs1 = self.sacnn(noisy_in)

                #Calculate total loss
                loss =  nn.MSELoss()(outputs1, phantom_in)

                # print statistics
                val_loss += loss.item()

            print('[%d, %5d] train_loss: %f val_loss: %f' %
                    (epoch + 1, i + 1, running_loss / int(i+1), val_loss / int(j+1) ))
            running_loss = 0.0
            val_loss = 0.0
            end = time.time()
            print('Time taken for epoch: '+str(end-start)+' seconds')

            if (epoch+1)%5 == 0:
                print('Saving model...')
                torch.save(self.sacnn.state_dict(), 'SACNN_'+str(epoch+1)+'.pth')#Please change this directory if you want to save elsewhere!!

        print('Training complete!')
