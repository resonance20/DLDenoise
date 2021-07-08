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
import matplotlib.pyplot as plt

from deployable.model import model

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)

#CNN based denoising
class cnn(model):

    def __init__(self):
        model.__init__(self)
        self.run_init()

    def run_init(self):
        self.gen = self._gen().to(self.device)
        params = sum([np.prod(p.size()) for p in self.gen.parameters()])
        print('Number of params in CNN10: %d'%(params))

    #CNN archtiecture
    class _gen(nn.Module):
        def __init__(self):
            super(cnn._gen, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size = (9, 9), stride = 1, padding = (4, 4))
            self.conv2 = nn.Conv2d(64, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
            self.conv3 = nn.Conv2d(32, 1, kernel_size = (5, 5), stride = 1, padding = (2, 2))

        def forward(self, x):
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.relu(self.conv3(x))

            return x

    def infer(self, x, fname = 'deployable/WGAN_gen_30.pth'):

        self.gen.load_state_dict(torch.load(fname))

        for p in self.gen.parameters():
            p.requires_grad = False

        self.gen.eval()
        patches = self._torchify(x)
        print(patches.shape)
        denoised_patches = torch.zeros_like(patches)

        for i in range(0, patches.shape[0]):
            with torch.no_grad():
                inp_patch = patches[i:i+1]
                denoised_patches[i] = self.gen(inp_patch.to(self.device))

        denoised_patches = torch.clamp(torch.round(denoised_patches), 0, 4096)
        print('Inference complete!')
        return np.squeeze(denoised_patches.cpu().detach().numpy(), axis = 1)

    def train(self, x, y, thickness = 1, epoch_number = 25, batch_size = 48, fname = 'WGAN'):

        for p in self.gen.parameters():
            p.requires_grad = True

        dataloader, valloader = self._prepare_training_data(x, y, thickness=thickness, bsize=batch_size)

        optim_gen = optim.Adam(self.gen.parameters(), lr=1e-4)       

        for epoch in range(epoch_number):
            running_loss = 0
            val_loss = 0
            self.gen.train() 

            start = time.time()

            #Train step
            for i, data in enumerate(dataloader, 0):

                #Load data and initialise optimisers
                phantom_in, noisy_in = data

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                optim_gen.zero_grad()

                #Gen update step
                outputs1 = self.gen(noisy_in)
                gen_loss = nn.MSELoss()(phantom_in, outputs1)
                gen_loss.backward()
                optim_gen.step()

                # print statistics
                running_loss += gen_loss.item()
                #print(gen_loss.item())

            #Val step
            self.gen.eval()

            for j, data in enumerate(valloader, 0):

                #Load validation data
                phantom_in, noisy_in = data

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                #Forward pass
                with torch.no_grad():
                    #Gen update step
                    outputs1 = self.gen(noisy_in)
                    gen_loss = nn.MSELoss()(phantom_in, outputs1)

                # print statistics
                val_loss += gen_loss.item()

            print('[%d, %5d] train_loss: %f val_loss: %f' %
                    (epoch + 1, i + 1, running_loss / int(i+1), val_loss / int(j+1) ))
            running_loss = 0.0
            val_loss = 0.0
            end = time.time()
            print('Time taken for epoch: '+str(end-start)+' seconds')

            if (epoch+1)%5 == 0:
                print('Saving model...')
                torch.save(self.gen.state_dict(), fname+'_'+str(epoch+1)+'.pth')

        print('Training complete!')