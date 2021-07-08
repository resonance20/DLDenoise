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

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.deterministic=True
torch.manual_seed(1)
#torch.cuda.empty_cache()

#%%Architecture segments
class gan_3d(model):

    def __init__(self):
        model.__init__(self)
        self.gen = self._gen().to(self.device)
        params = sum([np.prod(p.size()) for p in self.gen.parameters()])
        print('Number of params in GAN: %d'%(params))

    #Generator
    class _gen(nn.Module):
        def __init__(self):
            super(gan_3d._gen, self).__init__()
            self.conv1 = nn.Conv3d(1, 32, kernel_size = (3, 3, 3), stride = 1)
            self.bn1 = nn.BatchNorm3d(32)
            self.conv2 = nn.Conv3d(32, 32, kernel_size = (3, 3, 3), stride = 1)
            self.bn2 = nn.BatchNorm3d(32)
            self.conv3 = nn.Conv3d(32, 64, kernel_size = (3, 3, 3), stride = 1)
            self.bn3 = nn.BatchNorm3d(64)
            self.conv4 = nn.Conv3d(64, 64, kernel_size = (3, 3, 3), stride = 1)
            self.bn4 = nn.BatchNorm3d(64)
            self.conv5 = nn.Conv3d(64, 128, kernel_size = (3, 3, 3), stride = 1)
            self.bn5 = nn.BatchNorm3d(128)
            self.conv6 = nn.Conv3d(128, 128, kernel_size = (3, 3, 3), stride = 1)
            self.bn6 = nn.BatchNorm3d(128)
            self.conv7 = nn.Conv3d(128, 1, kernel_size = (3, 3, 3), stride = 1)
        
        def forward(self, x):
            im = x.clone()[:, :, 7:-7, 7:-7, 7:-7]
            x = F.leaky_relu(self.bn1(self.conv1(x)))
            x = F.leaky_relu(self.bn2(self.conv2(x)))
            x = F.leaky_relu(self.bn3(self.conv3(x)))
            x = F.leaky_relu(self.bn4(self.conv4(x)))
            x = F.leaky_relu(self.bn5(self.conv5(x)))
            x = F.leaky_relu(self.bn6(self.conv6(x)))
            return im - self.conv7(x)

    #Discriminator
    class _disc(nn.Module):
        def __init__(self):
            super(gan_3d._disc, self).__init__()

            self.conv1 = nn.Conv3d(1, 32, kernel_size = (3, 3, 3), stride = 1, padding = (0, 1, 1))
            self.conv2 = nn.Conv3d(32, 32, kernel_size = (3, 3, 3), stride = 1, padding = (0, 1, 1))
            self.conv3 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 2, padding = (1, 1))

            self.conv4 = nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 1, padding = (1, 1))
            self.conv5 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 1, padding = (1, 1))
            self.conv6 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 2, padding = (1, 1))

            self.conv7 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride = 1, padding = (1, 1))
            self.conv8 = nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 1, padding = (1, 1))
            self.conv9 = nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 2, padding = (1, 1))

            self.lin1 = nn.Linear(6272, 256)
            self.lin2 = nn.Linear(256, 1)

        def forward(self, x):
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = x.reshape(-1, x.shape[1], x.shape[3], x.shape[4])

            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = F.leaky_relu(self.conv5(x))

            x = F.leaky_relu(self.conv6(x))
            x = F.leaky_relu(self.conv7(x))
            x = F.leaky_relu(self.conv8(x))

            x = F.leaky_relu(self.conv9(x))

            x = x.view(x.shape[0], -1)
            x = F.leaky_relu(self.lin1(x))
            return torch.sigmoid(self.lin2(x))

    def infer(self, x, fname='deployable/GAN_gen_25.pth'):

        self.gen.load_state_dict(torch.load(fname))

        for p in self.gen.parameters():
            p.requires_grad = False

        self.gen.eval()
        patches = self._torchify(x)
        patches = patches.squeeze().unsqueeze(0).unsqueeze(0)
        denoised_patches = torch.zeros_like(patches)
        patches = F.pad(patches, (7, 7, 7, 7, 7, 7), mode='replicate')

        for i in range(7, patches.shape[2]-7):
            with torch.no_grad():
                inp_patch = patches[:, :, i-7:i+8]
                denoised_patches[:, :, i - 7] = self.gen(inp_patch.to(self.device))

        denoised_patches = torch.clamp(torch.round(denoised_patches), 0 , 4096)
        print('Inference complete!')
        return np.squeeze(denoised_patches.cpu().detach().numpy())

    
    def train(self, x, y, fname, thickness = 19, epoch_number = 25, batch_size = 48):

        for p in self.gen.parameters():
            p.requires_grad = True

        dataloader, valloader = self._prepare_training_data(x, y, thickness=thickness, bsize=batch_size)
        disc = self._disc().to(self.device)

        params = sum([np.prod(p.size()) for p in disc.parameters()])
        print('Number of params in discriminator: %d'%(params))

        optim_gen = optim.Adam(list(self.gen.parameters()) + list(disc.parameters()), lr=2e-4, betas=(0.5, 0.99))

        #%%Train branch
        for epoch in range(epoch_number):
            running_loss = 0
            val_loss = 0
            self.gen.train() 
            disc.train() 

            start = time.time()

            #Train step
            for i, data in enumerate(dataloader, 0):

                #Load data and initialise optimisers
                phantom_in, noisy_in = data

                #Rescale
                phantom_in = phantom_in[:, :, 7:-7, 7:-7, 7:-7]

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                optim_gen.zero_grad()

                #Forward pass
                outputs1 = self.gen(noisy_in)
                fake_res = disc(outputs1).squeeze()
                real_res = disc(phantom_in).squeeze()

                #Calculate perceptual loss
                mse_loss = nn.MSELoss()(outputs1, phantom_in)
                fake_loss = nn.BCELoss()(fake_res, torch.zeros(phantom_in.shape[0]).cuda())
                real_loss = nn.BCELoss()(real_res, torch.ones(phantom_in.shape[0]).cuda())
                adv_loss = nn.BCELoss()(fake_res, torch.ones(phantom_in.shape[0]).cuda())
                
                gen_loss =  (0.001 * mse_loss) + adv_loss
                disc_loss = real_loss + fake_loss
                loss = gen_loss + disc_loss

                #Backprop and update
                loss.backward()
                optim_gen.step()

                # print statistics
                running_loss += loss.item()

                #Clear memory
                #del phantom_in
                #del noisy_in
                #torch.cuda.empty_cache()

            #Val step
            self.gen.eval()
            disc.eval()
            #torch.cuda.empty_cache()

            for j, data in enumerate(valloader, 0):

                #Load validation data
                phantom_in, noisy_in = data

                #Rescale
                phantom_in = phantom_in[:, :, 7:-7, 7:-7, 7:-7]

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                #Forward pass in the validation phase
                with torch.no_grad():
                    outputs1 = self.gen(noisy_in)

                #Calculate perceptual loss
                loss = nn.MSELoss()(outputs1, phantom_in) 

                # print statistics
                val_loss += loss.item()

                #Clear memory
                #del phantom_in
                #del noisy_in
                #torch.cuda.empty_cache()

            print('[%d, %5d] train_loss: %.3f val_loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / int(i+1), val_loss / int(j+1) ))
            running_loss = 0.0
            val_loss = 0.0
            end = time.time()
            print('Time taken for epoch: '+str(end-start)+' seconds')
            #scheduler.step()

            if (epoch+1)%5 == 0:
                print('Saving model...')
                torch.save(self.gen.state_dict(), fname+'_gen_'+str(epoch+1)+'.pth')
                torch.save(disc.state_dict(), fname+'_disc_'+str(epoch+1)+'.pth')

        print('Training complete!')

#%%
