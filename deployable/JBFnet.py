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
from scipy.spatial import distance

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)
#torch.cuda.empty_cache()

#%%DJBFnet architecture
class _denoiser(nn.Module):
    
    def __init__(self):
        super(_denoiser, self).__init__()
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
        im1 = x.clone()[:,:,3:10,:,:]
        x = F.leaky_relu(self.conv2(x))
        im2 = x.clone()[:,:,2:9,:,:]
        x = F.leaky_relu(self.conv3(x))
        im3 = x.clone()[:,:,1:8,:,:]
        x = F.leaky_relu(self.conv4(x))
        #Deconv
        x = F.leaky_relu(self.upconv4(x))
        x = F.leaky_relu(self.pc1(torch.cat((x, im3), 1) ))
        x = F.leaky_relu(self.upconv3(x))
        x = F.leaky_relu(self.pc2(torch.cat((x, im2), 1) ))
        x = F.leaky_relu(self.upconv2(x))
        x = F.leaky_relu(self.pc3(torch.cat((x, im1), 1) ))
        return self.upconv1(x)

#Joint Bilateral Filtering block
class _JBF_block(nn.Module):
    
    def __init__(self):
        super(_JBF_block, self).__init__()
        self.range_coeffecients = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size = (3, 3, 3), stride = 1),
            nn.ReLU(),
            nn.Conv3d(1, 1, kernel_size = (3, 3, 3), stride = 1),
            nn.ReLU()
        )
        self.domain_coeffecients = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size = (3, 3, 3), stride = 1),
            nn.ReLU(),
            nn.Conv3d(1, 1, kernel_size = (3, 3, 3), stride = 1),
            nn.ReLU()
        )
        
    def forward(self, x, domain_neighbor, guide_im):
        
        #Store shape
        mat_size = (x.shape[0], x.shape[1], x.shape[2] - 4, x.shape[3], x.shape[4])
        
        #Estimate filter coeffecients
        domain_kernel = self.domain_coeffecients(domain_neighbor)
        range_kernel = self.range_coeffecients(guide_im)
        weights = (domain_kernel*range_kernel) + 1e-10
        
        #Apply bilateral filter
        x = F.pad(x, (1, 1, 1, 1, 0, 0), mode='constant')
        x = x.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1).reshape(-1, 1, 3, 3, 3)
        weighted_x = weights*x
        filtered_im = weighted_x.view(weighted_x.shape[0], 1, -1).sum(2) / weights.view(weights.shape[0], 1, -1).sum(2)
        
        #Reshape and upsample
        return filtered_im.view(mat_size)

#JBF net architecture
class _JBF_net(nn.Module):
    
    def __init__(self):
        super(_JBF_net, self).__init__()
        #Denoising

        self.spat_kernel = torch.zeros(1, 1, 7, 7, 7)
        for a in range(0, 7):
            for b in range(0, 7):
                for c in range(0, 7):
                    self.spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (3, 3, 3)) ] )
        self.net_denoiser = _denoiser()
        self.JBF_block1 = _JBF_block()
        self.JBF_block2 = _JBF_block()
        self.JBF_block3 = _JBF_block()
        self.JBF_block4 = _JBF_block()
        
        #Add in parameters
        self.alfa1 = nn.Conv3d(1, 1, kernel_size = (1, 3, 3), stride = 1, padding = (0, 1, 1))
        self.alfa2 = nn.Conv3d(1, 1, kernel_size = (1, 3, 3), stride = 1, padding = (0, 1, 1))
        self.alfa3 = nn.Conv3d(1, 1, kernel_size = (1, 3, 3), stride = 1, padding = (0, 1, 1))
        self.alfa4 = nn.Conv3d(1, 1, kernel_size = (1, 3, 3), stride = 1, padding = (0, 1, 1))
        

    def forward(self, x):

        #Compute guidance image
        guide_im = self.net_denoiser(x)
        prior = guide_im.clone()
        
        #Compute filter neighborhoods
        guide_im = F.pad(guide_im, (3, 3, 3, 3, 0, 0), mode='constant')
        guide_im = guide_im.unfold(2, 7, 1).unfold(3, 7, 1).unfold(4, 7, 1).reshape(-1, 1, 7, 7, 7)
        guide_im -= guide_im[:, 0, 3, 3, 3].view(guide_im.shape[0], 1, 1, 1, 1)
        guide_im = torch.abs(guide_im)
        
        #Extract relevant part
        inp = x.clone()
        x = x[:, :, 6:9, :, :]
        
        x = F.relu(self.JBF_block1(x, self.spat_kernel.clone(), guide_im))
        x = F.relu( x + self.alfa1( x - inp[:, :, 7:8, :, :]) * ( x - inp[:, :, 7:8, :, :]) )
        #f1 = x.clone()
        
        x = torch.cat((inp[:, :, 6:7, :, :], x, inp[:, :, 8:9, :, :]), dim = 2)
        x = F.relu(self.JBF_block1(x, self.spat_kernel.clone(), guide_im))
        x = F.relu( x + self.alfa2( x - inp[:, :, 7:8, :, :]) * ( x - inp[:, :, 7:8, :, :]) )
        #f2 = x.clone()

        x = torch.cat((inp[:, :, 6:7, :, :], x, inp[:, :, 8:9, :, :]), dim = 2)
        x = F.relu(self.JBF_block1(x, self.spat_kernel.clone(), guide_im))
        x = F.relu( x + self.alfa3( x - inp[:, :, 7:8, :, :]) * ( x - inp[:, :, 7:8, :, :]) )
        #f3 = x.clone()

        x = torch.cat((inp[:, :, 6:7, :, :], x, inp[:, :, 8:9, :, :]), dim = 2)
        x = F.relu(self.JBF_block1(x, self.spat_kernel.clone(), guide_im))
        x = F.relu( x + self.alfa4( x - inp[:, :, 7:8, :, :]) * ( x - inp[:, :, 7:8, :, :]) )

        return x

class jbfnet(model):

    def __init__(self):
        model.__init__(self)
        self.jbf_net = _JBF_net().to(self.device)
        self.jbf_net.spat_kernel = self.jbf_net.spat_kernel.to(self.device)
        params = sum([np.prod(p.size()) for p in self.jbf_net.parameters()])
        print('Number of params in JBFnet: %d'%(params))    

    def infer(self, x):

        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)

        x_pad = np.pad(x, ((7, 7), (0, 0), (0, 0)), 'edge')
        
        self.jbf_net.load_state_dict(torch.load('deployable/JBFnet_30.pth'))#Please change this directory if you change the model file location!!

        for p in self.jbf_net.parameters():
            p.requires_grad = False

        self.jbf_net.eval()
        patches = self._torchify(x_pad)
        print(patches.shape)
        denoised_patches = torch.from_numpy(np.zeros_like(x)).float().unsqueeze(1)

        for i in range(7, patches.shape[0] - 7):
            with torch.no_grad():
                inp_patch = patches[i - 7:i + 8, :, :, :].unsqueeze(0).transpose(1, 2)/4096
                denoised_patches[i - 7, :, :, :] = self.jbf_net(inp_patch.to(self.device)).squeeze(0)

        denoised_patches = denoised_patches*4096
        denoised_patches = torch.round(denoised_patches)
        print('Inference complete!')
        return np.squeeze(denoised_patches.cpu().detach().numpy(), axis = 1)

    def train(self, x, y, epoch_number = 25):
        
        print('Note, this function isn\'t implemented yet!')

        for p in self.jbf_net.parameters():
            p.requires_grad = True

        dataloader, valloader = self._prepare_training_data(x, y, thickness=15, bsize=32)

        optimiser = optim.Adam(self.jbf_net.parameters(), lr=1e-4)

        #%%Train branch
        for epoch in range(epoch_number):
            running_loss = 0
            val_loss = 0
            self.jbf_net.train() 

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
                outputs1 = self.jbf_net(noisy_in)

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
            self.jbf_net.eval()
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
                    outputs1 = self.jbf_net(noisy_in)

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
                torch.save(self.red_cnn.state_dict(), 'JBFnet_'+str(epoch+1)+'.pth')#Please change this directory if you want to save elsewhere!!

        print('Training complete!')

#%%
