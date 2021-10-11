#%%Imports
import matplotlib.pyplot as plt
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad, Variable
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

import math
import pytorch_ssim
import wandb

from deployable.model import model

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)

#WGAN - VGG Yang et al.
class wgan_vgg(model):

    def __init__(self):
        model.__init__(self)
        self.disc = None
        self.loss_net = None
        self.run_init()

    def run_init(self):
        self.gen = self._gen().to(self.device)
        params = sum([np.prod(p.size()) for p in self.gen.parameters()])
        print('Number of params in WGAN_VGG: %d'%(params))

    #WGAN-VGG archtiecture
    class _gen(nn.Module):
        def __init__(self):
            super(wgan_vgg._gen, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size = (3, 3), stride = 1)
            self.conv2 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1)
            self.conv3 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1)
            self.conv4 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1)
            self.conv5 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1)
            self.conv6 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1)
            self.conv7 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1)
            self.conv8 = nn.Conv2d(32, 1, kernel_size = (3, 3), stride = 1)

        def forward(self, x):
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = F.leaky_relu(self.conv5(x))
            x = F.leaky_relu(self.conv6(x))
            x = F.leaky_relu(self.conv7(x))
            x = F.relu(self.conv8(x))

            return x

    #Discriminator architecture
    class _disc(nn.Module):
        def __init__(self):
            super(wgan_vgg._disc, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size = (3, 3), stride = 1)
            self.conv2 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 2)
            self.conv3 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride = 1)
            self.conv4 = nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 2)
            self.conv5 = nn.Conv2d(128, 256, kernel_size = (3, 3), stride = 1)
            self.conv6 = nn.Conv2d(256, 256, kernel_size = (3, 3), stride = 2)
            self.fc1 = nn.Linear(256, 256)
            self.fc2 = nn.Linear(256, 1)

        def forward(self, x):
            if len(x.shape)==5:
                x = x.squeeze(2)
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            x = F.leaky_relu(self.conv5(x))
            x = F.leaky_relu(self.conv6(x))

            #x = torch.mean(x.view(x.shape[0], x.shape[1], -1), dim=2)
            x = torch.mean(x.view(x.size(0), x.size(1), -1), dim = 2)
            x = F.leaky_relu(self.fc1(x))
            x = self.fc2(x)

            return x
    
    def _gradient_penalty(self, real_data, generated_data):

        if len(real_data.shape)==5:
            real_data = real_data.squeeze(2)
        if len(generated_data.shape)==5:
            generated_data = generated_data.squeeze(2)
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data).to(self.device)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.disc(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                                grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                                create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return 10 * ((gradients_norm - 1) ** 2).mean()

    def _infer(self, x, fname = 'deployable/WGAN_gen_30.pth'):

        self.gen.load_state_dict(torch.load(fname))

        for p in self.gen.parameters():
            p.requires_grad = False

        self.gen.eval()
        patches = self._torchify(x)
        denoised_patches = torch.zeros_like(patches)
        patches = F.pad(patches, (8, 8, 8, 8))

        for i in range(0, patches.shape[0]):
            with torch.no_grad():
                inp_patch = patches[i:i+1]
                denoised_patches[i] = self.gen(inp_patch.to(self.device))

        denoised_patches = torch.clamp(torch.round(denoised_patches), 0, 4096)
        print('Inference complete!')
        return np.squeeze(denoised_patches.cpu().detach().numpy(), axis = 1)

    def train(self, train_dataset, val_dataset, epoch_number = 25, batch_size = 48, fname = 'WGAN'):

        for p in self.gen.parameters():
            p.requires_grad = True
            
        wandb.init(project="CT-Denoise", reinit=True)
        wandb.run.name = fname.split("/")[-1]
        wandb.run.save()
        wandb.watch(self.gen)
        wandb.config.update({'epochs':epoch_number, 'batch_size':batch_size, 'lr':1e-5, 'betas':(0.5, 0.9)})

        dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        
        self.disc = self._disc().to(self.device)

        params = sum([np.prod(p.size()) for p in self.disc.parameters()])
        print('Number of params in discriminator: %d'%(params))

        optim_gen = optim.Adam(self.gen.parameters(), lr=1e-5, betas=(0.5, 0.9))
        optim_disc = optim.Adam(self.disc.parameters(), lr=1e-5, betas=(0.5, 0.9))
        vgg_loss_fn = self._VGGLoss()

        for epoch in range(epoch_number):
            
            l1_loss = 0
            mse_loss = 0
            vgg_loss = 0
            ssim = 0

            self.gen.train() 

            start = time.time()

            #Train step
            for i, data in enumerate(dataloader, 0):

                #Load data and initialise optimisers
                phantom_in, noisy_in = data

                #Rescale
                if len(phantom_in.shape) == 5:
                    cont = math.floor((phantom_in.shape[2] - 1)/2)
                    phantom_in = phantom_in[:, :, cont:-cont]

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                #Disc update step
                outputs1 = self.gen(noisy_in)
                shrinkage = phantom_in.shape[-1] - outputs1.shape[-1]
                if shrinkage != 0:
                    cont = math.floor(shrinkage/2)
                    if thickness > 1:
                        phantom_in = phantom_in[:, :, :, cont:-cont, cont:-cont]
                    else:
                        phantom_in = phantom_in[:, :, cont:-cont, cont:-cont]
                fake_res = self.disc(outputs1).squeeze()
                real_res = self.disc(phantom_in).squeeze()

                if i%5!=0:
                    #Run update
                    optim_disc.zero_grad()
                    disc_loss = torch.mean(fake_res) - torch.mean(real_res) + self._gradient_penalty(phantom_in, outputs1)
                    disc_loss.backward()
                    optim_disc.step()

                else:
                    #Run update
                    optim_gen.zero_grad()
                    gen_loss = 0.1 * vgg_loss_fn(outputs1/4095, phantom_in/4095) - torch.mean(fake_res)
                    gen_loss.backward()
                    optim_gen.step()

            #Val step
            self.gen.eval()

            for j, data in enumerate(valloader, 0):

                #Load validation data
                phantom_in, noisy_in = data

                #Rescale
                if len(phantom_in.shape) == 5:
                    cont = math.floor((phantom_in.shape[2] - 1)/2)
                    phantom_in = phantom_in[:, :, cont:-cont]

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                #Forward pass
                with torch.no_grad():
                    outputs1 = self.gen(noisy_in)

                    shrinkage = phantom_in.shape[-1] - outputs1.shape[-1]
                    if shrinkage != 0:
                        cont = math.floor(shrinkage/2)
                        if thickness > 1:
                            phantom_in = phantom_in[:, :, :, cont:-cont, cont:-cont]
                        else:
                            phantom_in = phantom_in[:, :, cont:-cont, cont:-cont]

                    l1_loss += nn.L1Loss()(phantom_in, outputs1)
                    mse_loss += nn.MSELoss()(phantom_in, outputs1)
                    if len(phantom_in.shape)==5:
                        ssim += pytorch_ssim.SSIM()(phantom_in.squeeze(2), outputs1.squeeze(2))
                        vgg_loss += vgg_loss_fn(phantom_in.squeeze(2), outputs1.squeeze(2))
                    else:
                        ssim += pytorch_ssim.SSIM()(phantom_in, outputs1)
                        vgg_loss += vgg_loss_fn(phantom_in, outputs1)


            end = time.time()
            print('Time taken for epoch: '+str(end-start)+' seconds')
            
            wandb.log({
                'l1_loss':l1_loss/(j+1),
                'mse_loss':mse_loss/(j+1),
                'vgg_loss':vgg_loss/(j+1),
                'ssim':ssim/(j+1)
            })

            if (epoch+1)%5 == 0:
                print('Saving model...')
                torch.save(self.gen.state_dict(), fname+'_gen_'+str(epoch+1)+'.pth')
                torch.save(self.disc.state_dict(), fname+'_disc_'+str(epoch+1)+'.pth')

        print('Training complete!')
        wandb.run.finish()