#%%Imports
import matplotlib.pyplot as plt
import time
import numpy as np
from numpy.lib.type_check import real

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad, Variable
import torch.nn.functional as F
import torch.utils.data
import torchvision
import math

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

    #VGG features
    class _VGGfeat(nn.Module):
        def __init__(self):
            super(wgan_vgg._VGGfeat, self).__init__()
            vgg19_model = torchvision.models.vgg19(pretrained=True).eval()
            self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
            self.feature_extractor.eval()

        def forward(self, x):
            if len(x.shape)==5:
                x = x.squeeze(2)
            x = x.repeat(1, 3, 1, 1)
            out = self.feature_extractor(x)
            return out
    
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

    def infer(self, x, fname = 'deployable/WGAN_gen_30.pth'):

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

    def train(self, x, y, thickness = 1, epoch_number = 25, batch_size = 48, fname = 'WGAN'):

        for p in self.gen.parameters():
            p.requires_grad = True

        dataloader, valloader = self._prepare_training_data(x, y, thickness=thickness, bsize=batch_size)
        
        self.disc = self._disc().to(self.device)
        self.loss_net = self._VGGfeat().to(self.device)

        params = sum([np.prod(p.size()) for p in self.disc.parameters()])
        print('Number of params in discriminator: %d'%(params))

        optim_gen = optim.Adam(self.gen.parameters(), lr=1e-5, betas=(0.5, 0.9))
        optim_disc = optim.Adam(self.disc.parameters(), lr=1e-5, betas=(0.5, 0.9))        

        for epoch in range(epoch_number):
            running_loss = 0
            mse_loss_val = 0
            vgg_loss_val = 0
            self.gen.train() 

            start = time.time()
            iterator = iter(dataloader)

            #Train step
            for i in range(0, int(len(dataloader)/5)):

                for _ in range(4):

                    disc_loss = 0

                    #Load data and initialise optimisers
                    phantom_in, noisy_in = next(iterator)

                    #Rescale
                    if thickness > 1:
                        cont = math.floor((thickness - 1)/2)
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

                    #Run update
                    disc_loss = torch.mean(fake_res) - torch.mean(real_res) + self._gradient_penalty(phantom_in, outputs1)

                    optim_disc.zero_grad()
                    disc_loss.backward()
                    optim_disc.step()

                #Load data and initialise optimisers
                phantom_in, noisy_in = next(iterator)

                #Rescale
                if thickness > 1:
                    cont = math.floor((thickness - 1)/2)
                    phantom_in = phantom_in[:, :, cont:-cont]

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                #Gen update step
                outputs1 = self.gen(noisy_in)
                shrinkage = phantom_in.shape[-1] - outputs1.shape[-1]
                if shrinkage != 0:
                    cont = math.floor(shrinkage/2)
                    if thickness > 1:
                        phantom_in = phantom_in[:, :, :, cont:-cont, cont:-cont]
                    else:
                        phantom_in = phantom_in[:, :, cont:-cont, cont:-cont]
                fake_res = self.disc(outputs1).squeeze()
                """
                fig, axes = plt.subplots(1, 3)
                axes[0].imshow(noisy_in[0, 0, 4].cpu().detach().numpy(), cmap='gray')
                axes[1].imshow(phantom_in[0, 0, 0].cpu().detach().numpy(), cmap='gray')
                axes[2].imshow(noisy_in[0, 0, 2].cpu().detach().numpy(), cmap='gray')
                plt.show()
                """
                #Calculate perceptual loss
                feat1 = self.loss_net(outputs1/4095)
                feat2 = self.loss_net(phantom_in/4095)
                perc_loss = nn.MSELoss()(feat1, feat2)
                
                #Run update
                optim_gen.zero_grad()
                gen_loss = 0.1 * perc_loss - torch.mean(fake_res)
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

                #Rescale
                if thickness > 1:
                    cont = math.floor((thickness - 1)/2)
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
                    mse_loss = nn.MSELoss()(outputs1, phantom_in)

                    #Calculate perceptual loss
                    feat1 = self.loss_net(outputs1/4095*255)
                    feat2 = self.loss_net(phantom_in/4095*255)
                    perc_loss = nn.MSELoss()(feat1, feat2)

                # print statistics
                mse_loss_val += mse_loss.item()
                vgg_loss_val += perc_loss.item()

            print('[%d, %5d] train_loss: %f val_mse_loss: %f val_vgg_loss: %f' %
                    (epoch + 1, i + 1, running_loss / int(i+1), mse_loss_val / int(j+1), vgg_loss_val / int(j+1) ))
            running_loss = 0.0
            mse_loss_val = 0.0
            vgg_loss_val = 0.0

            end = time.time()
            print('Time taken for epoch: '+str(end-start)+' seconds')

            if (epoch+1)%5 == 0:
                print('Saving model...')
                torch.save(self.gen.state_dict(), fname+'_gen_'+str(epoch+1)+'.pth')
                torch.save(self.disc.state_dict(), fname+'_disc_'+str(epoch+1)+'.pth')

        print('Training complete!')