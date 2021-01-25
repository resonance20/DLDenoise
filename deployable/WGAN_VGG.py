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
import torchvision

from deployable.model import model

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)

#%%WGAN-VGG archtiecture
class _gen(nn.Module):
    def __init__(self):
        super(_gen, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv5 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv6 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv7 = nn.Conv2d(32, 32, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv8 = nn.Conv2d(32, 1, kernel_size = (3, 3), stride = 1, padding = (1, 1))

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
        super(_disc, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv5 = nn.Conv2d(128, 256, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.conv6 = nn.Conv2d(256, 256, kernel_size = (3, 3), stride = 1, padding = (1, 1))
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))

        x = torch.mean(x.view(x.shape[0], x.shape[1], -1), dim = 2)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        return x

#VGG features
class _VGGfeat(nn.Module):
    def __init__(self, resize=True):
        super(_VGGfeat, self).__init__()
        self.blocks = torchvision.models.vgg19(pretrained=True).features[-1].eval()
        for p in self.blocks.parameters():
            p.requires_grad = False
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        return self.blocks(input)

def _gradient_penalty(real_data, generated_data):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data).cuda()
    alpha = alpha.cuda()
    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = Variable(interpolated, requires_grad=True).cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = disc_wgan(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return 10 * ((gradients_norm - 1) ** 2).mean()
        
class wgan_vgg(model):

    def __init__(self):
        model.__init__(self)
        self.gen = _gen().to(self.device)
        params = sum([np.prod(p.size()) for p in self.gen.parameters()])
        print('Number of params in WGAN - VGG: %d'%(params))

    def infer(self, x):

        self.gen.load_state_dict(torch.load('deployable/WGAN_gen_30.pth'))#Please change this directory if you change the model file location!!

        for p in self.gen.parameters():
            p.requires_grad = False

        self.gen.eval()
        patches = self._torchify(x)
        print(patches.shape)
        denoised_patches = torch.zeros_like(patches)

        for i in range(0, patches.shape[0]):
            with torch.no_grad():
                inp_patch = patches[i:i+1]/4096
                denoised_patches[i] = self.gen(inp_patch.to(self.device))

        denoised_patches = denoised_patches*4096
        denoised_patches = torch.round(denoised_patches)
        print('Inference complete!')
        return np.squeeze(denoised_patches.cpu().detach().numpy(), axis = 1)

    def train(self, x, y, epoch_number = 25):

        for p in self.gen.parameters():
            p.requires_grad = True

        dataloader, valloader = self._prepare_training_data(x, y, thickness=1, bsize=48)

        disc = _disc().to(self.device)
        loss_network = _VGGfeat().to(self.device)

        optim_gen = optim.Adam(self.gen.parameters(), lr=1e-4)
        optim_disc = optim.Adam(disc.parameters(), lr=1e-4)        

        for epoch in range(epoch_number):
            running_loss = 0
            val_loss = 0
            self.gen.train() 

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

                optim_gen.zero_grad()
                optim_disc.zero_grad()

                #Forward pass
                outputs1 = self.gen(noisy_in)
                fake_res = disc(outputs1).squeeze()
                real_res = disc(phantom_in).squeeze()

                #Calculate perceptual loss
                with torch.no_grad():
                    feat1 = loss_net(outputs1)
                    feat2 = loss_net(phantom_in)
                
                gen_loss =  - torch.mean(fake_res) + 0.1 * perc_loss
                disc_loss = - (torch.mean(real_res) - torch.mean(fake_res)) + _gradient_penalty(phantom_in, outputs1)

                #Backprop and update
                gen_loss.backward(retain_graph=True)
                optim_gen.step()

                disc_loss.backward()
                optim_disc.step()

                # print statistics
                running_loss += gen_loss.item()

            #Val step
            self.gen.eval()

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
                    outputs1 = self.gen(noisy_in)
                    fake_res = disc(outputs1).squeeze()

                #Calculate perceptual loss
                with torch.no_grad():
                    feat1 = loss_net(outputs1)
                    feat2 = loss_net(phantom_in)
                perc_loss = nn.MSELoss()(feat1, feat2)

                #Calculate total loss
                gen_loss =  - torch.mean(fake_res) + 0.1 * perc_loss

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
                torch.save(self.gen.state_dict(), 'WGAN_gen_'+str(epoch+1)+'.pth')#Please change this directory if you want to save elsewhere!!
                torch.save(disc.state_dict(), 'WGAN_disc_'+str(epoch+1)+'.pth')#Please change this directory if you want to save elsewhere!!

        print('Training complete!')