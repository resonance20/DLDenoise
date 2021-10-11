#%%Imports
import os
import numpy as np
import time
import pytorch_ssim
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deployable.model import model
from scipy.spatial import distance

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)
#torch.cuda.empty_cache()

class jbfnet(model):

    def __init__(self):
        model.__init__(self)
        self.gen = self._gen().to(self.device)
        params = sum([np.prod(p.size()) for p in self.gen.parameters()])
        print('Number of params in JBFnet: %d'%(params))
        
    #%%DJBFnet architecture
    class _denoiser(nn.Module):

        def __init__(self):
            super(jbfnet._denoiser, self).__init__()
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
            super(jbfnet._JBF_block, self).__init__()
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
            weights = (domain_kernel*range_kernel)

            #Apply bilateral filter
            x = F.pad(x, (1, 1, 1, 1, 0, 0), mode='constant')
            x = x.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1).reshape(-1, 1, 3, 3, 3)
            weighted_x = weights*x
            filtered_im = weighted_x.view(weighted_x.shape[0], 1, -1).sum(2) / (weights.view(weights.shape[0], 1, -1).sum(2) + 1e-7)

            #Reshape and upsample
            return filtered_im.view(mat_size)

    #JBF net architecture
    class _gen(nn.Module):

        def __init__(self):
            super(jbfnet._gen, self).__init__()
            #Denoising

            self.spat_kernel = torch.zeros(1, 1, 7, 7, 7)
            for a in range(0, 7):
                for b in range(0, 7):
                    for c in range(0, 7):
                        self.spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (3, 3, 3)) ] )
            
            self.net_denoiser = jbfnet._denoiser()
            self.JBF_block1 = jbfnet._JBF_block()
            self.JBF_block2 = jbfnet._JBF_block()
            self.JBF_block3 = jbfnet._JBF_block()
            self.JBF_block4 = jbfnet._JBF_block()

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
            x = x[:, :, 6:-6, :, :]

            x = F.leaky_relu(self.JBF_block1(x, self.spat_kernel.clone(), guide_im))
            nm = F.leaky_relu(F.leaky_relu(self.alfa1( x - inp[:, :, 7:-7, :, :])) * ( x - inp[:, :, 7:-7, :, :]))
            x = F.leaky_relu( x + nm )
            f1 = F.relu(x).clone()

            x = torch.cat((inp[:, :, 6:7, :, :], x, inp[:, :, 8:9, :, :]), dim = 2)
            x = F.leaky_relu(self.JBF_block2(x, self.spat_kernel.clone(), guide_im))
            nm = F.leaky_relu(F.leaky_relu(self.alfa2( x - f1)) * ( x - f1))
            x = F.leaky_relu( x + nm )
            f2 = F.relu(x).clone()

            x = torch.cat((inp[:, :, 6:7, :, :], x, inp[:, :, 8:9, :, :]), dim = 2)
            x = F.leaky_relu(self.JBF_block3(x, self.spat_kernel.clone(), guide_im))
            nm = F.leaky_relu(F.leaky_relu(self.alfa3( x - f2)) * ( x - f2))
            x = F.leaky_relu( x + nm )
            f3 = F.relu(x).clone()

            x = torch.cat((inp[:, :, 6:7, :, :], x, inp[:, :, 8:9, :, :]), dim = 2)
            x = F.leaky_relu(self.JBF_block4(x, self.spat_kernel.clone(), guide_im))
            nm = F.leaky_relu(F.leaky_relu(self.alfa4( x - f3)) * ( x - f3))
            x = F.relu( x + nm )

            return x, prior, f1, f2, f3

    def _infer(self, x, fname = 'deployable/JBFnet_30.pth'):

        if len(x.shape)==2:
            x = np.expand_dims(x, axis=0)

        x_pad = np.pad(x, ((7, 7), (0, 0), (0, 0)), 'edge')
        
        self.gen.load_state_dict(torch.load(fname))
        self.gen.spat_kernel = self.gen.spat_kernel.to(self.device)

        for p in self.gen.parameters():
            p.requires_grad = False

        self.gen.eval()
        patches = self._torchify(x_pad)
        patches /= 4096
        denoised_patches = torch.zeros_like(patches).unsqueeze(1)[7:-7]

        for i in range(7, patches.shape[0] - 7):
            with torch.no_grad():
                inp_patch = patches[i - 7:i + 8, :, :, :].unsqueeze(0).transpose(1, 2)
                output, _, _, _, _ = self.gen(inp_patch.to(self.device))
                denoised_patches[i - 7, :, :, :] = output.squeeze(0)

        denoised_patches *= 4096
        denoised_patches = torch.round(denoised_patches)
        print('Inference complete!')
        return np.squeeze(denoised_patches.cpu().detach().numpy())
    

    def train(self, train_dataset, val_dataset, epoch_number = 25, batch_size = 48, fname = 'WGAN'):

        for p in self.gen.parameters():
            p.requires_grad = True
        self.gen.spat_kernel = self.gen.spat_kernel.to(self.device)

        wandb.init(project="CT-Denoise", reinit=True)
        wandb.run.name = fname.split("/")[-1]
        wandb.run.save()
        wandb.watch(self.gen)
        wandb.config.update({'epochs':epoch_number, 'batch_size':batch_size, 'lr':1e-4})

        dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

        optim_gen = optim.Adam(self.gen.parameters(), lr=1e-4)
        vgg_loss_fn = self._VGGLoss()

        #%%Train branch
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
                phantom_in /= 4096
                noisy_in /= 4096

                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                #Forward pass
                outputs1, prior, f1, f2, f3 = self.gen(noisy_in)

                #Calculate perceptual loss
                comb_loss = lambda x, y: nn.L1Loss()(x, y) + (1 - pytorch_ssim.SSIM()(x.squeeze(1), y.squeeze(1)))
                
                if epoch < 10:
                    
                    prior_loss = comb_loss(prior, phantom_in[:, :, 4:-4])
                    loss = prior_loss
                    
                else:  
                
                    main_loss = comb_loss(outputs1, phantom_in[:, :, 7:-7])
                    prior_loss = comb_loss(prior, phantom_in[:, :, 4:-4])
                    aux_loss = comb_loss(f1, phantom_in[:, :, 7:-7]) + comb_loss(f2, phantom_in[:, :, 7:-7]) + \
                        comb_loss(f3, phantom_in[:, :, 7:-7])
                    loss = main_loss + 0.1*prior_loss + 0.1*aux_loss

                #Backprop and update
                optim_gen.zero_grad()
                loss.backward()
                optim_gen.step()

            #Val step
            self.gen.eval()
            #torch.cuda.empty_cache()

            for j, data in enumerate(valloader, 0):

                #Load validation data
                phantom_in, noisy_in = data

                #Rescale
                noisy_in /= 4096
                phantom_in = phantom_in[:, :, 7:-7]
                
                phantom_in = phantom_in.to(self.device)
                noisy_in = noisy_in.to(self.device)

                #Forward pass in the validation phase
                with torch.no_grad():
                    
                    outputs1, _, _, _, _ = self.gen(noisy_in)
                    outputs1 *= 4096

                    l1_loss += nn.L1Loss()(phantom_in, outputs1)
                    mse_loss += nn.MSELoss()(phantom_in, outputs1)
                    ssim += pytorch_ssim.SSIM()(phantom_in.squeeze(1), outputs1.squeeze(1))
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

        print('Training complete!')
        wandb.run.finish()

#%%
