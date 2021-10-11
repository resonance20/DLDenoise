#%%Imports
import os
import numpy as np
from sklearn.model_selection import train_test_split
import time
import pytorch_ssim
import astra
import math
import copy
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.spatial import distance

from deployable.model import model

torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.backends.cudnn.deterministic=True
torch.manual_seed(1)
#torch.cuda.empty_cache()

#%%Projector class
class projector:
    
    def __init__(self):
        
        angles = np.linspace(0, 2*np.pi, num = 2304)
        det_row = 64
        det_col = 736
        det_spac_x = 1.285839319229126
        det_spac_y = 1.0947227478027344
        source_origin = 595
        source_det = 1085.5999755859375
        self.geom_size = 353
        
        self.proj_geom = astra.create_proj_geom('cone', det_spac_x, det_spac_y, det_row, det_col, angles, source_origin, source_det - source_origin)
        self.vol_geom = astra.create_vol_geom(self.geom_size, self.geom_size, 40)
        
    def project_forward(self, im):

        im = im.squeeze()

        im_np = im.cpu().detach().numpy()
        im_np = resize(im_np, (im.shape[0], self.geom_size, self.geom_size))
        _, sin = astra.create_sino3d_gpu(im_np, self.proj_geom, self.vol_geom)
        sin = np.transpose(sin, (1, 0, 2))
        return torch.from_numpy(sin).unsqueeze(1).float()
    
    def project_back(self, sin):
        
        sin = sin.squeeze()
        sin = sin.cpu().detach().numpy()
        sin = np.transpose(sin, (1, 0, 2))
        
        sin_id = astra.data3d.create('-sino', self.proj_geom, data=sin)
        rec_id = astra.data3d.create('-vol', self.vol_geom)

        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sin_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
        
        im = np.round(astra.data3d.get(rec_id))
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sin_id)
        
        return torch.from_numpy(im).unsqueeze(0).unsqueeze(0).float().cuda()

#%%Noisy Net
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())

#%%Deep Q Net
class deepQ(nn.Module):

    def __init__(self, support1, support2, atom_size=51):

        super(deepQ, self).__init__()

        #Arch params
        self.atom_size = atom_size
        self.support1 = support1
        self.support2 = support2

        #Common params
        self.conv1 = nn.Conv3d(1, 32, kernel_size = (3,3,3))
        self.conv2 = nn.Conv3d(32, 64, kernel_size = (3,3,3))

        #selection net params
        
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size = (3,3))
        self.fc1_1 = nn.Linear(576, 128)
        self.fc2_1 = NoisyLinear(128, 128)
        #Value Layer
        self.fc2_1V = NoisyLinear(128, 1*atom_size)
        #Advantage layer
        self.fc2_1A = NoisyLinear(128, 2*atom_size)

        #tuning net params
        self.conv3_2 = nn.Conv2d(64, 128, kernel_size = (3,3))
        self.conv4_2 = nn.Conv2d(128, 128, kernel_size = (3,3))
        self.fc1_2 = nn.Linear(128, 256)
        self.fc2_2 = NoisyLinear(256, 256)
        #Value layer
        self.fc2_2V = NoisyLinear(256, 1*self.atom_size)
        #Adventage layer
        self.fc2_2A = NoisyLinear(256, 5*self.atom_size)
        

    def forward(self, x):
        dist1, dist2 = self.layers(x)
        q1 = torch.sum(dist1*self.support1, dim=2)
        q2 = torch.sum(dist2*self.support2, dim=2)
        return q1, q2

    def layers(self, x):
        #Common features
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = x.squeeze()
        feat = x.clone()

        #Selection branch
        x = F.leaky_relu(self.conv3_1(x))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1_1(x))
        x = F.leaky_relu(self.fc2_1(x))

        #Dueling + categorical for selection
        xA = self.fc2_1A(x)
        xV = self.fc2_1V(x)
        xA = xA.view(-1, 2, self.atom_size)
        xV = xV.view(-1, 1, self.atom_size)
        q1 = xA + xV - xA.mean(dim=1, keepdim=True)
        dist1 = F.softmax(q1, dim=-1)
        dist1 = dist1.clamp(min=1e-3)

        #Action branch
        feat = F.leaky_relu(self.conv3_2(feat))
        feat = F.leaky_relu(self.conv4_2(feat))
        feat = feat.view(feat.size(0), -1)
        feat = F.leaky_relu(self.fc1_2(feat))
        feat = F.leaky_relu(self.fc2_2(feat))

        #Dueling + categorical for action
        featA = self.fc2_2A(feat)
        featV = self.fc2_2V(feat)
        featA = featA.view(-1, 5, self.atom_size)
        featV = featV.view(-1, 1, self.atom_size)
        q2 = featA + featV - featA.mean(dim=1, keepdim=True)
        dist2 = F.softmax(q2, dim=-1)
        dist2 = dist2.clamp(min=1e-3)

        return dist1, dist2
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.fc2_1.reset_noise()
        self.fc2_1V.reset_noise()
        self.fc2_1A.reset_noise()
        self.fc2_2.reset_noise()
        self.fc2_2V.reset_noise()
        self.fc2_2A.reset_noise()
        
class deepQproj(nn.Module):

    def __init__(self, support1, support2, atom_size=51):

        super(deepQproj, self).__init__()

        #Arch params
        self.atom_size = atom_size
        self.support1 = support1
        self.support2 = support2

        #Common params
        self.conv1 = nn.Conv2d(1, 16, kernel_size = (3,3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size = (3,3))

        #selection net params
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size = (3,3))
        self.fc1_1 = nn.Linear(32, 64)
        self.fc2_1 = NoisyLinear(64, 64)
        #Value Layer
        self.fc2_1V = NoisyLinear(64, 1*atom_size)
        #Advantage layer
        self.fc2_1A = NoisyLinear(64, 2*atom_size)
        
        #tuning net params
        self.conv3_2 = nn.Conv2d(32, 64, kernel_size = (3,3))
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size = (3,3))
        self.fc1_2 = nn.Linear(64, 128)
        self.fc2_2 = NoisyLinear(128, 128)
        #Value layer
        self.fc2_2V = NoisyLinear(128, 1*self.atom_size)
        #Adventage layer
        self.fc2_2A = NoisyLinear(128, 5*self.atom_size)

    def forward(self, x):
        dist1, dist2 = self.layers(x)
        q1 = torch.sum(dist1*self.support1, dim=2)
        q2 = torch.sum(dist2*self.support2, dim=2)
        return q1, q2

    def layers(self, x):
        #Common features
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        feat = x.clone()

        #Selection branch
        
        x = F.leaky_relu(self.conv3_1(x))
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        x = F.leaky_relu(self.fc1_1(x))
        x = F.leaky_relu(self.fc2_1(x))

        #Dueling + categorical for selection
        xA = self.fc2_1A(x)
        xV = self.fc2_1V(x)
        xA = xA.view(-1, 2, self.atom_size)
        xV = xV.view(-1, 1, self.atom_size)
        q1 = xA + xV - xA.mean(dim=1, keepdim=True)
        dist1 = F.softmax(q1, dim=-1)
        dist1 = dist1.clamp(min=1e-3)

        #Action branch
        feat = F.leaky_relu(self.conv3_2(feat))
        feat = F.leaky_relu(self.conv4_2(feat))
        feat = torch.mean(feat.view(x.size(0), x.size(1), -1), dim=2)
        feat = F.leaky_relu(self.fc1_2(feat))
        feat = F.leaky_relu(self.fc2_2(feat))

        #Dueling + categorical for action
        featA = self.fc2_2A(feat)
        featV = self.fc2_2V(feat)
        featA = featA.view(-1, 5, self.atom_size)
        featV = featV.view(-1, 1, self.atom_size)
        q2 = featA + featV - featA.mean(dim=1, keepdim=True)
        dist2 = F.softmax(q2, dim=-1)
        dist2 = dist2.clamp(min=1e-3)

        return dist1, dist2
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.fc2_1.reset_noise()
        self.fc2_1V.reset_noise()
        self.fc2_1A.reset_noise()
        self.fc2_2.reset_noise()
        self.fc2_2V.reset_noise()
        self.fc2_2A.reset_noise()

class rldn(model):

    def __init__(self):
        model.__init__(self)
        atom_size= 51
        support1 = torch.from_numpy(np.linspace(0, 100, atom_size)).float().cuda()
        support2 = torch.from_numpy(np.linspace(0, 200, atom_size)).float().cuda()
        self.gen = deepQ(support1, support2, atom_size=atom_size).to(self.device)
        self.gen_proj = deepQproj(support1, support2, atom_size=atom_size).to(self.device)

    def _gaussian(self, x, sigma):
        constant = (2 * sigma * sigma)
        return torch.exp(- torch.square(x) / constant)/constant

    #Apply simple bilateral filter
    def _bilateral_filter_simple(self, source, sigma_i, sigma_s, gs_base, blocksize = [37,37]):
        
        with torch.no_grad():
            #Temporary converter
            sigma_i = torch.from_numpy(sigma_i).float()
            sigma_s = torch.from_numpy(sigma_s).float()
            
            #Extract overlapping patches
            x_width = math.ceil(blocksize[0]/2) - 1
            y_width = math.ceil(blocksize[1]/2) - 1

            source_pad = F.pad(source, pad = (x_width, x_width, y_width, y_width), mode = 'replicate').cuda()
            source_pad = source_pad.unfold(2, blocksize[0], 1).unfold(3, blocksize[1], 1).reshape(source_pad.shape[0], -1, blocksize[0], blocksize[1])
            
            #Find intensity differences
            source_diff = source_pad - source_pad[:, :, x_width, y_width].view(source_pad.shape[0], source_pad.shape[1], 1, 1)
            
            #Calculate Gaussians
            sigma_i = sigma_i.view(-1, 1, 1, 1).cuda()
            sigma_s = sigma_s.view(-1, 1, 1, 1).cuda()
                
            weights = self._gaussian(gs_base, sigma_s)*self._gaussian(source_diff, sigma_i)
            
            #Apply weights, normalise, and reshape
            filtered_im = (torch.sum(source_pad*weights, dim=[2, 3]) + 1e-6)/(torch.sum(weights, dim=[2, 3]) + 1e-6)
            
            return filtered_im.view(source.shape).cpu()

    #Apply the bilateral filter
    def _bilateral_filter_adapt(self, source, sigma_i_array, sigma_s_array, gs_base, blocksize = [37,37,5]):
        
        with torch.no_grad():
            #Temporary converter
            sigma_i_array = torch.from_numpy(sigma_i_array).float()
            sigma_s_array = torch.from_numpy(sigma_s_array).float()
            filtered_im = torch.zeros( torch.numel(source) ).cuda()

            #Extract overlapping patches
            x_width = math.ceil(blocksize[0]/2) - 1
            y_width = math.ceil(blocksize[1]/2) - 1
            z_width = math.ceil(blocksize[2]/2) - 1

            source_pad = F.pad(source, pad = (x_width, x_width, y_width, y_width, z_width, z_width), mode = 'replicate')
            source_patch = source_pad.unfold(2, blocksize[2], 1).unfold(3, blocksize[0], 1).unfold(4, blocksize[1], 1).reshape(-1, 1, blocksize[2], blocksize[0], blocksize[1])
            
            #Find intensity differences
            source_patch_diff = source_patch - source_patch[:,:, z_width, x_width, y_width].view(source_patch.shape[0], 1, 1, 1, 1)

            #Calculate Gaussians
            sigma_i_reshape = sigma_i_array.view(-1, 1, 1, 1, 1).cuda()
            sigma_s_reshape = sigma_s_array.view(-1, 1, 1, 1, 1).cuda()
            
            batchify = 498436
            
            for i in range(0, source_patch.shape[0], batchify):
                
                weights = self._gaussian(gs_base.repeat(batchify, 1, 1, 1, 1).cuda(), sigma_s_reshape[i: i + batchify]) * \
                    self._gaussian(source_patch_diff[i: i + batchify].cuda(), sigma_i_reshape[i: i + batchify])
                
                #Apply weights, normalise, and reshape
                filtered_im[i: i + batchify] = (torch.sum(source_patch[i: i + batchify].cuda()*weights, dim=[1, 2, 3, 4]) + 1e-6)/(torch.sum(weights, dim=[1, 2, 3, 4]) + 1e-6)
            
            return filtered_im.view(source.shape).cpu()

    #%%Function to patchwise change parameter quality
    def _tune_param_proj(self, projs, sigma_i, sigma_s):
        
        outputs1 = torch.zeros(projs.shape[0], 2)
        outputs2 = torch.zeros(projs.shape[0], 5)
        
        batchify = 24
        
        for i in range(0, projs.shape[0], batchify):
            with torch.no_grad():
                outputs1_temp, outputs2_temp = self.gen_proj(projs[i : i + batchify].cuda())
            outputs1[i : i + batchify, :] = outputs1_temp.cpu()
            outputs2[i : i + batchify, :] = outputs2_temp.cpu()

        chosen_actions = outputs2.max(1)[1].detach().numpy()
        chosen_params = outputs1.max(1)[1].detach().numpy()
            
        chosen_actions_backup = np.copy(chosen_actions)
        chosen_actions = np.where(chosen_actions==0, -0.5, chosen_actions)
        chosen_actions = np.where(chosen_actions==1, -0.1, chosen_actions)
        chosen_actions = np.where(chosen_actions==2, 0, chosen_actions)
        chosen_actions = np.where(chosen_actions==3, 0.1, chosen_actions)
        chosen_actions = np.where(chosen_actions==4, 0.5, chosen_actions)
        
        sigma_i[chosen_params==0] += sigma_i[chosen_params==0]*chosen_actions[chosen_params==0]
        sigma_s[chosen_params==1] += sigma_s[chosen_params==1]*chosen_actions[chosen_params==1]
        
        sigma_i[sigma_i<0.5] = 0.5
        sigma_i[sigma_i>25] = 25
        sigma_s[sigma_s<0.5] = 0.5
        sigma_s[sigma_s>25] = 25
        
        return sigma_i, sigma_s, chosen_params, chosen_actions_backup
        
        
    def _tune_param(self, im, sigma_i, sigma_s, blocksize=[37, 37, 5]):
        
        #Extract overlapping blocks
        x_width = math.ceil(blocksize[0]/2) - 1
        y_width = math.ceil(blocksize[1]/2) - 1
        z_width = math.ceil(blocksize[2]/2) - 1
        
        #im = im.half()
        im_pad = F.pad(im, pad = (x_width, x_width, y_width, y_width, z_width, z_width)).cuda()
        im_overlap_patch = im_pad.unfold(2, blocksize[2], 1).unfold(3, blocksize[0], 1).unfold(4, blocksize[1], 1)
        im_overlap_patch = im_overlap_patch.reshape(-1, 1, blocksize[2], blocksize[0], blocksize[1])

        #Infrence of params and actions
        outputs1 = torch.zeros(im_overlap_patch.shape[0], 2).cuda()
        outputs2 = torch.zeros(im_overlap_patch.shape[0], 5).cuda()
        
        batchify = 32768
        
        for i in range(0, im_overlap_patch.shape[0], batchify):
            with torch.no_grad():
                outputs1_temp, outputs2_temp = self.gen(im_overlap_patch[i: i + batchify])
                
            outputs1[i : i + batchify, :] = outputs1_temp
            outputs2[i : i + batchify, :] = outputs2_temp

        chosen_actions = outputs2.max(1)[1].cpu().detach().numpy()
        chosen_params = outputs1.max(1)[1].cpu().detach().numpy()
        
        #Apply parameter updates
        sigma_i = np.reshape(sigma_i, -1)
        sigma_s = np.reshape(sigma_s, -1)

        chosen_actions_backup = np.copy(chosen_actions)
        chosen_actions = np.where(chosen_actions==0, -0.5, chosen_actions)
        chosen_actions = np.where(chosen_actions==1, -0.1, chosen_actions)
        chosen_actions = np.where(chosen_actions==2, 0, chosen_actions)
        chosen_actions = np.where(chosen_actions==3, 0.1, chosen_actions)
        chosen_actions = np.where(chosen_actions==4, 0.5, chosen_actions)

        sigma_i[chosen_params==0] += sigma_i[chosen_params==0]*chosen_actions[chosen_params==0]
        sigma_s[chosen_params==1] += sigma_s[chosen_params==1]*chosen_actions[chosen_params==1]
        
        sigma_i[sigma_i<0.5] = 0.5
        sigma_i[sigma_i>25] = 25
        sigma_s[sigma_s<0.5] = 0.5
        sigma_s[sigma_s>25] = 25

        sigma_i = np.reshape(sigma_i, im.shape)
        sigma_s = np.reshape(sigma_s, im.shape)

        return sigma_i, sigma_s, chosen_params, chosen_actions_backup

    def _denoise_im(self, im, num_steps=10):

        projector_op = projector()

        #Hyper parameters
        b_size = [9, 9, 5]
        filter_size = [5, 5, 5]

        sigma_i_list = []
        sigma_s_list = []
        reward_list = []
        
        #Initial guesses
        sigma_i_proj = np.zeros(2304) + 3
        sigma_s_proj = np.zeros(2304) + 3
        sigma_orig_proj = np.copy(sigma_i_proj)

        with torch.no_grad():

            #Pre compute spatial filter
            x_width = math.ceil(filter_size[0]/2) - 1
            y_width = math.ceil(filter_size[1]/2) - 1
            z_width = math.ceil(filter_size[2]/2) - 1
            gs_base = torch.zeros(1, 1, filter_size[2], filter_size[0], filter_size[1])
            for k in range(0, filter_size[0]):
                for l in range(0, filter_size[1]):
                    for m in range(0, filter_size[2]):
                        gs_base[:, 0, m,k,l] = torch.Tensor( [distance.euclidean((m, k, l), (z_width, x_width, y_width))] )
            gs_base = gs_base.cuda()
        
        #Estimate proj filter
        projections = projector_op.project_forward(im)
        projections_bc = projections.clone()
        values_i = np.unique(sigma_i_proj)
        values_s = np.unique(sigma_s_proj)
        
        batchify = 32
        gs_base_proj = gs_base[:, :, 3].repeat(batchify, 1, 1, 1).cuda()
        
        #stime = time.time()
        for recon in range(num_steps):
            
            with torch.no_grad():
                sigma_i_proj, sigma_s_proj, _, _ = self._tune_param_proj(projections_bc, sigma_i_proj, sigma_s_proj)
                filtered_proj = torch.zeros_like(projections)
                
                for i in range(0, projections.shape[0], batchify):
                    filtered_proj[i:i + batchify] = self._bilateral_filter_simple(projections[i:i + batchify], sigma_i_proj[i:i + batchify], \
                                                                            sigma_s_proj[i:i + batchify], gs_base_proj, blocksize=[5, 5])
                projections_bc = filtered_proj.clone()
                
                new_values_i = np.unique(sigma_i_proj)
                new_values_s = np.unique(sigma_s_proj)
                if np.array_equal(new_values_i, values_i) and np.array_equal(new_values_s, values_s):
                    break
                else:
                    values_i = new_values_i
                    values_s = new_values_s

        filtered_proj = torch.zeros_like(projections)
        for i in range(0, projections.shape[0], batchify):
            filtered_proj[i:i + batchify] = self._bilateral_filter_simple(projections[i:i + batchify], sigma_orig_proj[i:i + batchify], \
                                                                    sigma_orig_proj[i:i + batchify], gs_base_proj, blocksize=[5, 5])
        filtered_im = projector_op.project_back(filtered_proj).cpu()
        #print('Time taken for sinogram: %.3fs Steps: %d'%(time.time() - stime, recon + 1))

        #Replace image with filtered image
        im = filtered_im[:, :, 12:28].clone()

        #Initial guesses
        sigma_i_guess = np.zeros((im.shape[2], im.shape[3], im.shape[4])) + 13
        sigma_s_guess = np.zeros((im.shape[2], im.shape[3], im.shape[4])) + 13
        
        #Estimate image filter
        im_bc = im.clone()
        values_i = np.unique(sigma_i_guess)
        values_s = np.unique(sigma_s_guess)

        #Estimate filter
        #stime = time.time()
        for recon in range(num_steps):

            with torch.no_grad():
                sigma_i_guess, sigma_s_guess, _, _ = self._tune_param(im_bc, sigma_i_guess, sigma_s_guess, b_size)
                #sigma_i_list.append(copy.deepcopy(sigma_i_guess))
                #sigma_s_list.append(copy.deepcopy(sigma_s_guess))

                filtered_im_cuda = self._bilateral_filter_adapt(im, sigma_i_guess, sigma_s_guess, gs_base, filter_size)
                #reward_list.append(copy.deepcopy(filtered_im_cuda.cpu().detach().numpy()))
                im_bc = filtered_im_cuda.clone()
                
                new_values_i = np.unique(sigma_i_guess)
                new_values_s = np.unique(sigma_s_guess)
                if np.array_equal(new_values_i, values_i) and np.array_equal(new_values_s, values_s):
                    break
                else:
                    values_i = new_values_i
                    values_s = new_values_s
        
        #print('Time taken for volume: %.3fs Steps: %d'%(time.time() - stime, recon + 1))
        return im_bc

    def train(self, train_dataset, val_dataset, fname, batch_size, epoch_number):
        pass

    def _infer(self, x, fname, num_steps=10):
        
        ltime = time.time()
        assert len(fname)==2

        x = self._torchify(x).unsqueeze(0).transpose(1, 2)

        self.gen.load_state_dict(torch.load(fname[0]))
        self.gen.eval()
        self.gen_proj.load_state_dict(torch.load(fname[1]))
        self.gen_proj.eval()

        im_pad = x.clone()
        if x.shape[2]%16 != 0:
            im_pad = F.pad(im_pad, (0, 0, 0, 0, 0, 16 - x.shape[2]%16), mode = 'replicate')
        im_pad = F.pad(im_pad, (0, 0, 0, 0, 12, 12), mode = 'replicate')

        im_list = im_pad.unfold(2, 40, 16).view(-1, 1, x.shape[3], x.shape[4], 40).transpose(2, 4).transpose(3, 4)
        result = torch.zeros(1, 1, 1, 353, 353)

        for i, slab in enumerate(im_list):
            denoise_slab = self._denoise_im(slab.unsqueeze(0), num_steps=num_steps)
            result = torch.cat((result, denoise_slab), dim = 2)
        result = result[0, 0, 1:x.shape[2] + 1].detach().numpy()

        print('Inference complete!')
        print('Time taken: %.3f seconds'%(time.time() - ltime))

        return resize(result, (result.shape[0], x.shape[3], x.shape[4]))