#%%Imports
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathos.pools import ParallelPool

%reload_ext autoreload
%autoreload 2

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.external import tifffile as tif
from helpers import *

#%%Read and save LDCT challenge as .tiff files
ltime = time.time()
dir = '../reconstucted_ims/Images/LDCTchallenge/'
list_patients = os.listdir(dir)
for patient in list_patients:
    if '.' not in patient and 'L' in patient:
        noisy_data = dicom_read( os.path.join(dir, patient, 'quarter_1mm') )
        clean_data = dicom_read( os.path.join(dir, patient, 'full_1mm') )
        tif.imsave(os.path.join(dir, patient, 'quarter.tiff'), np.array(noisy_data, dtype=np.uint16))
        tif.imsave(os.path.join(dir, patient, 'full.tiff'), np.array(clean_data, dtype=np.uint16))

print('Time taken to save: %f seconds'%(time.time() - ltime))

#%%Read and save TCIA dataset as .tiff files
ltime = time.time()
dir = '../reconstucted_ims/Images/TCIAdataset/'
list_locs = os.listdir(dir)
for loc in list_locs:
    newdir = os.path.join(dir, loc, 'LDCT-and-Projection-data')
    list_patients = os.listdir(newdir)
    for patient in list_patients:
        if '.' not in patient:
            imdir = os.path.join(newdir, patient, os.listdir(os.path.join(newdir, patient))[0])
            dirs = os.listdir(imdir)
            for dir_set in dirs:
                if 'Full' in dir_set:
                    clean_data = dicom_read( os.path.join(imdir, dir_set) )
                    #tif.imsave(os.path.join(newdir, patient, 'full.tiff'), clean_data)
                    clean_data_ds = np.zeros((clean_data.shape[0], 256, 256))
                    for i, sl in enumerate(clean_data):
                        clean_data_ds[i] = cv2.resize(sl, (256, 256))
                    tif.imsave(os.path.join(newdir, patient, 'full_ds.tiff'), np.array(clean_data_ds, dtype=np.uint16))
                if 'Low' in dir_set:
                    noisy_data = dicom_read( os.path.join(imdir, dir_set) )
                    #tif.imsave(os.path.join(newdir, patient, 'quarter.tiff'), noisy_data)
                    noisy_data_ds = np.zeros((noisy_data.shape[0], 256, 256))
                    for i, sl in enumerate(noisy_data):
                        noisy_data_ds[i] = cv2.resize(sl, (256, 256))
                    tif.imsave(os.path.join(newdir, patient, 'quarter_ds.tiff'), np.array(noisy_data_ds, dtype=np.uint16))
        
print('Time taken to save: %f seconds'%(time.time() - ltime))

#%%Denoise each using BM3D
ltime = time.time()

dir = '../reconstucted_ims/Images/LDCTchallenge/'
list_patients = os.listdir(dir)
for patient in list_patients:
    if '.' not in patient:

        noisy_data = tif.imread(os.path.join(dir, patient, 'quarter.tiff'))
        noisy_slices_list = []
        sigma_list = []
        for slice in noisy_data:
            noisy_slices_list.append(slice)
            sigma_list.append(5)
        
        pool = ParallelPool(8)
        clean_by_bm3d = pool.map(bm3d.bm3d, noisy_slices_list, sigma_list)
        pool.close()
        pool.join()
        pool.clear()

        #clean_by_bm3d = np.zeros(noisy_data.shape)
        #for i, slice in enumerate(noisy_data):
        #    clean_by_bm3d[i, :, :] = bm3d.bm3d(slice, 20)    

        tif.imsave(os.path.join(dir, patient, 'bm3d.tiff'), np.array(clean_by_bm3d, dtype=np.float64))
        print('Denoised patient '+patient)

print('Time taken to denoise and save: %f seconds'%(time.time() - ltime))

# %%Test for BM3D
dir = 'reconstucted_ims/Images/LDCTchallenge/L067'
denoised_array = tif.imread(os.path.join(dir, 'bm3d.tiff'))
print(np.min(denoised_array))
print(np.max(denoised_array))

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
for ax in axes:
    ax.axis('off')
axes[0].imshow(denoised_array[30, :, :], cmap='gray')
axes[1].imshow(denoised_array[-30, :, :], cmap='gray')
plt.show()

#%%Denoise each using JBF and RL-DN
ltime = time.time()

#Include CPCE3D
cpce3d = redcnn().cuda()
cpce3d.load_state_dict(torch.load('../sota_implementation/models/CPCE3D/REDCNN_25.pth'))
cpce3d.eval()

#Include Deep Q network for parameter tuning
denoiser = deepQ().cuda()
denoiser.load_state_dict(torch.load('../denoise_dl/models/RL3D/RLNet_300.pth'))
denoiser.eval()

dir = '../reconstucted_ims/Images/LDCTchallenge/'
list_patients = os.listdir(dir)
for patient in list_patients:
    if '.' not in patient:
        noisy_data = tif.imread(os.path.join(dir, patient, 'quarter.tiff'))
        required_depth = noisy_data.shape[0]%7
        noisy_data_pad = np.pad(noisy_data, ((4, 4 + required_depth), (0, 0), (0, 0)), mode='edge')
        noisy_data_pad = torch.from_numpy(noisy_data_pad).unsqueeze(0).unsqueeze(0).float().cuda()
        
        clean_by_jbf = torch.zeros(1, 1, 1, noisy_data.shape[1] , noisy_data.shape[2] )
        #clean_by_rldn = torch.zeros(1, 1, 1, noisy_data.shape[1] , noisy_data.shape[2] )
        
        for ind in range(4, noisy_data_pad.shape[2] - 10, 7):
            
            im_cuda_rl = noisy_data_pad[:, :, ind : ind + 7, :, :]
            im_cuda_g = noisy_data_pad[:, :, ind - 4 : +ind + 11, :, :]
            im_cuda_jbf = im_cuda_rl.clone()

            sigma_i = np.zeros(im_cuda_rl.shape[2:]) + 10
            sigma_s = np.zeros(sigma_i.shape) + 2

            bsize = [3, 3, 3]
            #Pre compute spatial filter
            x_width = round(bsize[0]/2)
            y_width = round(bsize[1]/2)
            z_width = round(bsize[2]/2)
            gs_base = torch.zeros(im_cuda_rl.shape[0]* im_cuda_rl.shape[2]* im_cuda_rl.shape[3]* im_cuda_rl.shape[4], 1, bsize[2], bsize[0], bsize[1]).cuda()
            for k in range(0, bsize[0]):
                for l in range(0, bsize[1]):
                    for m in range(0, bsize[2]):
                        gs_base[:, 0, m,k,l] = torch.Tensor( [distance.euclidean((m, k, l), (z_width, x_width, y_width))] )

            #Estimate prior
            with torch.no_grad():
                prior = cpce3d(im_cuda_g)

            #Number of filtering steps
            for itr in range(4):
                
                #for loop in range(5):
                    #We choose an action for every pixel 
                #    sigma_i, sigma_s = tune_param(im_cuda_rl, sigma_i, sigma_s, denoiser, blocksize=[9, 9, 5])
                
                #Filter the images with tuned parameters
                #filtered_im = bilateral_filter_adapt(im_cuda_rl, prior, sigma_i, sigma_s, \
                #    gs_base, blocksize=[9, 9, 5])
                filtered_im_jbf = bilateral_filter_adapt(im_cuda_jbf, prior, np.zeros(sigma_i.shape) + 10, \
                    np.zeros(sigma_s.shape) + 2, gs_base, blocksize=[3, 3, 3])

                #Update
                #im_cuda_rl = filtered_im.clone()
                im_cuda_jbf = filtered_im_jbf.clone()
                #sigma_i = np.zeros(sigma_i.shape) + 10
                #sigma_s = np.zeros(sigma_s.shape) + 2
            
            #clean_by_rldn = torch.cat( (clean_by_rldn, im_cuda_rl.cpu().detach()), dim = 2 )
            clean_by_jbf = torch.cat( (clean_by_jbf, im_cuda_jbf.cpu().detach()), dim = 2 )

        print('Denoised patient '+patient)
        
        clean_by_jbf_num = clean_by_jbf.numpy()[0,0,1:clean_by_jbf.shape[2]-required_depth,:,:].round()
        #clean_by_rldn_num = clean_by_rldn.numpy()[0,0,1:clean_by_rldn.shape[2]-required_depth,:,:].round()
        tif.imsave(os.path.join(dir, patient, 'jbf.tiff'), np.array(clean_by_jbf_num, dtype=np.uint16))
        #tif.imsave(os.path.join(dir, patient, 'rlirqm.tiff'), np.array(clean_by_rldn_num, dtype=np.uint16))

print('Time taken to denoise and save: %f seconds'%(time.time() - ltime))

# %%Test for reinforcement learning denoisers
dir = 'reconstucted_ims/Images/LDCTchallenge/L096'
denoised_array = tif.imread(os.path.join(dir, 'jbf.tiff'))
print(np.min(denoised_array))
print(np.max(denoised_array))

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
for ax in axes:
    ax.axis('off')
axes[0].imshow(window(denoised_array[30, :, :]), cmap='gray')
axes[1].imshow(window(denoised_array[-30, :, :]), cmap='gray')
plt.show()

#%%Denoise using JBFnet and variants
ltime = time.time()

#Define the spatial kernel
spat_kernel = torch.zeros(1, 1, 7, 7, 7).cuda()
for a in range(0, 7):
    for b in range(0, 7):
        for c in range(0, 7):
            spat_kernel[0, 0, a, b, c] = torch.Tensor( [distance.euclidean((a, b, c), (3, 3, 3)) ] )

#Include JBFnet with different priors
jbf_net = JBF_net().cuda()
jbf_net.load_state_dict(torch.load('../denoise_dl/models/JBFNet_convAddIn/JBFNet_30.pth'))
jbf_net.eval()
jbf_net_fp = JBF_net().cuda()
jbf_net_fp.load_state_dict(torch.load('../denoise_dl/models/JBFNet_convAddIn_FrozenPrior/JBFNet_30.pth'))
jbf_net_fp.eval()
jbf_net_nopt = JBF_net().cuda()
jbf_net_nopt.load_state_dict(torch.load('../denoise_dl/models/JBFNet_convAddIn_PriorNoPt/JBFNet_30.pth'))
jbf_net_nopt.eval()

#Include JBFnet nomix and direct add
jbf_net_nomix = JBF_net_nomix().cuda()
jbf_net_nomix.load_state_dict(torch.load('../denoise_dl/models/JBFNet_noAddIn/JBFNet_20.pth'))
jbf_net_nomix.eval()
jbf_net_simplemix = JBF_net_simplemix().cuda()
jbf_net_simplemix.load_state_dict(torch.load('../denoise_dl/models/JBFNet_directAddIn/JBFNet_30.pth'))
jbf_net_simplemix.eval()

dir = '../reconstucted_ims/Images/LDCTchallenge/'
list_patients = os.listdir(dir)
for patient in list_patients:
    if '.' not in patient:
        noisy_data = tif.imread(os.path.join(dir, patient, 'quarter.tiff'))
        noisy_data_pad = np.pad(noisy_data, ((7, 7), (0, 0), (0, 0)), mode='edge')
        noisy_data_pad = torch.from_numpy(noisy_data_pad).unsqueeze(0).unsqueeze(0).float().cuda()
    
        clean_by_jbfnet = torch.zeros(1, 1, 1, noisy_data.shape[1] , noisy_data.shape[2] )
        clean_by_jbfnet_fp = torch.zeros(1, 1, 1, noisy_data.shape[1] , noisy_data.shape[2] )
        clean_by_jbfnet_nopt = torch.zeros(1, 1, 1, noisy_data.shape[1] , noisy_data.shape[2] )
        clean_by_jbfnet_nomix = torch.zeros(1, 1, 1, noisy_data.shape[1] , noisy_data.shape[2] )
        clean_by_jbfnet_simplemix = torch.zeros(1, 1, 1, noisy_data.shape[1] , noisy_data.shape[2] )
        
        for ind in range(7, noisy_data_pad.shape[2] - 7):
            with torch.no_grad():
                #jbfnet_denoised = jbf_net(noisy_data_pad[:, :, ind - 7 : ind + 8, :, :], spat_kernel)                
                jbfnet_fp_denoised = jbf_net_fp(noisy_data_pad[:, :, ind - 7 : ind + 8, :, :], spat_kernel)  
                #jbfnet_nopt_denoised = jbf_net_nopt(noisy_data_pad[:, :, ind - 7 : ind + 8, :, :], spat_kernel)  
                #jbfnet_nomix_denoised = jbf_net_nomix(noisy_data_pad[:, :, ind - 7 : ind + 8, :, :], spat_kernel)  
                #jbfnet_simplemix_denoised = jbf_net_simplemix(noisy_data_pad[:, :, ind - 7 : ind + 8, :, :], spat_kernel)  

            #clean_by_jbfnet = torch.cat( (clean_by_jbfnet, jbfnet_denoised.cpu().detach()), dim = 2 )
            clean_by_jbfnet_fp = torch.cat( (clean_by_jbfnet_fp, jbfnet_fp_denoised.cpu().detach()), dim = 2 )
            #clean_by_jbfnet_nopt = torch.cat( (clean_by_jbfnet_nopt, jbfnet_nopt_denoised.cpu().detach()), dim = 2 )
            #clean_by_jbfnet_nomix = torch.cat( (clean_by_jbfnet_nomix, jbfnet_nomix_denoised.cpu().detach()), dim = 2 )
            #clean_by_jbfnet_simplemix = torch.cat( (clean_by_jbfnet_simplemix, jbfnet_simplemix_denoised.cpu().detach()), dim = 2 )

        #clean_by_jbfnet = clean_by_jbfnet.numpy()[0,0,1:,:,:]
        clean_by_jbfnet_fp = clean_by_jbfnet_fp.numpy()[0,0,1:,:,:]
        #clean_by_jbfnet_nopt = clean_by_jbfnet_nopt.numpy()[0,0,1:,:,:]
        #clean_by_jbfnet_nomix = clean_by_jbfnet_nomix.numpy()[0,0,1:,:,:]
        #clean_by_jbfnet_simplemix = clean_by_jbfnet_simplemix.numpy()[0,0,1:,:,:]
        
        #tif.imsave(os.path.join(dir, patient, 'jbfnet.tiff'), np.array(clean_by_jbfnet, dtype=np.uint16))
        tif.imsave(os.path.join(dir, patient, 'jbfnetfp.tiff'), np.array(clean_by_jbfnet_fp, dtype=np.uint16))
        #tif.imsave(os.path.join(dir, patient, 'jbfnetnopt.tiff'), np.array(clean_by_jbfnet_nopt, dtype=np.uint16))
        #tif.imsave(os.path.join(dir, patient, 'jbfnetnomix.tiff'), np.array(clean_by_jbfnet_nomix, dtype=np.uint16))
        #tif.imsave(os.path.join(dir, patient, 'jbfnetsimplemix.tiff'), np.array(clean_by_jbfnet_simplemix, dtype=np.uint16))

        print('Denoised patient '+patient)

print('Time taken to denoise and save: %f seconds'%(time.time() - ltime))


# %%Denoise using GFnet
ltime = time.time()

gf_net = GF_net().cuda()
gf_net.load_state_dict(torch.load('../sota_implementation/models/GFNet/GFNet_30.pth'))

dir = '../reconstucted_ims/Images/LDCTchallenge/'
list_patients = os.listdir(dir)
for patient in list_patients:
    if '.' not in patient:
        noisy_data = tif.imread(os.path.join(dir, patient, 'quarter.tiff'))
        required_depth = noisy_data.shape[0]%7
        noisy_data= np.pad(noisy_data, ((4, 4 + required_depth), (0, 0), (0, 0)), mode='edge')
        noisy_data = torch.from_numpy(noisy_data).unsqueeze(0).unsqueeze(0).float().cuda()
    
        clean_by_gfnet = torch.zeros(1, 1, 1, noisy_data.shape[3] , noisy_data.shape[4] )
        
        for ind in range(4, noisy_data.shape[2] - 10, 7):
            with torch.no_grad():
                gf_denoised = gf_net(noisy_data[:, :, ind - 4:ind + 11, :, :])
            clean_by_gfnet = torch.cat( (clean_by_gfnet, gf_denoised.cpu().detach()), dim = 2 )

        clean_by_gfnet = clean_by_gfnet.numpy()[0, 0, 1:clean_by_gfnet.shape[2]-required_depth, :, :]
        tif.imsave(os.path.join(dir, patient, 'gfnet.tiff'), np.array(clean_by_gfnet, dtype=np.uint16))

        print('Denoised patient '+patient)

print('Time taken to denoise and save: %f seconds'%(time.time() - ltime))


# %%Denoise TCIA using SOTA methods
import time
import pandas as pd

%reload_ext autoreload
%autoreload 2
import os
os.chdir('../analysis')
from deployable.QAE import quadratic_autoencoder
from deployable.WGAN_VGG import wgan_vgg
from deployable.GAN import gan_3d
from deployable.CPCE3D import cpce3d
from deployable.JBFnet import jbfnet
from deployable.CNN import cnn
from deployable.REDCNN import red_cnn
from deployable.RLDN import rldn
from deployable.BM3D import bm3d_dl


model_wgan = wgan_vgg()
model_qae = quadratic_autoencoder()
model_cpce3d = cpce3d()
model_cnn = cnn()
model_gan = gan_3d()
model_redcnn = red_cnn()
model_jbfnet = jbfnet()
model_bm3d = bm3d_dl()
model_rldn = rldn()
step_cnter = pd.DataFrame()
ltime = time.time()
dir = '../reconstucted_ims/Images/TCIAdataset/'

newdir = os.path.join(dir, 'abdomen_scan', 'LDCT-and-Projection-data')
list_patients = os.listdir(newdir)
for patient in list_patients:
    if '.' not in patient:
        print(patient)
        imdir = os.path.join(newdir, patient, os.listdir(os.path.join(newdir, patient))[0])
        dirs = os.listdir(imdir)
        for dir_set in dirs:
            if 'Low' in dir_set:
                noisy_data = os.path.join(imdir, dir_set)
                model_cnn.denoise_dicom(in_folder=noisy_data, series_description='CNN10', \
                    out_folder=os.path.join(imdir, 'CNN10'), fname='deployable/CNN10/CNN_SMN_50.pth')
                model_wgan.denoise_dicom(in_folder=noisy_data, series_description='WGAN_VGG', \
                    out_folder=os.path.join(imdir, 'WGAN_VGG'), fname='deployable/WGAN_VGG/WGAN_SMN_gen_100.pth')
                model_qae.denoise_dicom(in_folder=noisy_data, series_description='QAE', \
                    out_folder=os.path.join(imdir, 'QAE'), fname='deployable/QAE/QAE_SMN_50.pth')
                model_gan.denoise_dicom(in_folder=noisy_data, series_description='GAN3D', \
                    out_folder=os.path.join(imdir, 'GAN3D'), fname='deployable/GAN3D/GAN3D_SMN_gen_100.pth')
                model_cpce3d.denoise_dicom(in_folder=noisy_data, series_description='CPCE3D', \
                    out_folder=os.path.join(imdir, 'CPCE3D'), fname='deployable/CPCE3D/CPCE3D_SMN_gen_100.pth')
                model_redcnn.denoise_dicom(in_folder=noisy_data, series_description='REDCNN', \
                    out_folder=os.path.join(imdir, 'REDCNN'), fname='deployable/REDCNN/REDCNN_SMN_50.pth')
                model_jbfnet.denoise_dicom(in_folder=noisy_data, series_description='JBFnet', \
                    out_folder=os.path.join(imdir, 'JBFnet'), fname='deployable/JBFnet/JBFnet_SMN_gen_40.pth')
                """ 
                for num_steps in range(10, 16):
                    model_rldn.denoise_dicom(in_folder=noisy_data, series_description='RLDN_Mid_'+str(num_steps), \
                                out_folder=os.path.join(imdir, 'RLDN_Mid', 'RLDN_' + str(num_steps)), fname=['../denoise_rl/models/RL3D_2domain/RLnetRainbow_20.pth',\
                                    '../denoise_rl/models/RL3D_2domain/RLnetRainbowProj_20.pth'], num_steps=num_steps)
                model_bm3d.denoise_dicom(in_folder=noisy_data, series_description='BM3D', \
                    out_folder=os.path.join(imdir, 'BM3D'), fname=None)
                """
        print('Completed '+patient)
print('Time taken to denoise using SOTA: %.3f seconds'%(time.time() - ltime))

# %% Denoise test using SOTA methods

import time

from deployable.QAE import quadratic_autoencoder
from deployable.WGAN_VGG import wgan_vgg
from deployable.GAN import gan_3d
from deployable.CPCE3D import cpce3d
from deployable.JBFnet import jbfnet
from deployable.CNN import cnn
from deployable.REDCNN import red_cnn
from deployable.RLDN import rldn
"""
model_wgan = wgan_vgg()
model_qae = quadratic_autoencoder()
model_cpce3d = cpce3d()
model_cnn = cnn()
model_gan = gan_3d()
model_redcnn = red_cnn()
model_jbfnet = jbfnet()
"""
model_rldn = rldn()
ltime = time.time()
dir = '../reconstucted_ims/test/'

list_patients = os.listdir(dir)
for patient in list_patients:
    if '.' not in patient:
        print(patient)
        imdir = os.path.join(dir, patient)
        dirs = os.listdir(imdir)
        for dir_set in dirs:
            if 'WFBP' in dir_set and os.path.isdir(os.path.join(imdir, dir_set)):
                noisy_datasets = os.path.join(imdir, dir_set)
                for dataset in os.listdir(noisy_datasets):
                    if os.path.isdir(os.path.join(noisy_datasets, dataset)) and '100' not in dataset:
                        noisy_data = os.path.join(noisy_datasets, dataset)
                        """
                        model_cnn.denoise_dicom(in_folder=noisy_data, series_description='CNN10', \
                            out_folder=os.path.join(imdir, dataset.split('_')[0]+'_CNN10'), fname='deployable/CNN10/CNN_SMN_50.pth')
                        model_wgan.denoise_dicom(in_folder=noisy_data, series_description='WGAN_VGG', \
                            out_folder=os.path.join(imdir, dataset.split('_')[0]+'_WGAN_VGG'), fname='deployable/WGAN_VGG/WGAN_SMN_gen_100.pth')
                        model_qae.denoise_dicom(in_folder=noisy_data, series_description='QAE', \
                            out_folder=os.path.join(imdir, dataset.split('_')[0]+'_QAE'), fname='deployable/QAE/QAE_SMN_50.pth')
                        model_gan.denoise_dicom(in_folder=noisy_data, series_description='GAN3D', \
                            out_folder=os.path.join(imdir, dataset.split('_')[0]+'_GAN3D'), fname='deployable/GAN3D/GAN3D_SMN_gen_100.pth')
                        model_cpce3d.denoise_dicom(in_folder=noisy_data, series_description='CPCE3D', \
                            out_folder=os.path.join(imdir, dataset.split('_')[0]+'_CPCE3D'), fname='deployable/CPCE3D/CPCE3D_SMN_gen_100.pth')
                        model_redcnn.denoise_dicom(in_folder=noisy_data, series_description='REDCNN', \
                            out_folder=os.path.join(imdir, dataset.split('_')[0]+'_REDCNN'), fname='deployable/REDCNN/REDCNN_SMN_50.pth')
                        model_jbfnet.denoise_dicom(in_folder=noisy_data, series_description='JBFnet', \
                            out_folder=os.path.join(imdir, dataset.split('_')[0]+'_JBFnet'), fname='deployable/JBFnet/JBFnet_SMN_gen_40.pth')
                        """
                        if '75' not in dataset:
                            model_rldn.denoise_dicom(in_folder=noisy_data, series_description='RLDN', \
                                out_folder=os.path.join(imdir, dataset.split('_')[0]+'_RLDN'), fname=['../denoise_rl/models/RL3D_2domain/RLnetRainbow_20.pth',\
                                    '../denoise_rl/models/RL3D_2domain/RLnetRainbowProj_20.pth'])                      
        print('Completed '+patient)
        
    
print('Time taken to denoise using SOTA: %.3f seconds'%(time.time() - ltime))
# %% Denoise task using SOTA methods

import time

from deployable.QAE import quadratic_autoencoder
from deployable.WGAN_VGG import wgan_vgg
from deployable.GAN import gan_3d
from deployable.CPCE3D import cpce3d
from deployable.JBFnet import jbfnet
from deployable.CNN import cnn
from deployable.REDCNN import red_cnn
from deployable.RLDN import rldn

model_wgan = wgan_vgg()
model_qae = quadratic_autoencoder()
model_cpce3d = cpce3d()
model_cnn = cnn()
model_gan = gan_3d()
model_redcnn = red_cnn()
model_jbfnet = jbfnet()
model_rldn = rldn()
ltime = time.time()
dir = '../reconstucted_ims/test/DetectionTask'

list_patients = os.listdir(dir)
for patient in list_patients:
    if '.' not in patient:
        print(patient)
        imdir = os.path.join(dir, patient)
        dirs = os.listdir(imdir)
        for dataset in dirs:
            if 'Low' in dataset and os.path.isdir(os.path.join(imdir, dataset)):
                    noisy_data = os.path.join(imdir, dataset)
                    
                    model_cnn.denoise_dicom(in_folder=noisy_data, series_description='CNN10', \
                        out_folder=os.path.join(imdir, 'CNN10'), fname='deployable/CNN10/CNN_SMN_50.pth')
                    model_wgan.denoise_dicom(in_folder=noisy_data, series_description='WGAN_VGG', \
                        out_folder=os.path.join(imdir, 'WGAN_VGG'), fname='deployable/WGAN_VGG/WGAN_SMN_gen_100.pth')
                    model_qae.denoise_dicom(in_folder=noisy_data, series_description='QAE', \
                        out_folder=os.path.join(imdir, 'QAE'), fname='deployable/QAE/QAE_SMN_50.pth')
                    model_gan.denoise_dicom(in_folder=noisy_data, series_description='GAN3D', \
                        out_folder=os.path.join(imdir, 'GAN3D'), fname='deployable/GAN3D/GAN3D_SMN_gen_100.pth')
                    model_cpce3d.denoise_dicom(in_folder=noisy_data, series_description='CPCE3D', \
                        out_folder=os.path.join(imdir, 'CPCE3D'), fname='deployable/CPCE3D/CPCE3D_SMN_gen_100.pth')
                    model_redcnn.denoise_dicom(in_folder=noisy_data, series_description='REDCNN', \
                        out_folder=os.path.join(imdir, 'REDCNN'), fname='deployable/REDCNN/REDCNN_SMN_50.pth')
                    model_jbfnet.denoise_dicom(in_folder=noisy_data, series_description='JBFnet', \
                        out_folder=os.path.join(imdir, 'JBFnet'), fname='deployable/JBFnet/JBFnet_SMN_gen_40.pth')
                    model_rldn.denoise_dicom(in_folder=noisy_data, series_description='RLDN', \
                        out_folder=os.path.join(imdir, 'RLDN'), fname=['../denoise_rl/models/RL3D_2domain/RLnetRainbow_20.pth',\
                            '../denoise_rl/models/RL3D_2domain/RLnetRainbowProj_20.pth'])                      
        print('Completed '+patient)
        
    
print('Time taken to denoise using SOTA: %.3f seconds'%(time.time() - ltime))# %%

# %%
