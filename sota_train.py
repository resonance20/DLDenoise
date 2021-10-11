#%%Imports
import numpy as np
import os
from skimage.external import tifffile as tif

import torch
from dataloader import DenoiseDataset

from deployable.WGAN_VGG import wgan_vgg
from deployable.CPCE3D import cpce3d
from deployable.QAE import quadratic_autoencoder
from deployable.CNN import cnn
from deployable.REDCNN import red_cnn
from deployable.GAN import gan_3d

#%%Load data
"""
x = []
y = []
dir = '../reconstucted_ims/Images/LDCTchallenge/dset'
patient_list = os.listdir(dir)
for pat in patient_list:
    x_name = os.path.join(dir, pat, 'quarter.tiff')
    y_name = os.path.join(dir, pat, 'full.tiff')
    x.append(tif.imread(x_name))
    y.append(tif.imread(y_name))
x_np = np.concatenate(x, axis=0).astype(np.float32)
y_np = np.concatenate(y, axis=0).astype(np.float32)
"""

#%%Load data
fold_100 = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/data_lake/train/100/'
fold_25 = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/data_lake/train/25/'
fold_5 = 'C:/Users/z003zv1a/Desktop/Code/reconstucted_ims/data_lake/train/5/'

dataset = DenoiseDataset(fold_100=fold_100, fold_25=fold_25, fold_5=fold_5, datatype='SD')
val_size = int(0.1*len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])

#%% Train 2D models
cnn_model = cnn()
cnn_model.train(train_dataset=train_dataset, val_dataset=val_dataset, batch_size = 1024, epoch_number = 100, fname = 'deployable/CNN10/CNN_SMN')
del cnn_model
torch.cuda.empty_cache()

qae_model = quadratic_autoencoder()
qae_model.train(train_dataset=train_dataset, val_dataset=val_dataset, batch_size = 1024, epoch_number = 100, fname = 'deployable/QAE/QAE_SMN')
del qae_model
torch.cuda.empty_cache()

redcnn_model = red_cnn()
redcnn_model.train(train_dataset=train_dataset, val_dataset=val_dataset, batch_size = 1024, epoch_number = 100, fname = 'deployable/REDCNN/REDCNN_SMN')
del redcnn_model
torch.cuda.empty_cache()
"""
wgan_model = wgan_vgg()
wgan_model.train(train_dataset=train_dataset, val_dataset=val_dataset, batch_size = 128, epoch_number = 100, fname = 'deployable/WGAN_VGG/WGAN_SMN')
del wgan_model
torch.cuda.empty_cache()
"""
#%%Train 3D models
"""
cpce3d_model = cpce3d()
cpce3d_model.train(x = x_np, y = y_np, thickness = 9, batch_size = 128, epoch_number = 100, fname = 'deployable/CPCE3D/CPCE3D_AAPM')
del cpce3d_model
torch.cuda.empty_cache()

gan_model = gan_3d()
gan_model.train(x = x_np, y = y_np, thickness = 19, batch_size = 32, epoch_number = 100, fname = 'deployable/GAN3D/GAN3D_AAPM')
del gan_model
torch.cuda.empty_cache()
"""
# %%