import numpy as np
import os
import pydicom
import cv2
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

import math
from scipy.spatial import distance
from skimage.metrics import structural_similarity, normalized_root_mse, peak_signal_noise_ratio
from pathos.pools import ParallelPool

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torchvision import models, transforms as T

from architectures import *

#Function to window an image
def window(im, centre = 40, width = 300, slope = 1, bias = -1024):

    im_modded = im*slope + bias
    im_modded[im_modded>(centre + width/2)] = centre + width/2
    im_modded[im_modded<(centre - width/2)] = centre - width/2

    return im_modded

#Slice wise structural similarity
def ssim(img1, img2, data_range=4096, clipping_ud=50, clipping_lr=20):
    return structural_similarity(img1[:, clipping_ud:-clipping_ud, clipping_lr:-clipping_lr], img2[:, clipping_ud:-clipping_ud, clipping_lr:-clipping_lr], data_range=data_range)

#Normalised root mean square error
def nrmse(img1, img2, clipping_ud=50, clipping_lr=20): 
    return normalized_root_mse(img1[:, clipping_ud:-clipping_ud, clipping_lr:-clipping_lr], img2[:, clipping_ud:-clipping_ud, clipping_lr:-clipping_lr])

#Peak signal to noise ratio
def psnr(img_true, img_comp, data_range=4096, clipping_ud=50, clipping_lr=20):
    return peak_signal_noise_ratio(img_true[:, clipping_ud:-clipping_ud, clipping_lr:-clipping_lr], img_comp[:, clipping_ud:-clipping_ud, clipping_lr:-clipping_lr], data_range=data_range)
    
#Function to read a DICOM folder
def dicom_read(path):

    #Load list of dicom files
    print(os.listdir(path))
    input('stahp')
    list_files = sorted(os.listdir(path))
    list_dicom = []
    for file in list_files:
        if file.endswith('.dcm') or file.endswith('.IMA'):
            list_dicom.append( os.path.join(path, file) )

    #Find reference values
    RefDs = pydicom.read_file(list_dicom[0])
    const_pixel_dims = (len(list_dicom), RefDs.Rows, RefDs.Columns)
    #const_pixel_dims = (len(list_dicom), 256, 256)

    #Create array and load values
    dicom_array = np.zeros(const_pixel_dims)
    for file in list_dicom:
        ds = pydicom.dcmread(file)
        im = np.array(ds.pixel_array, np.uint16)
        #Remove low frequency features
        dicom_array[list_dicom.index(file),:,:] = im
    return np.array(dicom_array, dtype=np.int16)

#sigmoid
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

#Function to find perceptual quality score
def irqm(image_array):

    image_array = np.array(image_array, dtype=np.float32)
    quality_net = IRQM_net().cuda()
    quality_net.load_state_dict(torch.load('../denoise_rl/models/IRQM3.pth'))
    quality_net.eval()

    if len(image_array.shape) == 2:
        torch_arr = torch.from_numpy(image_array).float().unsqueeze(0).unsqueeze(0)
        return quality_net(torch_arr.cuda()).squeeze().cpu().detach().numpy()

    qual = 0

    torch_arr = torch.from_numpy(image_array).float().unsqueeze(1)

    for item in torch_arr:
        with torch.no_grad():
            qual += quality_net(item.unsqueeze(1).cuda()).squeeze().cpu().detach().numpy()

    return qual/image_array.shape[0]

