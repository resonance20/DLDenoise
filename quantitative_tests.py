#%%Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import ttest_ind, ttest_rel
from skimage.external import tifffile as tif
from skimage.metrics import structural_similarity

from helpers import *
%reload_ext autoreload
%autoreload 2

torch.backends.cudnn.deterministic=True
torch.manual_seed(1)

#%%Load and ad metrics to dataframe
ltime = time.time()

clipping_ud=54
clipping_lr=22
relevant_window = [864, 1264]
#columns = ['quarter_ds', 'gan', 'cpce3d', 'wgan_vgg', 'qae', 'redcnn', 'cnn']
columns = ['rldn', 'rldn_onlysin', 'rldn_onlyvol', 'rldn_fixsin', 'rldn_fixvol', 'rldn_fix', 'rldngt']
dir = '../reconstucted_ims/Images/TCIAdataset/abdomen_scan/LDCT-and-Projection-data'

list_patients = os.listdir(dir)
list_patients = [patient for patient in list_patients if '.' not in patient]
data_frame = pd.DataFrame(index = list_patients, columns=columns)

for i, patient in enumerate(list_patients):
    #if i >= 20:
    #    break
    clean_data = tif.imread(os.path.join(dir, patient, 'full_ds.tiff'))
    clean_data = np.clip(clean_data, relevant_window[0], relevant_window[1])
    print("Number of slices : %d"%(clean_data.shape[0]))

    pat_dir = os.path.join(dir, patient, 'rlAbl')#Add ablation here
    list_methods = os.listdir(pat_dir)

    for file in list_methods:
        if file.endswith('.tiff') and any(c == file[:len(file) - 5] for c in columns): 
            loaded_data = tif.imread(os.path.join(pat_dir, file))
            
            denoiser_name = file[:len(file) - 5]

            #Padding errors quickfix
            if clean_data.shape[0] > loaded_data.shape[0]:
                clean_data = clean_data[:loaded_data.shape[0], :, :]
            elif clean_data.shape[0] < loaded_data.shape[0]:
                loaded_data = loaded_data[:clean_data.shape[0], :, :]

            #Calculate metrics
            irqm_val = irqm(loaded_data)
            loaded_data = np.clip(loaded_data, relevant_window[0], relevant_window[1])
            psnr_val = psnr(clean_data, loaded_data, data_range=relevant_window[1] - relevant_window[0], clipping_lr=clipping_lr, clipping_ud=clipping_ud)
            ssim_val = ssim(clean_data, loaded_data, data_range=relevant_window[1] - relevant_window[0], clipping_lr=clipping_lr, clipping_ud=clipping_ud)
            nrmse_val = nrmse(clean_data, loaded_data, clipping_lr=clipping_lr, clipping_ud=clipping_ud)
            
            #Store to DB
            dict_val = {'PSNR' : psnr_val,
                        'SSIM' : ssim_val,
                        'NRMSE' : nrmse_val,
                        'IRQM' : irqm_val}
            data_frame.loc[patient][denoiser_name] = dict_val
            print("method:%s psnr:%.3f ssim:%.4f nrmse:%.3f irqm:%.3f"%(file, psnr_val, ssim_val, nrmse_val, irqm_val))

    print('Collected data for patient '+patient)
    #break

data_frame.to_excel(os.path.join(dir, 'data_abla.xlsx'), columns = columns, index_label = list_patients)
print('Time taken to calculate: %f seconds'%(time.time() - ltime))

#%%Extract info from excel
def extract_metric(dataframe, denoiser, metric):
    
    series =  dataframe[denoiser]
    list_metric = np.zeros(series.size)
    for i in range(0, series.size):
        list_metric[i] = ast.literal_eval(series[i])[metric]

    return list_metric

# %% View means
dir = '../reconstucted_ims/Images/TCIAdataset/abdomen_scan/LDCT-and-Projection-data'
data_frame = pd.read_excel(os.path.join(dir, 'data_sota.xlsx'))
data_frame2 = pd.read_excel(os.path.join(dir, 'data_abla.xlsx'))

mt_name = 'SSIM'
m1 = extract_metric(data_frame, 'quarter_ds', mt_name)
m2 = extract_metric(data_frame2, 'rldn_onlysin', mt_name)
m3 = extract_metric(data_frame2, 'rldn_onlyvol', mt_name)
m4 = extract_metric(data_frame2, 'rldn_fix', mt_name)
m5 = extract_metric(data_frame2, 'rldn_fixsin', mt_name)
m6 = extract_metric(data_frame2, 'rldn_fixvol', mt_name)
m7 = extract_metric(data_frame2, 'rldngt', mt_name)
m8 = extract_metric(data_frame2, 'rldn', mt_name)

print("Mean: %.4f  Std. Dev.: %.4f"%(np.mean(m1), np.std(m1)) )
print("Mean: %.4f  Std. Dev.: %.4f"%(np.mean(m2), np.std(m2)) )
print("Mean: %.4f  Std. Dev.: %.4f"%(np.mean(m3), np.std(m3)) )
print("Mean: %.4f  Std. Dev.: %.4f"%(np.mean(m4), np.std(m4)) )
print("Mean: %.4f  Std. Dev.: %.4f"%(np.mean(m5), np.std(m5)) )
print("Mean: %.4f  Std. Dev.: %.4f"%(np.mean(m6), np.std(m6)) )
print("Mean: %.4f  Std. Dev.: %.4f"%(np.mean(m7), np.std(m7)) )
print("Mean: %.4f  Std. Dev.: %.4f"%(np.mean(m8), np.std(m8)) )

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 10))
plt.xlim(0.5, 7.5)
#plt.plot(np.linspace(1, 7, num = 7), \
#    [np.median(m1), np.median(m2), np.median(m3), np.median(m4), np.median(m5), np.median(m6), np.median(m7)])
medianprops = dict(color=(0,0,1,1), linewidth=3)
boxprops = dict(linewidth=3)
whiskerprops = dict(linewidth=3)
capprops = dict(linewidth=3)
plt.plot([0.5, 7.5], np.repeat(np.median([m1]), 2), 'r')
plt.boxplot([m2, m3, m4, m5, m6, m7, m8], showfliers=False, medianprops=medianprops, boxprops=boxprops, \
    whiskerprops=whiskerprops, capprops=capprops)
plt.xticks(ticks = np.linspace(1, 7, num = 7), labels=['GAN3D', 'CPCE\n3D', 'WGAN - \nVGG', 'QAE', 'CNN10', 'REDCNN', 'RLDN'])
plt.ylabel(mt_name)
plt.grid(which='both', axis = 'both')
plt.show()

# %%Statistical testing
dir = '../reconstucted_ims/Images/TCIAdataset/abdomen_scan/LDCT-and-Projection-data'
df1 = pd.read_excel(os.path.join(dir, 'data_sota.xlsx'))
df2 = pd.read_excel(os.path.join(dir, 'data_abla.xlsx'))
comp_par = ['rldngt', 'rldn']
#irqm1 = extract_metric(df2, comp_par[0], 'IRQM')
#irqm2 = extract_metric(df2, comp_par[1], 'IRQM')

#t, p = ttest_rel(irqm1, irqm2)

ssim1 = extract_metric(df2, comp_par[0], 'SSIM')
ssim2 = extract_metric(df2, comp_par[1], 'SSIM')

t2, p2 = ttest_rel(ssim1, ssim2)

psnr1 = extract_metric(df2, comp_par[0], 'PSNR')
psnr2 = extract_metric(df2, comp_par[1], 'PSNR')

t3, p3 = ttest_rel(psnr1, psnr2)

#print("IRQM t: %.4f p: %.4f"%(t, p) )
print("SSIM t: %.4f p: %.4f"%(t2, p2) )
print("PSNR t: %.4f p: %.4f"%(t3, p3) )

# %%SOTA comparison
dir = '../reconstucted_ims/Images/TCIAdataset/abdomen_scan/LDCT-and-Projection-data/L049'

num = 65
im_quarter = tif.imread(os.path.join(dir, 'quarter_ds.tiff'))[num, :, :]
im1 = tif.imread(os.path.join(dir, 'cpce3d.tiff'))[num, :, :]
im2 = tif.imread(os.path.join(dir, 'gan.tiff'))[num, :, :]
im3 = tif.imread(os.path.join(dir, 'wgan_vgg.tiff'))[num, :, :]
im4 = tif.imread(os.path.join(dir, 'qae.tiff'))[num, :, :]
im5 = tif.imread(os.path.join(dir, 'cnn.tiff'))[num, :, :]
im6 = tif.imread(os.path.join(dir, 'redcnn.tiff'))[num, :, :]
im7 = tif.imread(os.path.join(dir, 'rlAbl', 'rldn.tiff'))[num, :, :]
im_full = tif.imread(os.path.join(dir, 'full_ds.tiff'))[num, :, :]
#print(psnr(im5, cv2.resize(im_full, (256,256)) ))

plt.plot(np.sum(im_full, axis = 1))
print(np.sum(im_full, axis = 1))
plt.show()

title = ['(a)LDCT', '(b)CPCE3D', '(c)GAN3D', '(d)WGAN-VGG', '(e)QAE', '(f)CNN10', '(g)REDCNN', '(h)RLDN', '(i)SDCT']
im_array = [im_quarter, im1, im2, im3, im4, im5, im6, im7, im_full]

centre = 40
width = 400
plt.rcParams.update({'font.size': 30})
fig, axes = plt.subplots(3, 3, figsize=(30, 30))
for i, axr in enumerate(axes):
    for j, ax in enumerate(axr):
        if (i*3 + j) >= len(im_array):
            ax.axis('off')
            break
        im_s = im_array[i*3 + j][50:178, 80:208]
        ax.axis('off')
        ax.imshow(window(im_s, centre=centre, width=width), cmap='gray')
        ax.set_title(title[i*3 + j] + ', SSIM=%.4f'%( \
           structural_similarity(window(im_s, centre=centre, width=width) , \
               window(im_full[50:178, 80:208], centre=centre, width=width), data_range=width) ))
        #rect = patches.Rectangle((45,25),20,20,linewidth= 3 ,edgecolor='r',facecolor='none')
        #ax.add_patch(rect)
        #rect = patches.Rectangle((75,60),20,20,linewidth= 3 ,edgecolor='b',facecolor='none')
        #ax.add_patch(rect)
fig.subplots_adjust(hspace=0.1, wspace=0.0)
plt.show()

# %%Ablation
dir = '../reconstucted_ims/Images/TCIAdataset/abdomen_scan/LDCT-and-Projection-data/L150'
num = 44
im_quarter = tif.imread(os.path.join(dir, 'quarter_ds.tiff'))[num, :, :]
im1 = tif.imread(os.path.join(dir, 'rlAbl','rldn_onlysin.tiff'))[num, :, :]
im2 = tif.imread(os.path.join(dir, 'rlAbl','rldn_onlyvol.tiff'))[num, :, :]
im3 = tif.imread(os.path.join(dir, 'rlAbl','rldn_fix.tiff'))[num, :, :]
im4 = tif.imread(os.path.join(dir, 'rlAbl','rldn_fixsin.tiff'))[num, :, :]
im5 = tif.imread(os.path.join(dir, 'rlAbl', 'rldn_fixvol.tiff'))[num, :, :]
im6 = tif.imread(os.path.join(dir, 'rlAbl', 'rldngt.tiff'))[num, :, :]
im7 = tif.imread(os.path.join(dir, 'rlAbl','rldn.tiff'))[num, :, :]
im_full = tif.imread(os.path.join(dir, 'full_ds.tiff'))[num, :, :]
#print(psnr(im5, cv2.resize(im_full, (256,256)) ))

title = ['(a)LDCT', '(b)Only $filt_{sin}$', '(c)Only $filt_{vol}$', '(d)Fixed filters', \
    '(e)Fixed $filt_{sin}$', '(f)Fixed $filt_{vol}$', '(g)W/o $NET_{rew}$', '(h)RLDN', '(i)SDCT']
im_array = [im_quarter, im1, im2, im3, im4, im5, im6, im7, im_full]

centre = 50
width = 150
plt.rcParams.update({'font.size': 30})
fig, axes = plt.subplots(3, 3, figsize=(30, 30))
for i, axr in enumerate(axes):
    for j, ax in enumerate(axr):
        if (i*3 + j) >= len(im_array):
            ax.axis('off')
            break
        im_s = im_array[i*3 + j][50:178, 80:208]
        ax.axis('off')
        ax.imshow(window(im_s, centre=centre, width=width), cmap='gray')
        ax.set_title(title[i*3 + j] + ', SSIM=%.4f'%( \
           structural_similarity(window(im_s, centre=centre, width=width) , \
               window(im_full[50:178, 80:208], centre=centre, width=width), data_range=width) ))
        #rect = patches.Rectangle((45,25),20,20,linewidth= 3 ,edgecolor='r',facecolor='none')
        #ax.add_patch(rect)
        #rect = patches.Rectangle((75,60),20,20,linewidth= 3 ,edgecolor='b',facecolor='none')
        #ax.add_patch(rect)
fig.subplots_adjust(hspace=0.1, wspace=0.1)
plt.show()


# %%Head CT
dir = '../reconstucted_ims/Images/TCIAdataset/head_scan/LDCT-and-Projection-data/N100'

num = 15
im_quarter = tif.imread(os.path.join(dir, 'quarter_ds.tiff'))[num, :, :]
im1 = tif.imread(os.path.join(dir, 'cpce3d.tiff'))[num, :, :]
im2 = tif.imread(os.path.join(dir, 'gan.tiff'))[num, :, :]
im3 = tif.imread(os.path.join(dir, 'wgan_vgg.tiff'))[num, :, :]
im4 = tif.imread(os.path.join(dir, 'qae.tiff'))[num, :, :]
im5 = tif.imread(os.path.join(dir, 'cnn.tiff'))[num, :, :]
im6 = tif.imread(os.path.join(dir, 'redcnn.tiff'))[num, :, :]
im7 = tif.imread(os.path.join(dir, 'rlAbl', 'rldn.tiff'))[num, :, :]
im_full = tif.imread(os.path.join(dir, 'full_ds.tiff'))[num, :, :]
#print(psnr(im5, cv2.resize(im_full, (256,256)) ))

title = ['(a)LDCT', '(b)CPCE3D', '(c)GAN3D', '(d)WGAN-VGG', '(e)QAE', '(f)CNN10', '(g)REDCNN', '(h)RLDN', '(i)SDCT']
im_array = [im_quarter, im1, im2, im3, im4, im5, im6, im7, im_full]

centre = 32
width = 8
plt.rcParams.update({'font.size': 30})
fig, axes = plt.subplots(3, 3, figsize=(30, 30))
for i, axr in enumerate(axes):
    for j, ax in enumerate(axr):
        if (i*3 + j) >= len(im_array):
            ax.axis('off')
            break
        im_s = im_array[i*3 + j][70:198, 80:208]
        ax.axis('off')
        ax.imshow(window(im_s, centre=centre, width=width), cmap='gray')
        ax.set_title(title[i*3 + j] + ', SSIM=%.4f'%( \
            structural_similarity(window(im_s, centre=centre, width=width), \
               window(im_full[70:198, 80:208], centre=centre, width=width), data_range=width) ))
        #rect = patches.Rectangle((45,25),20,20,linewidth= 3 ,edgecolor='r',facecolor='none')
        #ax.add_patch(rect)
        #rect = patches.Rectangle((75,60),20,20,linewidth= 3 ,edgecolor='b',facecolor='none')
        #ax.add_patch(rect)
fig.subplots_adjust(hspace=0.1, wspace=0.0)
plt.show()

# %%
