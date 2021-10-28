# DLDenoise

A small library to collect trained low dose CT denoising models and make them easily usable and re-trainable.

## Models
This consists of the following models at the moment:

Model | Paper | Class Name | 2D or 3D | Train functionality tested
--- | --- | --- | --- |---
JBFnet | [Patwari et al.](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_49) | jbfnet | 3D | No
QAE| [Fan et al.](https://ieeexplore.ieee.org/document/8946589) | quadratic_autoencoder | 2D | Yes
CPCE3D| [Shan et al.](https://ieeexplore.ieee.org/document/8353466) | cpce3d | 3D | Yes
GAN| [Wolterink et al.](https://ieeexplore.ieee.org/document/7934380) | gan_3d | 3D | Yes
WGAN-VGG| [Yang et al.](https://ieeexplore.ieee.org/document/8340157) | wgan_vgg | 2D | Yes
CNN| [Chen et al.](https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-8-2-679) | cnn | 2D | Yes
REDCNN| [Chen et al.](https://ieeexplore.ieee.org/document/7947200) | red_cnn | 2D | Yes
BM3D| N/A | bm3d | 2D | N/A
RLDN| N/A | rldn | 3D | N/A

## Model Files
Trained model files on the AAPM Grand Challenge dataset are available in the folders with the respective model names.

## API Definition
There are three functions in the model APIs.
```python
@abstractmethod
def denoise_dicom(self, in_folder, out_folder, series_description, fname):
    """! Function to conduct inference on a given DICOM series. Calls _infer internally.
    @param in_folder  A string containing the location of the given DICOM folder.
    @param out_folder  A string containing the location where the denoised DICOM series will be saved.
    @param series_description   A string with the series description of the new denoised DICOM series.
    @return  None. The denoised DICOM series is written into a new folder.
    """

@abstractmethod
def _infer(self, x, fname):
    """! Function to conduct inference on a give NumPy array.
    @param x  The noisy NumPy array.
    @param fname   The location and name of the model file containing the weights. Is a string.
    @return  The denoised NumPy array.
    """
    
#Train function, to be implemented by each model
@abstractmethod
def train(self, train_dataset, val_dataset, fname, batch_size = 48, epoch_number = 30):
    """! Function to train a given model.
    @param train_dataset   The training dataset. Expected as a pair of (input, GT). This is a PyTorch dataset.
    @param val_dataset   The validation dataset. Expected as a pair of (input, GT). This is a PyTorch dataset.
    @param fname   The output file location and name. Do not append .pth to the filename, this is done automatically.
    @param batch_size  Size of each training batch.
    @param epoch_number  Number of training epochs.
    @return  None. the output models are saved with the epoch numbers appended to the filename.
    """
```

It is better practice to call ```denoise_dicom``` directly on the series instead of loading data and using ```_infer```. The resulting DICOM sreies, to the best of our knowledge, are fully DICOM compliant.

## Installation
This version can be installed and uninstalled using ```pip```. Clone the repository somewhere, navigate into the repository, and type ```pip install .```. The ```deployable``` package should now be importable system-wide.

## Example
Here is an example to denoise a single slice using QAE:
```python
from dldenoise.deployable import quadratic_autoencoder

model = quadratic_autoencoder()#No need to use CUDA, it should be automatically handled
denoised_im = model._infer(noisy_im, fname='models/QAE/QAE_30.pth')#You can also pass 3D numpy arrays, in which case you will recieve a denoised volume as output
```
The input should be a 2D NumPy array or 3D NumPy volume, of datatype float32. The values should range from 0 - 4096.

NOTE: You can pass 2D inputs to 3D networks and 3D inputs to 2D networks, this should be handled internally. Passing a 2D input to a 3D network will add a leading dimension to the output. I'll get around to fixing this later.

## Requirements
This will probably work with different platforms, but I only tested on Windows 10 with the following settings:

Python 3.7
CUDA 10.1
PyTorch 1.6

## Other Things
I'm working on including Self Attention CNN. Also, I haven't tested this thoroughly for bugs, so let me know if any breaking issues arise.
