# DLDenoise

A small library to collect trained low dose CT denoising models and make them easily usable and re-trainable.

## Models
This consists of the following models at the moment:

Model | Paper | Class Name | 2D or 3D | Train functionality tested
--- | --- | --- | --- |---
JBFnet | [Patwari et al.](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_49) | jbfnet | 3D | No
QAE| [Fan et al.](https://ieeexplore.ieee.org/document/8946589) | quadratic_autoencoder | 2D | No
CPCE3D| [Shan et al.](https://ieeexplore.ieee.org/document/8353466) | cpce3d | 3D | No
GAN| [Wolterink et al.](https://ieeexplore.ieee.org/document/7934380) | gan_3d | 3D | No
WGAN-VGG| [Yang et al.](https://ieeexplore.ieee.org/document/8340157) | wgan_vgg | 2D | No

## Example
Here is an example to denoise a single slice using QAE:
```python
from dldenoise.deployable import quadratic_autoencoder

model = quadratic_autoencoder()#No need to use CUDA, it should be automatically handled
denoised_im = model.infer(noisy_im)#You can also pass 3D numpy arrays, in which case you will recieve a denoised volume as output
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