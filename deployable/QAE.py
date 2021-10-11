#%%Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from deployable.CNN import cnn

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled=True
torch.manual_seed(1)

#QAE Fan et al.
class quadratic_autoencoder(cnn):

    def __init__(self):
        cnn.__init__(self)

    def run_init(self):
        self.gen = self._gen().to(self.device)
        params = sum([np.prod(p.size()) for p in self.gen.parameters()])
        print('Number of params in QAE: %d'%(params))

    class _quadConv(nn.Module):
        def __init__(self, inp, out, pad=True):
            super(quadratic_autoencoder._quadConv, self).__init__()
            if(pad):
                self.conv1 = nn.Conv2d(inp, out, kernel_size = (3, 3), stride = 1, padding=(1, 1))
                self.conv2 = nn.Conv2d(inp, out, kernel_size = (3, 3), stride = 1, padding=(1, 1))
                self.conv3 = nn.Conv2d(inp, out, kernel_size = (3, 3), stride = 1, padding=(1, 1))
            else:
                self.conv1 = nn.Conv2d(inp, out, kernel_size = (3, 3), stride = 1)
                self.conv2 = nn.Conv2d(inp, out, kernel_size = (3, 3), stride = 1)
                self.conv3 = nn.Conv2d(inp, out, kernel_size = (3, 3), stride = 1)

        def forward(self, x):
            quad_conv = torch.add( torch.multiply(self.conv1(x), self.conv2(x)), self.conv3(torch.square(x)) )

            return quad_conv

    class _quadDeconv(nn.Module):
        def __init__(self, inp, out, pad = True):
            super(quadratic_autoencoder._quadDeconv, self).__init__()
            if(pad):
                self.conv1 = nn.ConvTranspose2d(inp, out, kernel_size = (3, 3), stride = 1, padding=(1, 1))
                self.conv2 = nn.ConvTranspose2d(inp, out, kernel_size = (3, 3), stride = 1, padding=(1, 1))
                self.conv3 = nn.ConvTranspose2d(inp, out, kernel_size = (3, 3), stride = 1, padding=(1, 1))
            else:
                self.conv1 = nn.ConvTranspose2d(inp, out, kernel_size = (3, 3), stride = 1)
                self.conv2 = nn.ConvTranspose2d(inp, out, kernel_size = (3, 3), stride = 1)
                self.conv3 = nn.ConvTranspose2d(inp, out, kernel_size = (3, 3), stride = 1)

        def forward(self, x):
            quad_deconv = torch.add( torch.multiply(self.conv1(x), self.conv2(x)), self.conv3(torch.square(x)) )

            return quad_deconv

    class _gen(nn.Module):
        def __init__(self):
            super(quadratic_autoencoder._gen, self).__init__()
            self.conv1 = quadratic_autoencoder._quadConv(1, 15)
            self.conv2 = quadratic_autoencoder._quadConv(15, 15)
            self.conv3 = quadratic_autoencoder._quadConv(15, 15)
            self.conv4 = quadratic_autoencoder._quadConv(15, 15)
            self.conv5 = quadratic_autoencoder._quadConv(15, 15, pad = False)
            self.deconv5 = quadratic_autoencoder._quadDeconv(15, 15, pad = False)
            self.deconv4 = quadratic_autoencoder._quadDeconv(15, 15)
            self.deconv3 = quadratic_autoencoder._quadDeconv(15, 15)
            self.deconv2 = quadratic_autoencoder._quadDeconv(15, 15)
            self.deconv1 = quadratic_autoencoder._quadDeconv(15, 1)

        def forward(self, x):
            #Conv
            x /= 4096
            im = x.clone()
            x = F.leaky_relu(self.conv1(x))
            x = F.leaky_relu(self.conv2(x))
            im1 = x.clone()
            x = F.leaky_relu(self.conv3(x))
            x = F.leaky_relu(self.conv4(x))
            im2 = x.clone()
            x = F.leaky_relu(self.conv5(x))
            #Deconv
            x = F.leaky_relu(torch.add(self.deconv5(x), im2) )
            x = F.leaky_relu(self.deconv4(x))
            x = F.leaky_relu(torch.add(self.deconv3(x), im1) )
            x = F.leaky_relu(self.deconv2(x))
            x = F.relu(torch.add(self.deconv1(x), im) )
            return x*4096

