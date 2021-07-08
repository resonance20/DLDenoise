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
class red_cnn(cnn):

    def __init__(self):
        cnn.__init__(self)

    def run_init(self):
        self.gen = self._gen().to(self.device)
        params = sum([np.prod(p.size()) for p in self.gen.parameters()])
        print('Number of params in REDCNN: %d'%(params))

    class _gen(nn.Module):
        def __init__(self):
            super(red_cnn._gen, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size = (5, 5), stride = 1)
            self.conv2 = nn.Conv2d(32, 32, kernel_size = (5, 5), stride = 1)
            self.conv3 = nn.Conv2d(32, 32, kernel_size = (5, 5), stride = 1)
            self.conv4 = nn.Conv2d(32, 32, kernel_size = (5, 5), stride = 1)
            self.conv5 = nn.Conv2d(32, 32, kernel_size = (5, 5), stride = 1)
            self.deconv5 = nn.ConvTranspose2d(32, 32, kernel_size = (5, 5), stride = 1)
            self.deconv4 = nn.ConvTranspose2d(32, 32, kernel_size = (5, 5), stride = 1)
            self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size = (5, 5), stride = 1)
            self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size = (5, 5), stride = 1)
            self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size = (5, 5), stride = 1)

        def forward(self, x):
            #Conv
            im = x.clone()
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            im1 = x.clone()
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            im2 = x.clone()
            x = F.relu(self.conv5(x))
            #Deconv
            x = F.relu(self.deconv5(x) + im2)
            x = F.relu(self.deconv4(x))
            x = F.relu(self.deconv3(x) + im1)
            x = F.relu(self.deconv2(x))
            x = F.relu(self.deconv1(x) + im)
            return x