import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

def diff_x(input, r):
    assert input.dim() == 5

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

def diff_y(input, r):
    assert input.dim() == 5

    left   = input[:, :, :, :,         r:2 * r + 1]
    middle = input[:, :, :, :, 2 * r + 1:         ] - input[:, :, :, :,           :-2 * r - 1]
    right  = input[:, :, :, :,        -1:         ] - input[:, :, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=4)
    
def diff_z(input, r):
    assert input.dim() == 5

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 5

        return diff_z(diff_y(diff_x(x.cumsum(dim=3), self.r).cumsum(dim=4), self.r).cumsum(dim=2), self.r)
    
class ConvGuidedFilter(nn.Module):
    def __init__(self, radius=1, norm=nn.BatchNorm3d):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv3d(3, 3, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=3)
        self.conv_a = nn.Sequential(nn.Conv3d(6, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 3, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, _, d_lrx, h_lrx, w_lrx = x_lr.size()
        _, _, d_hrx, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, 3, d_lrx, h_lrx, w_lrx)).fill_(1.0))
        ## mean_x
        mean_x = self.box_filter(x_lr)/N
        ## mean_y
        mean_y = self.box_filter(y_lr)/N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b
        mean_A = F.interpolate(A, (d_hrx, h_hrx, w_hrx), mode='trilinear', align_corners=True)
        mean_b = F.interpolate(b, (d_hrx, h_hrx, w_hrx), mode='trilinear', align_corners=True)

        return mean_A * x_hr + mean_b