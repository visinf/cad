import torch
import random
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class CompressNet(nn.Module):
    def __init__(self):
        super(CompressNet, self).__init__()
        
        self.conv_last = nn.Conv2d(24,1,kernel_size=1,padding=0,stride=1)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.act(x)
        out = self.conv_last(x)
        return out

class FovSimModule(nn.Module):
    def __init__(self, in_channels=3, out_channels=24):
        # in_channels: num of channels corresponds to input image channels, e.g. 3
        # out_channels: num of channels corresponds to num of sclaes tested
        super(FovSimModule, self).__init__()
        BN_MOMENTUM = 0.1
        self.fov_expand_1 = nn.Conv2d(in_channels=in_channels, out_channels=8*out_channels, kernel_size=3, padding=1, bias=False)
        self.fov_expand_2 = nn.Conv2d(in_channels=8*out_channels, out_channels=8*out_channels, kernel_size=3, padding=1, bias=False)
        self.fov_squeeze_1 = nn.Conv2d(in_channels=8*out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        # bn
        self.norm1 = nn.BatchNorm2d(8*out_channels, momentum=BN_MOMENTUM)
        self.norm2 = nn.BatchNorm2d(8*out_channels, momentum=BN_MOMENTUM)
        self.norm3 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.act = nn.ReLU6(inplace=False)

    def forward(self, x, reset_grad=True, train_mode=True):
        layer1 = self.act(self.norm1(self.fov_expand_1(x)))
        layer2 = self.act(self.norm2(self.fov_expand_2(layer1)))
        layer3 = self.norm3(self.fov_squeeze_1(layer2))
        output = layer3
        return output


class SimpleModel(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleModel, self).__init__()
        BN_MOMENTUM = 0.1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1, bias=False)
        # bn
        self.norm1 = nn.BatchNorm2d(8, momentum=BN_MOMENTUM)
        self.norm2 = nn.BatchNorm2d(8, momentum=BN_MOMENTUM)

        self.act = nn.ReLU6(inplace=False)

    def forward(self, x):
        layer1 = self.act(self.norm1(self.conv1(x)))
        layer2 = self.act(self.norm2(self.conv2(layer1)))
        layer3 = self.conv3(layer2)
        output = layer3
        return output

def create_grid(x, grid_size_x, grid_size_y, padding_size_x, padding_size_y, P_basis, filter, segSize=None):
    #print('IN CREATE_GRID')
    #print(x.shape)
    #print(segSize)
    P = torch.autograd.Variable(torch.zeros(1,2,grid_size_x+2*padding_size_x, grid_size_y+2*padding_size_y, device=x.device),requires_grad=False)

    P[0,:,:,:] = P_basis.to(x.device) # [1,2,w,h], 2 corresponds to u(x,y) and v(x,y)
    P = P.expand(x.size(0),2,grid_size_x+2*padding_size_x, grid_size_y+2*padding_size_y)
    # input x is saliency map xs
    x_cat = torch.cat((x,x),1)
    # EXPLAIN: denominator of Eq. (3)
    p_filter = filter(x)
    x_mul = torch.mul(P,x_cat).view(-1,1,grid_size_x+2*padding_size_x,grid_size_y+2*padding_size_y)
    all_filter = filter(x_mul).view(-1,2,grid_size_x,grid_size_y)
    # EXPLAIN: numerator of Eq. (3)
    x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,grid_size_x,grid_size_y)
    y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,grid_size_x,grid_size_y)
    # EXPLAIN: Eq. (3)
    x_filter = x_filter/p_filter
    y_filter = y_filter/p_filter
    # EXPLAIN: fit F.grid_sample format (coordibates in the range [-1,1])
    xgrids = x_filter*2-1
    ygrids = y_filter*2-1
    xgrids = torch.clamp(xgrids,min=-1,max=1)
    ygrids = torch.clamp(ygrids,min=-1,max=1)
    # EXPLAIN: reshape
    xgrids = xgrids.view(-1,1,grid_size_x,grid_size_y)
    ygrids = ygrids.view(-1,1,grid_size_x,grid_size_y)
    grid = torch.cat((xgrids,ygrids),1)

    #print(self.input_size_net_infer, self.input_size_net)
    #if len(self.input_size_net_eval) != 0 and segSize is not None:# inference
    #    grid = nn.Upsample(size=self.input_size_net_infer, mode='bilinear')(grid)
    #else:
    grid = nn.Upsample(size=(grid_size_x, grid_size_y), mode='bilinear')(grid)
    # EXPLAIN: grid_y for downsampling label y, to handle segmentation architectures whose prediction are not same size with input x
    #if segSize is None:# training
    #    grid_y = nn.Upsample(size=tuple(np.array(self.input_size_net)//self.cfg.DATASET.segm_downsampling_rate), mode='bilinear')(grid)
    #else:# inference
    #    grid_y = nn.Upsample(size=tuple(np.array(self.input_size_net_infer)), mode='bilinear')(grid)

    grid = torch.transpose(grid,1,2)
    grid = torch.transpose(grid,2,3)

    #grid_y = torch.transpose(grid_y,1,2)
    #grid_y = torch.transpose(grid_y,2,3)

    return grid

def fast_resize(im, skips):
    return im[:,:,::skips,::skips]


def b_imresize(im, size, interp='bilinear'):
    return F.interpolate(im, size, mode=interp)

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
