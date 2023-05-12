import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .ap_utils import *

try: # for torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except: # for torchvision>=0.4
    from torch.hub import load_state_dict_from_url


__all__ = ['ad_e2e_resnet18', 'ad_e2e_resnet34', 'ad_e2e_resnet50', 'ad_e2e_resnet101',
           'ad_e2e_resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def ap_conv3x3(in_planes, out_planes, stride=1, dilation=1, number_pools=0):
    """3x3 convolution with padding"""
    return GraphConv2d(in_planes, out_planes, kernel_size=3, dilation = 1,number_pools=number_pools, padding_mode='zeros', bias=False)


def ap_conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class APBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, number_pools=0, downsample=None, norm_layer=None):
        super(APBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes)
        self.downsample = downsample
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = ap_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if self.downsample is not None:
            self.conv2 = ap_conv3x3(width, width, number_pools=number_pools-1)
        else:
            self.conv2 = ap_conv3x3(width, width, number_pools=number_pools)
        self.stride_down = IrregularDownsample2d(number_downsample=number_pools) # stride of two equals taking every second 
        self.bn2 = norm_layer(width)
        self.conv3 = ap_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):


        # first offsets work on higher res because it comes before the 'stride'
        # other offsets work for remaining blockes because they come after the 'stride'
        x, pooling_mask, offsets_first, offsets_others = x

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        if self.downsample is not None:
            out = self.conv2(out, offsets_first)
            out = self.stride_down(out, pooling_mask)
        else:
            out = self.conv2(out, offsets_others)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample((x, pooling_mask))

        out += identity
        out = self.relu(out)

        return (out, pooling_mask, offsets_first, offsets_others)

class APDownsample(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, expansion, number_pools=0, norm_layer=None):
        super(APDownsample, self).__init__()
        
        self.conv1 = ap_conv1x1(inplanes, planes * expansion)
        self.down = IrregularDownsample2d(number_downsample=number_pools) # stride of two equals taking every second 
        self.norm1 = norm_layer(planes * expansion)
        

    def forward(self, x):

        x, pooling_mask = x

        x = self.down(x, pooling_mask) # down can come first because of 1x1 conv
        x = self.conv1(x)
        x = self.norm1(x)

        return x

class APResNet(nn.Module):

    def __init__(self, block, ap_block, layers, num_classes=1000, zero_init_residual=False,
                replace_with_ap=None, norm_layer=None):
        super(APResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.number_pool = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=False)


        self.layer3 = self._make_ap_layer(ap_block, 256, layers[2], stride=2)
        self.layer4 = self._make_ap_layer(ap_block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.mask_estimator = nn.Sequential(
            nn.Conv2d(128*4, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        #else:
        #    downsample = nn.Sequential(
        #        conv1x1(self.inplanes, planes * block.expansion, stride),
        #        norm_layer(planes * block.expansion),
        #    )

        layers = []
        #print(previous_dilation)
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        print(self.dilation)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_ap_layer(self, block, planes, blocks, stride=1):
        #norm_layer = self._norm_layer
        norm_layer = nn.BatchNorm1d
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            assert stride == 1 or stride == 2
            if stride == 1:
                downsample = nn.Sequential(
                    ap_conv1x1(self.inplanes, planes * block.expansion),                    
                    norm_layer(planes * block.expansion),
                )
            elif stride == 2:
                downsample = APDownsample(self.inplanes, planes, block.expansion, self.number_pool, norm_layer)
            else:
                return NotImplementedError

        layers = []
        layers.append(block(self.inplanes, planes, number_pools=self.number_pool, downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                number_pools=self.number_pool, norm_layer=norm_layer))


        self.number_pool += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x) # shape B,C,H,W



        # estimate DS mask
        
        downsampling_mask = self.mask_estimator(x) # shape B,C,H/2,H/2
        downsampling_mask = F.gumbel_softmax(downsampling_mask, tau=1., hard=True, dim=1)
        downsampling_mask = downsampling_mask.repeat_interleave(2, dim=2)
        downsampling_mask = downsampling_mask.repeat_interleave(2, dim=3)
        
        downsampling_mask = downsampling_mask[:,1:2,:,:] # 1 where downsampling happens, 0 where no downsampling happens
        

        x = img2graph(x)

        offsets_first = self.layer3[0].conv2.get_offsets(downsampling_mask) # should correspond to custom conv
        offsets_others = self.layer3[1].conv2.get_offsets(downsampling_mask)
        x = self.layer3((x, downsampling_mask, offsets_first, offsets_others))[0]

        offsets_first = self.layer4[0].conv2.get_offsets(downsampling_mask) # should correspond to custom conv
        offsets_others = self.layer4[1].conv2.get_offsets(downsampling_mask)
        x = self.layer4((x, downsampling_mask, offsets_first, offsets_others))[0]
     
        features = graph2img(x, downsampling_mask, 2)


        out = OrderedDict()
        out['out'] = features
        out['mask'] = downsampling_mask

        return out


def _ap_resnet(arch, block, ap_block, layers, pretrained, progress, **kwargs):
    model = APResNet(block, ap_block, layers, **kwargs)
    
    return model



def ad_e2e_resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ap_resnet('resnet18', BasicBlock, APBottleneck, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def ad_e2e_resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ap_resnet('resnet34', BasicBlock, APBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def ad_e2e_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ap_resnet('resnet50', Bottleneck, APBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def ad_e2e_resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ap_resnet('resnet101', Bottleneck, APBottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def ad_e2e_resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _ap_resnet('resnet152', Bottleneck, APBottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


