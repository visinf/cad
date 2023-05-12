import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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


class BottleneckDebug(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckDebug, self).__init__()
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
        out = x
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# the model is hard coded to have OS = 8 -> 16
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=False)
        
        self.layer3_high_res = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=True)
        self.layer4_high_res = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=True)
        
        
        

        self.dilation = 1
        self.inplanes = 512
        self.layer3_low_res = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=False)
        self.layer4_low_res = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=True)
        
        
        print('SHARING WEIGHTS...')
        for param1,param2 in zip(self.layer3_high_res.named_parameters(), self.layer3_low_res.named_parameters()):
            #if 'downsample.0' in param2[0] or 'conv' in param2[0]: # remove weight sharing in batch_norm layers because in eval mode this causes problems
            if 'conv' in param2[0]: # remove weight sharing in batch_norm layers because in eval mode this causes problems
                #print(param2[0], param1[0])
                try:
                    idx, module_name, attribute = param2[0].split('.')
                    idx = int(idx)
                    setattr(getattr(self.layer3_low_res[idx], module_name), attribute, getattr(getattr(self.layer3_high_res[idx], module_name), attribute))
                except:
                    idx, module_name, idx2, attribute = param2[0].split('.')
                    idx = int(idx)
                    idx2 = int(idx2)
                    setattr(getattr(self.layer3_low_res[idx], module_name)[idx2], attribute, getattr(getattr(self.layer3_high_res[idx], module_name)[idx2], attribute))
            elif 'downsample.0' in param2[0]:
                idx, module_name, idx2, attribute = param2[0].split('.')
                idx = int(idx)
                idx2 = int(idx2)
                setattr(getattr(self.layer3_low_res[idx], module_name)[idx2], attribute, getattr(getattr(self.layer3_high_res[idx], module_name)[idx2], attribute))
            elif 'downsample.1' in param2[0]:
                idx, module_name, idx2, attribute = param2[0].split('.')
                idx = int(idx)
                idx2 = int(idx2)
                setattr(getattr(self.layer3_low_res[idx], module_name)[idx2], attribute, getattr(getattr(self.layer3_high_res[idx], module_name)[idx2], attribute))
            elif '.bn' in param2[0]:
                idx, module_name, attribute = param2[0].split('.')
                idx = int(idx)
                setattr(getattr(self.layer3_low_res[idx], module_name), attribute, getattr(getattr(self.layer3_high_res[idx], module_name), attribute))
            else:
                print(param2[0])

        for param1,param2 in zip(self.layer4_high_res.named_parameters(), self.layer4_low_res.named_parameters()):
            #if 'downsample.0' in param2[0] or 'conv' in param2[0]: # remove weight sharing in batch_norm layers because in eval mode this causes problems
            if 'conv' in param2[0]: # remove weight sharing in batch_norm layers because in eval mode this causes problems
                
                #print(param2[0], param1[0])
                try:
                    idx, module_name, attribute = param2[0].split('.')
                    idx = int(idx)
                    setattr(getattr(self.layer4_low_res[idx], module_name), attribute, getattr(getattr(self.layer4_high_res[idx], module_name), attribute))
                except:
                    idx, module_name, idx2, attribute = param2[0].split('.')
                    idx = int(idx)
                    idx2 = int(idx2)
                    setattr(getattr(self.layer4_low_res[idx], module_name)[idx2], attribute, getattr(getattr(self.layer4_high_res[idx], module_name)[idx2], attribute))
            elif 'downsample.0' in param2[0]:
                idx, module_name, idx2, attribute = param2[0].split('.')
                idx = int(idx)
                idx2 = int(idx2)
                setattr(getattr(self.layer4_low_res[idx], module_name)[idx2], attribute, getattr(getattr(self.layer4_high_res[idx], module_name)[idx2], attribute))
            elif 'downsample.1' in param2[0]:
                idx, module_name, idx2, attribute = param2[0].split('.')
                idx = int(idx)
                idx2 = int(idx2)
                setattr(getattr(self.layer4_low_res[idx], module_name)[idx2], attribute, getattr(getattr(self.layer4_high_res[idx], module_name)[idx2], attribute))
            elif '.bn' in param2[0]:
                idx, module_name, attribute = param2[0].split('.')
                idx = int(idx)
                setattr(getattr(self.layer4_low_res[idx], module_name), attribute, getattr(getattr(self.layer4_high_res[idx], module_name), attribute))
            else:
                print(param2[0])

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

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        print(self.dilation)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x) # shape B,C,H,W

        # regular path with downsampling
        x_down = self.layer3_low_res(x) # shape B,C,H/2,H/2
        #print('x_down', x_down[0,0,:8,:8])
        x_down = self.layer4_low_res(x_down)
        x_down = x_down.repeat_interleave(2, dim=2)
        x_down = x_down.repeat_interleave(2, dim=3)

        # estimate DS mask
        downsampling_mask = self.mask_estimator(x) # shape B,C,H/2,H/2
        downsampling_mask = F.gumbel_softmax(downsampling_mask, tau=1.0, hard=True, dim=1)
        downsampling_mask = downsampling_mask.repeat_interleave(2, dim=2)
        downsampling_mask = downsampling_mask.repeat_interleave(2, dim=3)

        x_high = self.layer3_high_res(x) # shape B,C,H,H
        x_high = self.layer4_high_res(x_high)
        x_down = x_down * downsampling_mask[:,1:2,:,:]
        x_high = x_high * downsampling_mask[:,0:1,:,:]

        features = x_high + x_down

        out = OrderedDict()
        out['out'] = features
        out['mask'] = downsampling_mask
        
        return out



def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        new_state_dict = {}
        for key in state_dict.keys():
            
            if 'layer3' in key:
                new_state_dict[key.replace('layer3', 'layer3_low_res')] = state_dict[key]
                new_state_dict[key.replace('layer3', 'layer3_high_res')] = state_dict[key]
            elif 'layer4' in key:
                new_state_dict[key.replace('layer4', 'layer4_low_res')] = state_dict[key]
                new_state_dict[key.replace('layer4', 'layer4_high_res')] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]

        model.load_state_dict(new_state_dict, strict=False)
    return model


def ad_e2e_resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def ad_e2e_resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def ad_e2e_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def ad_e2e_resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def ad_e2e_resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


