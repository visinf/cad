import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel

__all__ = ["FCN"]

class FCN(_SimpleSegmentationModel):
    """
    Implements FCN model from
    `"Fully Convolutional Networks for Semantic Segmentation"
    <https://arxiv.org/abs/1411.4038>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class FCNHead(nn.Sequential):
    # dilation is per default 1 when trained
    # when inference is done with different OS than training, dilation must be adjusted
    def __init__(self, in_channels, num_classes, dilation=1):
        super(FCNHead, self).__init__()

        inter_channels = in_channels // 4
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)