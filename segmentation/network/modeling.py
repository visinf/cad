import imp
from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from ._fcn import FCNHead, FCN 
from .backbone import resnet

from .backbone import ap_resnet
from .backbone import batch_ap_resnet


def _segm_resnet(name, backbone_name, num_classes, output_stride, output_stride_from_trained, pretrained_backbone, concat_mask_last_layer=False):

    if output_stride==4:
        replace_stride_with_dilation=[True, True, True]
        aspp_dilate = [24, 48, 72]
    elif output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    elif output_stride==16:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]
    elif output_stride==32:
        replace_stride_with_dilation=[False, False, False]
        aspp_dilate = [3, 6, 9]

    assert output_stride_from_trained >= output_stride
    classifier_dilation = int(output_stride_from_trained / output_stride)
    print('classifier_dilation', classifier_dilation)

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    if backbone_name == 'resnet34':
        if not concat_mask_last_layer:
            inplanes = 512
        else:
            inplanes = 513
    else:
        if not concat_mask_last_layer:
            inplanes = 2048
        else:
            inplanes = 2049
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate, classifier_dilation)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate, classifier_dilation)
    elif name=='fcn':
        return_layers = {'layer4': 'out'}
        classifier = FCNHead(inplanes, num_classes, classifier_dilation)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, concat_mask_last_layer = concat_mask_last_layer)

    if name=='deeplabv3plus' or name=='deeplabv3':
        model = DeepLabV3(backbone, classifier)
    elif name=='fcn':
        model = FCN(backbone, classifier)
    return model

def _segm_ap_resnet(name, backbone_name, num_classes, output_stride, output_stride_from_trained, pretrained_backbone):

    if output_stride==4:
        replace_with_ap=[True, True, True]
        aspp_dilate = [24, 48, 72]
    elif output_stride==8:
        replace_with_ap=[False, True, True]
        aspp_dilate = [12, 24, 36]
    elif output_stride==16:
        replace_with_ap=[False, False, True]
        aspp_dilate = [6, 12, 18]
    elif output_stride==32:
        replace_with_ap=[False, False, False]
        aspp_dilate = [3, 6, 9]

    assert output_stride_from_trained >= output_stride
    classifier_dilation = int(output_stride_from_trained / output_stride)
    print('classifier_dilation', classifier_dilation)

    backbone = ap_resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_with_ap=replace_with_ap)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate, classifier_dilation)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate, classifier_dilation)
    elif name=='fcn':
        return_layers = {'layer4': 'out'}
        classifier = FCNHead(inplanes, num_classes, classifier_dilation)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if name=='deeplabv3plus' or name=='deeplabv3':
        model = DeepLabV3(backbone, classifier)
    elif name=='fcn':
        model = FCN(backbone, classifier)
    return model


def _segm_batch_ap_resnet(name, backbone_name, num_classes, output_stride, output_stride_from_trained, pretrained_backbone, concat_mask_last_layer=False):

    if output_stride==4:
        replace_with_ap=[True, True, True]
        aspp_dilate = [24, 48, 72]
    elif output_stride==8:
        replace_with_ap=[False, True, True]
        aspp_dilate = [12, 24, 36]
    elif output_stride==16:
        replace_with_ap=[False, False, True]
        aspp_dilate = [6, 12, 18]
    elif output_stride==32:
        replace_with_ap=[False, False, False]
        aspp_dilate = [3, 6, 9]

    assert output_stride_from_trained >= output_stride
    classifier_dilation = int(output_stride_from_trained / output_stride)
    print('classifier_dilation', classifier_dilation)

    backbone = batch_ap_resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_with_ap=replace_with_ap)
    
    if not concat_mask_last_layer:
        inplanes = 2048
    else:
        inplanes = 2049
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate, classifier_dilation)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate, classifier_dilation)
    elif name=='fcn':
        return_layers = {'layer4': 'out'}
        classifier = FCNHead(inplanes, num_classes, classifier_dilation)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, concat_mask_last_layer = concat_mask_last_layer)

    if name=='deeplabv3plus' or name=='deeplabv3':
        model = DeepLabV3(backbone, classifier)
    elif name=='fcn':
        model = FCN(backbone, classifier)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, output_stride_from_trained, pretrained_backbone, concat_mask_last_layer = False):

    
    if backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer=concat_mask_last_layer)
    elif backbone.startswith('ap_resnet'):
        model = _segm_ap_resnet(arch_type, backbone, num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('batch_ap_resnet'):
        model = _segm_batch_ap_resnet(arch_type, backbone, num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer=concat_mask_last_layer)
    
    else:
        raise NotImplementedError
    return model

# Deeplab v3
def deeplabv3_resnet50(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer = concat_mask_last_layer)

def deeplabv3_ap_resnet50(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3', 'ap_resnet50', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone)

def deeplabv3_batch_ap_resnet50(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3', 'batch_ap_resnet50', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer = concat_mask_last_layer)

def deeplabv3_resnet101(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer = concat_mask_last_layer)

def deeplabv3_ap_resnet101(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3', 'ap_resnet101', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone)

def deeplabv3_batch_ap_resnet101(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3', 'batch_ap_resnet101', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer=concat_mask_last_layer)

def deeplabv3_resnet152(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3', 'resnet152', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer = concat_mask_last_layer)

def deeplabv3_ap_resnet152(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3', 'ap_resnet152', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer=concat_mask_last_layer)

def deeplabv3_batch_ap_resnet152(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3', 'batch_ap_resnet152', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer=concat_mask_last_layer)


# Deeplab v3+
def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet101(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer = concat_mask_last_layer)

def deeplabv3plus_ap_resnet101(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3plus', 'ap_resnet101', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_batch_ap_resnet101(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('deeplabv3plus', 'batch_ap_resnet101', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer = concat_mask_last_layer)

# FCN
def fcn_resnet50(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True):
    """Constructs a FCN model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for fcn.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('fcn', 'resnet50', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone)

def fcn_resnet101(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a FCN model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for fcn.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('fcn', 'resnet101', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer = concat_mask_last_layer)

def fcn_ap_resnet50(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True):
    """Constructs a FCN model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for fcn.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('fcn', 'ap_resnet50', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone)

def fcn_ap_resnet101(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True):
    """Constructs a FCN model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for fcn.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('fcn', 'ap_resnet101', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone)

def fcn_batch_ap_resnet101(num_classes=21, output_stride=8, output_stride_from_trained=None, pretrained_backbone=True, concat_mask_last_layer = False):
    """Constructs a FCN model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for fcn.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    if output_stride_from_trained == None:
        output_stride_from_trained = output_stride
    return _load_model('fcn', 'batch_ap_resnet101', num_classes, output_stride=output_stride, output_stride_from_trained=output_stride_from_trained, pretrained_backbone=pretrained_backbone, concat_mask_last_layer = concat_mask_last_layer)