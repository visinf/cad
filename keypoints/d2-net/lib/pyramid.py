import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.exceptions import EmptyTensorError
from lib.utils import interpolate_dense_features, upscale_positions
from lib.ap_utils import create_pooling_mask


def process_det_des(image, det, des, args=None):
    b, _, h_init, w_init = image.size()
    device = image.device
    assert(b == 1)

    saliency = det(image)
    y,x = det.get_keypoints_from_saliency(saliency)
    
    scores = saliency[0,0,y,x]

    if args.des == 'APD2Net':
        pooling_mask = create_pooling_mask(saliency, args.dilations, output_stride=args.output_stride)
        dense_features = des.dense_feature_extraction(image, pooling_mask)
    elif args.des == 'D2Net':
        dense_features = des.dense_feature_extraction(image)
    
    scale_factor = ((image.shape[2]-1)/(dense_features.shape[2]-1), (image.shape[3]-1)/(dense_features.shape[3]-1))
    y_down = (y / scale_factor[0]).round().long()
    x_down = (x / scale_factor[1]).round().long()
    
    descriptors = dense_features[0,:,y_down,x_down].t()
    
    placeholder_scale = torch.ones_like(y).float()
    y, x = y.unsqueeze(-1), x.unsqueeze(-1)
    placeholder_scale = placeholder_scale.unsqueeze(-1)
    keypoints = torch.stack([x,y, placeholder_scale], dim=1)
    keypoints = keypoints.squeeze(-1).float()
    
    scores = scores.squeeze(-1)
    keypoints = keypoints.cpu().numpy()
    scores = scores.cpu().numpy()
    descriptors = descriptors.cpu().numpy()

    return keypoints, scores, descriptors