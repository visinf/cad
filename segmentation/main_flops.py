from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import Cityscapes
from utils import ext_transforms as et
from network.backbone.ap_utils import mask2poolingMask

import torch
import torch.nn as nn
from utils.flops_counter import get_model_complexity_info



def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=available_models, help='model name')
    parser.add_argument("--output_stride", type=int, default=16, choices=[4, 8, 16, 32])
    parser.add_argument("--output_stride_from_trained", type=int, default=None, choices=[4, 8, 16, 32])
    parser.add_argument("--trimap_width", type=int, default=7,
                        help="trimap_width for edge evaluation")
    parser.add_argument("--pooling_mask_mode", default=None, type=str, choices=['edges', 'edges_os16till8', 'zeros', 'ones'],
                        help="pooling_mask_mode")
    parser.add_argument("--pooling_mask_edge_detection_treshold", type=float, default=None,
                        help="trimap_width that model uses")
                        
    # Train Options
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst, val_dst


def count_flops(opts, model, loader, device):
    """Do validation and return specified samples"""

    flops = []
    flops_mask = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            assert images.shape[0] == 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            input_shape = images.shape[-2:]

            if '_ap_' in opts.model:
                assert images.shape[0] == 1


                if '_ap_' in opts.model:
                    if opts.pooling_mask_mode == 'edges' or opts.pooling_mask_mode == 'edges_os16till8':
                        if opts.dataset == 'cityscapes' or opts.dataset == 'voc':
                            dilation = nn.Conv2d(1, 1, opts.trimap_width, padding=opts.trimap_width//2, bias=False).to(device)
                            weight = torch.ones((1,1,opts.trimap_width,opts.trimap_width)).to(device)
                            dilation.weight = nn.Parameter(weight)

                            edge_detection_x = nn.Conv2d(1, 1, 3, padding=1, bias=False).to(device)
                            weight = torch.zeros((1,1,3,3)).to(device)
                            weight[0,0,1,0] = -1.
                            weight[0,0,1,2] = +1.
                            edge_detection_x.weight = nn.Parameter(weight)

                            edge_detection_y = nn.Conv2d(1, 1, 3, padding=1, bias=False).to(device)
                            weight = torch.zeros((1,1,3,3)).to(device)
                            weight[0,0,0,1] = -1.
                            weight[0,0,2,1] = +1.
                            edge_detection_y.weight = nn.Parameter(weight)


                if '_ap_' in opts.model:
                    if opts.dataset == 'cityscapes':
                        if opts.pooling_mask_mode == 'edges' or opts.pooling_mask_mode == 'edges_os16till8':
                            images_gray = 0.299 * images[:,0,:,:] + 0.587 * images[:,1,:,:] + 0.114 * images[:,2,:,:]
                            images_gray = images_gray.unsqueeze(1)

                            edges_x = (edge_detection_x(images_gray) >= opts.pooling_mask_edge_detection_treshold).float()
                            edges_y = (edge_detection_y(images_gray) >= opts.pooling_mask_edge_detection_treshold).float()
                            edges = ((edges_x + edges_y) != 0).float()
                            edges = dilation(edges)

                if opts.pooling_mask_mode == 'edges':
                    if opts.output_stride == 16:
                        edges = edges[:,:,::16,::16]
                        pooling_mask = mask2poolingMask(edges, 1)
                        #pooling_mask = torch.zeros((edge.shape)).to(device)
                    elif opts.output_stride == 8:
                        edges = edges[:,:,::8,::8]
                        pooling_mask = mask2poolingMask(edges, 2)
                    else:
                        return NotImplementedError
                elif opts.pooling_mask_mode == 'zeros':
                    pooling_mask = torch.zeros((1,1,images.shape[2], images.shape[3])).to(images.device)
                    pooling_mask = pooling_mask[:,:,::opts.output_stride, ::opts.output_stride]
                elif opts.pooling_mask_mode == 'ones':
                    pooling_mask = torch.ones((1,1,images.shape[2], images.shape[3])).to(images.device)
                    pooling_mask = pooling_mask[:,:,::opts.output_stride, ::opts.output_stride]
                elif opts.pooling_mask_mode == 'edges_os16till8':
                    edges = edges[:,:,::8,::8]
                    pooling_mask = mask2poolingMask(edges, 2, base_os16=True)


                def input_constructor(input_res):
                    batch = torch.ones(()).new_empty((1,input_res[0],input_res[1],input_res[2])).to(device)
                    return batch, pooling_mask

            else: # no ap module
                def input_constructor(input_res):
                    batch = torch.ones(()).new_empty((1,input_res[0],input_res[1],input_res[2])).to(device)
                    return batch
            

            if '_ap_' in opts.model:
                if opts.pooling_mask_mode == 'edges' or opts.pooling_mask_mode == 'edges_os16till8':

                    def input_constructor_mask(input_res):
                        batch = torch.ones(()).new_empty((1,input_res[0],input_res[1],input_res[2])).to(device)
                        return batch

                    nr_flops_mask1, _ = get_model_complexity_info(edge_detection_x, (1, images.shape[2], images.shape[3]), input_constructor=input_constructor_mask, custom_modules_hooks={}, as_strings=False,
                                                print_per_layer_stat=False, verbose=False)
                    nr_flops_mask2, _ = get_model_complexity_info(edge_detection_y, (1, images.shape[2], images.shape[3]), input_constructor=input_constructor_mask, custom_modules_hooks={}, as_strings=False,
                                                print_per_layer_stat=False, verbose=False)
                    nr_flops_mask = nr_flops_mask1 + nr_flops_mask2

                
            if opts.pooling_mask_mode == 'zeros'  or opts.pooling_mask_mode == 'ones':
                nr_flops_mask = 0



            nr_flops, _ = get_model_complexity_info(model.backbone, (images.shape[1], images.shape[2], images.shape[3]), input_constructor=input_constructor, custom_modules_hooks={}, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)

            flops.append(nr_flops)
            flops_mask.append(nr_flops_mask)
            
    return flops, flops_mask


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

   
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)
    
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Val set: %d" %
          (opts.dataset, len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, output_stride_from_trained=opts.output_stride_from_trained, pretrained_backbone=False)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    model.to(device)
    
    
    model.eval()

    flops, flops_mask = count_flops(
        opts=opts, model=model, loader=val_loader, device=device)

    flops = np.array(flops)
    print('FLOPS backbone')
    print(np.mean(flops, axis=0))
    print(np.std(flops, axis=0))

    print('FLOPS mask')
    print(np.mean(flops_mask, axis=0))
    print(np.std(flops_mask, axis=0))
    
    return

if __name__ == '__main__':
    main()
