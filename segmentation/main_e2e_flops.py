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
    available_models = sorted(name for name in network.modeling_e2e.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling_e2e.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    
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

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

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

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            assert images.shape[0] == 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            input_shape = images.shape[-2:]

    
            def input_constructor_mask(input_res):
                return images

            nr_flops, _ = get_model_complexity_info(model.backbone, (images.shape[1], int(images.shape[2] / 8), int(images.shape[3] / 8)), input_constructor=input_constructor_mask, custom_modules_hooks={}, as_strings=False,
                                        print_per_layer_stat=False, verbose=False)

            flops.append(nr_flops)
            
    return flops


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

    # Set up model (all models are 'constructed at network.modeling_e2e)
    model = network.modeling_e2e.__dict__[opts.model](num_classes=opts.num_classes, eval=True)
    model_debug = network.modeling_e2e.__dict__[opts.model](num_classes=opts.num_classes, eval=False)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # register forward hook to part layers
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output
        return hook

    #for module in model.backbone.children():
    #    print(module)
    part_activations = model.backbone.mask_estimator.register_forward_hook(getActivation('mask'))

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        state_dict = checkpoint["model_state"]
        print(checkpoint["best_score"])
        model_debug.load_state_dict(state_dict, strict=True)

        new_state_dict = {}
        for key in state_dict.keys():
            new_key = key
            new_param = state_dict[key]
            if 'layer3_low_res' in key:
                
                if 'downsample.0' in key:
                    new_key = new_key.replace('downsample.0', 'downsample.conv1')

                if 'downsample.1' in key:
                    new_key = new_key.replace('downsample.1', 'downsample.norm1')

                if 'conv2' in key:
                    new_key = new_key.replace('conv2', 'conv2.conv')
                    c1,c2,_,_ = new_param.shape
                    new_param = new_param.view(c1,c2,9,1)

                if 'conv1' in new_key or 'conv3' in new_key:
                    new_param = new_param.squeeze(-1)
                
                #if 'downsample.0' in key:
                #    new_key = new_key.replace('downsample.0', 'downsample.conv1')
                new_key = new_key.replace('layer3_low_res', 'layer3')
            elif 'layer4_low_res' in key:
                
                if 'downsample.0' in key:
                    new_key = new_key.replace('downsample.0', 'downsample.conv1')

                if 'downsample.1' in key:
                    new_key = new_key.replace('downsample.1', 'downsample.norm1')

                if 'conv2' in key:
                    new_key = new_key.replace('conv2', 'conv2.conv')
                    c1,c2,_,_ = new_param.shape
                    new_param = new_param.view(c1,c2,9,1)

                if 'conv1' in new_key or 'conv3' in new_key:
                    new_param = new_param.squeeze(-1)

                new_key = new_key.replace('layer4_low_res', 'layer4')
            elif 'high_res' in key:
                continue
            else:
                do_nothing = 1
            new_state_dict[new_key] = new_param

        
        model.load_state_dict(new_state_dict, strict=True)
        
        model.to(device)

        model_debug = nn.DataParallel(model_debug)
        model_debug.to(device)
        
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("Checkpoint not valid")
        return 

    flops = count_flops(
        opts=opts, model=model, loader=val_loader, device=device)

    flops = np.array(flops)
    print('FLOPS backbone')
    print(np.mean(flops, axis=0))
    print(np.std(flops, axis=0))
    
    return

if __name__ == '__main__':
    main()
