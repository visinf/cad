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
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    parser.add_argument("--model", type=str, default='deeplabv3_resnet50',
                        choices=available_models, help='model name')
    
    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--exp_name", type=str, required=False,
                        help="experiment name")

    parser.add_argument("--tau", type=float, default=1, choices=[1], # is hardcoded in backbone code, so adjust manually
                        help='tau for gumbel_softmax (default: 1)')
    parser.add_argument("--low_res_active", type=float, default=0.3,
                        help='tau for gumbel_softmax (default: 1)')

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


def validate(opts, model, loader, device, metrics, ret_samples_ids=None, activation=None, criterion=None):
    """Do validation and return specified samples"""
    print('Validating...')
    metrics.reset()
    ret_samples = []
    
    interval_segm_loss = 0
    interval_mask_low_res_active = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):


            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            
            outputs = model(images)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            # get losses
            mask = activation['mask']
            mask = F.gumbel_softmax(mask, tau=opts.tau, hard=True, dim=1)
            mask_low_res_active = mask[:,1:2,:,:].sum() / mask[:,1:2,:,:].numel()
            np_mask_low_res_active = mask_low_res_active.detach().cpu().numpy()
            interval_mask_low_res_active += np_mask_low_res_active
            segm_loss = criterion(outputs, labels)            
            np_segm_loss = segm_loss.detach().cpu().numpy()
            interval_segm_loss += np_segm_loss

            if (i) % 10 == 0:
                interval_segm_loss = interval_segm_loss / 10
                interval_mask_low_res_active = interval_mask_low_res_active / 10
                print("Itrs %d, Seg. Loss=%f, Mask DS active=%f" %
                      (i, interval_segm_loss, interval_mask_low_res_active))
                interval_segm_loss = 0.0
                interval_mask_low_res_active = 0.0

        score = metrics.get_results()
    return score, ret_samples


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
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling_e2e.__dict__[opts.model](num_classes=opts.num_classes)
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

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    #print(list(model.backbone.parameters()))
    backbone_params_wo_mask_estimator = []
    for p in model.backbone.named_parameters():
        if not 'mask_estimator' in p[0]:
            backbone_params_wo_mask_estimator.append(p[1])
            

    optimizer = torch.optim.SGD(params=[
        {'params': backbone_params_wo_mask_estimator, 'lr': 0.1 * opts.lr},
        {'params': model.backbone.mask_estimator.parameters(), 'lr': 0.01*opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        
        state_dict = checkpoint["model_state"]
        model.load_state_dict(state_dict, strict=True)
        
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)


    

    # ==========   Train Loop   ==========#
    vis_sample_id = None  # sample idxs for visualization

    if opts.test_only:
        model.eval()
        
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id, activation=activation, criterion=criterion)
        print('Regular Eval')
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    interval_segm_loss = 0
    interval_mask_low_res_active = 0

    
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            
            outputs = model(images)
            
            mask = activation['mask']
            mask = F.gumbel_softmax(mask, tau=opts.tau, hard=True, dim=1)
            mask_low_res_active = mask[:,1:2,:,:].sum() / mask[:,1:2,:,:].numel()
            
            loss_mask = (mask_low_res_active - opts.low_res_active).square() 
            
            np_mask_low_res_active = mask_low_res_active.detach().cpu().numpy()
            interval_mask_low_res_active += np_mask_low_res_active


            segm_loss = criterion(outputs, labels)
            loss = segm_loss + loss_mask
            loss.backward()
            optimizer.step()

            
            np_segm_loss = segm_loss.detach().cpu().numpy()
            interval_segm_loss += np_segm_loss

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            
            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                interval_segm_loss = interval_segm_loss / 10
                interval_mask_low_res_active = interval_mask_low_res_active / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f, Seg. Loss=%f, Mask DS active=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss, interval_segm_loss, interval_mask_low_res_active))
                interval_loss = 0.0
                interval_segm_loss = 0.0
                interval_mask_low_res_active = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                
                
                save_ckpt('./latest_%s_%s_mode%s_seed%d_%s.pth' %
                        (opts.model, opts.dataset, 'end2end', opts.random_seed, opts.exp_name))
                
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id, activation=activation, criterion=criterion)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    
                    save_ckpt('./best_%s_%s_mode%s_seed%d_%s.pth' %
                        (opts.model, opts.dataset, 'end2end', opts.random_seed, opts.exp_name))
                

                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
