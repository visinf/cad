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

from network.backbone.ap_utils import batch_mask2poolingMask

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
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16, 32])
    parser.add_argument("--output_stride_from_trained", type=int, default=None, choices=[8, 16, 32])
    parser.add_argument("--pooling_mask_mode", type=str, default='gt_edges', choices=['edges', 'edges_os16till8'],
                        help="pooling_mask_mode")
    parser.add_argument("--trimap_width", type=int, default=7,
                        help="trimap_width")
    parser.add_argument("--pooling_mask_edge_detection_treshold", type=float, default=0.25,
                                    help="pooling_mask_edge_detection_treshold")

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


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    print('Validating...')
    metrics.reset()
    ret_samples = []

    if '_ap_' in opts.model:
        if opts.pooling_mask_mode == 'edges' or opts.pooling_mask_mode == 'edges_os16till8':
            if opts.dataset == 'cityscapes':
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
            else:
                return NotImplementedError
        


    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):


            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            if '_ap_' in opts.model:
                if opts.dataset == 'cityscapes':
                    if opts.pooling_mask_mode == 'edges' or opts.pooling_mask_mode == 'edges_os16till8':
                        images_gray = 0.299 * images[:,0,:,:] + 0.587 * images[:,1,:,:] + 0.114 * images[:,2,:,:] 
                        images_gray = images_gray.unsqueeze(1)

                        edges_x = (edge_detection_x(images_gray) >= opts.pooling_mask_edge_detection_treshold).float()
                        edges_y = (edge_detection_y(images_gray) >= opts.pooling_mask_edge_detection_treshold).float()
                        edges = ((edges_x + edges_y) != 0).float()
                        edges = dilation(edges)
                    
                    else:
                        return NotImplementedError

                else:
                    return NotImplementedError

                if opts.pooling_mask_mode == 'edges':
                    if opts.output_stride == 16:
                        edges = edges[:,:,::16,::16]
                        pooling_mask = batch_mask2poolingMask(edges, 1)
                    elif opts.output_stride == 8:
                        edges = edges[:,:,::8,::8]
                        pooling_mask = batch_mask2poolingMask(edges, 2)
                    else:
                        return NotImplementedError
                elif opts.pooling_mask_mode == 'edges_os16till8':
                    assert opts.output_stride == 8
                    edges = edges[:,:,::8,::8]
                    pooling_mask = batch_mask2poolingMask(edges, 2, base_os16=True)
                
                else:
                    return NotImplementedError


            
            if '_ap_' in opts.model:
                outputs = model(images, pooling_mask)
            else:
                outputs = model(images)

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            
        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
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
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, output_stride_from_trained=opts.output_stride_from_trained, concat_mask_last_layer = False)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    
    # Set up optimizer
   
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
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

    #utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        print(checkpoint["best_score"])
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)


    # ==========   Train Loop   ==========#
    vis_sample_id = None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print('Regular Eval')
        print(metrics.to_str(val_score))
        return

    interval_loss = 0

    if '_ap_' in opts.model:
        if opts.pooling_mask_mode == 'edges' or opts.pooling_mask_mode == 'edges_os16till8':
            if opts.dataset == 'cityscapes':
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
            else:
                return NotImplementedError


    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            if '_ap_' in opts.model:
                if opts.dataset == 'cityscapes':
                    if opts.pooling_mask_mode == 'edges' or opts.pooling_mask_mode == 'edges_os16till8':
                        images_gray = 0.299 * images[:,0,:,:] + 0.587 * images[:,1,:,:] + 0.114 * images[:,2,:,:] 
                        images_gray = images_gray.unsqueeze(1)

                        edges_x = (edge_detection_x(images_gray) >= opts.pooling_mask_edge_detection_treshold).float()
                        edges_y = (edge_detection_y(images_gray) >= opts.pooling_mask_edge_detection_treshold).float()
                        edges = ((edges_x + edges_y) != 0).float()
                        edges = dilation(edges)
                    
                    else:
                        return NotImplementedError
                
                else:
                    return NotImplementedError

                if opts.pooling_mask_mode == 'edges':
                    if opts.output_stride == 16:
                        edges = edges[:,:,::16,::16]
                        pooling_mask = batch_mask2poolingMask(edges, 1)
                    elif opts.output_stride == 8:
                        edges = edges[:,:,::8,::8]
                        pooling_mask = batch_mask2poolingMask(edges, 2)
                    else:
                        return NotImplementedError
                elif opts.pooling_mask_mode == 'edges_os16till8':
                    assert opts.output_stride == 8
                    edges = edges[:,:,::8,::8]
                    pooling_mask = batch_mask2poolingMask(edges, 2, base_os16=True)
                

            

            
            if '_ap_' in opts.model:
                outputs = model(images, pooling_mask)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                
                if '_ap_' in opts.model:
                    save_ckpt('./latest_%s_%s_os%d_mode%s_seed%d_%s.pth' %
                            (opts.model, opts.dataset, opts.output_stride, opts.pooling_mask_mode, opts.random_seed, opts.exp_name))
                else:
                    if opts.output_stride_from_trained is not None:
                        classifier_dilation = int(opts.output_stride_from_trained / opts.output_stride)
                    else:
                        classifier_dilation = 1
                    
                    if classifier_dilation == 1:
                        save_ckpt('./latest_%s_%s_os%d_seed%d.pth' %
                            (opts.model, opts.dataset, opts.output_stride, opts.random_seed))
                    else:
                        save_ckpt('./latest_%s_%s_os%d_dilation%d.pth' %
                            (opts.model, opts.dataset, opts.output_stride, classifier_dilation))
                
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    
                    if '_ap_' in opts.model:
                        save_ckpt('./best_%s_%s_os%d_mode%s_seed%d_%s.pth' %
                            (opts.model, opts.dataset, opts.output_stride, opts.pooling_mask_mode, opts.random_seed, opts.exp_name))
                    else:
                        if opts.output_stride_from_trained is not None:
                            classifier_dilation = int(opts.output_stride_from_trained / opts.output_stride)
                        else:
                            classifier_dilation = 1
                        if classifier_dilation == 1:
                            save_ckpt('./best_%s_%s_os%d_seed%d.pth' %
                                    (opts.model, opts.dataset, opts.output_stride, opts.random_seed))
                        else:
                            save_ckpt('./best_%s_%s_os%d_dilation%d.pth' %
                                    (opts.model, opts.dataset, opts.output_stride, classifier_dilation))
                
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
