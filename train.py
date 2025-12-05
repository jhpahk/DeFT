"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import sys
import torch

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
from optimizer import forgiving_state_restore
import datasets
import loss
import network
import optimizer
import time
import numpy as np
import random
import re

import wandb

import copy
import torchvision.transforms as tr


from augmentation import SimpleAugRGB, SimpleAugLAB

# ImageNet normalization/denormalization
Norm = tr.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
DeNorm = tr.Compose([ tr.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                      tr.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]), ])


# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR50V3PlusD',
                    help='Network architecture. ResNet50 backbone')
parser.add_argument('--dataset', nargs='*', type=str, default=['gtav'],
                    help='a list of datasets; cityscapes, gtav, mapillary, synthia')    # training에 사용할 dataset
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list consists of cityscapes, mapillary, gtav, synthia')    # extra validation sets
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=False)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)
parser.add_argument('--adamw', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')
parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')

parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=30000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')

parser.add_argument('--bs_mult', type=int, default=4,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=1,
                    help='Batch size for Validation per gpu')
parser.add_argument('--bs_mult_pre', type=int, default=4,
                    help='Batch size for pre-training per gpu')

parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='train or test')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')

# args for multiprocessing
parser.add_argument('--master_port', type=int, default=29500)

################################
### args for proposed method ###
parser.add_argument('--lr_enc', type=float, default=1e-2)
parser.add_argument('--lr_dec', type=float, default=1e-2)
parser.add_argument('--lr_pre', type=float, default=1e-2)

parser.add_argument('--weight_decay_enc', type=float, default=5e-4)
parser.add_argument('--weight_decay_dec', type=float, default=5e-4)
parser.add_argument('--weight_decay_pre', type=float, default=5e-4)

parser.add_argument('--pre_epoch', type=int, default=20)
# parser.add_argument('--update_momentum', type=float, default=0.9999)
parser.add_argument('--momentum_enc', type=float, default=0.9999)
parser.add_argument('--momentum_dec', type=float, default=0.9999)
parser.add_argument('--pre_enc_snapshot', type=str, default=None)
parser.add_argument('--pre_dec_snapshot', type=str, default=None)
################################
################################


args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
# torch.backends.cudnn.benchmark = True

# random seed 고정
random_seed = cfg.RANDOM_SEED  #304
print(f"random seed: {random_seed}")
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

args.local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)

# Initialize distributed communication
# args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)
args.dist_url = 'env://'


torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)


def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)

    pre_train_loader, _, pre_train_obj, _ = datasets.setup_loaders(args, mode='pre')
    train_loader, val_loaders, train_obj, extra_val_loaders = datasets.setup_loaders(args)

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    
    # net = network.get_net(args)
    enc, dec = network.get_net(args)
    optim_enc, scheduler_enc = optimizer.get_optimizer(args, enc, update='enc')
    optim_dec, scheduler_dec = optimizer.get_optimizer(args, dec, update='dec')
    
    enc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(enc)
    enc = network.warp_network_in_dataparallel(enc, args.local_rank)
    
    dec = torch.nn.SyncBatchNorm.convert_sync_batchnorm(dec)
    dec = network.warp_network_in_dataparallel(dec, args.local_rank)
    
    enc_init = copy.deepcopy(enc)
    dec_init = copy.deepcopy(dec)
    
    for param_enc in enc_init.parameters():
        param_enc.detach_()
    for param_dec in dec_init.parameters():
        param_dec.detach_()
        
    enc_frozen = copy.deepcopy(enc)
    dec_frozen = copy.deepcopy(dec)
    
    for param_enc in enc_frozen.parameters():
        param_enc.requires_grad_(False)
    # for param_dec in dec_final.parameters():
    #     param_dec.requires_grad_(False)
    
    optim_enc_pre, scheduler_enc_pre = optimizer.get_optimizer(args, enc_frozen, update='pre')
    optim_dec_pre, scheduler_dec_pre = optimizer.get_optimizer(args, dec_frozen, update='pre')
        
    
    if args.pre_enc_snapshot is not None:
        enc_ckpt = torch.load(args.pre_enc_snapshot, map_location=torch.device('cpu'))
        forgiving_state_restore(enc_frozen, enc_ckpt['state_dict'])
    if args.pre_dec_snapshot is not None:
        dec_ckpt = torch.load(args.pre_dec_snapshot, map_location=torch.device('cpu'))
        forgiving_state_restore(dec_frozen, dec_ckpt['state_dict'])

    
    epoch = 0
    i = 0
    
    best_miou_extra = [0] * len(args.val_dataset)
    best_epoch_extra = [0] * len(args.val_dataset)
    
    best_miou_val = 0
    best_epoch_val = 0
    
    pre_epoch = 0
    pre_iter = 0

    print("#### iteration", i)
    torch.cuda.empty_cache()
    # Main Loop
    # for epoch in range(args.start_epoch, args.max_epoch):
    
    ckpt_path = re.sub(r"[/.]", "", args.exp)
    ckpt_path = os.path.join('ckpt', args.date, ckpt_path)
    # ckpt_path = args.exp
    if not os.path.exists(ckpt_path):
        if args.local_rank == 0:
            os.makedirs(ckpt_path)

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)
        
        # Stage 1: Decoder Warm-Up
        if pre_epoch < args.pre_epoch:
            if args.local_rank == 0:
                print("\n### pre-epoch ###\n")
            
            pre_iter = train_pre(pre_train_loader, (enc_frozen, dec_frozen), \
                (optim_enc_pre, optim_dec_pre), pre_epoch, pre_iter, writer, (scheduler_enc_pre, scheduler_dec_pre), args.max_iter, criterion)
            
            pre_train_loader.sampler.set_epoch(pre_epoch + 1)
            
            if args.class_uniform_pct:
                if pre_epoch >= args.max_cu_epoch:
                    pre_train_obj.build_epoch(cut=True)
                    pre_train_loader.sampler.set_num_samples()
                else:
                    pre_train_obj.build_epoch()
            
            if pre_epoch % 5 == 4:
                last_snapshot_dec = f'LABaugmented_pretrained_dec_epoch_{pre_epoch+1}.pth'
                last_snapshot_dec = os.path.join(f'ckpt/r50_{args.dataset[0]}', last_snapshot_dec)
                torch.save({
                    'state_dict': dec_frozen.state_dict(),
                    'command': ' '.join(sys.argv[1:])
                }, last_snapshot_dec)
                
            pre_epoch += 1
                
            continue
        
        # Stage 2: Decoupled Finetuning with EMA updates
        if args.local_rank == 0:
            print("\n### main epoch ###\n")

        if epoch == 0:
            # sync dec with dec_frozen
            dec.load_state_dict(dec_frozen.state_dict())
            dec_init.load_state_dict(dec_frozen.state_dict())
            
            print("Decoder state_dict is loaded!")
            
            for param_enc in enc_frozen.parameters():
                param_enc.requires_grad_(False)
            for param_dec in dec_frozen.parameters():
                param_dec.requires_grad_(False)
        
        i = train_both(train_loader, (enc, dec), (enc_frozen, dec_frozen), (enc_init, dec_init), \
            (optim_enc, optim_dec), epoch, i, writer, (scheduler_enc, scheduler_dec), args.max_iter, criterion)

        train_loader.sampler.set_epoch(epoch + 1)

        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()

                
        net = (enc_frozen, dec_frozen)
        # validation on target domains
        for dataset_idx, (dataset, val_loader) in enumerate(extra_val_loaders.items()):
            miou = validate(val_loader, dataset, net, criterion_val, epoch, writer, i, save_pth=False)
            
            if args.local_rank == 0:
                if miou > best_miou_extra[dataset_idx]:
                    best_miou_extra[dataset_idx] = miou
                    best_epoch_extra[dataset_idx] = epoch
                    
                    # save the best checkpoint
                    best_snapshot_enc = f'enc_best_ckpt_{dataset}.pth'
                    best_snapshot_enc = os.path.join(ckpt_path, best_snapshot_enc)
                    torch.save({
                        'state_dict': net[0].state_dict(),
                        'epoch': epoch,
                    }, best_snapshot_enc)
                    
                    best_snapshot_dec = f'dec_best_ckpt_{dataset}.pth'
                    best_snapshot_dec = os.path.join(ckpt_path, best_snapshot_dec)
                    torch.save({
                        'state_dict': net[1].state_dict(),
                        'epoch': epoch
                    }, best_snapshot_dec)
                
                msg = 'Best Epoch:{}, Best mIoU:{:.5f} for {} dataset'.format(best_epoch_extra[dataset_idx], best_miou_extra[dataset_idx], dataset)
                logging.info(msg)
                print(msg)
        
        epoch += 1
        

def train_pre(train_loader, net, optims, curr_epoch, curr_iter, writer, schedulers, max_iter, criterion):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard
    return:
    """
    enc, dec = net
    optim_enc, optim_dec = optims
    scheduler_enc, scheduler_dec = schedulers
    
    enc.train()
    dec.train()

    train_total_loss = AverageMeter()
    
    time_meter = AverageMeter()
    
    curr_iter = curr_iter

    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break

        # single-source DG
        input, gt, _, aux_gt = data

        B, C, H, W = input.shape
        batch_pixel_size = C * H * W

        start_ts = time.time()

        img_gt = None
        # input, gt, aux_gt = input.cuda(), gt.cuda(), aux_gt.cuda()
        input, gt = input.cuda(), gt.cuda()

        # input = SimpleAugRGB(input, curr_iter)
        input = SimpleAugLAB(input, curr_iter) 

        # optim_enc.zero_grad()
        optim_dec.zero_grad()

        enc_feat, low_level = enc(input)
        output = dec(enc_feat.detach(), low_level.detach(), (H, W))
        
        # main_loss = criterion(output, gt)
        main_loss = criterion(output, torch.cat([gt] * 2))
        total_loss = main_loss
       
        if torch.isnan(total_loss).any():
            print("NAN!!!")
            exit(-1)

        # update loss AverageMeters
        log_total_loss = total_loss.clone().detach_()
        torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
        log_total_loss = log_total_loss / args.world_size
        train_total_loss.update(log_total_loss.item(), batch_pixel_size)

        total_loss.backward()
        
        # optim_enc.step()
        optim_dec.step()

        time_meter.update(time.time() - start_ts)

        del total_loss, log_total_loss

        if args.local_rank == 0:
            if i % 50 == 49:
                msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                    curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg,
                    optim_dec.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)

                logging.info(msg)

                # Log tensorboard metrics for each iteration of the training phase
                # writer.add_scalar(f'loss/pre_train_loss', (train_total_loss.avg),
                #                 curr_iter)
                train_total_loss.reset()
                time_meter.reset()

        curr_iter += 1
        # scheduler_enc.step()
        scheduler_dec.step()

    return curr_iter


def train_both(train_loader, net, net_frozen, net_init, optims, curr_epoch, curr_iter, writer, schedulers, max_iter, criterion):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard
    return:
    """
    
    # unpacking
    enc, dec = net
    enc_frozen, dec_frozen = net_frozen
    enc_init, dec_init = net_init
    
    optim_enc, optim_dec = optims
    scheduler_enc, scheduler_dec = schedulers
    
    enc.train()
    dec.train()
    
    enc_frozen.train()
    dec_frozen.train()

    train_total_loss_enc = AverageMeter()
    train_total_loss_dec = AverageMeter()
    
    time_meter = AverageMeter()
    
    curr_iter = curr_iter

    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break
        
        optim_enc.zero_grad()
        optim_dec.zero_grad()

        # single-source DG
        input, gt, _, aux_gt = data

        B, C, H, W = input.shape
        batch_pixel_size = C * H * W

        start_ts = time.time()

        img_gt = None
        # input, gt, aux_gt = input.cuda(), gt.cuda(), aux_gt.cuda()
        input, gt = input.cuda(), gt.cuda()

        # input = SimpleAugRGB(input, curr_iter)
        input = SimpleAugLAB(input, curr_iter)
        
        # get outputs
        x, low_level = enc(input)
        out_enc = dec_frozen(x, low_level, (H, W))
        
        x_frozen, low_level_frozen = enc_frozen(input)
        out_dec = dec(x_frozen.detach(), low_level_frozen.detach(), (H, W))
        
        # main loss
        main_loss_enc = criterion(out_enc, torch.cat([gt] * 2, dim=0))
        main_loss_dec = criterion(out_dec, torch.cat([gt] * 2, dim=0))
        # main_loss_enc = criterion(out_enc, gt)
        # main_loss_dec = criterion(out_dec, gt)
        
        total_loss_enc = main_loss_enc
        total_loss_dec = main_loss_dec

        if torch.isnan(total_loss_enc).any() or torch.isnan(total_loss_dec):
            print("NAN!!!")
            exit(-1)

        # update loss AverageMeters (encoder)
        log_total_loss_enc = total_loss_enc.clone().detach_()
        torch.distributed.all_reduce(log_total_loss_enc, torch.distributed.ReduceOp.SUM)
        log_total_loss_enc = log_total_loss_enc / args.world_size
        train_total_loss_enc.update(log_total_loss_enc.item(), batch_pixel_size)
        
        # update loss AverageMeters (decoder)
        log_total_loss_dec = total_loss_dec.clone().detach_()
        torch.distributed.all_reduce(log_total_loss_dec, torch.distributed.ReduceOp.SUM)
        log_total_loss_dec = log_total_loss_dec / args.world_size
        train_total_loss_dec.update(log_total_loss_dec.item(), batch_pixel_size)

        total_loss_enc.backward()
        total_loss_dec.backward()
        
        optim_enc.step()
        optim_dec.step()

        time_meter.update(time.time() - start_ts)

        del total_loss_enc, log_total_loss_enc
        del total_loss_dec, log_total_loss_dec                
        
        for frozen_param, param in zip(enc_frozen.parameters(), enc.parameters()):
            frozen_param.data.mul_(args.momentum_enc).add_(param.data, alpha=1-args.momentum_enc)
        for frozen_param, param in zip(dec_frozen.parameters(), dec.parameters()):
            frozen_param.data.mul_(args.momentum_dec).add_(param.data, alpha=1-args.momentum_dec)

        # ## debiased EMA ##
        # for frozen_param, param in zip(enc_frozen.parameters(), enc.parameters()):
        #     frozen_param.data.mul_(1 - args.momentum_enc**(curr_iter + 1))
        #     frozen_param.data.mul_(args.momentum_enc).add_(param.data, alpha=1-args.momentum_enc)
        #     frozen_param.data.div_(1 - args.momentum_enc**(curr_iter + 2))
        # for frozen_param, param in zip(dec_frozen.parameters(), dec.parameters()):
        #     frozen_param.data.mul_(1 - args.momentum_enc**(curr_iter + 1))
        #     frozen_param.data.mul_(args.momentum_dec).add_(param.data, alpha=1-args.momentum_dec)
        #     frozen_param.data.div_(1 - args.momentum_enc**(curr_iter + 2))
        
        if args.local_rank == 0:
            if i % 50 == 49:
                msg_enc = '\n<<Encoder>> [epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                    curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss_enc.avg,
                    optim_enc.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)
                msg_dec = '<<Decoder>> [epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                    curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss_dec.avg,
                    optim_dec.param_groups[-1]['lr'], time_meter.avg / args.train_batch_size)
                msg = msg_enc + '\n' + msg_dec

                logging.info(msg)

                # # Log tensorboard metrics for each iteration of the training phase
                # writer.add_scalar(f'loss/train_loss_enc', (train_total_loss_enc.avg), curr_iter)
                # writer.add_scalar(f'loss/train_loss_dec', (train_total_loss_dec.avg), curr_iter)
                
                train_total_loss_enc.reset()
                train_total_loss_dec.reset()
                time_meter.reset()

        curr_iter += 1
        scheduler_enc.step()
        scheduler_dec.step()

        if i > 5 and args.test_mode:
            return curr_iter

    return curr_iter


def validate(val_loader, dataset, net, criterion, curr_epoch, writer, curr_iter, save_pth=False):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """    
    enc, dec = net
    enc.eval()
    dec.eval()
    # net.eval()
    
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        # input        = torch.Size([1, 3, 713, 713])
        # gt_image     = torch.Size([1, 713, 713])
        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.reshape(-1, C, H, W)
            gt_image = gt_image.reshape(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            enc_feat, low_level = enc(inputs)
            output = dec(enc_feat, low_level, inputs.size()[2:])

        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        miou = evaluate_eval(args, net, None, None, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=False)
        if save_pth:
            last_snapshot_enc = f'encoder_last_{dataset}_epoch_{curr_epoch}_mean-iu_{miou:.5f}.pth'
            last_snapshot_enc = os.path.join(args.exp_path, last_snapshot_enc)
            torch.save({
                'state_dict': enc.state_dict(),
                'epoch': curr_epoch,
                'mean_iu': miou,
                'command': ' '.join(sys.argv[1:])
            }, last_snapshot_enc)
            
            last_snapshot_dec = f'decoder_last_{dataset}_epoch_{curr_epoch}_mean-iu_{miou:.5f}.pth'
            last_snapshot_dec = os.path.join(args.exp_path, last_snapshot_dec)
            torch.save({
                'state_dict': dec.state_dict(),
                'epoch': curr_epoch,
                'mean_iu': miou,
                'command': ' '.join(sys.argv[1:])
            }, last_snapshot_dec)

    else:
        miou = 0

    return miou



if __name__ == '__main__':
    main()