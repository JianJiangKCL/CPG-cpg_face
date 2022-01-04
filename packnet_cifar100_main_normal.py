"""Main entry point for doing all stuff."""
import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter

import logging
import os
import pdb
import math
from tqdm import tqdm
import sys
import numpy as np

import utils
from utils import Optimizers
from utils.packnet_manager import Manager
import utils.cifar100_dataset as dataset
import packnet_models


# To prevent PIL warnings.
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='vgg16_bn_cifar100',
                   help='Architectures')
parser.add_argument('--num_classes', type=int, default=-1,
                   help='Num outputs for dataset')

# Optimization options.
parser.add_argument('--lr', type=float, default=0.1,
                   help='Learning rate for parameters, used for baselines')

parser.add_argument('--batch_size', type=int, default=32,
                   help='input batch size for training')
parser.add_argument('--val_batch_size', type=int, default=100,
                   help='input batch size for validation')
parser.add_argument('--workers', type=int, default=24, help='')
parser.add_argument('--weight_decay', type=float, default=4e-5,
                   help='Weight decay')

# Paths.
parser.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
parser.add_argument('--path', type=str, default='',
                   help='path to the dataset')
parser.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
parser.add_argument('--val_path', type=str, default='',
                   help='Location of test data')

# Other.
parser.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')

parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--checkpoint_format', type=str,
                    default='./{save_folder}/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--epochs', type=int, default=160,
                    help='number of epochs to train')
parser.add_argument('--restore_epoch', type=int, default=0, help='')
parser.add_argument('--save_folder', type=str,
                    help='folder name inside one_check folder')
parser.add_argument('--load_folder', default='', help='')
parser.add_argument('--one_shot_prune_perc', type=float, default=0.5,
                   help='% of neurons to prune per layer')
parser.add_argument('--mode',
                   choices=['finetune', 'prune', 'inference'],
                   help='Run mode')
parser.add_argument('--logfile', type=str, help='file to save baseline accuracy')
parser.add_argument('--initial_from_task', type=str, help="")


def main():
    """Do stuff."""
    args = parser.parse_args()
    if args.save_folder and not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        args.cuda = False

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    resume_folder = args.load_folder
    # Set default train and test path if not provided as input.
    utils.set_dataset_paths(args)

    dataset_history = []
    dataset2num_classes = {}
    masks = {}
    shared_layer_info = {}
    if args.arch == 'vgg16_bn_cifar100':
        model = packnet_models.__dict__[args.arch](pretrained=False, dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
    elif args.arch == 'resnet18':
        model = packnet_models.__dict__[args.arch](dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)

    # Add and set the model dataset
    model.add_dataset(args.dataset, args.num_classes)
    model.set_dataset(args.dataset)

    model = nn.DataParallel(model)
    model = model.cuda()

    train_loader = dataset.cifar100_train_loader(args.path, args.dataset, args.batch_size)
    val_loader = dataset.cifar100_val_loader(args.path, args.dataset, args.val_batch_size)
    start_epoch = 0
    manager = Manager(args, model, shared_layer_info, masks, train_loader, val_loader)


    lr = args.lr
    # update all layers
    named_params = dict(model.named_parameters())
    params_to_optimize_via_SGD = []
    named_params_to_optimize_via_SGD = []
    masks_to_optimize_via_SGD = []
    named_masks_to_optimize_via_SGD = []

    for tuple_ in named_params.items():
        if 'classifiers' in tuple_[0]:
            if '.{}.'.format(model.module.datasets.index(args.dataset)) in tuple_[0]:
                params_to_optimize_via_SGD.append(tuple_[1])
                named_params_to_optimize_via_SGD.append(tuple_)
            continue
        else:
            params_to_optimize_via_SGD.append(tuple_[1])
            named_params_to_optimize_via_SGD.append(tuple_)

    # here we must set weight decay to 0.0,
    # because the weight decay strategy in build-in step() function will change every weight elem in the tensor,
    # which will hurt previous tasks' accuracy. (Instead, we do weight decay ourself in the `prune.py`)
    optimizer_network = optim.SGD(params_to_optimize_via_SGD, lr=lr,
                          weight_decay=args.weight_decay, momentum=0.9, nesterov=True)

    optimizers = Optimizers()
    optimizers.add(optimizer_network, lr)

    manager.load_checkpoint(optimizers, resume_from_epoch, resume_folder)

    """Performs training."""
    curr_lrs = []
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            curr_lrs.append(param_group['lr'])
            break


    for epoch_idx in range(start_epoch, args.epochs):
        avg_train_acc = manager.train(optimizers, epoch_idx, curr_lrs)
        avg_val_acc = manager.validate(epoch_idx)

        if args.mode == 'finetune':
            if epoch_idx + 1 == 50 or epoch_idx + 1 == 80:
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] *= 0.1
                curr_lrs[0] = param_group['lr']



if __name__ == '__main__':
    main()
