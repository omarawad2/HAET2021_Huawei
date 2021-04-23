'''
 * ============================================================================
 *
 * Copyright (C) 2020, Huawei Technologies Co., Ltd. All Rights Reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     1 Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *
 *     2 Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimer in the documentation
 *       and/or other materials provided with the distribution.
 *
 *     3 Neither the names of the copyright holders nor the names of the
 *       contributors may be used to endorse or promote products derived from this
 *       software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */
'''

'''Train CIFAR10 with PyTorch.'''
from time import time
start_time = time()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as tv_transforms
from torchvision import transforms
import os,sys
import argparse

import numpy as np

#from torch.utils.tensorboard import SummaryWriter

from models import resnet8bot as bot
from utils import progress_bar, SampleCIFAR10, smooth_crossentropy, getDataSubset, boolean_string
from opts.sam import SAM

from read_mini_imagenet import get_mini_imagenet_data
from skimage.transform import resize

import read_mini_imagenet as loader

DATA_PATH = '/data/cifar10/'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--batch_size", default=256, type=int, help="Batch size used in the training and validation loop.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
parser.add_argument("--train_seeds", default=None, type=int, help="The default seed for sampling dataset")
parser.add_argument("--use_subset", default=False, type=boolean_string, help="Whether to use 10\% subset of the CIFAR10 dataset for training")
parser.add_argument("--test_portion", default=100, type=int, help="The percentage of samples to be used for testing")
parser.add_argument("--test_seeds", default=None, type=int, help="The default seed")
parser.add_argument("--model_path", default="./checkpoint/ckpt.pth", type=str, help="The full path of the model")
parser.add_argument("--gpus", type=str, default="0", help='GPU Device IDs. If not specified, use all GPUs')
parser.add_argument("--optimizer", default="sam", type=str, help="The type of optimizer: sgd, sam")
parser.add_argument("--lr_scheduler", default="PiecewiseLinear", type=str, help="The type of learning_schedule: PiecewiseLinear, Piecewise, cosineAnnealingLR")
parser.add_argument("--net_save", default="./checkpoint/ckpt.pth", type=str, help="The full path of the model")

args = parser.parse_args()

# Data
print('==> Preparing data')

# get CIFAR training data, and get 10% subset
if not os.path.isdir(DATA_PATH): os.makedirs(DATA_PATH)

#####################################################################################
################## Change the dataset here

#train_dataset = torchvision.datasets.CIFAR10(DATA_PATH, train=True, download=True)
train_dataset, test_dataset = get_mini_imagenet_data()

train_transforms = [   
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.ToTensor()
    #ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
    ]
transform_train = transforms.Compose(train_transforms)
train_dataset.data = np.array([255*transform_train(img).permute(1,2,0).numpy() for img in train_dataset.data]).astype(np.uint8)

#####################################################################################
args.model_path = args.net_save
model_path = args.model_path
#Ensure the model path is created
mpath = os.path.split(model_path)[0]
if not os.path.isdir(mpath):
    os.makedirs(mpath)

model_name = 'ResNet18_BagOfTricks'

# set GPU device for use
device_ids = []
if args.gpus is not None and len(args.gpus) > 0:
    gpu_no = args.gpus.split(",")
    device_ids = []
    for g in gpu_no:
        device_ids.append(int(g))

if torch.cuda.is_available():
    if len(device_ids)>1:
        device = 'cuda'
    elif len(device_ids)==1:
        device = 'cuda:' + str(device_ids[0])
    else:
        device = 'cuda'
else:
    device = 'cpu'
device = 'cuda:0'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epochs = args.epochs

#optm_type = args.optimizer #'sgd'

# no cross-validation splitting, use all for training.
# just need a valid_dataset object for the model interface, but it's not validating against different data from training data.
valid_dataset = train_dataset


# if accuracy is low, likely due to transform_train, since bot package will also perform transformation on data
total_train_steps_per_epoch = len(train_dataset) // args.batch_size

# organize data
dataset_dict = {k: {'data': torch.tensor(v.data), 'targets': torch.tensor(v.targets)} 
                    for k,v in [('train', train_dataset), ('valid', valid_dataset)]}
# move data to GPU
dataset_dict = bot.map_nested(bot.to(device), dataset_dict)

print('=====> Data moved to GPU')

# get data statistics for normalizing data
mean = tuple(np.mean(train_dataset.data, axis=(0,1,2)))
std = tuple(np.std(train_dataset.data, axis=(0,1,2)))
mean, std = [torch.tensor(x, device=device, dtype=torch.float16) for x in (mean, std)]
normalize = lambda data, mean=mean, std=std: (data-mean)/std

# preprocess data on gpu
train_set = bot.preprocess(dataset_dict['train'], [bot.partial(bot.pad, border=4), bot.transpose, bot.normalise, bot.to(torch.float16)]) # 
valid_set = bot.preprocess(dataset_dict['valid'], [ bot.transpose, normalize, bot.to(torch.float16)]) #

if args.use_subset:
    # use only subset of the data (10%)
    train_set['data'],train_set['targets'] = bot.get_subset(train_set, 0.1)
    valid_set['data'],valid_set['targets'] = train_set['data'],train_set['targets']
    #valid_set['data'],valid_set['targets'] = bot.get_subset(valid_set, 0.01)

print('=====> Data preprocessed (on GPU)')

# create batching lambda function
train_batches = bot.partial(bot.Batches, dataset=train_set, shuffle=True,  drop_last=True, max_options=200)
valid_batches = bot.partial(bot.Batches, dataset=valid_set, shuffle=False, drop_last=False)


print('=====> Input whitening')

# create input whitening network
Λ, V = bot.eigens(bot.patches(train_set['data'][:10000,:,4:-4,4:-4]))
input_whitening_net = bot.network(conv_pool_block=bot.conv_pool_block_pre, prep_block=bot.partial(bot.whitening_block, Λ=Λ, V=V), scale=1/16, types={
    nn.ReLU: bot.partial(nn.CELU, 0.3),
    bot.BatchNorm: bot.partial(bot.GhostBatchNorm, num_splits=16, weight=False)})


print('=====> Building model (with input whitening network)')
net = bot.getResNet8BOT(input_whitening_net)


print('=====> Preparing for training')
epochs, batch_size, ema_epochs=args.epochs, args.batch_size, 2 #epochs = 10
transforms = (bot.Crop(32,32), bot.FlipLR())

#learning rate schedules
if args.lr_scheduler == 'PiecewiseLinear':  
    lr_schedule = lambda knots, vals, batch_size: bot.PiecewiseLinear(np.array(knots)*len(train_batches(batch_size)), np.array(vals)/batch_size)
    opt_params = {'lr': lr_schedule([0, epochs/5, epochs - ema_epochs], [0.0, 1.0, 0.1], batch_size), 'weight_decay': bot.Const(args.weight_decay*batch_size), 'momentum': bot.Const(0.9)}
    opt_params_bias = {'lr': lr_schedule([0, epochs/5, epochs - ema_epochs], [0.0, 1.0*64, 0.1*64], batch_size), 'weight_decay': bot.Const(args.weight_decay*batch_size/64), 'momentum': bot.Const(0.9)}
elif args.lr_scheduler == 'Piecewise':
    base = 0.9
    bot.opt_flag = False
    lr_schedule2 = lambda base, total_epochs: bot.Piecewise(base, epochs)
    opt_params = {'lr': lr_schedule2(base, epochs), 'weight_decay': bot.Const(args.weight_decay*batch_size), 'momentum': bot.Const(0.9)}
    opt_params_bias = {'lr': lr_schedule2(base, epochs), 'weight_decay': bot.Const(args.weight_decay*batch_size/64), 'momentum': bot.Const(0.9)}
else: #SAM
    bot.opt_flag = False
    lr_schedule3 = lambda base_lr, T_max, eta_min: bot.cosineAnnealingLR(base_lr, T_max, eta_min)
    base_lr, T_max, eta_min = 0.1, epochs, 0
    opt_params = {'lr': lr_schedule3(base_lr, T_max, eta_min), 'weight_decay': bot.Const(args.weight_decay*batch_size), 'momentum': bot.Const(0.9)}
    opt_params_bias = {'lr': lr_schedule3(base_lr, T_max, eta_min), 'weight_decay': bot.Const(args.weight_decay*batch_size/64), 'momentum': bot.Const(0.9)}

is_bias = bot.group_by_key(('bias' in k, v) for k, v in bot.trainable_params(net).items())

if args.optimizer == 'sgd':
    state, timer = {bot.MODEL: net, bot.VALID_MODEL: bot.copy.deepcopy(net), bot.OPTS: [bot.SGD(is_bias[False], opt_params), bot.SGD(is_bias[True], opt_params_bias)]}, bot.Timer(torch.cuda.synchronize)
else: #SAM
    state, timer = {bot.MODEL: net, bot.VALID_MODEL: bot.copy.deepcopy(net), bot.OPTS: [bot.SAM1(is_bias[False], opt_params, sam_flag=True)]}, bot.Timer(torch.cuda.synchronize)
#, bot.SAM2(is_bias[False], opt_params, sam_flag=True)
print('=====> Training...')

# Training
def train(epoch, max_acc, total_steps_per_epoch=0):
    total = len(train_set['data'])

    if args.optimizer == 'sgd':
        res = bot.train_epoch(epoch, state, timer, train_batches(batch_size, transforms), valid_batches(batch_size), 
            valid_steps=bot.valid_steps_tta, train_steps=(*bot.train_steps, bot.update_ema(momentum=0.99, update_freq=5)))
    else: #SAM
        res = bot.train_epoch(epoch, state, timer, train_batches(batch_size, transforms), valid_batches(batch_size), 
            valid_steps=bot.valid_steps_tta, train_steps=(*bot.train_steps_sam, bot.update_ema(momentum=0.99, update_freq=5)))


    res = res['train']
    train_loss = res['loss']
    train_acc = res['acc']
    correct = int(train_acc * total)

    print('Epoch', epoch)
    progress_bar(total-1, total, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (train_loss, 100.*train_acc, correct, total))

    if(train_acc >= max_acc):
        max_acc = train_acc
        print('Saving..')
        net_state = {
            'net': net.state_dict(),
            'acc': train_acc,
            'epoch': epoch,
        }
        torch.save(net_state, model_path)

if __name__=='__main__':
    max_acc = 0
    # run training
    for epoch in range(start_epoch, start_epoch+total_epochs):
        train(epoch, max_acc)
        #test(epoch, total_steps_per_epoch=total_test_steps_per_epoch)
    
        curr_time = time()
        if curr_time - start_time > 600.:
            print('Reached 10 minutes, aborting..')
            break
    