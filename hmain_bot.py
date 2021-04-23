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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as tv_transforms

import os,sys
import argparse
from time import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from models import resnet8bot as bot
from utils import progress_bar, SampleCIFAR10, smooth_crossentropy, getDataSubset, boolean_string
from opts.sam import SAM

DATA_PATH = '/data/cifar10/'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
parser.add_argument("--train_seeds", default=None, type=int, help="The default seed for sampling dataset")
parser.add_argument("--use_subset", default=True, type=boolean_string, help="Whether to use 10\% subset of the CIFAR10 dataset for training")
parser.add_argument("--test_portion", default=100, type=int, help="The percentage of samples to be used for testing")
parser.add_argument("--test_seeds", default=None, type=int, help="The default seed")
parser.add_argument("--model_path", default="./checkpoint/sam_efnet_ckpt.pth", type=str, help="The full path of the model")
parser.add_argument("--optimizer", default="sgd", type=str, help="The type of optimizer: sgd, sam")
parser.add_argument("--gpus", type=str, default="1", help='GPU Device IDs. If not specified, use all GPUs')
parser.add_argument("--logdir", default="", type=str, help="The path to store log for tensorboard")

args = parser.parse_args()

model_path = args.model_path
#Ensure the model path is created
mpath = os.path.split(model_path)[0]
if not os.path.isdir(mpath):
    os.makedirs(mpath)

tb_log_dir = args.logdir
model_name = 'ResNet18_BagOfTricks'

train_portion=10 if args.use_subset else 100
#Automatically generate and create the log according to model, batch, epochs, 
#train_portion, test_portion, optimizer, dropout
if args.logdir is None or len(args.logdir)<1:
    tb_log_dir = "./tblog/s_{}_b{}_epc{}_tp{}_vp{}_{}_d{:.2f}".format(model_name, args.batch_size, 
                                                          args.epochs, train_portion, 
                                                          args.test_portion, args.optimizer, 
                                                          args.dropout)
    if not os.path.isdir(tb_log_dir):
        os.makedirs(tb_log_dir)


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

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epochs = args.epochs

#optm_type = args.optimizer #'sgd'
# Data
print('==> Preparing data..')
transform_train = tv_transforms.Compose([
    tv_transforms.RandomCrop(32, padding=4),
    tv_transforms.RandomHorizontalFlip(),
    #tv_transforms.RandomResizedCrop(32),
    #tv_transforms.RandomAffine(45),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = tv_transforms.Compose([
    tv_transforms.ToTensor(),
    tv_transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

total_train_samples = 0
total_test_samples = 0

# get CIFAR training data, and get 10% subset
print('=====> Grabbing dataset')
if not os.path.isdir(DATA_PATH): os.makedirs(DATA_PATH)
train_dataset = torchvision.datasets.CIFAR10(DATA_PATH, 
											 train=True, download=True) 
											 #transform=transform_train)
valid_dataset = torchvision.datasets.CIFAR10(DATA_PATH,
											 train=False, download=True)
											 #transform=transform_test)
# if accuracy is low, likely due to transform_train, since bot package will also perform transformation on data
total_train_steps_per_epoch = len(train_dataset) // args.batch_size

# organize data
dataset_dict = {k: {'data': torch.tensor(v.data), 'targets': torch.tensor(v.targets)} 
            		for k,v in [('train', train_dataset), ('valid', valid_dataset)]}
# move data to GPU
dataset_dict = bot.map_nested(bot.to(device), dataset_dict)

print('=====> Data moved to GPU')

# preprocess data on gpu
train_set = bot.preprocess(dataset_dict['train'], [bot.partial(bot.pad, border=4), bot.transpose, bot.normalise, bot.to(torch.float16)])
valid_set = bot.preprocess(dataset_dict['valid'], [bot.transpose, bot.normalise, bot.to(torch.float16)])

if args.use_subset:
	# use only subset of the data (10%)
	train_set['data'],train_set['targets'] = bot.get_subset(train_set, 0.1)
	valid_set['data'],valid_set['targets'] = bot.get_subset(valid_set, 0.1)

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
epochs, batch_size, ema_epochs=200, args.batch_size, 2 #epochs = 10
transforms = (bot.Crop(32, 32), bot.FlipLR())

lr_schedule = lambda knots, vals, batch_size: bot.PiecewiseLinear(np.array(knots)*len(train_batches(batch_size)), np.array(vals)/batch_size)
opt_params = {'lr': lr_schedule([0, epochs/5, epochs - ema_epochs], [0.0, 1.0, 0.1], batch_size), 'weight_decay': bot.Const(5e-4*batch_size), 'momentum': bot.Const(0.9)}
opt_params_bias = {'lr': lr_schedule([0, epochs/5, epochs - ema_epochs], [0.0, 1.0*64, 0.1*64], batch_size), 'weight_decay': bot.Const(5e-4*batch_size/64), 'momentum': bot.Const(0.9)}

is_bias = bot.group_by_key(('bias' in k, v) for k, v in bot.trainable_params(net).items())
state, timer = {bot.MODEL: net, bot.VALID_MODEL: bot.copy.deepcopy(net), bot.OPTS: [bot.SGD(is_bias[False], opt_params), bot.SGD(is_bias[True], opt_params_bias)]}, bot.Timer(torch.cuda.synchronize)

print('=====> Training...')

train_times = []
train_accs = []
test_accs = []

# Training
logs = bot.Table()

def train(epoch, total_steps_per_epoch=0):
    global train_accs

    #print('\nEpoch: %d' % epoch)
    #net.train()
    total = len(train_set['data'])
    #tbwriter = SummaryWriter(log_dir=tb_log_dir)

    logs.append(bot.union({'run': 1, 'epoch': epoch},
    	bot.train_epoch(state, timer, train_batches(batch_size, transforms), valid_batches(batch_size), 
    	valid_steps=bot.valid_steps_tta, train_steps=(*bot.train_steps, bot.update_ema(momentum=0.99, update_freq=5)))))

    #epoch_res = bot.reduce(train_batches(batch_size, transforms), state, bot.train_steps)
    #print(type(epoch_res))
    #print(epoch_res.keys())
    #print(type(epoch_res['output']))
    #print(epoch_res['output'].keys())
    #print(type(epoch_res['output']['loss']))
    #print(type(epoch_res['output']['acc']))
    #print(type(epoch_res['output']['loss'].cpu()))
    #print(type(epoch_res['output']['acc'].cpu()))
    #train_loss = np.mean(epoch_res['output']['loss'].detach().cpu().numpy())
    #correct = np.count_nonzero(epoch_res['output']['acc'].detach().cpu().numpy() == True)

    #progress_bar(total, total, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #             % (train_loss, 100.*correct/total, correct, total))
    
    #global_step = total_steps_per_epoch * epoch
    #tbwriter.add_scalar("Loss/train", train_loss, global_step)
    #tbwriter.add_scalar("Accuracy/train", 100.*correct/total, global_step)

    #train_accs.append(100.*correct/total)

    #tbwriter.flush()
    #tbwriter.close()


def test(epoch, total_steps_per_epoch=0):
    global best_acc
    global test_accs

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    tbwriter = SummaryWriter(log_dir=tb_log_dir)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            if batch_idx % 2 == 0:
                global_step = total_steps_per_epoch * epoch + batch_idx
                tbwriter.add_scalar("Loss/test", test_loss/(batch_idx+1), global_step)
                tbwriter.add_scalar("Accuracy/test", 100.*correct/total, global_step)

    tbwriter.flush()
    tbwriter.close()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, model_path)
        best_acc = acc

    test_accs.append(100.*correct/total)

best_10min_acc = 0.
best_20min_acc = 0.


# run training
start_time = time()
for epoch in range(start_epoch, start_epoch+total_epochs):
    train(epoch)
    #test(epoch, total_steps_per_epoch=total_test_steps_per_epoch)

    curr_time = time()
    if curr_time - start_time < 600.:
        best_10min_acc = best_acc
        best_20min_acc = best_acc
    elif curr_time - start_time < 1200.:
        best_20min_acc = best_acc
    else:
        print('10 min best :', best_10min_acc, '     20 min best : ', best_20min_acc)
        print('training exceeded 20 mins, aborting...')
        break

    #scheduler.step()
    iter_time = time()
    train_times.append(iter_time - start_time)

print('10 min best :', best_10min_acc, '     20 min best : ', best_20min_acc)
print()
print(train_times)
print(train_accs)
print(test_accs)

# save training and test accuracies over time in numpy text file
if not os.path.isdir('./training_graphs'):
	os.mkdir('./training_graphs')
#np.savetxt("./training_graphs/"+model_name+'_bs_'+str(args.batch_size+'_training_graph.npy', np.asarray([train_times, train_accs, test_accs]))