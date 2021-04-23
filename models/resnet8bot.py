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

import IPython

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
import copy
import inspect
from collections import namedtuple, defaultdict
from functools import partial
import functools
from itertools import chain, count, islice as take
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device('cpu')
from torchsummary import summary

import IPython.display as display
from inspect import signature   

from functools import partial
import torchvision
from functools import lru_cache as cache
import math
import time

import altair as alt
alt.renderers.enable('colab')
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import SVG

MODEL = 'model'
VALID_MODEL = 'valid_model'
OUTPUT = 'output'
OPTS = 'optimisers'
ACT_LOG = 'activation_log'
WEIGHT_LOG = 'weight_log'

lr_global = 0

# UTILITIES

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}
make_tuple = lambda path: (path,) if isinstance(path, str) else path
map_values = lambda func, dct: {k: func(v) for k,v in dct.items()}
reorder = lambda dct, keys: {k: dct[k] for k in keys}


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict): yield from path_iter(val, pfx+make_tuple(name))
        else: yield (pfx+make_tuple(name), val)  

def map_nested(func, nested_dict):
    return {k: map_nested(func, v) if isinstance(v, dict) else func(v) for k,v in nested_dict.items()}

def group_by_key(seq):
    res = defaultdict(list)
    for k, v in seq: 
        res[k].append(v) 
    return res
def identity(value): return value

def build_graph(net, path_map='_'.join):
    net = {path: node if len(node) is 3 else (*node, None) for path, node in path_iter(net)}
    default_inputs = chain([('input',)], net.keys())
    resolve_path = lambda path, pfx: pfx+path if (pfx+path in net or not pfx) else resolve_path(net, path, pfx[:-1])
    return {path_map(path): (typ, value, ([path_map(default)] if inputs is None else [path_map(resolve_path(make_tuple(k), path[:-1])) for k in inputs])) 
            for (path, (typ, value, inputs)), default in zip(net.items(), default_inputs)}

class ColorMap(dict):
    palette = (
        'bebada,ffffb3,fb8072,8dd3c7,80b1d3,fdb462,b3de69,fccde5,bc80bd,ccebc5,ffed6f,1f78b4,33a02c,e31a1c,ff7f00,'
        '4dddf8,e66493,b07b87,4e90e3,dea05e,d0c281,f0e189,e9e8b1,e0eb71,bbd2a4,6ed641,57eb9c,3ca4d4,92d5e7,b15928'
    ).split(',')
 
    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]

def make_pydot(nodes, edges, direction='LR', sep='_', **kwargs):
    from pydot import Dot, Cluster, Node, Edge
    class Subgraphs(dict):
        def __missing__(self, path):
            *parent, label = path
            subgraph = Cluster(sep.join(path), label=label, style='rounded, filled', fillcolor='#77777744')
            self[tuple(parent)].add_subgraph(subgraph)
            return subgraph
    g = Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(
        shape='box', style='rounded, filled', fillcolor='#ffffff')
    subgraphs = Subgraphs({(): g})
    for path, attr in nodes:
        *parent, label = path.split(sep)
        subgraphs[tuple(parent)].add_node(
            Node(name=path, label=label, **attr))
    for src, dst, attr in edges:
        g.add_edge(Edge(src, dst, **attr))
    return g

class DotGraph():
    colors = ColorMap()   
    def __init__(self, graph, size=15, direction='LR'):
        self.nodes = [(k, {
            'tooltip': '%s %.1000r' % (typ, value), 
            'fillcolor': '#'+self.colors[typ],
        }) for k, (typ, value, inputs) in graph.items()] 
        self.edges = [(src, k, {}) for (k, (_,_,inputs)) in graph.items() for src in inputs]
        self.size, self.direction = size, direction
    def dot_graph(self, **kwargs):
        return make_pydot(self.nodes, self.edges, size=self.size, 
                            direction=self.direction, **kwargs)
    def svg(self, **kwargs):
        return self.dot_graph(**kwargs).create(format='svg').decode('utf-8')
    try:
        import pydot
        def _repr_svg_(self):
            return self.svg()
    except ImportError:
        def __repr__(self):
            return 'pydot is needed for network visualisation'


# NETWORK DEFINITION
class Network(nn.Module):
    def __init__(self, net, loss=None):
        super().__init__()
        self.graph = {path: (typ, typ(**params), inputs) for path, (typ, params, inputs) in build_graph(net).items()}
        self.loss = loss or identity
        for path, (_,node,_) in self.graph.items(): 
            setattr(self, path, node)
    
    def nodes(self):
        return (node for _,node,_ in self.graph.values())
    
    def forward(self, inputs):
        outputs = dict(inputs)
        for k, (_, node, ins) in self.graph.items():
            outputs[k] = node(*[outputs[x] for x in ins])
        return outputs
    
    def half(self):
        for node in self.nodes():
            if isinstance(node, nn.Module) and not isinstance(node, nn.BatchNorm2d):
                node.half()
        return self

build_model = lambda network, loss: Network(network, loss).half().to(device)

class Add(namedtuple('Add', [])):
    def __call__(self, x, y): return x + y 

class AddWeighted(namedtuple('AddWeighted', ['wx', 'wy'])):
    def __call__(self, x, y): return self.wx*x + self.wy*y 
    
class Identity(namedtuple('Identity', [])):
    def __call__(self, x): return x

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features*self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features*self.num_splits))
    def train(self, mode=True):
        if (self.training is True) and (mode is False): #lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
        return super().train(mode)        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C*self.num_splits, H, W), self.running_mean, self.running_var, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W) 
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features], 
                self.weight, self.bias, False, self.momentum, self.eps)
        
class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight

class Flatten(nn.Module):
    def forward(self, x): 
        return x.view(x.size(0), x.size(1))

# Losses
class CrossEntropyLoss(namedtuple('CrossEntropyLoss', [])):
    def __call__(self, log_probs, target):
        return torch.nn.functional.nll_loss(log_probs, target, reduction='none')

class KLLoss(namedtuple('KLLoss', [])):        
    def __call__(self, log_probs):
        return -log_probs.mean(dim=1)

class Correct(namedtuple('Correct', [])):
    def __call__(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

class LogSoftmax(namedtuple('LogSoftmax', ['dim'])):
    def __call__(self, x):
        return torch.nn.functional.log_softmax(x, self.dim, _stacklevel=5)


# NODE DEFINITIONS

# empty_signature = inspect.Signature()

class node_def(namedtuple('node_def', ['type'])):
    def __call__(self, *args, **kwargs):
        return (self.type, dict(signature(self.type).bind(*args, **kwargs).arguments))

conv = node_def(nn.Conv2d)
linear = node_def(nn.Linear)
batch_norm = node_def(BatchNorm)
pool = node_def(nn.MaxPool2d)
relu = node_def(nn.ReLU)

def map_types(mapping, net):
    def f(node):
        typ, *rest = node
        return (mapping.get(typ, typ), *rest)
    return map_nested(f, net) 

# COMPATIBILITY
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()  
    return x

def flip_lr(x):
    if isinstance(x, torch.Tensor):
        return torch.flip(x, [-1]) 
    return x[..., ::-1].copy()


# OPTIMIZERS
trainable_params = lambda model: {k:p for k,p in model.named_parameters() if p.requires_grad}
norm = lambda x: torch.norm(x.reshape(x.size(0),-1).float(), dim=1)[:,None,None,None]

def sam_first_step(epoch, w, dw, e_w, v, lr, weight_decay, momentum):
    grad_norm = torch.norm(dw) 
    scale = 0.006 / (grad_norm + 1e-12)
    e_w2 = dw * scale.to(w)
    if(epoch%6 == 0):
        w.add_(e_w2)
        e_w.mul_(0).add_(e_w2)

def sam_second_step(epoch, w, dw, e_w, v, lr, weight_decay, momentum):
    if(epoch%6 == 0):
        w.sub_(e_w)
    #SGD
    dw.add_(weight_decay, w).mul_(-lr)
    v.mul_(momentum).add_(dw)
    w.add_(dw.add_(momentum, v))
    e_w.mul_(0)

def nesterov_update(w, dw, v, lr, weight_decay, momentum):
    dw.add_(weight_decay, w).mul_(-lr)
    v.mul_(momentum).add_(dw)
    w.add_(dw.add_(momentum, v))


def LARS_update(w, dw, v, lr, weight_decay, momentum):
    nesterov_update(w, dw, v, lr*(norm(w)/(norm(dw)+1e-2)).to(w.dtype), weight_decay, momentum)

def zeros_like(weights):
    return [torch.zeros_like(w) for w in weights]

def optimiser(weights, param_schedule, update, state_init, sam_flag=False):
    weights = list(weights)
    e_w = state_init(weights) if(sam_flag) else None
    return {'update': update, 'param_schedule': param_schedule, 'step_number': 0, 'weights': weights, 'opt_state': state_init(weights), 'e_w': e_w}

opt_flag = True

def opt_step(epoch, update, param_schedule, step_number, weights, opt_state, e_w):
    global opt_flag 
    step_number += 1
    param_values = {k: f(step_number) for k, f in param_schedule.items()} #if(opt_flag) else {k: f(epoch) for k, f in param_schedule.items()}

    if e_w is None:
        for w, v in zip(weights, opt_state):
            if w.requires_grad:
                if torch.isnan(w.grad.data).any():
                    print('###SGD w.grad nan detected, step_number: ', step_number, epoch)
                update(w.data, w.grad.data, v, **param_values)
    else:
        for w, v, e_w_elem in zip(weights, opt_state, e_w):
            if w.requires_grad:
                if torch.isnan(w.grad.data).any():
                    print('###SAM w.grad nan detected, step_number: ', step_number, epoch)
                update(epoch, w.data, w.grad.data, e_w_elem.data, v, **param_values)

    return {'update': update, 'param_schedule': param_schedule, 'step_number': step_number, 'weights': weights, 'opt_state': opt_state, 'e_w': e_w}

LARS = partial(optimiser, update=LARS_update, state_init=zeros_like)
SGD = partial(optimiser, update=nesterov_update, state_init=zeros_like)
SAM1 = partial(optimiser, update=sam_first_step, state_init=zeros_like)
SAM2 = partial(optimiser, update=sam_second_step, state_init=zeros_like)

# LEARNING SCHEDULES

class Piecewise(namedtuple('Piecewise', ('base', 'total_epochs'))):
    def __call__(self, epoch):
        if epoch < self.total_epochs * 1/10:
            lr = self.base*0.2
        elif epoch < self.total_epochs * 2/10:
            lr = self.base*0.2**2
        elif epoch < self.total_epochs * 3/10:
            lr = self.base * 0.1 ** 2
        else:
            lr = self.base * 0.2 ** 3
        return lr

class cosineAnnealingLR(namedtuple('cosineAnnealingLR', ('base_lr', 'T_max', 'eta_min'))):
    def __call__(self, last_epoch):
        global lr_global
        if last_epoch == 0:
            lr_global = self.base_lr
        elif (last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            lr_global = lr_global + (self.base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                
        else:
            lr_global = (1 + math.cos(math.pi * last_epoch / self.T_max)) / \
                (1 + math.cos(math.pi * (last_epoch - 1) / self.T_max)) * \
                (lr_global - self.eta_min) + self.eta_min
        return lr_global

  
class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        lr = np.interp([t], self.knots, self.vals)[0]
        return lr

class Const(namedtuple('Const', ['val'])):
    def __call__(self, x):
        return self.val


# DATA
@cache(None)
def cifar10(root='./data'):
    download = lambda train: torchvision.datasets.CIFAR10(root=root, train=train, download=True)
    return {k: {'data': torch.tensor(v.data), 'targets': torch.tensor(v.targets)} 
            for k,v in [('train', download(True)), ('valid', download(cFalse))]}

cifar10_mean, cifar10_std = [
    (125.31, 122.95, 113.87), # equals np.mean(cifar10()['train']['data'], axis=(0,1,2)) 
    (62.99, 62.09, 66.70), # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
]
cifar10_classes= 'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck'.split(', ')


# PREPROCESSING
mean, std = [torch.tensor(x, device=device, dtype=torch.float16) for x in (cifar10_mean, cifar10_std)]
# above command takes a bit of time


normalise = lambda data, mean=mean, std=std: (data - mean)/std
unnormalise = lambda data, mean=mean, std=std: data*std + mean
pad = lambda data, border: nn.ReflectionPad2d(border)(data)
transpose = lambda x, source='NHWC', target='NCHW': x.permute([source.index(d) for d in target]) 
to = lambda *args, **kwargs: (lambda x: x.to(*args, **kwargs))

def preprocess(dataset, transforms):
    dataset = copy.copy(dataset)
    for transform in reversed(transforms):
        dataset['data'] = transform(dataset['data'])
    return dataset


# DATA AUGMENTATION    
chunks = lambda data, splits: (data[start:end] for (start, end) in zip(splits, splits[1:]))
even_splits = lambda N, num_chunks: np.cumsum([0] + [(N//num_chunks)+1]*(N % num_chunks)  + [N//num_chunks]*(num_chunks - (N % num_chunks)))

def shuffled(xs, inplace=False):
    xs = xs if inplace else copy.copy(xs) 
    np.random.shuffle(xs)
    return xs

def transformed(data, targets, transform, max_options=None, unshuffle=False):
    i = torch.randperm(len(data), device=device)
    data = data[i]
    options = shuffled(transform.options(data.shape), inplace=True)[:max_options]
    data = torch.cat([transform.apply(x, **choice) for choice, x in zip(options, chunks(data, even_splits(len(data), len(options))))])
    return (data[torch.argsort(i)], targets) if unshuffle else (data, targets[i])

class Batches():
    def __init__(self, batch_size, transforms=(), dataset=None, shuffle=True, drop_last=False, max_options=None):
        self.dataset, self.transforms, self.shuffle, self.max_options = dataset, transforms, shuffle, max_options
        N = len(dataset['data'])
        self.splits = list(range(0, N+1, batch_size))
        if not drop_last and self.splits[-1] != N:
            self.splits.append(N)
    def __iter__(self):
        data, targets = self.dataset['data'], self.dataset['targets']
        for transform in self.transforms:
            data, targets = transformed(data, targets, transform, max_options=self.max_options, unshuffle=not self.shuffle)
        if self.shuffle:
            i = torch.randperm(len(data), device=device)
            data, targets = data[i], targets[i]
        return ({'input': x.clone(), 'target': y} for (x, y) in zip(chunks(data, self.splits), chunks(targets, self.splits)))
    def __len__(self): 
        return len(self.splits) - 1    


# MORE DATA AUGMENTATIONS
class Crop(namedtuple('Crop', ('h', 'w'))):
    def apply(self, x, x0, y0):
        return x[..., y0:y0+self.h, x0:x0+self.w] 
    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]


class FlipLR(namedtuple('FlipLR', ())):
    def apply(self, x, choice):
        return flip_lr(x) if choice else x 
    def options(self, shape):
        return [{'choice': b} for b in [True, False]]


class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def apply(self, x, x0, y0):
        x[..., y0:y0+self.h, x0:x0+self.w] = 0.0
        return x
    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]

#class Resize(namedtuple('Resize', ('size'))):
#    def apply(self, x):
#
#    def options(self, shape):
        


# ACTUAL TRAINING

class Timer():
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.perf_counter()]
        self.total_time = 0.0
    def __call__(self, update_total=True):
        self.synch()
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if update_total:
            self.total_time += delta_t
        return delta_t

default_table_formats = {float: '{:{w}.4f}', str: '{:>{w}s}', 'default': '{:{w}}', 'title': '{:>{w}s}'}

def table_formatter(val, is_title=False, col_width=12, formats=None):
    formats = formats or default_table_formats
    type_ = lambda val: float if isinstance(val, (float, np.float)) else type(val)
    return (formats['title'] if is_title else formats.get(type_(val), formats['default'])).format(val, w=col_width)

every = lambda n, col: (lambda data: data[col] % n == 0)

class Table():
    def __init__(self, keys=None, report=(lambda data: True), formatter=table_formatter):
        self.keys, self.report, self.formatter = keys, report, formatter
        self.log = []
    def append(self, data):
        self.log.append(data)
        data = {' '.join(p): v for p,v in path_iter(data)}
        self.keys = self.keys or data.keys()
        if len(self.log) is 1:
            print(*(self.formatter(k, True) for k in self.keys))
        if self.report(data):
            print(*(self.formatter(data[k]) for k in self.keys))
    def df(self):
        return pd.DataFrame([{'_'.join(p): v for p,v in path_iter(row)} for row in self.log])     

def reduce(epoch, batches, state, steps):
    #state: is a dictionary
    #steps: are functions that take (batch, state)
    #and return a dictionary of updates to the state (or None)
    
    for batch in chain(batches, [None]): 
    #we send an extra batch=None at the end for steps that 
    #need to do some tidying-up (e.g. log_activations)
        for step in steps:
            updates = step(epoch, batch, state)
            if updates:
                for k,v in updates.items():
                    state[k] = v 
    return state

#step definitions
def forward(training_mode):
    def step(epoch, batch, state):
        if not batch: return
        model = state[MODEL] if training_mode or (VALID_MODEL not in state) else state[VALID_MODEL]
        if model.training != training_mode: #without the guard it's slow!
            model.train(training_mode)
        return {OUTPUT: model.loss(model(batch))}
    return step

def forward_tta(tta_transforms):
    def step(epoch, batch, state):
        if not batch: return
        model = state[MODEL] if (VALID_MODEL not in state) else state[VALID_MODEL]
        if model.training:
            model.train(False)
        logits = torch.mean(torch.stack([model({'input': transform(batch['input'].clone())})['logits'].detach() for transform in tta_transforms], dim=0), dim=0)
        return {OUTPUT: model.loss(dict(batch, logits=logits))}
    return step

def backward(dtype=torch.float16, sam_first_step=False):
    def step(epoch, batch, state):
        state[MODEL].zero_grad()
        if not batch: return
        state[OUTPUT]['loss'].to(dtype).sum().backward(retain_graph=sam_first_step)
    return step

def opt_steps(epoch, batch, state, sam_flag=False, sam_opt_step=0):
    if not batch: return
    if sam_flag:
        state[OPTS][0]['update'] = sam_first_step if(sam_opt_step == 0) else sam_second_step
        return {OPTS: [opt_step(epoch, **state[OPTS][0])]}

    return {OPTS: [opt_step(epoch, **opt) for opt in state[OPTS]]}

def log_activations(node_names=('loss', 'acc')):
    logs = []
    def step(epoch, batch, state):
        if batch:
            logs.extend((k, state[OUTPUT][k].detach()) for k in node_names)
        else:
            res = map_values((lambda xs: to_numpy(torch.cat(xs)).astype(np.float)), group_by_key(logs))
            logs.clear()
            return {ACT_LOG: res}
    return step

def update_ema(momentum, update_freq=1):
    n = iter(count())
    rho = momentum**update_freq
    def step(epoch, batch, state):
        if not batch: return
        if (next(n) % update_freq) != 0: return
        for v, ema_v in zip(state[MODEL].state_dict().values(), state[VALID_MODEL].state_dict().values()):
            if ema_v.dtype == torch.int64:
                ema_v = ema_v.float()
            ema_v *= rho
            ema_v += (1-rho)*v
    return step

# below 2 lines don't take any time
train_steps = (forward(training_mode=True), log_activations(('loss', 'acc')), backward(), opt_steps)
valid_steps = (forward(training_mode=False), log_activations(('loss', 'acc')))

sam_opt_step1 = partial(opt_steps, sam_flag=True, sam_opt_step=0)
sam_opt_step2 = partial(opt_steps, sam_flag=True, sam_opt_step=1)
train_steps_sam = (forward(training_mode=True), log_activations(('loss', 'acc')), backward(sam_first_step=True), sam_opt_step1, backward(sam_first_step=False), sam_opt_step2) 


epoch_stats = lambda state: {k: np.mean(v) for k, v in state[ACT_LOG].items()}

def train_epoch(epoch, state, timer, train_batches, valid_batches, train_steps=train_steps, valid_steps=valid_steps, on_epoch_end=identity):
    train_summary = epoch_stats(on_epoch_end(reduce(epoch, train_batches, state, train_steps)))
    train_time = timer()
    #valid_summary = epoch_stats(reduce(epoch, valid_batches, state, valid_steps))
    #valid_time = timer(update_total=False) #DAWNBench rules
    return {
        'train': union({'time': train_time}, train_summary), 
        #'valid': union({'time': valid_time}, valid_summary), 
        'total time': timer.total_time
    }

summary = lambda logs, cols=['valid_acc']: logs.df().query('epoch==epoch.max()')[cols].describe().transpose().astype({'count': int})[
    ['count', 'mean', 'min', 'max', 'std']]

def log_weights(state, weights):
    state[WEIGHT_LOG] = state.get(WEIGHT_LOG, [])
    state[WEIGHT_LOG].append({k: to_numpy(v.data) for k,v in weights.items()})
    return state

def fine_tune_bn_stats(state, batches, model_key=VALID_MODEL):
    reduce(batches, {MODEL: state[model_key]}, [forward(True)])
    return state

#misc
def warmup_cudnn(model, batch):
    #run forward and backward pass of the model
    #to allow benchmarking of cudnn kernels 
    reduce([batch], {MODEL: model}, [forward(True), backward()])
    torch.cuda.synchronize()

# PLOTTING
def empty_plot(ax, **kw):
    ax.axis('off')
    return ax

def image_plot(ax, img, title):
    ax.imshow(to_numpy(unnormalise(transpose(img, 'CHW', 'HWC'))).astype(np.int))
    ax.set_title(title)
    ax.axis('off')

def layout(figures, sharex=False, sharey=False, figure_title=None, col_width=4, row_height = 3.25, **kw):
    nrows, ncols = np.array(figures).shape
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=(col_width*ncols, row_height*nrows))
    axs = [figure(ax, **kw) for row in zip(np.array(axs).reshape(nrows, ncols), figures) for ax, figure in zip(*row)]
    fig.suptitle(figure_title)
    return fig, axs

# NETWORK
conv_block = lambda c_in, c_out: {
    'conv': conv(in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
    'norm': batch_norm(c_out), 
    'act':  relu(),
}

conv_pool_block = lambda c_in, c_out: dict(conv_block(c_in, c_out), pool=pool(2))
conv_pool_block_pre = lambda c_in, c_out: reorder(conv_pool_block(c_in, c_out), ('conv', 'pool', 'norm', 'act'))

residual = lambda c, conv_block: {
    'in': (Identity, {}),
    'res1': conv_block(c, c),
    'res2': conv_block(c, c),
    'out': (Identity, {}),
    'add': (Add, {}, ['in', 'out']),
}

def build_network(channels, extra_layers, res_layers, scale, conv_block=conv_block, 
                  prep_block=conv_block, conv_pool_block=conv_pool_block, types=None): 
    net = {
        'prep': prep_block(3, channels['prep']),
        'layer1': conv_pool_block(channels['prep'], channels['layer1']),
        'layer2': conv_pool_block(channels['layer1'], channels['layer2']),
        'layer3': conv_pool_block(channels['layer2'], channels['layer3']),
        'pool': pool(4),
        'classifier': {
            'flatten': (Flatten, {}),
            'conv': linear(channels['layer3'], 10, bias=False),
            'scale': (Mul, {'weight': scale}),
        },
        'logits': (Identity, {}),
    }
    for layer in res_layers:
        net[layer]['residual'] = residual(channels[layer], conv_block)
    for layer in extra_layers:
        net[layer]['extra'] = conv_block(channels[layer], channels[layer])     
    if types: net = map_types(types, net)
    return net

channels={'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
network = partial(build_network, channels=channels, extra_layers=(), res_layers=('layer1', 'layer3'), scale=1/8)   
# above line does not take any time

x_ent_loss = Network({
  'loss':  (nn.CrossEntropyLoss, {'reduction': 'none'}, ['logits', 'target']),
  'acc': (Correct, {}, ['logits', 'target'])
})
# above command does not take any time

label_smoothing_loss = lambda alpha: Network({
        'logprobs': (LogSoftmax, {'dim': 1}, ['logits']),
        'KL':  (KLLoss, {}, ['logprobs']),
        'xent':  (CrossEntropyLoss, {}, ['logprobs', 'target']),
        'loss': (AddWeighted, {'wx': 1-alpha, 'wy': alpha}, ['xent', 'KL']),
        'acc': (Correct, {}, ['logits', 'target']),
    })

# name == main

#N_RUNS = 1
#print('******************************Preprocessing on the GPU******************************')

#dataset = cifar10() # download cifar10 dataset if not already present

#print('Starting timer')
#t = Timer(synch=torch.cuda.synchronize)
#dataset = map_nested(to(device), dataset)


def get_subset(dataset, x):
    n_classes = len(cifar10_classes)
    samplesPerClass = int(dataset['data'].size()[0]/n_classes)
    class_rand_idx = np.random.choice(np.arange(0,samplesPerClass), int(samplesPerClass*x), replace=False)
    class_rand_idx = np.tile(class_rand_idx, (n_classes, 1))
    strides = np.array([i*samplesPerClass for i in range(0,n_classes)])[:,np.newaxis]
    class_rand_idx += strides
    class_rand_idx = class_rand_idx.flatten()
    return dataset['data'][class_rand_idx], dataset['targets'][class_rand_idx]

# preprocess data on GPU

#train_set = preprocess(dataset['train'], [partial(pad, border=4), transpose, normalise, to(torch.float16)])
#valid_set = preprocess(dataset['valid'], [transpose, normalise, to(torch.float16)])

#train_set['data'],train_set['targets'] = get_subset(train_set, 0.1)
#valid_set['data'],valid_set['targets'] = get_subset(valid_set, 0.1)

#print(f'Data preprocessing:\t{t():.3f}s')
#map_nested(to(cpu), {'train': train_set, 'valid': valid_set})
#print(f'Transfer to CPU:\t{t():.3f}s')

#print('train_set: ', len(train_set['data']), 'valid_set: ', len(valid_set['data']))

#train_batches = partial(Batches, dataset=train_set, shuffle=True,  drop_last=True, max_options=200)
#valid_batches = partial(Batches, dataset=valid_set, shuffle=False, drop_last=False)

def cov(X):
    X = X/np.sqrt(X.size(0) - 1)
    return X.t() @ X

def patches(data, patch_size=(3, 3), dtype=torch.float32):
    h, w = patch_size
    c = data.size(1)
    return data.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1, c, h, w).to(dtype)

def eigens(patches):
    n,c,h,w = patches.shape
    Σ = cov(patches.reshape(n, c*h*w))
    Λ, V = torch.symeig(Σ, eigenvectors=True)
    return Λ.flip(0), V.t().reshape(c*h*w, c, h, w).flip(0)

# below line takes a bit of time
#Λ, V = eigens(patches(train_set['data'][:10000,:,4:-4,4:-4]))

def whitening_block(c_in, c_out, Λ=None, V=None, eps=1e-2):
    filt = nn.Conv2d(3, 27, kernel_size=(3,3), padding=(1,1), bias=False)
    filt.weight.data = (V/torch.sqrt(Λ+eps)[:,None,None,None])
    filt.weight.requires_grad = False 
    return {
        'whiten': (identity, {'value': filt}),
        'conv': conv(27, c_out, kernel_size=(1, 1), bias=False),
        'norm': batch_norm(c_out), 
        'act':  relu(),
    }

#input_whitening_net = network(conv_pool_block=conv_pool_block_pre, prep_block=partial(whitening_block, Λ=Λ, V=V), scale=1/16, types={
#    nn.ReLU: partial(nn.CELU, 0.3),
#    BatchNorm: partial(GhostBatchNorm, num_splits=16, weight=False)
#})

#print('******************************Test-time augmentation******************************')

valid_steps_tta = (forward_tta([identity, flip_lr]), log_activations(('loss', 'acc')))

def getResNet8BOT(input_whitening_net):
    model = build_model(input_whitening_net, label_smoothing_loss(0.2))
    return model







