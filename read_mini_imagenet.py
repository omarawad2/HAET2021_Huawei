import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import pickle
import time
import datetime
from torchvision import datasets, transforms
batch_size = 60

from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple
from PIL import Image


root = "./"

def load_datasets(root):
    datasets = {}
    class_index = 0
    for subset in ["train"]:
        f = open(root + "mini-imagenet-cache-" + subset + ".pkl", "rb")
        dataset = pickle.load(f)
        data = dataset['image_data']
        target = torch.zeros(data.shape[0], dtype=int)
        for cl in dataset['class_dict'].keys():
            for elt in dataset['class_dict'][cl]:
                target[elt] = class_index
            class_index += 1
        datasets[subset] = [data, target]
    return datasets

class MiniImageNet(VisionDataset):
    def __init__(
            self,
            root : str,
            subset = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:

        super(MiniImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        self.data: Any = []
        self.targets = []

        datasets = load_datasets(root)
        
        self.data = datasets[subset][0]
        self.targets = datasets[subset][1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.data.shape[0]
from torchvision import datasets, transforms
batch_size = 64

from PIL import ImageEnhance

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out

train_transforms = [   
    transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406) ,(0.229, 0.224, 0.225))
    #ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
    #transforms.RandomHorizontalFlip()
    ] # used for standard data augmentation during training

test_transforms = [
    transforms.Resize([int(80*1.15), int(80*1.15)]),
    transforms.CenterCrop(80)
]

standard_transforms = [
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406) ,(0.229, 0.224, 0.225))
    ] # used for all data

transform_train = transforms.Compose(train_transforms) # + standard_transforms)
transform_all = transforms.Compose(test_transforms + standard_transforms)


a_train = MiniImageNet(root, subset="train", transform=transform_train)
a_test = MiniImageNet(root, subset="train", transform=transform_train)

idx_train = (a_train.targets==0) | (a_train.targets==1) | (a_train.targets==2) | (a_train.targets==3) | (a_train.targets==4) | (a_train.targets==5) | (a_train.targets==6) | (a_train.targets==7) | (a_train.targets==8) | (a_train.targets==9) 
idx_test = (a_test.targets==0) | (a_test.targets==1) | (a_test.targets==2) | (a_test.targets==3) | (a_test.targets==4) | (a_test.targets==5) | (a_test.targets==6) | (a_test.targets==7) | (a_test.targets==8) | (a_test.targets==9) 

a_train.targets = a_train.targets[idx_train]
a_train.data = a_train.data[idx_train]

a_test.targets = a_test.targets[idx_test]
a_test.data = a_test.data[idx_test]

idx_train  = (a_train.targets==0) | (a_train.targets==1) | (a_train.targets==2) | (a_train.targets==3) | (a_train.targets==4) | (a_train.targets==5) | (a_train.targets==6) | (a_train.targets==7) | (a_train.targets==8) | (a_train.targets==9)
idx_test  =  (a_train.targets==0) | (a_train.targets==1) | (a_train.targets==2) | (a_train.targets==3) | (a_train.targets==4) | (a_train.targets==5) | (a_train.targets==6) | (a_train.targets==7) | (a_train.targets==8) | (a_train.targets==9)
for i in range(len(idx_train)):
    if (0<=i<500 or 599<i<1100 or 1199<i<1700 or 1799<i<2300 or 2399<i<2900 or 2999<i<3500 or 3599<i<4100 or 4199<i<4700 or 4799<i<5300 or 5399<i<5900):
        idx_train[i]=True
        idx_test[i]=False
    else:
        idx_train[i]=False
        idx_test[i] =True 


a_train.data = a_train.data[idx_train]
a_train.targets = a_train.targets[idx_train]

a_test.data = a_test.data[idx_test]
a_test.targets = a_test.targets[idx_test]

print(a_train.targets.shape)
print(a_test.targets.shape)

train_loader = torch.utils.data.DataLoader(
    a_train,
    batch_size=128, shuffle=True, num_workers = 4)

test_loader = torch.utils.data.DataLoader(
    a_test,
    batch_size=128, shuffle=True, num_workers = 4)

def get_data_loaders(batch_size=128):
    train = torch.utils.data.DataLoader(
        a_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test = torch.utils.data.DataLoader(
        a_test, batch_size=batch_size, shuffle=True, num_workers=4
    )

    return train, test

def get_mini_imagenet_data(train_transforms=None, test_transforms=None):
    
    a_train = MiniImageNet(root, subset="train", transform=train_transforms)
    a_test = MiniImageNet(root, subset="train", transform=test_transforms)

    idx_train = (a_train.targets==0) | (a_train.targets==1) | (a_train.targets==2) | (a_train.targets==3) | (a_train.targets==4) | (a_train.targets==5) | (a_train.targets==6) | (a_train.targets==7) | (a_train.targets==8) | (a_train.targets==9) 
    idx_test = (a_test.targets==0) | (a_test.targets==1) | (a_test.targets==2) | (a_test.targets==3) | (a_test.targets==4) | (a_test.targets==5) | (a_test.targets==6) | (a_test.targets==7) | (a_test.targets==8) | (a_test.targets==9) 

    a_train.targets = a_train.targets[idx_train]
    a_train.data = a_train.data[idx_train]

    a_test.targets = a_test.targets[idx_test]
    a_test.data = a_test.data[idx_test]

    idx_train  = (a_train.targets==0) | (a_train.targets==1) | (a_train.targets==2) | (a_train.targets==3) | (a_train.targets==4) | (a_train.targets==5) | (a_train.targets==6) | (a_train.targets==7) | (a_train.targets==8) | (a_train.targets==9)
    idx_test  =  (a_train.targets==0) | (a_train.targets==1) | (a_train.targets==2) | (a_train.targets==3) | (a_train.targets==4) | (a_train.targets==5) | (a_train.targets==6) | (a_train.targets==7) | (a_train.targets==8) | (a_train.targets==9)
    for i in range(len(idx_train)):
        if (0<=i<500 or 599<i<1100 or 1199<i<1700 or 1799<i<2300 or 2399<i<2900 or 2999<i<3500 or 3599<i<4100 or 4199<i<4700 or 4799<i<5300 or 5399<i<5900):
            idx_train[i]=True
            idx_test[i]=False
        else:
            idx_train[i]=False
            idx_test[i] =True 


    a_train.data = a_train.data[idx_train]
    a_train.targets = a_train.targets[idx_train]

    a_test.data = a_test.data[idx_test]
    a_test.targets = a_test.targets[idx_test]

    print(a_train.targets.shape)
    print(a_test.targets.shape)

    return a_train, a_test