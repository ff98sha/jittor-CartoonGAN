import jittor as jt
from jittor import init
import itertools, imageio, torch, random
import matplotlib.pyplot as plt
import numpy as np
from jittor import nn
import jittor.dataset as datasets

def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]
    n = 0
    for i in range(len(dset.imgs)):
        if (ind != dset.imgs[n][1]):
            del dset.imgs[n]
            n -= 1
        n += 1
    dset.set_attrs(total_len=len(dset.imgs))
    print(len(dset.imgs))
    return dset.set_attrs(batch_size = batch_size, num_workers = 8)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print(('Total number of parameters: %d' % num_params))

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv):
            m.weight.gauss_(0, 0.02)
            m.bias.zero_()
        elif isinstance(m, nn.ConvTranspose):
            m.weight.gauss_(0, 0.02)
            m.bias.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.gauss_(0, 0.02)
            m.bias.zero_()
        elif isinstance(m, nn.BatchNorm):
            m.weight.fill_(1)
            m.bias.zero_()