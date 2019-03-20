#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provides training and testing module for Bayesian Inference used in visual imitation learning.
=============================================================================
* paper: Robot Visual Imitation Learning using Variational Bayesian Inference
* Author: Jun Jin (jjin5@ualberta.ca), University of Alberta
* TimeMark: Dec 11, 2018
* Status: Dev
* License: check LICENSE in parent folder
=============================================================================
- input: training sample
- output: network module + testing visualizations
- comments: network module will further used in the control module.
"""

# code structure based on https://github.com/pytorch/examples/blob/master/vae/main.py
from __future__ import print_function
import sys, copy
sys.path.insert(0,'../Lib/')
import numpy as np
import torch_atlas as t_atlas
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


# <editor-fold desc="Lfd_Bayesian class + instance + optimizer">
class AE(nn.Module):
    """
    auto encoder
    """
    # a simple MLP
    def __init__(self, dim_Zt, input_channel):
        """
        Construct layers for Encoder and Decoder respectively
        """
        super(AE, self).__init__()
        self.dim_Zt = dim_Zt
        self.input_channel = input_channel

        # layers for encoder
        self.encoder_part1 = nn.Sequential(
            nn.Conv2d(self.input_channel, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            t_atlas.View((-1, 512*7*7)),
            nn.Linear(512*7*7, self.dim_Zt)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.dim_Zt, 512*7*7),
            t_atlas.View((-1, 512, 7, 7)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, self.input_channel, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def encode(self, inputs):
        """
        encoder
        """
        h1 = self.encoder_part1(inputs)
        # what it encodes is not variance, but log of variance.
        return h1

    def decode(self, z):
        """
        decoder
        """
        return self.decoder(z)

    def forward(self, inputs):
        """
        forward pass
        """
        zt = self.encode(inputs)
        return self.decode(zt), zt

    def load_weights(self, params):
        i = 0
        assert (len(list(self.parameters())) == len(params), "grad update failed, length not match!")

        for f in self.parameters():  # 获取当前的weights
            f.data = copy.deepcopy(params[i].data)
            i += 1

    def load_weights_from_file(self, params_file_path):
        self.load_weights(torch.load(params_file_path))

    def save(self, save_name):
        # save model
        ppps = list(self.parameters())
        torch.save(ppps, 'params/'+save_name)


# <editor-fold desc="Utilities + loss function">
def loss_function(recon_x, x):
    """
    construct the cost function. Gradient is automatically calculated based on this cost function.
    """
    # not using the mean error
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, args.dim_at), size_average=False)  # minimize the reconstruction error
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # print('BCE  ' + str(BCE.detach().to('cpu').numpy()))
    # the cost function that needs to be minimized
    return BCE, BCE.detach().to('cpu').numpy()
# </editor-fold>
