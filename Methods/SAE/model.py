#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys, copy
sys.path.insert(0,'../Lib/')
import numpy as np
import torch_atlas as t_atlas
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import spatial_soft_max
from torchvision import transforms

# <editor-fold desc="Lfd_Bayesian class + instance + optimizer">
class SpatialAutoEncoder(nn.Module):
    """
    LfD_Bayesian Class: similar to Conditional VAE, but prior assumptions are different
    """
    # a simple MLP
    def __init__(self, dim_Zt, input_channel):
        """
        Construct layers for Encoder and Decoder respectively
        """
        super(SpatialAutoEncoder, self).__init__()
        self.input_channel = input_channel
        #self.output_dim = output_dim
        self.dim_Zt = dim_Zt
        self.scale = int(dim_Zt/32)
        # layers for encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, 7, 2, 0),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 5, 1, 0),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(32, int(self.dim_Zt/2), 5, 1, 0),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            # now spatial softmax layer
            spatial_soft_max.SpatialSoftmax(109, 109, int(self.dim_Zt/2))
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.dim_Zt, 512 * 7 * 7),
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
        return self.encoder(inputs)

    def decode(self, z):
        """
        decoder
        """
        return self.decoder(z)

    def forward(self, inputs):
        z = self.encoder(inputs)
        reconn = self.decoder(z)
        return reconn, z

    def load_weights(self, params):
        i = 0

        for f in self.parameters():  # get current weights
            f.data = copy.deepcopy(params[i].data)
            i += 1

    def load_weights_from_file(self, params_file_path):
        self.load_weights(torch.load(params_file_path))

    def save(self, save_name):
        # save model
        ppps = list(self.parameters())
        torch.save(ppps, 'params/'+save_name)



# <editor-fold desc="Utilities + loss function">
# Reconstruction + KL divergence losses summed over all elements and batch
# def loss_function(recon_x, x):
#     """
#     1 downsampling, x
#     2 flat x
#     3 calc mse
#     """
#
#     transform_grey = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor()
#     ])
#     x_grey = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3]).to('cuda')
#     for i in range(x.shape[0]):
#         x_grey[i] = transform_grey(x[i].to('cpu')).to('cuda')
#     # print('x_grey size')
#     # print(x_grey.shape)
#     x_downsample = F.interpolate(x_grey, size=(60, 60))
#     x_flat = x_downsample.view(-1, 60*60)
#     # print('reconn x size')
#     # print(recon_x.shape)
#     # print('x_flat size')
#     # print(x_flat.shape)
#     MSE = F.mse_loss(recon_x, x_flat, size_average=False)
#     # print('MSE  ' + str(MSE.detach().to('cpu').numpy()))
#     # print('KLD  ' + str(KLD.detach().to('cpu').numpy()))
#     # the cost function that needs to be minimized
#     return MSE, MSE.detach().to('cpu').numpy()

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
