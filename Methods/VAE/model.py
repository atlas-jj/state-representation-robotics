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


# <editor-fold desc="Lfd_Bayesian class + instance + optimizer">
class VAE(nn.Module):
    """
    LfD_Bayesian Class: similar to Conditional VAE, but prior assumptions are different
    """
    # a simple MLP
    def __init__(self, dim_Zt, input_channel):
        """
        Construct layers for Encoder and Decoder respectively
        """
        super(VAE, self).__init__()
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
            t_atlas.View((-1, 512*7*7))
        )
        # now encoder part2 for reparameterization
        self.fc11 = nn.Linear(512*7*7, self.dim_Zt)  # Z_mean
        self.fc12 = nn.Linear(512*7*7, self.dim_Zt)  # Z_log_variance

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
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        """
        the same reparameterization trick as shown in original VAE paper
        :param mu:
        :param logvar:
        :return:
        """
        std = torch.exp(0.5*logvar)  # from log variance to std
        eps = torch.randn_like(std)  # resample from N(0,1), std is only used to determine its size, i.e., std.size(), ,i.e., 20
        return eps.mul(std).add_(mu)  # back to the real Normal Distribution.

    def decode(self, z):
        """
        decoder
        """
        return self.decoder(z)

    def forward(self, inputs):
        """
        forward pass
        :param at: generalized action vector
        :param St: the condition vector. state vector.
        :return: reconstructed x, z_mu, z_log(variance)
        """
        mu, logvar = self.encode(inputs)
        zt = self.reparameterize(mu, logvar)  # sampliing
        return self.decode(zt), zt, mu, logvar

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
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    """
    construct the cost function. Gradient is automatically calculated based on this cost function.
    :param recon_x: a variable
    :param x: a variable
    :param mu: a variable
    :param logvar: a variable
    :param variance0: a constant
    :return: the cost function that needs to minimize.
    """
    # not using the mean error
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, args.dim_at), size_average=False)  # minimize the reconstruction error
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # mse_loss
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KL divergence of this normal distribution to a standard normal distribution.
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print('MSE  ' + str(BCE.detach().to('cpu').numpy()))
    # print('KLD  ' + str(KLD.detach().to('cpu').numpy()))
    # the cost function that needs to be minimized
    return BCE + KLD, BCE.detach().to('cpu').numpy(), KLD.detach().to('cpu').numpy()
# </editor-fold>
