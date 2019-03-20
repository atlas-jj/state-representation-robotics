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

from __future__ import print_function
import dis_models as models
import plot_result as pr
import os, sys, math, time
sys.path.insert(0,'../Lib/')
import Utils as uts
import dis_utils
import torch_atlas as t_atlas
import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.optim as optim
import gc

from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.colors as colors

device = torch.device("cuda")

dimz = 300
alpha = 0.05

model_params_path = '../Disentangling/params/BlocksV2_h1r1_dimZ_'+str(dimz)+'alpha_' + str(alpha)
final_states_folder = '../Experiments/Dataset/BlocksV2/final_states'
image_size = 240
final_states_num = 40
# test codes
model = models.beta_VAE(dimz, 3).to(device)
# model load pretrained params
model.load_weights_from_file(model_params_path)
save_folder = '../Experiments/Dataset/BlocksV2'


all_inputs = uts.load_images_sequence(final_states_folder, final_states_num, image_size)
# plot_r = pr.plot_task_map('../Experiments/Dataset/Toy_Example', 100, 0.2, 345.6, sample_sequence, device)
plot_r = pr.plot_task_map(save_folder, dimz, alpha, 172.8, all_inputs, device)

plot_r.save_results_4(model, 'eva_final_states_dim' + str(dimz) + '_' + str(alpha), False)
