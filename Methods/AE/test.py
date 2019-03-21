#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import model as models
import plot_result as pr
import os, sys, math, time
sys.path.insert(0,'../Lib/')
import Utils as uts

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

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda")
image_size = 240
dimz = 50
# test codes
model = models.AE(dimz, 3).to(device)
# model load pretrained params
# model.load_weights_from_file('../Disentangling/params/ClothV2_env_all_dimZ_100alpha_0.05')BlocksV3_env_all_dimZ_100alpha_0.03
model.load_weights_from_file('params/AE_Cloth_prev_human12_dimZ_50')
mixedHR = uts.load_images_sequence('../../Experiments/Dataset/Cloth_prev/mixedHR', 200, image_size)
finalImages = uts.load_images_sequence('../../Experiments/Dataset/Cloth_prev/final', 20, image_size)
plot_r = pr.plot_task_map('../../Experiments/Dataset/Cloth_prev', None, device)

plot_r.save_results_41(model, "AE_mixedHR", mixedHR, "mixedHR")
plot_r.save_results_41(model, "AE_final_states", finalImages, "final states")

# plot_r.save_results_4(model, 'save_test')
# plot_r.compare_exp(model, 'another_test', 144)
input("Press Enter to continue...")
# plot_r.animation_exp(model, zt_marks, '../Experiments/Dataset/Blocks/results/raw_12.jpg', image_size,
#                      '../Experiments/Dataset/Blocks/results/animation', 20, 1)

# sequence = uts.load_images_sequence('../Experiments/Dataset/ClothV2/human2', 135, image_size)
# plot_r = pr.plot_task_map('../Experiments/Dataset/ClothV2', dimz, 0.05, 172.8, sequence, device)
# plot_r.save_results_4(model, 'save_test')
# zt_marks = [0,1]
#
# def time_varying(state, zt_marks):
#     """
#     get the time varying units from a state
#     :param state: state vector, 1d
#     :param zt_marks: mark vector, 1d
#     :return:
#     """
#     dim_action = len(zt_marks)
#     var_state = np.zeros(dim_action)
#     for i in range(dim_action):
#         var_state[i] = state[zt_marks[i]]
#     return var_state
#
# def time_varying_states(states, zt_marks):
#     states_num = states.shape[0]
#     dim_action = len(zt_marks)
#     lean_states = np.zeros((states_num, dim_action))
#     for i in range(dim_action):
#         lean_states[:, i] = states[:, zt_marks[i]]
#     return lean_states
#
# def lean_zts(img_sequence_folder, img_num, image_size, zt_marks, model):
#     seqs = uts.load_images_sequence(img_sequence_folder, img_num, image_size)
#     with torch.no_grad():
#         Zts = model.encode(seqs.to(device))
#         lean_states = time_varying_states(Zts.to('cpu').numpy(), zt_marks)
#         return lean_states
#
# def plot_fig(list_lean_states, dim_lean):
#     num_trajs = len(list_lean_states)
#     for i in range(dim_lean):
#         plt.subplot(dim_lean, 1, i+1)
#         for j in range(num_trajs):
#             this_traj = list_lean_states[j]
#             plt.plot(this_traj[:, i], label=str(j+1))
#         # plt.legend(loc='upper right')
#     # plt.ylim(-5, 5)
#
#
# base_folder = '../../Experiments/Dataset/Toy_Example/s_shape/'
# num_trajs = 11
# num_steps = 149
# list_lean_states = []
# not_included = [4,5]
#
# def valid(index):
#     for i in range(len(not_included)):
#         if index is not_included[i]:
#             return False
#     return True
#
# for i in range(num_trajs):
#     if valid(i):
#         lean_states = lean_zts(base_folder + str(i+1), num_steps, image_size, zt_marks, model)
#         list_lean_states.append(lean_states)
#
# plot_fig(list_lean_states, len(zt_marks))
# plt.show()
#
