#!/usr/bin/env python
# -*- coding: utf-8 -*-

# code structure based on https://github.com/pytorch/examples/blob/master/vae/main.py
from __future__ import print_function


import os, sys, math
sys.path.insert(0,'../Lib/')
import Utils as uts
import numpy as np
import torch
import torch.utils.data
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.nn import functional as F

class plot_task_map:
    def __init__(self, sample_folder, dim_Zt, alpha, beta, sample_sequence, device):
        self.dim_Zt = dim_Zt
        self.alpha = alpha
        self.beta = beta
        self.sample_sequence = sample_sequence
        self.sample_folder = sample_folder + "/results"
        self.device = device
        if os.path.exists(self.sample_folder) is not True:
            os.mkdir(self.sample_folder)
        if os.path.exists(self.sample_folder +"/latent_space") is not True:
            os.mkdir(self.sample_folder +"/latent_space")
        if os.path.exists(self.sample_folder +"/reconstruction") is not True:
            os.mkdir(self.sample_folder +"/reconstruction")
        if os.path.exists(self.sample_folder +"/free_sampling") is not True:
            os.mkdir(self.sample_folder +"/free_sampling")

    def save_reconstruction(self, comparison, filename, row_number):
        save_image(comparison, self.sample_folder + '/reconstruction/' + filename + '.png', nrow=row_number)

    def save_zt_vis(self, zt_numpy, epoch):
        c = plt.pcolor(np.transpose(zt_numpy), norm=colors.Normalize(vmin=-1, vmax=1), cmap='BdRu')
        # c = plt.pcolor(np.transpose(zt_numpy), cmap='RdBu')
        plt.colorbar(c)
        plt.savefig(self.sample_folder + '/latent_space/train_' + str(epoch) + '.jpg')
        plt.gcf().clear()
        plt.clf()
        plt.cla()

    def save_delta_zt_vis(self, zt_numpy, epoch):
        n = zt_numpy.shape[0]
        d_zt = np.zeros((n - 1, zt_numpy.shape[1]))
        for i in range(n - 1):
            d_zt[i, :] = zt_numpy[i + 1, :] - zt_numpy[i, :]
        self.save_zt_vis(d_zt, epoch)

    def save_zt_sequencial(self, epoch, model):
        with torch.no_grad():
            recon_batch, zt, mu, logvar = model(self.sample_sequence.to(self.device))
            # save_zt_vis(mu.detach().to('cpu').numpy(), epoch)
            self.save_delta_zt_vis(mu.detach().to('cpu').numpy(), epoch)

    def compare_exp(self, model, save_name, sample_size=9):
        with torch.no_grad():
            recon_batch, zt, mu, logvar = model(self.sample_sequence.to(self.device))
            # save mu and its diffs
            mu_numpy_t = np.transpose(mu.detach().to('cpu').numpy())
            zt_numpy_t = np.transpose(zt.detach().to('cpu').numpy())
            std_t = np.transpose(torch.exp(0.5*logvar).detach().to('cpu').numpy())
            n = zt_numpy_t.shape[1]  # time steps
            dim_zt = zt_numpy_t.shape[0]  # dim of Zt
            # generate some images
            self.reconn_from_zt_numpy(np.transpose(zt_numpy_t), model, save_name + "_reconn_originZ", sample_size)

            for t in range(n): # time steps
                for i in range(dim_zt):  # zt dim
                    if std_t[i, t] > 0.5:
                        zt_numpy_t[i, t] = mu_numpy_t[i, t]
            self.reconn_from_zt_numpy(np.transpose(zt_numpy_t), model, save_name + "_reconn_reg_Z", sample_size)
            plt.subplot(1, 2, 1)
            c = plt.pcolor(zt_numpy_t, norm=colors.Normalize(vmin=-1, vmax=1), cmap='RdBu')
            plt.colorbar(c)
            plt.title('Zt')
            n = zt_numpy_t.shape[1]  # time steps
            d_zt = np.zeros((zt_numpy_t.shape[0], n - 1))
            for i in range(n - 1):
                d_zt[:, i] = zt_numpy_t[:, i + 1] - zt_numpy_t[:, i]
            plt.subplot(1, 2, 2)
            c = plt.pcolor(d_zt, norm=colors.Normalize(vmin=-1, vmax=1), cmap='RdBu')
            plt.colorbar(c)
            plt.title('diff_Zt')
            plt.savefig(self.sample_folder + '/latent_space/' + save_name + '_zts.jpg')
            plt.gcf().clear()
            plt.clf()
            plt.cla()

    def animation_exp(self, model, z_marks, sample_img_path, img_size, save_folder, gen_size, w_range=2):
        """
        generate animation images by manipulate Z time-varying units
        :param model:
        :param z_marks: list, e.g., [0,67,90]
        :param sample_img_path:
        :param save_folder:
        :param gen_size:
        :return:
        """
        with torch.no_grad():
            img = uts.get_image_tensor(sample_img_path, img_size)
            recon_batch, zt, mu, logvar = model(img.to(self.device))
            std_t = (torch.exp(0.5 * logvar).detach().to('cpu').numpy())
            zt = zt.detach()
            # analyze Zt time changing part
            dim_zt = zt.shape[1]
            n_count = 0
            if z_marks is None:
                z_marks = []
                for i in range(100):
                    if std_t[0,i] < 0.5:
                        z_marks.append(i)
                        n_count +=1
            else:
                n_count = len(z_marks)
            print("time varying units Number: " + str(n_count))
            print(z_marks)
            #  now changing experiment paras
            for i in range(n_count):
                ori = zt[0, i]
                for j in range(gen_size):
                    t = (j - gen_size/2)*w_range/gen_size
                    zt[0, z_marks[i]] = ori + t
                    reconn = model.decode(zt.to(self.device))
                    save_image(reconn.detach().cpu().float(),  # if normalized to [-1,1], * 0.5 + 0.5
                               save_folder + "/aniZ_" + str(z_marks[i]) + "_" + str(j) + "_" + uts.d2s(ori + t, 3)
                               + '.png', nrow=1)


    def reconn_from_zt_numpy(self, zt_numpy, model, save_name, sample_size=9):
        """

        :param zt_numpy: is the original version, not the Zt_numpy_t
        :return:
        """
        with torch.no_grad():
            zts = torch.from_numpy(zt_numpy)
            reconn = model.decode(zts.to(self.device))
            BCE = F.binary_cross_entropy(reconn, self.sample_sequence.to(self.device), size_average=True)
            save_image(reconn[0:sample_size].cpu().float(),  # if normalized to [-1,1], * 0.5 + 0.5
                       self.sample_folder + '/latent_space/' + save_name + '_BCE_' + uts.d2s(BCE.to('cpu').numpy(), 6)
                       + '.png', nrow=int(math.sqrt(sample_size)))
            return reconn

    def save_results_41(self, model, save_name, sequence, plt_title='', normalize=True):
        self.sample_sequence = sequence
        self.save_results_4(model, save_name, plt_title, normalize)

    def save_results_4(self, model, save_name, plt_title,normalize=True):
        with torch.no_grad():
            recon_batch, zt, mu, logvar = model(self.sample_sequence.to(self.device))
            # save mu and its diffs
            mu_numpy_t = np.transpose(mu.detach().to('cpu').numpy())
            zt_numpy_t = np.transpose(zt.detach().to('cpu').numpy())
            std_t = np.transpose(torch.exp(0.5*logvar).detach().to('cpu').numpy())

            plt.rcParams["figure.figsize"] = (15, 10)
            fig = plt.figure()
            fig.suptitle(plt_title, fontsize=20)
            plt.subplot(3, 2, 1)
            if normalize:
                c = plt.pcolor(mu_numpy_t, norm=colors.Normalize(vmin=-1, vmax=1), cmap='RdBu')
                plt.colorbar(c)
            else:
                c = plt.pcolor(mu_numpy_t, cmap='RdBu')
                plt.colorbar(c)
            plt.title('mu')
            n = mu_numpy_t.shape[1] # time steps
            d_mut = np.zeros((mu_numpy_t.shape[0], n-1))
            for i in range(n-1):
                d_mut[:, i] = mu_numpy_t[:, i + 1] - mu_numpy_t[:, i]
            plt.subplot(3, 2, 2)
            if normalize:
                c = plt.pcolor(d_mut, norm=colors.Normalize(vmin=-1, vmax=1), cmap='RdBu')
                plt.colorbar(c)
            else:
                c = plt.pcolor(d_mut, cmap='RdBu')
                plt.colorbar(c)

            plt.title('diff_mu')

            # # save Zt values and diff Zt
            # plt.subplot(3, 2, 3)
            # c = plt.pcolor(zt_numpy_t, norm=colors.Normalize(vmin=-1, vmax=1), cmap='RdBu')
            # plt.colorbar(c)
            # plt.title('Zt')
            # n = zt_numpy_t.shape[1]  # time steps
            # d_zt = np.zeros((zt_numpy_t.shape[0], n - 1))
            # for i in range(n - 1):
            #     d_zt[:, i] = zt_numpy_t[:, i + 1] - zt_numpy_t[:, i]
            # plt.subplot(3, 2, 4)
            # c = plt.pcolor(d_zt, norm=colors.Normalize(vmin=-1, vmax=1), cmap='RdBu')
            # plt.colorbar(c)
            # plt.title('diff_Zt')
            # free smapling use N(0,1)
            zt1 = model.reparameterize(torch.zeros((2, self.dim_Zt)), torch.zeros((2, self.dim_Zt)))
            reconn = model.decode(zt1.to(self.device))
            reconn = reconn.cpu().float()  # * 0.5 + 0.5
            reconn = reconn.numpy()
            reconn2 = np.zeros((reconn.shape[0], reconn.shape[2], reconn.shape[3], 3))
            reconn2[:,:,:,0] = reconn[:,0,:,:]
            reconn2[:, :, :, 1] = reconn[:, 1, :, :]
            reconn2[:, :, :, 2] = reconn[:, 2, :, :]
            plt.subplot(3, 2, 3)
            plt.imshow(reconn2[0])
            plt.subplot(3, 2, 4)
            plt.imshow(reconn2[1])

            # save its mu hist vis
            # sort mu_numpy_t of each row values
            for i in range(mu_numpy_t.shape[0]):
                mu_numpy_t[i, :] = np.sort(mu_numpy_t[i, :])

            plt.subplot(3, 2, 5)
            if normalize:
                c = plt.pcolor(mu_numpy_t, norm=colors.Normalize(vmin=-1, vmax=1), cmap='RdBu')
                plt.colorbar(c)
            else:
                c = plt.pcolor(mu_numpy_t, cmap='RdBu')
                plt.colorbar(c)
            plt.title('distr_mu')
            # save its std hist vis
            # sort
            for i in range(std_t.shape[0]):
                std_t[i, :] = np.sort(std_t[i, :])
            plt.subplot(3, 2, 6)
            c = plt.pcolor(std_t, cmap='RdBu')
            plt.colorbar(c)
            plt.title('distr_std')

            plt.savefig(self.sample_folder + '/latent_space/' + save_name + '.jpg')
            plt.gcf().clear()
            plt.clf()
            plt.cla()


    def free_sampling(self, model, filename):
        with torch.no_grad():
            sample_size = 9
            # free smapling use N(0,1)
            zts = model.reparameterize(torch.zeros((sample_size, self.dim_Zt)), torch.zeros((sample_size, self.dim_Zt)))
            reconn = model.decode(zts.to(self.device))
            save_image(reconn[0:sample_size].cpu().float(), # if normalized to [-1,1], * 0.5 + 0.5
                       self.sample_folder + '/free_sampling/test_' + filename + '.png', nrow=int(sample_size / 3))
