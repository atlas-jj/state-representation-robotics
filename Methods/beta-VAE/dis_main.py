#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import dis_models as models
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

# <editor-fold desc="Parameters Setting">
parser = argparse.ArgumentParser(description='LfD Bayesian Inference')
# parser.add_argument('--task-name', type=str, default='Blocks', metavar='S',
#                     help='task name')
task_names = ['Cloth_prev'] #['Blocks', 'Plug', 'Toy_Example]

data_set1 = ['136'] #['1', '2', '3', '4', '5', '6']#['bh1', 'bh2', 'bh3', 'bh4']
sample_size1 = [462] #[60, 43, 43, 39]

data_set2 = ['human12'] #['p1', 'p2', 'p3']
sample_size2 = [410]

data_set3 = ['human12'] #['p1', 'p2', 'p3']
sample_size3 = [216]

data_sets = [data_set3]  #[data_set1, data_set2]
sample_sizes = [sample_size3]  #[sample_size1, sample_size2]
sample_sequence_folders = ['human2']
sample_sequence_numbers = [111]

dim_Zs = [100]  # [50, 100, 200]
alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

epoch_interval = 100
# parser.add_argument('--image-folder', type=str, default='Blocks/bh1', metavar='S',
#                     help='training dataset path')
# parser.add_argument('--sample-size', type=int, default=60, metavar='N',
#                     help='sample size: 60')
parser.add_argument('--image-size', type=int, default=240, metavar='N',
                    help='input image size: 240*240')
parser.add_argument('--input-channel', type=int, default=3, metavar='N',
                    help='input-channel: 3')


# parser.add_argument('--dim-Zt', type=int, default=100, metavar='N',
#                     help='dimension of Zt')
# parser.add_argument('--alpha', type=float, default=0.01, metavar='N',
#                     help='alpha')
# parser.add_argument('--beta', type=float, default=700, metavar='N',
#                     help='beta parameter for disentangling')

parser.add_argument('--batch-size', type=int, default=290, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 25)')


parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
# args.image_folder = '../Experiments/Dataset/'+args.image_folder

args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# </editor-fold>

# <editor-fold desc="Initialization + load data">

device = torch.device("cuda" if args.cuda else "cpu")

# </editor-fold>


# create instance of current class

# </editor-fold>


# <editor-fold desc="Utilities">
def generate_file_name(task_name, data_set, dim_Zt, alpha, beta, epoch, bce, kld):
    return "betaVAE_" + task_name+'_'+data_set+'_dim'+str(dim_Zt)+'_' + uts.d2s(alpha, 2) + '_' + uts.d2s(beta, 1)+'_'+str(epoch)+'_'\
           + uts.d2s(bce, 2)+'_'+uts.d2s(kld, 2)


# </editor-fold>


# <editor-fold desc="train + test">

# now generate samples.
train_losss = []
train_bce = []
train_kld = []
def train(epoch, train_loader, model, optimizer, alpha, beta, task_name, data_set, dim_Zt, plot_r):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        data = data[0].to(device)
        optimizer.zero_grad()
        recon_batch, zt, mu, logvar = model(data)   # invoke the forward function
        loss, bce, kld = models.loss_function(beta, recon_batch, data, mu, logvar)  # from loss function to back prop the variables.
        loss.backward()
        this_loss = loss.item()/mu.shape[0]
        train_loss += this_loss
        train_losss.append(this_loss)
        train_bce.append(bce / mu.shape[0])
        train_kld.append(kld / mu.shape[0])
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                'Train:{} {} dim {} beta {:.1f} Epoch:{} [{}/{} ({:.0f}%)]'.format(
                    task_name, data_set, dim_Zt, beta,
                    epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                           100. * (batch_idx + 1) / len(train_loader)))

    train_loss /= len(train_loader.dataset)
    print('Train:{} {} dim {} beta {:.1f} Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t[BCE: {:.6f}]'.format(
        task_name, data_set, dim_Zt, beta, epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                                            100. * (batch_idx + 1) / len(train_loader),
        train_loss, train_loss))
    if epoch % epoch_interval == 0:
        plot_r.save_results_4(model, generate_file_name(task_name, data_set, dim_Zt, alpha, beta, epoch,
                                                            bce / mu.shape[0], kld / mu.shape[0]),
                              task_name + " epoch " + str(epoch))
    return train_loss
    # print('====> Epoch: {} Average loss: {:.4f}'.format(
    #       epoch, train_loss / len(train_loader.dataset)))


# li_mu = []
# li_logvar = []
test_loss = []
test_bce = []
test_kld = []
def test(epoch, test_loader, model, alpha, beta, task_name, data_set, dim_Zt, plot_r):

    model.eval()
    test_loss = 0
    bce_loss = 0
    kld_loss = 0
    # li_mu[:]=[]
    # li_logvar[:]=[]
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            data = data[0].to(device)
            recon_batch, zt, mu, logvar = model(data)
            loss, bce, kld = models.loss_function(beta, recon_batch, data, mu, logvar)
            test_loss += loss.item()
            bce_loss += bce.item()
            kld_loss += kld.item()
            # li_mu.append(mu.detach().to('cpu').numpy())
            # li_logvar.append(logvar.detach().to('cpu').numpy())
            if i == 0 and epoch % epoch_interval == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(-1, 3, 240, 240)[:n]])
                plot_r.save_reconstruction(comparison.cpu(), generate_file_name(task_name, data_set, dim_Zt, alpha,
                                                                                beta, epoch, 0, 0), n)
                ## when save to image, need to de-normalize image * std + mean
                # save_image(comparison.cpu()*0.5 + 0.5, 'results/reconstruction/test_' +
                #            generate_file_name(task_name, data_set, dim_Zt, alpha, beta, epoch, 0, 0) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    bce_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    # print('====> Test set loss: {:.6f}\t[MSE: {:.6f}  KLD: {:.6f}]'.format(test_loss, mse_loss, kld_loss))
    # saveHist(epoch, li_mu, li_logvar)
    if epoch % epoch_interval == 0:
        plot_r.free_sampling(model, generate_file_name(task_name, data_set, dim_Zt, alpha, beta, epoch, 0, 0))


# def test_cases(i):
#     list_datas = list(enumerate(train_loader))
#
#     # list by batches.
#     # for i in range(len(list_datas)):
#     batch_id, datas = list_datas[i]
#
#     rec, z, mu, logvar = model(datas[0].to(device))
#
#     return datas[0], rec, z, mu
#


if __name__ == "__main__":
    for i in range(len(task_names)): # task_level
        task_name = task_names[i]
        data_set = data_sets[i]
        sample_size = sample_sizes[i]
        sample_sequence_folder = '../Experiments/Dataset/' + task_name + '/' + sample_sequence_folders[i]
        sample_sequence = uts.load_images_sequence(sample_sequence_folder, sample_sequence_numbers[i], args.image_size)
        for j in range(len(data_set)): #data_set level
            data_set_name = data_set[j]
            sample_size_t = sample_size[j]
            # load data
            image_folder = '../Experiments/Dataset/' + task_name + '/' + data_set_name
            # train_loader, test_loader = uts.wrap_data_loader_images(image_folder, args.image_size, kwargs, 0, args.batch_size)
            train_loader = uts.wrap_data_loader_images(image_folder, args.image_size, kwargs, 0,
                                                       args.batch_size)
            # sample_sequence = uts.load_images_sequence(image_folder, sample_size_t, args.image_size)
            # sample_sequence = uts.load_images_sequence('../Experiments/Dataset/' + task_name + '/final_states', 40, args.image_size)
            for m in range(len(dim_Zs)):  # dim Z level
                dim_Zt = dim_Zs[m]
                for n in range(len(alphas)):  # alpha level, beta level
                    alpha = alphas[n]
                    beta = alpha * args.image_size * args.image_size * args.input_channel / dim_Zt
                    torch.manual_seed(args.seed)  # manually set random seed for CPU random number generation.
                    # construct model and optimizer
                    model = models.beta_VAE(dim_Zt, args.input_channel).to(device)
                    # initialize an optimizer
                    optimizer = optim.Adam(model.parameters(), lr=5e-4)
                    train_losss = []
                    plot_r = pr.plot_task_map('../Experiments/Dataset/' + task_name, dim_Zt, alpha, beta, sample_sequence, device)
                    for epoch in range(1, args.epochs + 1):
                        trainLoss = train(epoch, train_loader, model, optimizer, alpha, beta, task_name, data_set_name, dim_Zt, plot_r)
                        train_losss.append(trainLoss)
                        # test(epoch, test_loader, model, alpha, beta, task_name, data_set_name, dim_Zt, plot_r)
                    # save the model
                    model.save("betaVAE_" + task_name + "_" + data_set_name + "_dimZ_" + str(dim_Zt) + "alpha_" + str(alpha))
                    np.save("train_loss_betaVAE_" + task_name + "_" + data_set_name + "_dimZ_" + str(dim_Zt), train_losss)
                    # del model and plot_r
                    del model, optimizer, plot_r
                    torch.cuda.empty_cache()
                    gc.collect()
            # del train_loader, test_loader, sample_sequence
            del train_loader, sample_sequence
            torch.cuda.empty_cache()
            gc.collect()
            print('sorry, system fell asleep 30s to cool down our expensive GPU...')
            time.sleep(20)  # freez the GPU
