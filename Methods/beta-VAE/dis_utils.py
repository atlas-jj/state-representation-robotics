import os, sys
sys.path.insert(0,'../Lib/')
import Utils as uts
import numpy as np
import torch
import copy, gc
import torch.utils.data as Data


def get_time_varying_marks(inputs, model, device, thres=0.5):
    """
    return zt_marks for the time varing factors, and its score, and the range of its state Zt, min, and max
    :param self:
    :param inputs:
    :param model:
    :param thres:
    :return:
    """
    with torch.no_grad():
        _, Zts, meanz, logvar = model(inputs.to(device))
        std_t = np.transpose(torch.exp(0.5 * logvar).detach().to('cpu').numpy())
        Zts_t = np.transpose(Zts.detach().to('cpu').numpy())
        zt_marks = []
        zt_ranges = []
        n = std_t.shape[1]  # time steps
        dim_zt = std_t.shape[0]  # dim of Zt vector
        std_z_units = np.zeros(dim_zt)
        for i in range(dim_zt):
            avg_z_unit = np.average(std_t[i, :])
            if avg_z_unit < thres:
                zt_marks.append(i)
                zt_ranges.append([Zts_t[i, :].min(), Zts_t[i, :].max()])

            std_z_units[i] = np.std(std_t[i, :])

        score = np.average(std_z_units)
        return zt_marks, score, zt_ranges
