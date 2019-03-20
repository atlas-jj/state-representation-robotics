import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class ChrisClamp(nn.Module):
    def __init__(self, low, high):
        super(ChrisClamp, self).__init__()
        self.low = low
        self.high = high

    def forward(self, tensor):
        return tensor.clamp(min=self.low, max=self.high)

class TransSigmoid(nn.Module):
    def __init__(self, low, high):
        super(TransSigmoid, self).__init__()
        self.low = low
        self.high = high

    def forward(self, tensor):
        m = nn.Sigmoid()
        return m(tensor) * (self.high - self.low) + self.low

class TransTanh(nn.Module):
    def __init__(self, range_tensor):
        super(TransTanh, self).__init__()
        self.range_tensor = range_tensor

    def forward(self, tensor):
        m = nn.Tanh()
        return m(tensor) * self.range_tensor


def F_TransSigmoid(tensor, low, high):
    m = nn.Sigmoid()
    return m(tensor) * (high - low) + low
