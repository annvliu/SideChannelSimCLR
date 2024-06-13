import numpy as np
import torch
import random
import copy
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)


class DataWinFilter(object):
    """flip a sample on CPU"""

    def __init__(self, filter_weight):
        self.filter_weight = filter_weight
        print('window filter', filter_weight)

    def __call__(self, trs):
        trs_npy = trs.numpy()
        new_trs = copy.deepcopy(trs_npy)
        window_size = random.randint(1, self.filter_weight)

        if window_size != 1:
            kernel = np.ones(window_size) / window_size
            new_trs = np.convolve(new_trs, kernel, mode='same')
        return torch.from_numpy(new_trs)
