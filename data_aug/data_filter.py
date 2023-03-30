import numpy as np
import torch
import random
import copy
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)


class DataFilter(object):
    """flip a sample on CPU"""

    def __init__(self, filter_weight):
        self.filter_weight = filter_weight
        print('filter', filter_weight)

    def __call__(self, trs):
        intrand = random.randint(1, 100)
        trs_len = trs.size()[0]
        trs_npy = trs.numpy()
        new_trs = copy.deepcopy(trs_npy)

        if intrand % 2:
            for i in range(1, trs_len):
                new_trs[i] = (new_trs[i] + self.filter_weight * new_trs[i - 1]) / (self.filter_weight + 1)
            for i in range(trs_len - 2, -1, -1):
                new_trs[i] = (new_trs[i] + self.filter_weight * new_trs[i + 1]) / (self.filter_weight + 1)
        return torch.from_numpy(new_trs)
