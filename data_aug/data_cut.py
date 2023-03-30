import numpy as np
import torch
from scipy import interpolate
import random
import copy
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)


class DataCut(object):
    """flip a sample on CPU"""

    def __init__(self, size):
        self.size = size
        print('cut', size)

    def __call__(self, trs):
        trs_len = trs.size()[0]
        trs_npy = trs.numpy()
        new_trs = copy.deepcopy(trs_npy)

        intrand = random.randint(1, 100)

        if intrand % 2:
            addr = random.randint(0, trs_len - self.size)
            temp = np.concatenate((trs_npy[0:addr], trs_npy[addr + self.size:]))

            seg_s_seg = int(len(temp) / (self.size + 1))
            for k in range(self.size):
                new_trs[k + seg_s_seg * k: k + seg_s_seg * (k + 1)] = temp[seg_s_seg * k: seg_s_seg * (k + 1)]
                new_trs[k + seg_s_seg * (k + 1)] = (temp[seg_s_seg * (k + 1) - 1] + temp[
                    seg_s_seg * (k + 1)]) / 2
            new_trs[self.size + seg_s_seg * self.size:] = temp[seg_s_seg * self.size:]

        return torch.from_numpy(new_trs)
