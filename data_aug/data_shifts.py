import random
import numpy as np
import torch
import copy
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)


class DataShift(object):
    """flip a sample on CPU"""

    def __init__(self, delay_num_of_operation, points_num_of_clock=1):
        self.points_num_of_clock = points_num_of_clock
        self.delay_num_of_operation = delay_num_of_operation
        print('shift', delay_num_of_operation)

    def __call__(self, trs):
        shift = random.randint(-self.delay_num_of_operation, self.delay_num_of_operation) * self.points_num_of_clock

        trs_len = trs.size()[0]
        trs_npy = trs.numpy()
        new_trs = copy.deepcopy(trs_npy)

        if shift > 0:
            new_trs[0:trs_len - shift] = trs_npy[shift:trs_len]
            new_trs[trs_len - shift:trs_len] = trs_npy[0:shift]
        else:
            new_trs[0:abs(shift)] = trs_npy[trs_len - abs(shift):trs_len]
            new_trs[abs(shift):trs_len] = trs_npy[0:trs_len - abs(shift)]

        return torch.from_numpy(new_trs)