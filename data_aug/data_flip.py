import numpy as np
import torch
import random
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)


class DataFlip(object):
    """flip a sample on CPU"""

    def __init__(self):
        pass

    def __call__(self, trs):
        intrand = random.randint(1, 100)
        if intrand % 2:
            trs = torch.flip(trs, dims=[0])
        return trs
