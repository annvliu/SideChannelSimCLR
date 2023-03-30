import numpy as np
import torch
import random
import scipy
from scipy import signal
from torch import nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import copy
from scipy import fftpack

np.random.seed(0)


class DataEnvelope(object):

    def __init__(self, distance):
        self.distance = distance

    def peak_envelope(self, trs):
        # 接收ndarry (n,)，是要进行包络的波形，distance是包多少个尖峰
        # 返回u,l函数，其中u是上包络线函数，l是下包络线函数
        # x = np.arange(np.shape(sig)[0])
        # u(x)为上包络线结果，l(x)是下包络结果

        u_x = range(np.shape(trs)[0])
        l_x = range(np.shape(trs)[0])
        u_y = trs
        l_y = -1 * trs

        # find upper and lower peaks
        u_peaks, _ = scipy.signal.find_peaks(u_y, distance=self.distance)
        l_peaks, _ = scipy.signal.find_peaks(l_y, distance=self.distance)

        # use peaks and peak values to make envelope
        u_x = u_peaks
        u_y = trs[u_peaks]
        l_x = l_peaks
        l_y = trs[l_peaks]

        # add start and end of signal to allow proper indexing
        end = len(trs)
        u_x = np.concatenate((u_x, [0, end]))
        u_y = np.concatenate((u_y, [0, 0]))
        l_x = np.concatenate((l_x, [0, end]))
        l_y = np.concatenate((l_y, [0, 0]))

        # create envelope functions
        u = scipy.interpolate.interp1d(u_x, u_y)
        l = scipy.interpolate.interp1d(l_x, l_y)

        return u, l

    def __call__(self, trs):
        intrand = random.randint(1, 100)
        trs_len = trs.size()[0]
        trs_npy = trs.numpy()

        if intrand % 2:
            u, l = self.peak_envelope(trs_npy)
            new_trs = l(np.arange(trs_len))
            return torch.from_numpy(new_trs)
        else:
            return trs
