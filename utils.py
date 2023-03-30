import os
import socket
from datetime import datetime
import pdb
import numpy as np
import shutil
import networkx

import torch
import yaml


def create_folder():
    current_time = datetime.now().strftime('%Y%b%d_%H-%M-%S-%f')
    log_dir = os.path.join('runs/' + current_time + '_' + socket.gethostname())
    os.makedirs(log_dir, exist_ok=True)
    return log_dir + '/'


def save_dataset_file(dataset, path):
    augmented_trs = np.asarray([[sample[0][0].numpy(), sample[0][1].numpy()] for sample in dataset])
    label = np.asarray([sample[1].numpy() for sample in dataset])
    np.save(path + 'dataset_data', augmented_trs)
    np.save(path + 'dataset_label', label)


def load_config_data(path: str) -> dict:
    """
    Load a config data from a given path
    :param path: the path as a string
    :return: the config as a dict
    """
    with open(path) as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
