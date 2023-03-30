import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import load_config_data, create_folder
from data_aug.dataset import FineTuningDataset
from neural_net.choose_net import simclr_net
from phrase_procedure.network import DeepLearning

np.set_printoptions(threshold=np.inf)


def network_main(init_cfg, GE_list=None):
    cfg = dict()
    cfg['common'] = init_cfg['common']
    cfg['common']['experiment_type'] = 'network'
    cfg.update(init_cfg['tuning'])

    cfg['GE_epoch'] = GE_list
    cfg['GE_epoch'].append(cfg['epoch'])
    cfg['outfile'] = create_folder()

    dataset = FineTuningDataset(cfg)
    train_dataset, test_dataset = dataset.get_dataset(train_num=cfg['train_num'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['common']['batch_size'], shuffle=True,
                                               pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['common']['batch_size'], shuffle=True,
                                              pin_memory=True, drop_last=True)

    cfg['out_dim'] = cfg['common']['classification']
    model = simclr_net(config=cfg)

    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])

    cnn = DeepLearning(model=model, optimizer=optimizer, config=cfg)
    return cnn.demo(train_loader, test_loader)
