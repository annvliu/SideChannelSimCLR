import numpy as np
import torch
import torch.backends.cudnn as cudnn

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
    cfg['GE_epoch'].append(cfg['epoch']) if cfg['GE_epoch'][-1] != cfg['epoch'] else cfg['GE_epoch']
    cfg['outfile'] = create_folder()
    cfg['out_dim'] = cfg['common']['classification']

    # check if gpu training is available
    if not cfg['common']['disable_cuda'] and torch.cuda.is_available():
        cfg['common']['device'] = "cuda:" + str(cfg['common']['gpu_index'])
        # cudnn.deterministic = True
        # cudnn.benchmark = True
        print(cfg['common']['device'])
    else:
        cfg['common']['device'] = "cpu"
        cfg['common']['gpu_index'] = -1

    network_datasets = FineTuningDataset(cfg)
    network_dataloaders = [torch.utils.data.DataLoader(dataset, batch_size=cfg['common']['batch_size'], shuffle=True,
                                                       pin_memory=True, drop_last=True) for dataset in network_datasets]

    model = simclr_net(config=cfg)

    optimizer = torch.optim.Adam(model.parameters(), cfg['lr'])
    if 'optim' in cfg and cfg['optim'] == 'RMSprop':
        print('optim RMSprop')
        optimizer = torch.optim.RMSprop(model.parameters(), cfg['lr'])

    scheduler = None
    if 'scheduler' in cfg and cfg['scheduler'] == 'MECNN OneCycleLR':
        # MECNN config
        print('MECNN OneCycleLR')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg['lr'], epochs=cfg['epoch'],
                                                        steps_per_epoch=cfg['train_num'] // cfg['common']['batch_size'],
                                                        pct_start=0.4, anneal_strategy='linear', div_factor=10,
                                                        final_div_factor=100, three_phase=True)

    elif 'scheduler' in cfg and cfg['scheduler'] == 'BilinearCNN OneCycleLR':
        # BilinearCNN config
        print('BilinearCNN OneCycleLR')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg['lr'],
                                                        steps_per_epoch=cfg['train_num'] // cfg['common']['batch_size'],
                                                        pct_start=0.2, anneal_strategy='linear', cycle_momentum=False,
                                                        epochs=cfg['epoch'], div_factor=10, verbose=False)

    cnn = DeepLearning(model=model, optimizer=optimizer, scheduler=scheduler, config=cfg)
    return cnn.demo(network_dataloaders)
