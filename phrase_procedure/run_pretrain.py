import argparse
import torch
import copy
import torch.backends.cudnn as cudnn
from torchvision import models

from utils import create_folder, save_dataset_file
from data_aug.dataset import ContrastiveLearningDataset
from neural_net.change_net import add_projection_head
from phrase_procedure.simclr import SimCLR
from neural_net.choose_net import simclr_net


def pretrain_main(init_cfg):
    cfg = dict()
    cfg['common'] = init_cfg['common']
    cfg['common']['experiment_type'] = 'pretrain'
    cfg.update(init_cfg['pretrain'])
    cfg['outfile'] = create_folder()

    # check if gpu training is available
    if not cfg['common']['disable_cuda'] and torch.cuda.is_available():
        cfg['common']['device'] = torch.device("cuda:" + str(cfg['common']['gpu_index']))  # 增加device
        cudnn.deterministic = True
        cudnn.benchmark = True
        print("cuda!", cfg['common']['gpu_index'])
    else:
        cfg['common']['device'] = torch.device("cpu")
        cfg['common']['gpu_index'] = -1

    dataset = ContrastiveLearningDataset(cfg)

    train_dataset = dataset.get_dataset(cfg['n_views'])

    dataset_copy = copy.deepcopy(train_dataset)
    save_dataset_file(dataset_copy, cfg['outfile'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['common']['batch_size'], shuffle=True,
                                               pin_memory=True, drop_last=True)

    model = simclr_net(config=cfg)
    model = add_projection_head(model)

    optimizer = torch.optim.Adam(model.parameters(), cfg['lr'], weight_decay=cfg['wd'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    experiment_no = None
    with torch.cuda.device(cfg['common']['gpu_index']):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=cfg)
        experiment_no = simclr.train(train_loader)
    return experiment_no
