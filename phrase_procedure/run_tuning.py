# from torchlars import LARS
import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models

from data_aug.dataset import FineTuningDataset
from neural_net.change_net import copy_model_for_classification
from neural_net.choose_net import simclr_net
from phrase_procedure.fine_tuning import FineTuning
from utils import load_config_data, create_folder


def tuning_main(init_cfg, pretrain_path, GE_list=None):
    cfg = dict()
    cfg['common'] = init_cfg['common']
    cfg['common']['experiment_type'] = 'tuning'
    cfg.update(init_cfg['tuning'])

    cfg['pretrain_path'] = pretrain_path
    cfg['GE_epoch'] = GE_list
    cfg['GE_epoch'].append(cfg['epoch'])
    cfg['out_dim'] = cfg['common']['classification']
    cfg['outfile'] = create_folder()

    # check if gpu training is available
    if not cfg['common']['disable_cuda'] and torch.cuda.is_available():
        cfg['common']['device'] = torch.device("cuda:0")  # 增加device
        cudnn.deterministic = True
        cudnn.benchmark = True
        print("cuda!")
    else:
        cfg['common']['device'] = torch.device("cpu")
        cfg['common']['gpu_index'] = -1

    dataset = FineTuningDataset(cfg)
    train_dataset, test_dataset = dataset.get_dataset(train_num=cfg['train_num'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['common']['batch_size'], shuffle=True,
                                               pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['common']['batch_size'], shuffle=True,
                                              pin_memory=True,
                                              drop_last=True)

    model = simclr_net(config=cfg)
    model = copy_model_for_classification(model, cfg['pretrain_path'], frozen=cfg['frozen'], add_dense=cfg['add_dense_bool'])

    # optimizer = LARS(
    #     torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr, weight_decay=args.weight_decay)
    #     )
    # https://pythonawesome.com/a-lars-implementation-in-pytorch/
    # https: // github.com / kakaobrain / torchlars

    optimizer = torch.optim.Adam(model.parameters(), cfg['lr'])
    if 'optim' in cfg and cfg['optim'] == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), cfg['lr'])

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    experiment_no = None
    with torch.cuda.device(cfg['common']['gpu_index']):
        finetuing = FineTuning(model=model, optimizer=optimizer, args=cfg)
        experiment_no = finetuing.demo(train_loader, test_loader)
    return experiment_no
