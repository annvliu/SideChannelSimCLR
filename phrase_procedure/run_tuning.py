import torch
import torch.backends.cudnn as cudnn

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
        cfg['common']['device'] = "cuda:" + str(cfg['common']['gpu_index'])
        cudnn.deterministic = True
        cudnn.benchmark = True
        print(cfg['common']['device'])
    else:
        cfg['common']['device'] = "cpu"
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

    optimizer = torch.optim.Adam(model.parameters(), cfg['lr'])
    if 'optim' in cfg and cfg['optim'] == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), cfg['lr'])

    scheduler = None
    if 'scheduler' in cfg and cfg['scheduler'] == 'OneCycleLR':
        # optimal ascad config
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg['lr'], epochs=cfg['epoch'],
        #                                                 steps_per_epoch=cfg['train_num'] // cfg['common']['batch_size'],
        #                                                 final_div_factor=4, verbose=False)

        # MECNN config
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg['lr'], epochs=cfg['epoch'],
                                                        steps_per_epoch=cfg['train_num'] // cfg['common']['batch_size'],
                                                        pct_start=0.4, anneal_strategy='linear', div_factor=10,
                                                        final_div_factor=100)

    finetuing = FineTuning(model=model, optimizer=optimizer, scheduler=scheduler, args=cfg)
    return finetuing.demo(train_loader, test_loader)
