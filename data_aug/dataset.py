from torchvision.transforms import transforms
from torchvision import transforms, datasets
import numpy as np
import pdb
from torch.utils.data import Dataset
import torch
from data_aug.view_generator import ContrastiveLearningViewGenerator
from data_aug.data_envelope import DataEnvelope
from data_aug.data_cut import DataCut
from data_aug.data_filter import DataFilter
from data_aug.data_shifts import DataShift
from data_aug.data_window_filter import DataWinFilter


class NetDataset(Dataset):
    def __init__(self, train_data, label_data, plain_data, transform=None):
        super().__init__()
        self.len = train_data.shape[0]
        self.trs = torch.from_numpy(train_data)
        self.label = torch.from_numpy(label_data)
        self.label = self.label.long()
        self.plain = torch.from_numpy(plain_data)
        self.transform = transform

    def __getitem__(self, index):
        single_data = self.trs[index]
        if self.transform is not None:
            single_data = self.transform(single_data)
        return single_data, self.label[index], self.plain[index]

    def __len__(self):
        return self.len


class ContrastiveLearningDataset:
    def __init__(self, config):
        self.init_data = np.load(config['common']['init_data_folder'] + config['common']['trs_fname'])
        self.init_label = np.load(config['common']['init_data_folder'] + config['common']['label_fname'])  # seg_label
        if len(self.init_label.shape) > 1:
            self.init_label = self.init_label[:, 1]
        self.init_plain = np.load(config['common']['init_data_folder'] + config['common']['plain_fname'])
        self.init_data = self.init_data.astype('float32')

        self.aug = config["augmentation"]

        if config["train_num"] != 0:
            self.init_data = self.init_data[:config["train_num"]]
            self.init_label = self.init_label[:config["train_num"]]
            self.init_plain = self.init_plain[:config["train_num"]]

        print(self.init_data.shape, self.init_label.shape, self.init_plain.shape)

    @staticmethod
    def get_simclr_pipeline_transform(aug):  # size, s=1
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        trans_list = []
        for key in aug:
            if key == 'data_filter' and aug[key] is not None:
                trans_list.append(DataFilter(filter_weight=aug[key]))
            elif key == 'data_window_filter' and aug[key] is not None:
                trans_list.append(DataWinFilter(filter_weight=aug[key]))
            elif key == 'data_shift' and aug[key] is not None:
                trans_list.append(DataShift(delay_num_of_operation=aug[key]))
            elif key == 'data_cut' and aug[key] is not None:
                trans_list.append(DataCut(size=aug[key]))
        data_transforms = transforms.Compose(trans_list)
        return data_transforms

    def get_dataset(self, n_views):
        print("train num:", self.init_label.shape[0])
        valid_datasets = NetDataset(self.init_data, self.init_label, self.init_plain,
                                    transform=ContrastiveLearningViewGenerator(
                                        self.get_simclr_pipeline_transform(self.aug),
                                        n_views))
        return valid_datasets


class LinearEvaluationDataset:
    def __init__(self, config):
        self.init_data = np.load(config['init_data_folder'] + config['trs_fname'])
        self.init_label = np.load(config['init_data_folder'] + config['label_fname'])  # seg_label
        if len(self.init_label.shape) > 1:
            self.init_label = self.init_label[:, 1]
        self.init_plain = np.load(config['init_data_folder'] + config['plain_fname'])

    def get_dataset(self):
        print("train num:", self.init_label.shape[0])
        valid_datasets = NetDataset(self.init_data, self.init_label, self.init_plain)

        return valid_datasets


def FineTuningDataset(cfg):
    init_data = np.load(cfg['common']['init_data_folder'] + cfg['common']['trs_fname'])
    init_label = np.load(cfg['common']['init_data_folder'] + cfg['common']['label_fname'])
    init_plain = np.load(cfg['common']['init_data_folder'] + cfg['common']['plain_fname'])
    init_data = init_data.astype('float32')

    train_num, valid_num = cfg['train_num'], 0 if 'valid_num' not in cfg else cfg['valid_num']
    print("train num:", train_num, "valid num:", valid_num)

    network_datasets = [NetDataset(init_data[:train_num], init_label[:train_num], init_plain[:train_num])]  # train
    network_datasets += [] if valid_num == 0 else [NetDataset(init_data[train_num:train_num + valid_num],  # train
                                                              init_label[train_num:train_num + valid_num],
                                                              init_plain[train_num:train_num + valid_num])]
    network_datasets += [NetDataset(init_data[train_num + valid_num:], init_label[train_num + valid_num:],  # train
                                    init_plain[train_num + valid_num:])]
    return network_datasets
