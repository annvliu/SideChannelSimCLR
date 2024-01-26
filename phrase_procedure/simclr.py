import logging
import sys
import pdb

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import save_config_file, accuracy
from sqlite_command import insert_pretrain

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.optimize = None
        self.config = kwargs['args']
        self.model = kwargs['model'].to(self.config['common']['device'])
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.config['common']['device'])

    # def info_nce_loss(self, features):
    #     labels = torch.cat([torch.arange(self.config['common']['batch_size']) for i in range(self.config['n_views'])], dim=0)
    #     labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    #     labels = labels.to(self.config['common']['device'])
    #
    #     features = F.normalize(features, dim=1)
    #
    #     similarity_matrix = torch.matmul(features, features.T)
    #     # assert similarity_matrix.shape == (
    #     #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    #     # assert similarity_matrix.shape == labels.shape
    #
    #     # discard the main diagonal from both: labels and similarities matrix
    #     mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.config['common']['device'])
    #     labels = labels[~mask].view(labels.shape[0], -1)
    #     similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    #     # assert similarity_matrix.shape == labels.shape
    #
    #     # select and combine multiple positives
    #     positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    #
    #     # select only the negatives
    #     negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    #
    #     pdb.set_trace()
    #
    #     logits = torch.cat([positives, negatives], dim=1)
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.config['common']['device'])
    #
    #     logits = logits / self.config['temperature']
    #     return logits, labels

    def info_nce_loss(self, features):
        sample_num = self.config['common']['batch_size'] * self.config['n_views']

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        iden_matrix = np.identity(sample_num, dtype=np.bool8)
        from_same_matrix = np.tile(np.identity(self.config['common']['batch_size']),
                                   (self.config['n_views'], self.config['n_views']))
        
        positives_labels = torch.from_numpy(np.logical_xor(from_same_matrix, iden_matrix)).to(self.config['common']['device'])
        negatives_labels = torch.from_numpy(np.logical_not(from_same_matrix)).to(self.config['common']['device'])

        positives = similarity_matrix[positives_labels].view(positives_labels.shape[0], -1)
        negatives = similarity_matrix[negatives_labels].view(negatives_labels.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.config['temperature']
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.config['common']['device'])
        return logits, labels

    def train(self, train_loader):
        self.model.train()

        acc_list = []
        loss_list = []
        lr_list = []
        for epoch_counter in range(self.config['epoch']):
            for images, _, _ in tqdm(train_loader):

                images1 = images[0].to(self.config['common']['device'])
                images2 = images[1].to(self.config['common']['device'])

                features1 = self.model(images1)
                features2 = self.model(images2)
                features = torch.cat((features1, features2), dim=0)
                logits, labels = self.info_nce_loss(features)

                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc_list.append(accuracy(logits, labels)[0][0].cpu().numpy())
                loss_list.append(loss.item())
                lr_list.append(self.scheduler.get_last_lr()[0])

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

        torch.save({
            'epoch': self.config['epoch'],
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.config['outfile'] + 'checkpoint.tar')

        np.save(self.config['outfile'] + 'acc_for_batch', np.asarray(acc_list))
        np.save(self.config['outfile'] + 'loss_for_batch', np.asarray(loss_list))
        np.save(self.config['outfile'] + 'lr_for_batch', np.asarray(lr_list))

        # sqlite
        experiment_no = insert_pretrain(config=self.config)
        self.config['common']['experiment_no'] = experiment_no

        # save dict
        save_config_file(self.config['outfile'], self.config)

        return experiment_no
