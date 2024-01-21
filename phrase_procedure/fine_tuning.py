import logging
import os
import sys
import pdb
from datetime import datetime
import numpy as np
import socket
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import save_config_file, accuracy
from sqlite_command import insert_tuning

torch.manual_seed(0)


class FineTuning(object):
    def __init__(self, *args, **kwargs):
        self.config = kwargs['args']
        self.model = kwargs['model'].to(self.config['common']['device'])
        self.optimizer = kwargs['optimizer']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.config['common']['device'])

        self.test_loss = []
        self.train_loss = []

        self.best_ave_GE = 0x3f3f3f3f3f
        self.best_ave_GE_epoch = -1

    def demo(self, train_loader, test_loader):

        train_acc = np.empty(self.config['epoch'], dtype=float)
        test_acc = np.empty(self.config['epoch'], dtype=float)
        for epoch_counter in range(self.config['epoch']):
            train_acc[epoch_counter] = self.train(epoch_counter, train_loader)
            test_acc[epoch_counter] = self.test(epoch_counter, test_loader)

        torch.save({
            'epoch': self.config['epoch'],
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.config['outfile'] + 'checkpoint.tar')

        np.save(self.config['outfile'] + 'train_acc.npy', train_acc)
        np.save(self.config['outfile'] + 'test_acc.npy', test_acc)
        np.save(self.config['outfile'] + 'train_loss_for_batch', self.train_loss)
        np.save(self.config['outfile'] + 'test_loss_for_batch', self.test_loss)

        # sqlite
        experiment_no = insert_tuning(config=self.config, best_GE=self.best_ave_GE, best_GE_epoch=self.best_ave_GE_epoch)
        self.config['common']['experiment_no'] = experiment_no

        # save config file
        save_config_file(self.config['outfile'], self.config)

        return experiment_no

    def test(self, epoch_counter, test_loader):
        if self.config['model_eval']:
            self.model.eval()

        with torch.no_grad():

            all_outputs = torch.empty((0, self.config['out_dim']), dtype=torch.float32).to( self.config['common']['device'])
            all_labels = torch.empty(0, dtype=torch.int64).to(self.config['common']['device'])

            predict_proba = np.zeros((0, self.config['out_dim']), dtype=float)
            plain_GE = np.zeros(0, dtype=int)

            for images, labels, plain in tqdm(test_loader):

                images = images.to(self.config['common']['device'])
                labels = labels.to(self.config['common']['device'])
                all_labels = torch.cat((all_labels, labels), dim=0)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                all_outputs = torch.cat((all_outputs, outputs), dim=0)

                self.test_loss.append(loss.item())

                if epoch_counter + 1 in self.config['GE_epoch']:
                    predict_proba = np.concatenate((predict_proba, outputs.cpu().detach().numpy()))
                    plain_GE = np.concatenate((plain_GE, plain.numpy()))

            topn = accuracy(all_outputs, all_labels, (1, 5))

            if epoch_counter + 1 in self.config['GE_epoch']:
                proba_plain = np.hstack((predict_proba, plain_GE.reshape(-1, 1)))
                np.save(self.config['outfile'] + 'proba_plain_' + str(epoch_counter + 1) + '.npy', proba_plain)

                # GE_trs = GE_plot(predict_proba, plain_GE, self.config)
                # np.save(self.config['outfile'] + 'GE_' + str(epoch_counter + 1) + '.npy', GE_trs)
                #
                # if np.min(GE_trs.mean(axis=0)) < self.best_ave_GE:
                #     self.best_ave_GE = np.min(GE_trs.mean(axis=0))
                #     self.best_ave_GE_epoch = epoch_counter + 1

        return topn[0]

    def train(self, epoch_counter, train_loader):
        if self.config['model_eval']:
            self.model.train()

        all_outputs = torch.empty((0, self.config['out_dim']), dtype=torch.float32).to(self.config['common']['device'])
        all_labels = torch.empty(0, dtype=torch.int64).to(self.config['common']['device'])

        for images, labels, _ in tqdm(train_loader):
            images = images.to(self.config['common']['device'])
            labels = labels.to(self.config['common']['device'])
            all_labels = torch.cat((all_labels, labels), dim=0)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            all_outputs = torch.cat((all_outputs, outputs), dim=0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_loss.append(loss.item())

        top1 = accuracy(all_outputs, all_labels)

        return top1[0]
