import os
import time
import socket
import logging
import numpy as np
from datetime import datetime

import torch

from utils import save_config_file
from sqlite_command import insert_network
from rank import GE_plot


class DeepLearning:
    def __init__(self, *args, **kwargs):

        self.config = kwargs['config']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = kwargs['model'].to(self.device)
        self.optimizer = kwargs['optimizer']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.best_ave_GE = 0x3f3f3f3f3f
        self.best_ave_GE_epoch = -1

    def demo(self, train_loader, test_loader):

        train_acc = np.empty(self.config['epoch'], dtype=float)
        test_acc = np.empty(self.config['epoch'], dtype=float)

        begin_time = time.time()

        for epoch_counter in range(self.config['epoch']):
            train_acc[epoch_counter] = self.train(epoch_counter, train_loader)
            test_acc[epoch_counter] = self.test(epoch_counter, test_loader)

            end_time = time.time()
            time_consumed = end_time - begin_time
            time_expected = time_consumed / ((epoch_counter + 1) / self.config['epoch']) - time_consumed
            sec_consumed = time_consumed % 60
            min_consumed = ((time_consumed - time_consumed % 60) % 3600) / 60
            hour_consumed = (time_consumed - time_consumed % 3600) / 3600
            sec_expected = time_expected % 60
            min_expected = ((time_expected - time_expected % 60) % 3600) / 60
            hour_expected = (time_expected - time_expected % 3600) / 3600
            print("已用时{}时{}分{}秒,预计还需{}时{}分{}秒".format(int(hour_consumed), int(min_consumed),
                                                      int(sec_consumed),
                                                      int(hour_expected), int(min_expected),
                                                      int(sec_expected)))

        np.save(self.config['outfile'] + 'test_acc.npy', test_acc)
        np.save(self.config['outfile'] + 'train_acc.npy', train_acc)

        torch.save({
            'epoch': self.config['epoch'],
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, self.config['outfile'] + 'checkpoint.tar')

        experiment_no = insert_network(config=self.config, best_GE=self.best_ave_GE, best_GE_epoch=self.best_ave_GE_epoch)
        self.config['common']['experiment_no'] = experiment_no

        # save config file
        save_config_file(self.config['outfile'], self.config)

        return experiment_no

    def train(self, epoch_counter, train_loader):
        if self.config['model_eval']:
            self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, data in enumerate(train_loader, 0):

            inputs, target, _ = data
            inputs, target = inputs.to(self.device), target.to(self.device)
            outputs = self.model(inputs)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, dim=1)
            running_loss += loss.item()
            if batch_idx % 1 == 0:
                print('[%d, %5d] loss %.16f' % (epoch_counter + 1, batch_idx + 1, running_loss / (batch_idx + 1)))

            total += target.size(0)
            correct += (predicted == target).sum().item()

        print('Accuracy on train set: %f %% [%d/%d]' % (100 * correct / total, correct, total))
        return correct / total

    def test(self, epoch_counter, test_loader):
        if self.config['model_eval']:
            self.model.eval()

        with torch.no_grad():
            correct = 0
            total = 0

            predict_proba = np.zeros((0, self.config['out_dim']), dtype=float)
            cipher_GE = np.empty(0, dtype=int)

            for i, data in enumerate(test_loader, 0):
                inputs, target, cipher = data
                inputs, target = inputs.to(self.device), target.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)

                if epoch_counter + 1 in self.config['GE_epoch']:
                    predict_proba = np.concatenate((predict_proba, outputs.cpu().detach().numpy()))
                    cipher_GE = np.concatenate((cipher_GE, cipher.numpy()))

                total += target.size(0)
                correct += (predicted == target).sum().item()

        if epoch_counter + 1 in self.config['GE_epoch']:
            GE_trs = GE_plot(predict_proba, cipher_GE, self.config)
            np.save(self.config['outfile'] + 'GE_' + str(epoch_counter + 1) + '.npy', GE_trs)

            if np.min(GE_trs.mean(axis=0)) < self.best_ave_GE:
                self.best_ave_GE = np.min(GE_trs.mean(axis=0))
                self.best_ave_GE_epoch = epoch_counter + 1

        print('Accuracy on test set: %f %% [%d/%d]' % (100 * correct / total, correct, total))
        return correct / total

