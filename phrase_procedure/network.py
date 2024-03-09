import os
import pdb
import time
import socket
import logging
import numpy as np
from datetime import datetime

import torch
from tqdm import tqdm

from utils import save_config_file, accuracy
from sqlite_command import insert_network


class DeepLearning:
    def __init__(self, *args, **kwargs):

        self.config = kwargs['config']
        self.device = torch.device(self.config['common']['device'])
        self.model = kwargs['model'].to(self.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.valid_loss = 0x3f3f3f3f3f
        self.valid_True = 'valid_num' in self.config and self.config['valid_num'] != 0
        print('valid: ', self.valid_True)

        self.test_loss = []
        self.train_loss = []
        self.lr_batch = []

        self.best_ave_GE = 0x3f3f3f3f3f
        self.best_ave_GE_epoch = -1

    def demo(self, network_dataloaders):
        train_loader = network_dataloaders[0]
        test_loader = network_dataloaders[-1]
        valid_loader = None if not self.valid_True else network_dataloaders[1]

        train_acc = np.empty(self.config['epoch'], dtype=float)
        test_acc = np.empty(self.config['epoch'], dtype=float)
        for epoch_counter in range(self.config['epoch']):
            train_acc[epoch_counter] = self.train(epoch_counter, train_loader)
            if self.valid_True:
                self.valid(epoch_counter, valid_loader, test_loader)
            else:
                test_acc[epoch_counter] = self.test(epoch_counter, test_loader)

        np.save(self.config['outfile'] + 'test_acc.npy', test_acc)
        np.save(self.config['outfile'] + 'train_acc.npy', train_acc)
        np.save(self.config['outfile'] + 'train_loss_for_batch', self.train_loss)
        np.save(self.config['outfile'] + 'test_loss_for_batch', self.test_loss)
        np.save(self.config['outfile'] + 'lr_for_batch.npy', self.lr_batch)

        # sqlite
        experiment_no = insert_network(config=self.config, best_GE=self.best_ave_GE, best_GE_epoch=self.best_ave_GE_epoch)
        self.config['common']['experiment_no'] = experiment_no

        # save config file
        save_config_file(self.config['outfile'], self.config)

        return experiment_no

    def valid(self, epoch_counter, valid_loader, test_loader):
        with torch.no_grad():

            valid_criterion = torch.nn.CrossEntropyLoss().to(self.device)
            for inputs, target, plain in tqdm(valid_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)
                outputs = self.model(inputs)
                valid_loss = valid_criterion(outputs, target)

            valid_loss /= len(valid_loader)
            if valid_loss < self.valid_loss:
                self.valid_loss = valid_loss
                self.test(epoch_counter, test_loader, valid_test=True)

    def train(self, epoch_counter, train_loader):
        if self.config['model_eval']:
            self.model.train()

        all_outputs = torch.empty((0, self.config['out_dim']), dtype=torch.float32).to(self.config['common']['device'])
        all_labels = torch.empty(0, dtype=torch.int64).to(self.config['common']['device'])

        for inputs, target, _ in tqdm(train_loader):

            inputs, target = inputs.to(self.device), target.to(self.device)
            all_labels = torch.cat((all_labels, target), dim=0)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, target)
            all_outputs = torch.cat((all_outputs, outputs), dim=0)
            self.lr_batch.append(self.optimizer.state_dict()['param_groups'][0]['lr'])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.train_loss.append(loss.item())

        top1 = accuracy(all_outputs, all_labels)

        return top1[0]

    def test(self, epoch_counter, test_loader, valid_test=False):
        if self.config['model_eval']:
            self.model.eval()

        with torch.no_grad():

            all_outputs = torch.empty((0, self.config['out_dim']), dtype=torch.float32).to(self.config['common']['device'])
            all_labels = torch.empty(0, dtype=torch.int64).to(self.config['common']['device'])

            predict_proba = np.zeros((0, self.config['out_dim']), dtype=float)
            plain_GE = None

            for inputs, target, plain in tqdm(test_loader):

                inputs, target = inputs.to(self.device), target.to(self.device)
                all_labels = torch.cat((all_labels, target), dim=0)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, target)
                all_outputs = torch.cat((all_outputs, outputs), dim=0)

                self.test_loss.append(loss.item())

                if epoch_counter + 1 in self.config['GE_epoch'] or valid_test:
                    predict_proba = np.concatenate((predict_proba, outputs.cpu().detach().numpy()))
                    plain_numpy = plain.numpy().reshape(self.config['common']['batch_size'], -1)
                    plain_GE = np.vstack((plain_GE, plain_numpy)) if plain_GE is not None else plain_numpy

            topn = accuracy(all_outputs, all_labels, (1, 5))

            if epoch_counter + 1 in self.config['GE_epoch'] or valid_test:
                proba_plain = np.hstack((predict_proba, plain_GE))
                fname = self.config['outfile'] + 'proba_plain_' + (str(epoch_counter + 1) if not valid_test else 'valid')
                np.save(fname, proba_plain)

            if epoch_counter + 1 == self.config['GE_epoch'] or valid_test:
                torch.save({
                    'epoch': epoch_counter + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, self.config['outfile'] + 'checkpoint.tar')

        return topn[0]
