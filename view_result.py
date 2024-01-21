import matplotlib.pyplot as plt
import numpy as np
from sqlite_command import select_path_from_no, change_GE
from utils import load_config_data
from rank import search_min


def view_tuning_network(experiment_no, change_GE_bool=False):
    path = select_path_from_no(experiment_no)

    train_acc = np.load(path + 'tuning_train_acc.npy')
    test_acc = np.load(path + 'tuning_test_acc.npy')
    print('max train acc', max(train_acc))
    plt.plot(train_acc, label='train')
    plt.plot(test_acc, label='test')
    plt.legend()
    plt.show()

    cfg = load_config_data(path + 'config.yml')

    min_GE = 0x3F3F3F3F
    min_trsnum = None
    min_GE_epoch = -1
    for GE_epoch in cfg['GE_epoch']:
        GE = np.load(path + 'GE_' + str(GE_epoch) + '.npy')
        GE_ave = GE.mean(axis=0)

        for once in GE:
            plt.plot([i * cfg['GE_every'] for i in range(GE_ave.shape[0])], once, c='gray')
        plt.plot([i * cfg['GE_every'] for i in range(GE_ave.shape[0])], GE_ave, c='b')
        plt.show()

        this_min_GE, this_trs_num = search_min(GE_ave, cfg['GE_every'])
        print('epoch:', GE_epoch, 'GE:', this_min_GE, 'trs num:', this_trs_num)
        if this_min_GE < min_GE:
            min_GE = this_min_GE
            min_GE_epoch = GE_epoch
        if this_min_GE == 0 and (min_trsnum is None or this_trs_num < min_trsnum):
            min_trsnum = this_trs_num
            min_GE_epoch = GE_epoch

    if change_GE_bool:
        change_GE(experiment_no, min_GE, min_GE_epoch)


def view_pretrain(experiment_no):
    path = select_path_from_no(experiment_no)

    acc = np.load(path + 'acc_for_batch.npy')
    plt.plot(acc, label='acc')
    plt.legend()
    plt.show()

    loss = np.load(path + 'loss_for_batch.npy')
    plt.plot(loss, label='loss')
    plt.legend()
    plt.show()


def main():
    # view_pretrain(17)
    view_tuning_network(794)


if __name__ == '__main__':
    main()
