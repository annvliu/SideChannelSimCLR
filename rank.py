import numpy as np
import torch
import time
import multiprocessing

from sqlite_command import select_path_from_no, change_GE, find_no_for_GE
from utils import load_config_data


class Sbox:
    def __init__(self):
        self.SBox_value = [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB,
                           0x76, 0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4,
                           0x72, 0xC0, 0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71,
                           0xD8, 0x31, 0x15, 0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2,
                           0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6,
                           0xB3, 0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB,
                           0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF, 0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45,
                           0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
                           0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44,
                           0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73, 0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A,
                           0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32, 0x3A, 0x0A, 0x49,
                           0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8, 0x37, 0x6D,
                           0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08, 0xBA, 0x78, 0x25,
                           0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A, 0x70, 0x3E,
                           0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1,
                           0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
                           0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB,
                           0x16]

    def s_box(self, x):
        return self.SBox_value[x]

    def inverse_s_box(self, x):
        return np.where(self.SBox_value == x)[0][0]


FESH_Sbox = [3, 13, 15, 10, 0, 7, 12, 1, 4, 2, 9, 5, 11, 14, 6, 8]


def compute_leakage(dataset_name, plain, key):
    sbox = Sbox()
    if 'ascad' in dataset_name:
        return sbox.s_box(int(plain[0]) ^ int(key))

    elif dataset_name.startswith('EM'):
        sbox_output = plain[0] ^ key
        sbox_input = sbox.inverse_s_box(sbox_output)
        HD_model = plain[0] ^ sbox_input

        res = str('')
        while HD_model != 0:
            if HD_model % 2 == 0:
                res = res + '0'
                HD_model = HD_model / 2
            else:
                res = res + '1'
                HD_model = (HD_model - 1) / 2
        return res.count('1')

    elif 'AES_HD' in dataset_name:
        return sbox.inverse_s_box(plain[1] ^ key) ^ plain[0]

    elif 'FESH' in dataset_name:
        print(plain)
        return FESH_Sbox[plain[0] ^ key]


def once_GE(no, process_no, process_num, trs_num, key_proba, config, epoch, result):
    for run_time in range(int(config['GE_run_time'] / process_num)):
        trs_random = np.arange(trs_num)
        np.random.seed(run_time * process_no)
        np.random.shuffle(trs_random)

        this_run_GE = np.zeros(int(trs_num / config['GE_every']), dtype=float)
        for i in range(1, int(trs_num / config['GE_every']) + 1):
            if i % 100 == 0:
                print("正在计算实验", no, "epoch", epoch, "第",
                      int(config['GE_run_time'] / process_num) * process_no + run_time, "次攻击",
                      i * config['GE_every'], "条波形的GE")

            tmp_id = trs_random[:i * config['GE_every']]
            tmp_proba = np.sum(key_proba[tmp_id], axis=0)
            rank = sum(tmp_proba > tmp_proba[config['common']['true_key']])
            this_run_GE[i - 1] = rank

        result.put(this_run_GE)


def GE_plot_multiprocess(no, epoch, probability, plain, config: dict, process_num=10):
    trs_num = plain.shape[0]
    key_guess_num = 16 if 'FESH' in config['common']['dataset_name'] else 256

    key_proba = np.zeros((trs_num, key_guess_num), dtype=float)
    log_softmax = torch.nn.LogSoftmax(dim=1)
    attack_proba = log_softmax(torch.from_numpy(probability))
    for j in range(trs_num):
        for candidate_key in range(key_guess_num):
            key_proba[j][candidate_key] = attack_proba[j][
                compute_leakage(config['common']['dataset_name'], plain[j], candidate_key)]

    result = multiprocessing.Queue()
    process_list = []
    for process_no in range(process_num):
        new_process = multiprocessing.Process(target=once_GE,
                                              args=(
                                                  no, process_no, process_num, trs_num, key_proba, config, epoch,
                                                  result))
        process_list.append(new_process)
        new_process.start()

    GE = []
    for new_process in process_list:
        while new_process.is_alive():
            while not result.empty():
                GE.append(result.get())

    for new_process in process_list:
        new_process.join()

    GE = np.asarray(GE)
    print(GE.shape)

    return GE


def search_min(GE_trs, GE_every):
    if np.min(GE_trs) > 0:
        return np.min(GE_trs), None
    else:
        for trsnum, value in enumerate(GE_trs):
            if value == 0:
                return 0, trsnum * GE_every


def calculate_tuning_GE(no, calculated_epoch_list=None):
    path = select_path_from_no(no)
    cfg = load_config_data(path + 'config.yml')

    min_GE = 0x3F3F3F3F
    min_trsnum = None
    min_GE_epoch = -1

    epoch_list = cfg['GE_epoch'] if calculated_epoch_list is None else calculated_epoch_list
    epoch_list = ['valid'] if 'valid_num' in cfg and cfg['valid_num'] != 0 else epoch_list

    for GE_epoch in epoch_list:
        proba_plain = np.load(path + 'proba_plain_' + str(GE_epoch) + '.npy')[:]
        GE_trnum = proba_plain.shape[0] if 'GE_trsnum' not in cfg or cfg['GE_trsnum'] == 0 or cfg['GE_trsnum'] > \
                                           proba_plain.shape[0] else cfg['GE_trsnum']

        proba = proba_plain[:GE_trnum, :cfg['out_dim']]
        plain = np.asarray(proba_plain[:GE_trnum, cfg['out_dim']:], dtype=int)
        GE = GE_plot_multiprocess(no, GE_epoch, proba, plain, cfg)
        np.save(path + 'GE_' + str(GE_epoch) + '.npy', GE)

        GE_ave = GE.mean(axis=0)
        this_min_GE, this_trs_num = search_min(GE_ave, cfg['GE_every'])
        if this_min_GE < min_GE:
            min_GE = this_min_GE
            min_GE_epoch = GE_epoch
        if this_min_GE == 0 and (min_trsnum is None or this_trs_num < min_trsnum):
            min_trsnum = this_trs_num
            min_GE_epoch = GE_epoch

    if epoch_list != ['valid']:
        change_GE(no, min_GE, min_GE_epoch)


def subprocess_tuning_experiment(min_no, stop_event, i):
    while not stop_event.is_set():
        result = find_no_for_GE(min_no=min_no)
        # result = None
        if result:
            calculate_tuning_GE(result)
        else:
            print("进程", i, "正在监听……")
            time.sleep(10)
    print("子进程", i, "结束！")


def monitor_GE_process(process_num=3, min_no=1):
    # 创建一个事件对象用于控制进程停止
    stop_event = multiprocessing.Event()
    processes = []
    for i in range(process_num):
        process = multiprocessing.Process(target=subprocess_tuning_experiment, args=(min_no, stop_event, i))
        processes.append(process)
        process.start()
        time.sleep(3)

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        # 设置事件，通知所有监听进程停止
        stop_event.set()
        for process in processes:
            if process.is_alive():
                print("子进程", process.pid, "未响应停止信号，强制终止...")
                process.terminate()
                process.join()
