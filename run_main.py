import matplotlib.pyplot as plt
import yaml
import numpy as np
import sqlite3
from pyinstrument import Profiler

from init_data.data_info import get_data_info
from phrase_procedure.run_pretrain import pretrain_main
from phrase_procedure.run_tuning import tuning_main
from phrase_procedure.run_network import network_main
from utils import load_config_data
from sqlite_command import select_path_from_no, select_tuning_from_pretrain

from rank import calculate_tuning_GE

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    """ main func """
    profiler = Profiler()
    print(type(profiler))
    profiler.start()

    """ init config """
    cfg = load_config_data('init_data/FESH_config.yml')

    """ pretrain """
    cfg['pretrain']['augmentation'] = {'data_window_filter': 3}
    cfg['pretrain']['train_num'] = 50000
    pretrain_no = pretrain_main(cfg)

    """ tuning """
    cfg['tuning']['epoch'] = 200
    GE_epoch_list = [i for i in range(1, 201, 1)]
    pretrain_path = select_path_from_no(pretrain_no)
    tuning_no = tuning_main(cfg, pretrain_path, GE_epoch_list)

    profiler.stop()
    result = profiler.output_text(unicode=True, color=True, show_all=True, )
    print(result)

    profiler.write_html('profile_cost.html', show_all=True)

    """ network """
    # for i in range(2):
    #     network_no = network_main(cfg, [20, 30, 40, 50, 60, 70])
    #     print(network_no)

    network_no = network_main(cfg, [20, 30, 40, 50, 60, 70])
    print(network_no)

    # for i in range(7):
    #     network_no = network_main(cfg, [40, 50, 60, 80, 100])  # EM
    #     print(network_no)

    # network_no = network_main(cfg, [20, 40, 60, 80, 100])  # EM mlp
    # print(network_no)

    """ change GE """
    no_list = select_tuning_from_pretrain(pretrain_no)
    # print(len(no_list))
    calculate_tuning_GE(no)


if __name__ == '__main__':
    main()
