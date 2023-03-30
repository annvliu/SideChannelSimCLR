import h5py
import numpy as np
import pdb
from torch.utils.tensorboard import SummaryWriter
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys

from sqlite_command import select_path_from_no

sys.setrecursionlimit(100000)  # 修改递归深度


class max_clique:
    def __init__(self, graph):
        self.graph = graph
        self.cnt = np.zeros(graph.shape[0], dtype=int)  # 第i个数表示包含点i，点在i~n内的团的最大点数
        self.clique = np.zeros(graph.shape[0], dtype=int)  # 存储结果的团的点
        self.visit = np.zeros(graph.shape[0], dtype=int)  # 当前团中点的访问顺序
        self.clique_num = None

    def dfs(self, pos, num):
        """
        深搜 查询从pos开始，包含pos，从pos~n的所有团内点
        :param pos: 当前团的最新点
        :param num: 当前团包含点数
        :return: 是否构成团
        """
        for i in range(pos + 1, self.graph.shape[0], 1):
            if self.cnt[i] + num <= self.clique_num:
                return False

            if self.graph[pos][i] == 1:
                j = 0
                while j < self.graph.shape[0]:
                    if self.graph[i][self.visit[j]] == 0:
                        break
                    j += 1
                if j == self.graph.shape[0]:
                    self.visit[num] = i
                    if self.dfs(i, num + 1):
                        return True

        # 只有pos为团内最后一个点才会进入该if
        if num > self.clique_num:
            for i in range(0, num):
                self.clique[i] = self.visit[i]
            self.clique_num = num
            return True

        return False

    def demo(self):
        self.clique_num = -1
        for i in range(self.graph.shape[0] - 1, -1, -1):
            self.visit[0] = i
            self.dfs(i, 1)
            self.cnt[i] = self.clique_num
        return self.clique_num


def sample_distance(sample1, sample2):
    if len(sample1.shape) > 1:  # 有增强视图
        view_nums1 = sample1.shape[0]
        min_result = min([np.min(np.linalg.norm(sample1[i] - sample2, axis=1, ord=2)) for i in range(view_nums1)])
        return min_result
    else:
        return np.linalg.norm(sample1 - sample2, ord=2)


def distance_matrix(samples):
    sample_num = samples.shape[0]
    dis_matrix = np.zeros((sample_num, sample_num), dtype=float)
    for i in range(sample_num):
        for j in range(i, sample_num):
            if i == j:
                dis_matrix[i, i] = 0
            else:
                dis_matrix[i, j] = sample_distance(samples[i], samples[j])
                dis_matrix[j, i] = dis_matrix[i, j]
    return dis_matrix


def calculate_dataset_info(sample, label, path):
    # trs分类
    classified_trs = [sample[label == i] for i in range(np.max(label) + 1)]
    print("分类后trs shape", [item.shape for item in classified_trs])

    # 计算距离矩阵
    classified_dis_matrix = []
    for item in tqdm(classified_trs):
        classified_dis_matrix.append(distance_matrix(item))

    with h5py.File(path + "trs" + str(sample.shape[0]) + "_classified_dis_matrix.h5", 'a') as f:
        for i, item in enumerate(classified_dis_matrix):
            f.create_dataset('label' + str(i), data=item)

    max_dis = [np.max(item) for item in classified_dis_matrix]
    np.save(path + 'classified_max_dis', np.asarray(max_dis))
    print('max_dis_for_classified', max_dis)

    # 计算均值
    classified_trs_mean = np.asarray([item.reshape(-1, item.shape[2]).mean(axis=0) for item in classified_trs])
    print("mean shape:", classified_trs_mean.shape)

    mean_dis_matrix = distance_matrix(np.asarray(classified_trs_mean))
    np.save(path + 'class_dis', mean_dis_matrix)


def calculate_gener(classified_dis_matrix, outfile, delta):
    # 给定距离情况下计算最大团
    graph_result = []
    for item in tqdm(classified_dis_matrix):
        graph = np.asarray(
            [[1 if item[i, j] <= delta and i != j else 0 for j in range(item.shape[1])] for i in range(item.shape[0])])
        clique = max_clique(graph)
        graph_result.append(clique.demo() / graph.shape[0])

    print(graph_result)
    np.save(outfile + 'clique_for_class_delta_' + str(delta), np.asarray(graph_result))
    print(np.asarray(graph_result).mean())


def main():
    # delta = 0.0324
    # delta = 0.0114
    delta = 0.0154
    for no in [59, 60, 61]:
        path = select_path_from_no(no)

        # sample = np.load(path + 'dataset_data.npy')
        # label = np.load(path + 'dataset_label.npy')
        #
        # num = 1500
        # gener_sample = np.empty((0, sample.shape[1], sample.shape[2]), dtype=float)
        # gener_label = np.empty(0, dtype=int)
        # for label_i in range(max(label) + 1):
        #     sample_label_i = sample[label == label_i]
        #     this_num = min(num, sample_label_i.shape[0])
        #
        #     gener_sample = np.concatenate((gener_sample, sample_label_i[:this_num]))
        #     gener_label = np.concatenate((gener_label, np.asarray([label_i] * this_num)))
        #     print(gener_sample.shape, gener_label.shape)
        #
        # calculate_dataset_info(gener_sample, gener_label, path)

        dis_matrix_list = []
        with h5py.File(path + 'trs10681_classified_dis_matrix.h5', 'r') as file:
            for key in file.keys():
                dis_matrix_list.append(np.asarray(file[key]))
        calculate_gener(dis_matrix_list, path, delta)

        # clique_rate = np.load(pretrain_path + 'clique_for_class_delta_0.0248.npy')[2:7]
        # class_dis = np.load(pretrain_path + 'class_dis.npy')[2:7, 2:7]
        #
        # clique_rate = np.load(pretrain_path + 'clique_for_class_delta_0.0248.npy')[3:6]
        # class_dis = np.load(pretrain_path + 'class_dis.npy')[3:6, 3:6]
        #
        # print(pretrain_no, clique_rate.mean(), np.mean(class_dis), np.mean(class_dis) * 100 * clique_rate.mean())


if __name__ == '__main__':
    main()
