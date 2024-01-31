import torch.nn as nn
import torch


class Classic_MLP(nn.Module):  # VGG
    def __init__(self, out_dim, point_num):
        super().__init__()

        self.dense1 = nn.Sequential(
            nn.Linear(point_num, 256),
            nn.ReLU(),
        )

        self.dense2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.dense3 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
        )

        self.dense4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.fc_end = nn.Linear(1024, out_dim)

    def forward(self, x):
        out = x.to(torch.float32)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.dense3(out)
        out = self.dense4(out)
        out = self.fc_end(out)
        return out


class cnn_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cnn_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=11, stride=1, padding=5),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )

    def forward(self, x):
        y = self.conv1(x)
        return y


class ascad_cnn_best(nn.Module):
    def __init__(self, out_dim, point_num, dense_input):
        super(ascad_cnn_best, self).__init__()

        self.bn = nn.Sequential(
            nn.BatchNorm1d(point_num)
        )

        self.cnn1 = cnn_block(1, 64)
        self.cnn2 = cnn_block(64, 128)
        self.cnn3 = cnn_block(128, 256)
        self.cnn4 = cnn_block(256, 512)
        self.cnn5 = cnn_block(512, 512)

        self.fullc1 = nn.Sequential(
            nn.Linear(dense_input, 4096),
            nn.ReLU(),
        )
        self.fullc2 = nn.Sequential(
            nn.Linear(4096, 4096),
        )
        self.fullc2_relu = nn.Sequential(
            nn.ReLU(),
        )
        self.fc_end = nn.Linear(4096, out_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(torch.float32)
        x = self.bn(x)
        x = x.view(batch_size, 1, -1)

        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)

        x = x.view(batch_size, -1)
        x = self.fullc1(x)
        x = self.fullc2(x)
        x = self.fullc2_relu(x)
        x = self.fc_end(x)
        return x


class cnn_block_BN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cnn_block_BN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.AvgPool1d(2)
        )

    def forward(self, x):
        y = self.conv1(x)
        return y


class ascad_cnn_best_BN(nn.Module):
    def __init__(self, out_dim, point_num, dense_input):
        super(ascad_cnn_best_BN, self).__init__()

        self.bn = nn.Sequential(
            nn.BatchNorm1d(point_num)
        )

        self.cnn1 = cnn_block_BN(1, 64)
        self.cnn2 = cnn_block_BN(64, 128)
        self.cnn3 = cnn_block_BN(128, 256)
        self.cnn4 = cnn_block_BN(256, 512)
        self.cnn5 = cnn_block_BN(512, 512)

        self.fullc1 = nn.Sequential(
            nn.Linear(dense_input, 4096),
            nn.ReLU(),
        )
        self.fullc2 = nn.Sequential(
            nn.Linear(4096, 4096),
        )
        self.fullc2_relu = nn.Sequential(
            nn.ReLU(),
        )
        self.fc_end = nn.Linear(4096, out_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(torch.float32)
        x = self.bn(x)
        x = x.view(batch_size, 1, -1)

        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)

        x = x.view(batch_size, -1)
        x = self.fullc1(x)
        x = self.fullc2(x)
        x = self.fullc2_relu(x)
        x = self.fc_end(x)
        return x


class MECNN_N0(nn.Module):  # Methodology for Efficient CNN Architectures in Profiling Attacks
    def __init__(self, out_dim, dense_input):
        super(MECNN_N0, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.SELU(),
            nn.AvgPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=50, stride=1, padding=25),
            nn.BatchNorm1d(64),
            nn.SELU(),
            nn.AvgPool1d(50)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.SELU(),
            nn.AvgPool1d(2)
        )

        self.fullc1 = nn.Sequential(
            nn.Linear(dense_input, 20),
            nn.SELU(),
        )
        self.fullc2 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU(),
        )
        self.fullc3 = nn.Sequential(
            nn.Linear(20, 20),
            nn.SELU(),
        )
        self.fc_end = nn.Linear(20, out_dim)

        self.initialize()

    def initialize(self):
        for layers in self.modules():
            if isinstance(layers, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(layers.weight)
                nn.init.constant_(layers.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(torch.float32)
        x = x.view(batch_size, 1, -1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(batch_size, -1)
        x = self.fullc1(x)
        x = self.fullc2(x)
        x = self.fullc3(x)
        x = self.fc_end(x)
        return x


def simclr_net(config: dict):
    """ Choose model with model_type """
    net_dict = {"ascad_cnn_block_anti_bn": ascad_cnn_best(out_dim=config["out_dim"],
                                                          point_num=config['common']["feature_num"],
                                                          dense_input=config['common'][
                                                              "ascad_cnn_block_anti_bn_dense_input"]),
                "ascad_cnn": ascad_cnn_best_BN(out_dim=config["out_dim"],
                                               point_num=config['common']["feature_num"],
                                               dense_input=config['common']["ascad_cnn_dense_input"]),
                "classic_mlp": Classic_MLP(out_dim=config["out_dim"],
                                           point_num=config['common']["feature_num"]),
                "MECNN_N0": MECNN_N0(out_dim=config["out_dim"],
                                     dense_input=config['common']["MECNN_N0_dense_input"])}

    model = net_dict[config['common']["model_name"]]
    return model
