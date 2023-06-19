import torch.nn as nn
import torch


class twolayers_mlp(nn.Module):  # VGG
    def __init__(self, out_dim, batch_size):
        super(twolayers_mlp, self).__init__()

        self.batch_size = batch_size

        self.fc = nn.Sequential(
            nn.Linear(1084, 128),
            nn.ReLU(),
        )
        self.fc_end = nn.Linear(128, out_dim)

    def forward(self, x):
        out = x.to(torch.float32)
        # out = out.view(x.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        out = self.fc_end(out)
        return out


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


class ches_model(nn.Module):  # VGG
    def __init__(self, out_dim, batch_size):
        super(ches_model, self).__init__()

        self.batch_size = batch_size

        self.avepool_cswap_arith_gv = nn.Sequential(
            nn.AvgPool1d(kernel_size=4, stride=4)
        )

        self.bn_cswap_arith_gv = nn.Sequential(
            nn.BatchNorm1d(271)
        )

        self.cnn_cswap_arith_gv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=(20,), stride=(1,)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,)),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=(20,), stride=(1,)),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(20,), stride=(1,)),
            nn.ReLU()
        )

        self.fc_cswap_arith_gv = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2816, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fc_end = nn.Linear(128, out_dim)

    def forward(self, x):
        out = x.view(self.batch_size, 1, -1)
        out = self.avepool_cswap_arith_gv(out)
        out = out.view(self.batch_size, -1)
        # print("shape after ave")
        # print(out.shape)
        out = out.to(torch.float32)
        out = self.bn_cswap_arith_gv(out)
        out = out.view(self.batch_size, 1, -1)
        out = self.cnn_cswap_arith_gv(out)
        out = out.view(x.size(0), -1)
        # print("shape before fc")
        # print(out.shape)
        out = self.fc_cswap_arith_gv(out)
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
                                           point_num=config['common']["feature_num"])}

    model = net_dict[config['common']["model_name"]]
    return model
