import torch
import torch.nn as nn


def add_projection_head(model, config):
    projection_list = []
    dim_mlp = model.fc_end.in_features
    if config["projection_head_layer"] > 0:
        projection_list.append(nn.Linear(dim_mlp, config["out_dim"]))
        for i in range(config["projection_head_layer"] - 1):
            projection_list.append(nn.ReLU())
            projection_list.append(nn.Linear(config["out_dim"], config["out_dim"]))
    else:
        model.fullc2_relu = nn.Sequential()
        config["out_dim"] = dim_mlp
    print(projection_list)
    model.fc_end = nn.Sequential(*projection_list)
    return model


def copy_model_for_classification(model, refer_model_file, frozen, add_dense=False):
    # add two denses layer
    if add_dense:
        dim_mlp = model.fc_end.in_features
        model.fc_end = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc_end)

    if refer_model_file is not None:
        model_dict = model.state_dict()
        pretrain_dict = torch.load(refer_model_file + 'checkpoint.tar', map_location=torch.device('cpu'))['state_dict']

        # create dict contains pretrain_state_dict without fc_end, and contains fc_end of base_model_state_dict
        new_dict = {k: v for k, v in pretrain_dict.items() if not k.startswith("fc_end")}
        new_dict.update({k: v for k, v in model_dict.items() if k.startswith("fc_end")})

        # use new_dict to update model_dict whose type is OrderedDict
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

    # frozen layers except fc_end
    if frozen:
        for name, value in model.named_parameters():
            if not name.startswith('fc_end'):
                value.requires_grad = False

    return model


def copy_model(model, net_dir):
    model_dict = model.state_dict()
    pretrain_dict = torch.load(net_dir)['state_dict']

    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    return model
