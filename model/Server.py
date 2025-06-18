import torch
import os
import numpy as np
import h5py
import copy
import torch.nn.functional as F
import time
import torch.nn as nn
from utils.Get_model import define_model
from utils.Get_data import define_data
from utils.trainer_util import get_optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ServerBase:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info(' '.join(f'--{k}={v} \n' for k, v in vars(args).items()))
        self.device = device
        require_image_function = True if self.args.fed_method in ['fed_dm', 'fed_af'] else False
        image_size = args.image_size if args.fed_method == 'fed_desa' else args.syn_size
        self.train_dataset, self.test_dataset, self.n_clients = define_data(args,
                                                                            logger,
                                                                            require_image=require_image_function,
                                                                            image_size=args.image_size,
                                                                            top_k=args.top_k)
        self.get_data_weight()
        self.clients = []
        self.global_model = define_model(args)
        self.optimizer = get_optim(self.args, self.global_model)
        # print the number of parameters in the model
        no_trainable_parameters = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
        print('Number of trainable parameters:', no_trainable_parameters)
        self.scv = define_model(args)  # server control variate for FedScaffold
        print('Number of clients:', len(self.n_clients), self.weight_list)

    def get_number_of_parameters(self):
        pytorch_total_params = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
        return pytorch_total_params

    def get_data_weight(self):
        n_clnt = len(self.train_dataset)
        weight_list = np.asarray([len(self.train_dataset[i]) for i in range(n_clnt)])
        # normalize the weight to sum to 1
        self.weight_list = weight_list / np.sum(weight_list) # normalize the weight

    def send_parameters(self):
        return self.global_model.state_dict()

    def aggregate_parameter(self, model_weight, method='average', coeff=None, norm_grad=None):
        if method == 'average':
            w_avg = copy.deepcopy(model_weight[0])
            for key in w_avg.keys():
                for i in range(1, len(model_weight)):
                    w_avg[key] += model_weight[i][key]
                w_avg[key] = torch.div(w_avg[key], len(model_weight))
            self.global_model.load_state_dict(w_avg)
        elif method == 'direct':
            dict_param = copy.deepcopy(dict(self.global_model.named_parameters()))
            idx = 0
            for name, param in self.global_model.named_parameters():
                weights = param.data
                length = len(weights.reshape(-1))
                dict_param[name].data.copy_(torch.tensor(model_weight[idx:idx + length].reshape(weights.shape)).to(device))
                idx += length
            self.global_model.load_state_dict(dict_param, strict=False)
        elif method == 'weighted':
            global_w = self.global_model.state_dict()
            for net_id, net_para in enumerate(model_weight):
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * self.weight_list[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * self.weight_list[net_id]
            self.global_model.load_state_dict(global_w, strict=False)
        elif method == 'noise':
            default_noise = 0.001 # follow HistoFL setup
            global_w = self.global_model.state_dict()
            for net_id, net_para in enumerate(model_weight):
                for key in net_para:
                    if net_id == 0:
                        if 'bias' not in key:
                            noise = default_noise * torch.empty(net_para[key].size()).normal_(mean=0,
                                                                                              std=net_para[key].reshape(
                                                                                                  -1).float().std())
                            global_w[key] = net_para[key] * self.weight_list[net_id] + noise.to(self.device)
                        else:
                            global_w[key] = net_para[key] * self.weight_list[net_id]
                    else:
                        if 'bias' not in key:
                            noise = default_noise * torch.empty(net_para[key].size()).normal_(mean=0,
                                                                                              std=net_para[key].reshape(
                                                                                                  -1).float().std())
                            global_w[key] += net_para[key] * self.weight_list[net_id] + noise.to(self.device)
                        else:
                            global_w[key] += net_para[key] * self.weight_list[net_id]
            self.global_model.load_state_dict(global_w, strict=False)
        elif method == 'nova':
            global_w = self.global_model.state_dict()
            nova_model_state = copy.deepcopy(global_w)
            coeff_cum = 0.0
            for net_id, net_para in enumerate(model_weight):
                coeff_cum = coeff_cum + coeff[net_id] * self.weight_list[net_id]
                if net_id == 0:
                    for key in net_para:
                        nova_model_state[key] = norm_grad[net_id][key] * self.weight_list[net_id]
                else:
                    for key in net_para:
                        nova_model_state[key] += norm_grad[net_id][key] * self.weight_list[net_id]
            for key in global_w:
                try:
                    global_w[key] = global_w[key] - coeff_cum * nova_model_state[key]
                except:
                    global_w[key] = global_w[key]
            self.global_model.load_state_dict(global_w, strict=False)
        elif method == 'scaffold':
            # i use norm_grad to represent the delta_ccv_state
            global_w = self.global_model.state_dict()
            new_scv_state = copy.deepcopy(global_w)
            for net_id, net_para in enumerate(model_weight):
                for key in net_para:
                    if net_id == 0:
                        global_w[key] = net_para[key] * self.weight_list[net_id]
                        new_scv_state[key] = norm_grad[net_id][key] * self.weight_list[net_id]
                    else:
                        global_w[key] += net_para[key] * self.weight_list[net_id]
                        new_scv_state[key] += norm_grad[net_id][key] * self.weight_list[net_id]
            # self.scv_state = self.scv.state_dict()
            self.global_model.load_state_dict(global_w)
            self.scv.load_state_dict(new_scv_state)





