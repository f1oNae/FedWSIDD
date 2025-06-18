from copy import deepcopy
import numpy as np
import torch
from model.Server import ServerBase
from model.FedProto.FedProtoClient import FedProtoAgent
import os


def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label

class FedProto(ServerBase):
    def __init__(self, args, logger):
        super(FedProto, self).__init__(args, logger)
        self.setup_clients()

    def setup_clients(self):
        MIL_pool = None
        init_model = deepcopy(self.global_model)
        if self.args.heter_model:
            MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']  # more will be added
            init_model = None
        for idx in self.n_clients:
            curr_client = FedProtoAgent(self.args, init_model, self.logger, MIL_pool)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)
            print(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            self.logger.info(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            if self.args.heter_model:
                MIL_pool.remove(curr_client.local_model_name)

    def run(self, iter):
        # include training and testing
        global_protos = []
        train_loss = []
        best_accuracy = 0.
        train_acc_wt = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_weights, local_losses, local_protos = [], [], {}
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            for idx in self.n_clients:
                w, agent_loss, protos = self.clients[idx].local_train(idx, global_protos)
                local_weights.append(deepcopy(w))
                local_losses.append(deepcopy(agent_loss))
                agg_protos = agg_func(protos)
                local_protos[idx] = agg_protos
                print(f'Agent {idx} has protos of size {agg_protos[0].size()}')

            # update global weights
            # self.aggregate_parameter(local_weights, method='average')
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # update global weights
            global_protos = proto_aggregation(local_protos)
            for cls_i in global_protos:
                print(f'Class {cls_i} has {len(global_protos[cls_i])} protos of size {global_protos[cls_i][0].size()}')
            # Calculate avg training accuracy over all users at every epoch

        # test without proto
        list_acc, list_loss = [], []
        for idx in self.n_clients:
            agent_loss, agent_error, fpr, tpr = self.clients[idx].local_test()
            list_acc.append(1-agent_error)
            list_loss.append(agent_loss)
            agent_fpr_save_pth = os.path.join(self.args.results_dir, f'agent_{idx}_iter_{iter}_fpr.npy')
            agent_tpr_save_pth = os.path.join(self.args.results_dir, f'agent_{idx}_iter_{iter}_tpr.npy')
            np.save(agent_fpr_save_pth, fpr)
            np.save(agent_tpr_save_pth, tpr)

        train_acc_noproto = sum(list_acc) / len(list_acc)
        self.logger.info(f' \n[WITHOUT PROTO]:')
        self.logger.info(f'Training Loss : {np.mean(np.array(train_loss))}')
        self.logger.info('Train Accuracy: {:.2f}% \n'.format(100 * train_acc_noproto))
        list_acc_wt = [0] * len(self.n_clients)
        for i in range(len(self.n_clients)):
            list_acc_wt[i] = list_acc[i] * self.weight_list[i]
        train_acc_wt = sum(list_acc_wt)

        # test with proto
        # list_acc, list_loss = [], []
        # for idx in self.n_clients:
        #     agent_loss, agent_error = self.clients[idx].local_test_proto(global_protos)
        #     agent_acc = 1 - agent_error
        #     list_acc.append(agent_acc)
        #     list_loss.append(agent_loss)
        # train_acc_proto = sum(list_acc) / len(list_acc)
        # self.logger.info(f' \n[USING PROTO]')
        # self.logger.info(f'Training Loss : {np.mean(np.array(train_loss))}')
        # self.logger.info('Train Accuracy: {:.2f}% \n'.format(100 * train_acc_proto))

        best_model = deepcopy(self.global_model)
        torch.save(best_model.state_dict(), best_model_save_pth)

        return train_acc_noproto, train_acc_wt, list_acc