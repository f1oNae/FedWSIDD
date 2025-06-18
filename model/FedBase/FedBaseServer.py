from copy import deepcopy
import numpy as np
import torch
from model.prompter import Prompter
from model.Server import ServerBase
from model.FedBase.FedBaseClient import FedBaseAgent
from model.FedPrompt.FedPromptClient import FedPromptAgent
import os


class FedBase(ServerBase):
    def __init__(self, args, logger):
        super(FedBase, self).__init__(args, logger)
        self.setup_clients()
        self.get_data_weight()

    def get_data_weight(self):
        n_clnt = len(self.train_dataset)
        weight_list = np.asarray([len(self.train_dataset[i]) for i in range(n_clnt)])
        self.weight_list = weight_list / np.sum(weight_list) * n_clnt

    def setup_clients(self):
        print('==============Setting up clients==============')
        MIL_pool = []
        init_model = deepcopy(self.global_model)
        if self.args.heter_model:
            MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']  # more will be added
            init_model = None
        for idx in self.n_clients:
            curr_client = FedBaseAgent(self.args, init_model, self.logger, MIL_pool)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)
            print(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            self.logger.info(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            if self.args.heter_model and len(MIL_pool) > 0:
                MIL_pool.remove(curr_client.local_model_name)
            if len(MIL_pool) == 0:
                MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']


    def run(self, iter):
        # include training and testing
        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        train_acc_wt = 0.
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        local_weights = []
        self.logger.info(f'| Local Training Round |')
        local_model_train = True
        for idx in self.n_clients:
            w, agent_loss = self.clients[idx].local_train(idx)
            local_weights.append(deepcopy(w))

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        for idx in self.n_clients:
            agent_loss, agent_error, fpr, tpr = self.clients[idx].local_test()
            list_acc.append(1-agent_error)
            list_loss.append(agent_loss)
            agent_fpr_save_pth = os.path.join(self.args.results_dir, f'agent_{idx}_iter_{iter}_fpr.npy')
            agent_tpr_save_pth = os.path.join(self.args.results_dir, f'agent_{idx}_iter_{iter}_tpr.npy')
            np.save(agent_fpr_save_pth, fpr)
            np.save(agent_tpr_save_pth, tpr)


        train_acc = sum(list_acc) / len(list_acc)
        self.logger.info(f'Training Loss : {np.mean(np.array(train_loss))}')
        self.logger.info('Train Accuracy: {:.2f}% \n'.format(100 * train_acc))
        if train_acc > best_accuracy:
            list_acc_wt = [0] * len(self.n_clients)
            for i in range(len(self.n_clients)):
                list_acc_wt[i] = list_acc[i] * self.weight_list[i]
            train_acc_wt = sum(list_acc_wt)
            best_accuracy = train_acc
            best_accuracy_per_agent = list_acc
            best_model = deepcopy(self.global_model)
            torch.save(best_model.state_dict(), best_model_save_pth)

        return best_accuracy, train_acc_wt, best_accuracy_per_agent