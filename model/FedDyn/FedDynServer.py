from copy import deepcopy
import numpy as np
import torch
from model.Server import ServerBase
from model.FedDyn.FedDynClient import FedDynAgent
from utils.trainer_util import get_mdl_params
import os


class FedDyn(ServerBase):
    def __init__(self, args, logger):
        super(FedDyn, self).__init__(args, logger)
        self.setup_clients()

    def setup_clients(self):
        for idx in self.n_clients:
            curr_client = FedDynAgent(self.args, self.global_model, self.logger)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)

    def run(self, iter):
        # include training and testing
        self.global_model.to(self.device)
        n_par = len(get_mdl_params([self.global_model])[0])
        local_param_list = np.zeros((len(self.n_clients), n_par)).astype('float32')  # [n_clnt X n_par]
        init_par_list = get_mdl_params([self.global_model], n_par)[0]
        clnt_params_list = np.ones(len(self.n_clients)).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                    -1)  # [n_clnt X n_par]
        cld_mdl_param = get_mdl_params([self.global_model], n_par)[0]
        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        train_acc_wt = 0.
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_weights, local_losses = [], []
            # get current global model parameters
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=self.device)  # [n_par]
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            self.global_model.train()
            for idx in self.n_clients:
                # Warm start from current global model
                self.clients[idx].update_local_model(self.global_model)
                alpha_coef_adpt = self.args.alpha_coef / self.weight_list[idx]  # adaptive alpha coef
                local_param_list_curr = torch.tensor(local_param_list[idx], dtype=torch.float32, device=self.device)
                w, agent_loss = self.clients[idx].local_train(idx,
                                                              alpha_coef_adpt,
                                                              cld_mdl_param_tensor,
                                                              local_param_list_curr)
                local_weights.append(deepcopy(w))
                local_losses.append(deepcopy(agent_loss))
                curr_model_par = get_mdl_params([self.clients[idx].local_model], n_par)[0]
                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                local_param_list[idx] += curr_model_par-cld_mdl_param
                clnt_params_list[idx] = curr_model_par

            # update global weights
            avg_mdl_param = np.mean(clnt_params_list, axis=0)
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)
            self.aggregate_parameter(avg_mdl_param, method='direct')
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.global_model.eval()
            for idx in self.n_clients:
                # test using local copy of global model
                agent_loss, agent_error, fpr, tpr = self.clients[idx].local_test()
                list_acc.append(1 - agent_error)
                list_loss.append(agent_loss)
                agent_fpr_save_pth = os.path.join(self.args.results_dir, f'agent_{idx}_iter_{iter}_fpr.npy')
                agent_tpr_save_pth = os.path.join(self.args.results_dir, f'agent_{idx}_iter_{iter}_tpr.npy')
                np.save(agent_fpr_save_pth, fpr)
                np.save(agent_tpr_save_pth, tpr)

            train_acc = sum(list_acc) / len(list_acc)
            if (epoch + 1) % 1 == 0:
                self.logger.info(f' \nAvg Training Stats after {epoch + 1} global rounds:')
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