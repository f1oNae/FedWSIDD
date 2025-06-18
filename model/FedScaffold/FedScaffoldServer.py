from copy import deepcopy
import numpy as np
import torch
from model.Server import ServerBase
from model.FedScaffold.FedScaffoldClient import FedScaffoldAgent
import os


class FedScaffold(ServerBase):
    def __init__(self, args, logger):
        super(FedScaffold, self).__init__(args, logger)
        self.setup_clients()

    def setup_clients(self):
        for idx in self.n_clients:
            curr_client = FedScaffoldAgent(self.args, self.global_model, self.logger)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)

    def run(self, iter):
        # include training and testing
        self.global_model.to(self.device)
        train_loss = []
        best_accuracy = 0.
        train_acc_wt = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)

        for epoch in range(self.args.global_epochs):
            local_weights, local_losses, local_delta_ccv = [], [], []
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            self.global_model.train()
            for idx in self.n_clients:
                w, agent_loss, delta_ccv_state = self.clients[idx].local_train(idx)
                local_weights.append(deepcopy(w))
                local_losses.append(deepcopy(agent_loss))
                local_delta_ccv.append(deepcopy(delta_ccv_state))
                # calculate the number of parameter in delta_ccv_state
                print(f'Number of parameters in delta_ccv_state: {sum(p.numel() for p in delta_ccv_state.values())}')

            # update global weights
            self.aggregate_parameter(local_weights, method='scaffold', norm_grad=local_delta_ccv)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # dispatch global model to all clients
            for idx in self.n_clients:
                self.clients[idx].update(self.global_model.state_dict(), self.scv.state_dict())

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.global_model.eval()
            for idx in self.n_clients:
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