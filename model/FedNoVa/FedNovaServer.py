from copy import deepcopy
import numpy as np
import torch
from model.Server import ServerBase
from model.FedNoVa.FedNovaClient import FedNovaAgent
import os


class FedNoVa(ServerBase):
    def __init__(self, args, logger):
        super(FedNoVa, self).__init__(args, logger)
        self.setup_clients()

    def setup_clients(self):
        for idx in self.n_clients:
            curr_client = FedNovaAgent(self.args, self.global_model, self.logger)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)

    def run(self, iter):
        # include training and testing
        self.global_model.to(self.device)
        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)

        for epoch in range(self.args.global_epochs):
            local_weights, local_losses, local_coef, local_norm_grad = [], [], [], []
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            self.global_model.train()
            for idx in self.n_clients:
                w, agent_loss, coeff, norm_grad = self.clients[idx].local_train(idx)
                local_weights.append(deepcopy(w))
                local_losses.append(deepcopy(agent_loss))
                local_coef.append(deepcopy(coeff))
                local_norm_grad.append(deepcopy(norm_grad))

            # update global weights
            self.aggregate_parameter(local_weights, method='nova', coeff=local_coef, norm_grad=local_norm_grad)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # dispatch global model to all clients
            for idx in self.n_clients:
                self.clients[idx].update_local_model(self.global_model)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.global_model.eval()
            for idx in self.n_clients:
                agent_loss, agent_error = self.clients[idx].local_test()
                list_acc.append(1-agent_error)
                list_loss.append(agent_loss)
            train_acc = sum(list_acc) / len(list_acc)
            if (epoch + 1) % 1 == 0:
                self.logger.info(f' \nAvg Training Stats after {epoch + 1} global rounds:')
                self.logger.info(f'Training Loss : {np.mean(np.array(train_loss))}')
                self.logger.info('Train Accuracy: {:.2f}% \n'.format(100 * train_acc))
                if train_acc > best_accuracy:
                    best_accuracy = train_acc
                    best_accuracy_per_agent = list_acc
                    best_model = deepcopy(self.global_model)
                    torch.save(best_model.state_dict(), best_model_save_pth)

        return best_accuracy, best_accuracy_per_agent