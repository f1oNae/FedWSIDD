from copy import deepcopy
import numpy as np
import random
import torch
from model.Server import ServerBase
from model.FedMut.FedMutClient import FedMutAgent
import os


class FedMut(ServerBase):
    def __init__(self, args, logger):
        super(FedMut, self).__init__(args, logger)
        self.setup_clients()

    def setup_clients(self):
        for idx in self.n_clients:
            curr_client = FedMutAgent(self.args, self.global_model, self.logger)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)

    def Aggregation(self, w, lens):
        w_avg = None
        if lens == None:
            total_count = len(w)
            lens = []
            for i in range(len(w)):
                lens.append(1.0)
        else:
            total_count = sum(lens)

        for i in range(0, len(w)):
            if i == 0:
                w_avg = deepcopy(w[0])
                for k in w_avg.keys():
                    w_avg[k] = w[i][k] * lens[i]
            else:
                for k in w_avg.keys():
                    w_avg[k] += w[i][k] * lens[i]

        for k in w_avg.keys():
            w_avg[k] = torch.div(w_avg[k], total_count)

        return w_avg

    def FedSub(self, w, w_old, weight):
        w_sub = deepcopy(w)
        for k in w_sub.keys():
            w_sub[k] = (w[k] - w_old[k]) * weight
        return w_sub

    def delta_rank(self, args, delta_dict):
        cnt = 0
        dict_a = torch.Tensor(0)
        s = 0
        for p in delta_dict.keys():
            a = delta_dict[p]
            a = a.view(-1)
            if cnt == 0:
                dict_a = a
            else:
                dict_a = torch.cat((dict_a, a), dim=0)

            cnt += 1
            # print(sim)
        s = torch.norm(dict_a, dim=0)
        return s

    def mutation_spread(self, args, iter, w_glob, w_old, w_locals, m, w_delta, alpha):
        w_locals_new = []
        ctrl_cmd_list = []
        ctrl_rate = args.mut_acc_rate * (1.0 - min(iter * 1.0 / args.mut_bound, 1.0))

        for k in w_glob.keys():
            ctrl_list = []
            for i in range(0, int(m / 2)):
                ctrl = random.random()
                if ctrl > 0.5:
                    ctrl_list.append(1.0)
                    ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
                else:
                    ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
                    ctrl_list.append(1.0)
            random.shuffle(ctrl_list)
            ctrl_cmd_list.append(ctrl_list)
        cnt = 0
        for j in range(m):
            w_sub = deepcopy(w_glob)
            if not (cnt == m - 1 and m % 2 == 1):
                ind = 0
                for k in w_sub.keys():
                    w_sub[k] = w_sub[k] + w_delta[k] * ctrl_cmd_list[ind][j] * alpha
                    ind += 1
            cnt += 1
            w_locals_new.append(w_sub)

        return w_locals_new

    def run(self, iter):
        # include training and testing
        self.global_model.to(self.device)
        train_loss = []
        w_locals = []
        max_rank = 0
        for idx in self.n_clients:
            w_locals.append(deepcopy(self.global_model.state_dict()))
        best_accuracy = 0.
        train_acc_wt = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_weights, local_losses = [], []
            w_old = deepcopy(self.global_model.state_dict())
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            self.global_model.train()
            for idx in self.n_clients:
                self.global_model.load_state_dict(w_locals[idx])
                self.clients[idx].update_local_model(self.global_model)
                w, agent_loss = self.clients[idx].local_train(idx)
                w_locals[idx] = deepcopy(w)
                local_losses.append(deepcopy(agent_loss))
                # calculate number of element in w
                n_elements = 0
                for key in w.keys():
                    n_elements += w[key].numel()
                print(f'FedMut Comm Costs: {n_elements}')

            # update global weights
            w_glob = self.Aggregation(w_locals, None)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # dispatch global model to all clients
            self.global_model.load_state_dict(w_glob)
            w_delta = self.FedSub(w_glob, w_old, 1.0)
            rank = self.delta_rank(self.args, w_delta)
            print(rank)
            if rank > max_rank:
                max_rank = rank
            alpha = self.args.radius
            w_locals = self.mutation_spread(self.args, iter, w_glob, w_old, w_locals, len(self.n_clients), w_delta, alpha)
            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.global_model.eval()
            for idx in self.n_clients:
                self.global_model.load_state_dict(w_locals[idx])
                self.clients[idx].update_local_model(self.global_model)
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