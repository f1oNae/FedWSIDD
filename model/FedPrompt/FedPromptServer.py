from copy import deepcopy
import numpy as np
import torch
from model.prompter import Prompter
from model.Server import ServerBase
from model.FedPrompt.FedPromptClient import FedPromptAgent
import os


class FedPrompt(ServerBase):
    def __init__(self, args, logger):
        super(FedPrompt, self).__init__(args, logger)
        self.setup_clients()
        self.get_data_weight()
        dfp_dict = {'init': args.prompt_initialisation,
                    'number_prompts': args.number_prompts,
                    'prompt_aggregation': args.prompt_aggregation,
                    'prompt_size': self.global_model.size[0]}
        self.global_prompt = Prompter(dfp_dict)

    def get_data_weight(self):
        n_clnt = len(self.train_dataset)
        weight_list = np.asarray([len(self.train_dataset[i]) for i in range(n_clnt)])
        self.weight_list = weight_list / np.sum(weight_list) * n_clnt

    def setup_clients(self):
        MIL_pool = []
        init_model = deepcopy(self.global_model)
        self.client_init_model = []
        if self.args.heter_model:
            MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']  # more will be added
            init_model = None
        for idx in self.n_clients:
            curr_client = FedPromptAgent(self.args, init_model, self.logger, MIL_pool)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)
            print(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            self.logger.info(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            if self.args.heter_model and len(MIL_pool) > 0:
                MIL_pool.remove(curr_client.local_model_name)
            if len(MIL_pool) == 0:
                MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']
            self.client_init_model.append(deepcopy(curr_client.local_model))

    def aggregate_prompt(self, prompt_weight, method='average', weighted=False):
        if method == 'average':
            for client_idx, client_prompt in enumerate(prompt_weight):
                if client_idx == 0:
                    self.global_prompt.prompt_embeddings.data = client_prompt[0].prompt_embeddings.data * self.weight_list[client_idx] if weighted else client_prompt[0].prompt_embeddings.data
                else:
                    self.global_prompt.prompt_embeddings.data += client_prompt[0].prompt_embeddings.data * self.weight_list[client_idx] if weighted else client_prompt[0].prompt_embeddings.data
        else:
            raise NotImplementedError

    def run(self, iter):
        # include training and testing
        train_loss = []
        best_accuracy = 0.
        train_acc_wt = 0.
        best_accuracy_per_agent = []
        normalised_weight_list = self.weight_list / np.sum(self.weight_list)
        print('Global epochs ', self.args.global_epochs)
        for epoch in range(self.args.global_epochs):
            global_info = f'===========Global epoch {epoch}===========' if epoch > 0 else f'===========Global epoch {epoch} [Init Prompt]==========='
            print(global_info)
            self.logger.info(global_info)

            local_weights, local_prompt, local_acc, local_fpr, local_tpr = [], [], [], [], []
            self.logger.info(f'| Training local model + prompt |')
            # each client trains their local model+local prompt
            # local_model_train = True if epoch < 1 else False
            local_model_train = True
            for idx in self.n_clients:
                if epoch > 0:
                    self.clients[idx].update_prompt(self.global_prompt)
                # local_lr = 3e-4 if idx == 0 else 3e-3
                # local_lr = 1e-3 if idx == 0 else 1e-3
                # local_lr = 1e-2 if idx == 0 else 1e-2
                # local_lr = 1e-1 if idx == 0 else 1e-1
                local_lr = self.args.prompt_lr
                w, prompt_w, agent_acc, agent_fpr, agent_tpr = self.clients[idx].local_train(idx,
                                                                       normalised_weight_list[idx],
                                                                       epoch=None,
                                                                       local_model_train=local_model_train,
                                                                       local_lr=local_lr)
                local_weights.append(deepcopy(w))
                local_acc.append(deepcopy(agent_acc))
                local_prompt.append(deepcopy(prompt_w))
                local_fpr.append(deepcopy(agent_fpr))
                local_tpr.append(deepcopy(agent_tpr))
                logging_info = f'Agent: {idx}, Test Acc: {agent_acc}'
                self.logger.info(logging_info)
            self.logger.info(f' Avg Test Acc: {np.mean(local_acc)}\n')
            # server aggregates the local prompt
            # TODO： Efficient Prompt Aggregation
            #   1. Single average prompt (current)
            #   2. Prompt aggregation with attention
            #   3. Prompt aggregation with learning (Existed in the paper)
            self.aggregate_prompt(local_prompt, method='average', weighted=True)
            if np.mean(local_acc) > best_accuracy:
                best_accuracy = np.mean(local_acc)
                best_accuracy_per_agent = local_acc
                list_acc_wt = [0] * len(self.n_clients)
                for i in range(len(self.n_clients)):
                    list_acc_wt[i] = local_acc[i] * self.weight_list[i]
                train_acc_wt = sum(list_acc_wt)
                for idx in self.n_clients:
                    best_model = deepcopy(self.clients[idx].local_model)
                    best_prompt = deepcopy(self.clients[idx].prompter_gather)
                    best_model_save_pth = os.path.join(self.args.results_dir, "best_model_client_%d.pt" % idx)
                    best_prompt_save_pth = os.path.join(self.args.results_dir, "best_prompt_client_%d.pt" % idx)
                    torch.save(best_model.state_dict(), best_model_save_pth)
                    torch.save(best_prompt, best_prompt_save_pth)
                    agent_fpr_save_pth = os.path.join(self.args.results_dir, f'agent_{idx}_iter_{iter}_fpr.npy')
                    agent_tpr_save_pth = os.path.join(self.args.results_dir, f'agent_{idx}_iter_{iter}_tpr.npy')
                    np.save(agent_fpr_save_pth, local_fpr[idx])
                    np.save(agent_tpr_save_pth, local_tpr[idx])

            if self.args.renew_train:
                for idx in self.n_clients:
                    self.clients[idx].local_model = deepcopy(self.client_init_model[idx])

        # self.logger.info(f'| Direct Test with global prompt |')
        # list_acc, list_loss = [], []
        # for idx in self.n_clients:
        #     self.clients[idx].load_local_model(torch.load(best_model_save_pth))
        #     self.clients[idx].update_prompt(torch.load(best_prompt_save_pth))
        #     agent_loss, agent_error = self.clients[idx].local_test()
        #     logging_info = f'Agent: {idx}, Test Acc: {1 - agent_error}'
        #     list_acc.append(1-agent_error)
        #     self.logger.info(logging_info)
        # self.logger.info(f' Avg Test Acc: {np.mean(list_acc)}\n')
        #
        # self.logger.info(f'| Quick Finetune with global prompt |')
        # list_acc, list_loss = [], []
        # for idx in self.n_clients:
        #     self.clients[idx].load_local_model(torch.load(best_model_save_pth))
        #     self.clients[idx].update_prompt(torch.load(best_prompt_save_pth))
        #     w, prompt_w, agent_acc = self.clients[idx].local_train(idx, normalised_weight_list[idx], epoch=5)
        #     logging_info = f'Agent: {idx}, Test Acc: {agent_acc}'
        #     list_acc.append(agent_acc)
        #     self.logger.info(logging_info)
        # self.logger.info(f' Avg Test Acc: {np.mean(list_acc)}\n')

        return best_accuracy, train_acc_wt, best_accuracy_per_agent