from copy import deepcopy
import numpy as np
import torch
from collections import defaultdict
from model.Server import ServerBase
from model.SGPT.SGPTClient import SGPTAgent
from model.FExtractor.vit_prompt import VisionTransformer_m, CONFIGS
import os

def group_ratio_func(keys_dict, selected):
    group_ratio = {}
    group_total = {}
    for r in selected:
        cluster_r = keys_dict[r]['cluster_size']
        for key in cluster_r.keys():
            if key in group_total:
                group_total[key] += cluster_r[key]
            else:
                group_total[key] = cluster_r[key]

    for r in selected:
        group_ratio[r] = {}
        cluster_r = keys_dict[r]['cluster_size']
        for key,val in cluster_r.items():
            if group_total[key] <= 50:
                group_ratio[r][key] = 1/len(selected)
            else:
                group_ratio[r][key] = val/group_total[key]
    return group_ratio

def aggregation_func(keys_dict,global_para,selected,fed_avg_freqs,group_ratio,args):
    unique_dict = {}
    # unique_dict['prompt_keys'] = copy.deepcopy(global_para['prompt_keys'])
    # for idx,r in enumerate(selected):
    for idx in selected:
        net_para = keys_dict[idx]
        if idx == 0:
            for key in net_para:
                if  'prompt_embeddings' in key or 'prompt_keys' in key :
                    unique_dict[key] = deepcopy(global_para[key])
                if ('head' in key or 'prompt_embeddings' in key  or 'prompt_common' in key or 'running_mean' in key) and ('prompt_common_g' not in key) :
                    global_para[key] = deepcopy(net_para[key]) * fed_avg_freqs[idx]
                elif ( 'prompt_keys' in key ) and group_ratio is not None:
                    for ii, gs in enumerate(group_ratio[idx].keys()):
                        global_para[key][ii:ii+1] = net_para[key][ii:ii+1]*group_ratio[idx][gs]
        else:
            # or 'prompt_embeddings' in key
            for key in net_para:
                if ('head' in key  or 'prompt_embeddings' in key  or 'prompt_common' in key or 'running_mean' in key)  and ('prompt_common_g' not in key):
                    global_para[key] += deepcopy(net_para[key]) * fed_avg_freqs[idx]
                elif ( 'prompt_keys' in key ) and group_ratio is not None:
                    for ii, gs in enumerate(group_ratio[idx].keys()):
                        global_para[key][ii:ii+1] += net_para[key][ii:ii+1]*group_ratio[idx][gs]
    for key in unique_dict.keys():
        #### momentum
        if  'prompt_embeddings' in key:
            global_para[key] = 0.5*unique_dict[key] + (1-0.5)*global_para[key]
        if  'prompt_keys' in key :
            # global_para[key] = args.moment*unique_dict[key] + (1-args.moment)*global_para[key]
            global_para[key] = global_para[key]

    return global_para

class SGPT(ServerBase):
    def __init__(self, args, logger):
        super(SGPT, self).__init__(args, logger)
        self.setup_clients()

    def setup_clients(self):
        MIL_pool = None
        init_model = deepcopy(self.global_model)
        if self.args.heter_model:
            MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']  # more will be added
            init_model = None
        for idx in self.n_clients:
            curr_client = SGPTAgent(self.args, init_model, self.logger, MIL_pool)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)
            print(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            self.logger.info(f'=====> Agent {idx} uses {curr_client.local_model_name}')
            if self.args.heter_model:
                MIL_pool.remove(curr_client.local_model_name)

    def run(self, iter):
        # include training and testing
        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)

        local_weights, local_losses = [], []
        self.logger.info(f'| Local Training Round |')
        for idx in self.n_clients:
            w, agent_loss = self.clients[idx].local_train(idx)
            local_weights.append(deepcopy(w))
            local_losses.append(deepcopy(agent_loss))

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        for idx in self.n_clients:
            agent_loss, agent_error, fpr, tpr = self.clients[idx].local_test()
            list_acc.append(1 - agent_error)
            list_loss.append(agent_loss)
            agent_fpr_save_pth = os.path.join(self.args.results_dir, f'agent_{idx}_iter_{iter}_fpr.npy')
            agent_tpr_save_pth = os.path.join(self.args.results_dir, f'agent_{idx}_iter_{iter}_tpr.npy')
            np.save(agent_fpr_save_pth, fpr)
            np.save(agent_tpr_save_pth, tpr)

        train_acc = sum(list_acc) / len(list_acc)

        self.logger.info(f'Training Loss : {np.mean(np.array(train_loss))}')
        self.logger.info('Train Accuracy: {:.2f}% \n'.format(100 * train_acc))
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            best_accuracy_per_agent = list_acc
            best_model = deepcopy(self.global_model)
            torch.save(best_model.state_dict(), best_model_save_pth)

        return best_accuracy, best_accuracy_per_agent
    # def __init__(self, args, logger):
    #     super(SGPT, self).__init__(args, logger)
    #     self.setup_clients()
    #     ###### Model related ######
    #     config = CONFIGS[args.ft_model]
    #     self.global_model = VisionTransformer_m(config, args.image_size, num_classes=args.n_classes, vis=True, args=args)
    #     self.global_model.load_from(np.load(args.pretrained_dir))
    #     self.global_model.freeze()
    #     self.global_model.to(self.device)
    #     self.global_model.selection = True
    #
    # def setup_clients(self):
    #     for idx in self.n_clients:
    #         curr_client = SGPTAgent(self.args, self.global_model, self.logger)
    #         curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
    #         self.clients.append(curr_client)
    #
    # def run(self, it):
    #     global_para = {k: deepcopy(v) for k, v in self.global_model.state_dict().items() if 'head' in k or 'prompt' in k}
    #     keys_dict = {}
    #     cluster_dict = {}
    #     cluster_dict_all = {}
    #     best_accuracy = 0.
    #     best_accuracy_per_agent = []
    #     for net_id in self.n_clients:
    #         keys_dict[net_id] = keys_dict[net_id] = {k: deepcopy(v) for k, v in global_para.items()}
    #         cluster_dict[net_id] = {}
    #         cluster_dict_all[net_id] = {i: 0 for i in range(self.args.key_prompt)}
    #     global_select = {i: 0 for i in range(self.args.key_prompt)}
    #     embedding_dicts = {i: {} for i in range(len(self.n_clients))}
    #     dict_loss = {}
    #     results_dict = defaultdict(list)
    #     groups_list = []
    #     pr_label_pr = {}
    #     for epoch in range(self.args.global_epochs):
    #         self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
    #         for idx in self.n_clients: # since we dont have many clients, we use all clients
    #             param_dict = {}
    #             keys_dict[idx] = {k: deepcopy(v) for k, v in global_para.items()}
    #             self.global_model.load_state_dict(keys_dict[idx], strict=False)
    #             self.global_model.cluster_size = {i: 0 for i in range(self.args.key_prompt)}
    #             self.global_model.cluster_size_g = global_select
    #             self.global_model.cluster_size_l = cluster_dict_all[idx]
    #             avg_acc = 0.0
    #
    #             param_dict['dict_loss'] = dict_loss
    #             param_dict['round'] = epoch
    #             param_dict['group_label'] = pr_label_pr
    #             param_dict['embedding_dict'] = embedding_dicts[idx]
    #
    #             embedding_dict = self.clients[idx].local_train(self.global_model, param_dict)
    #             embedding_dicts[idx] = deepcopy(embedding_dict)
    #             for k, v in self.global_model.state_dict().items():
    #                 if 'head' in k or 'prompt' in k:
    #                     keys_dict[idx][k] = deepcopy(v)
    #             cluster_dict[idx]['cluster_size'] = deepcopy(self.global_model.cluster_size)
    #             for key in cluster_dict[idx]['cluster_size'].keys():
    #                 global_select[key] += cluster_dict[idx]['cluster_size'][key]
    #                 cluster_dict_all[idx][key] = cluster_dict[idx]['cluster_size'][key]
    #             # save local model
    #             best_model_save_pth = os.path.join(self.args.results_dir, f"last_model_client_{idx}_iter_{it}.pt")
    #             torch.save(self.clients[idx].local_model.state_dict(), best_model_save_pth)
    #
    #         group_ratio = group_ratio_func(cluster_dict, self.n_clients)
    #         global_para = aggregation_func(keys_dict, global_para, self.n_clients, self.weight_list, group_ratio, self.args)
    #         self.global_model.load_state_dict(global_para, strict=False)
    #         best_model_save_pth = os.path.join(self.args.results_dir, f"last_global_vit_prompt_iter_{it}.pt")
    #         torch.save(self.global_model.state_dict(), best_model_save_pth)
    #
    #         # list_acc, list_loss = [], []
    #         # for idx in self.n_clients:
    #         #     agent_loss, agent_error = self.clients[idx].local_test(self.global_model)
    #         #     list_acc.append(1-agent_error)
    #         #     list_loss.append(agent_loss)
    #         # train_acc = sum(list_acc) / len(list_acc)
    #         # if (epoch + 1) % 1 == 0:
    #         #     self.logger.info(f' \nAvg Training Stats after {epoch + 1} global rounds:')
    #         #     self.logger.info('Train Accuracy: {:.2f}% \n'.format(100 * train_acc))
    #         #     if train_acc > best_accuracy:
    #         #         best_accuracy = train_acc
    #         #         best_accuracy_per_agent = list_acc
    #         #         best_model = deepcopy(self.global_model)
    #
    #     return best_accuracy, best_accuracy_per_agent


